"""Anomaly scoring and explanation generation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.calibration import ScoreCalibrator
from engine.config_loader import (
    get_alert_bands,
    get_ensemble_weights,
    get_label,
    normalize_ensemble_weights,
    resolve_feature_list,
)


logger = logging.getLogger(__name__)


class AnomalyScorer:
    """Score records with optional calibration and weight overrides."""

    def __init__(
        self,
        config,
        models,
        *,
        calibration_artifact=None,
        weights=None,
        metadata=None,
    ):
        self.config = config
        self.models = models
        self.features = list(
            getattr(models, "feature_names", None)
            or getattr(models, "features", None)
            or []
        )
        self.raw_features = list(
            getattr(models, "raw_feature_names", None)
            or self.features
        )
        self.weights = normalize_ensemble_weights(weights or get_ensemble_weights(config))
        self.top_n = config.get("scoring", {}).get("top_n_reasons", 3)
        self.z_threshold = config.get("scoring", {}).get("univariate_z_threshold", 2.5)
        self.bands = get_alert_bands(config)
        self.score_min = float(config.get("scoring", {}).get("score_min", 0))
        self.score_max = float(config.get("scoring", {}).get("score_max", 100))
        self.metadata = metadata or {}
        self.calibration_artifact = calibration_artifact
        self.calibrator = ScoreCalibrator(config) if calibration_artifact else None

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        id_col = self.config["pipeline"]["id_column"]
        time_col = self.config["pipeline"]["time_column"]
        segment_col = self.config.get("development", {}).get("segment_column")
        self.features = self._resolve_features(df)

        X = self.models.transform(df)
        n_rows = len(X)

        component = self.component_scores(df)
        expected_X = self.models.ae_reconstruct(X)
        ae_weight = float(self.weights["autoencoder"])
        baseline_X = ae_weight * expected_X
        actual_values = self.models.actual_values(df)
        expected_values = self.models.inverse_transform(baseline_X)

        ae_c = self.models.ae_contribution(X)
        if_c = self.models.if_contribution(X)
        md_c = self.models.md_contribution(X)

        unified = (
            self.weights["autoencoder"] * ae_c
            + self.weights["isolation_forest"] * if_c
            + self.weights["mahalanobis"] * md_c
        )

        z_abs = np.abs(X)
        uni_flag_count = (z_abs > self.z_threshold).sum(axis=1)

        reasons = []
        details = []
        full_details = []
        for row_index in range(n_rows):
            ordered_idx = np.argsort(unified[row_index])[::-1]
            top_idx = np.argsort(unified[row_index])[::-1][: self.top_n]
            parts = []
            detail_payload = {}
            full_detail_payload = {}
            for rank, feature_index in enumerate(ordered_idx, 1):
                feature_name = self.features[feature_index]
                label = get_label(self.config, feature_name)
                contribution_pct = unified[row_index, feature_index] * 100

                actual = actual_values[row_index, feature_index]
                expected = expected_values[row_index, feature_index]
                if abs(expected) > 1e-6:
                    pct_change = ((actual - expected) / abs(expected)) * 100
                else:
                    pct_change = 0.0 if abs(actual) < 1e-6 else 999.0

                detail_record = {
                    "label": label,
                    "beklenen": round(float(expected), 2),
                    "gerceklesen": round(float(actual), 2),
                    "degisim_pct": round(float(pct_change), 1),
                    "katki_pct": round(float(contribution_pct), 1),
                    "rank": rank,
                    "is_top_reason": rank <= self.top_n,
                }
                full_detail_payload[feature_name] = detail_record

                if feature_index in top_idx:
                    role = "ana etken" if rank == 1 else f"katki %{contribution_pct:.0f}"
                    direction = "UP" if pct_change > 0 else "DN"
                    parts.append(
                        f"{label}: {expected:.2f}->{actual:.2f} ({direction}%{abs(pct_change):.0f}, {role})"
                    )
                    detail_payload[feature_name] = detail_record

            reasons.append(" | ".join(parts))
            details.append(detail_payload)
            full_details.append(full_detail_payload)

        result = df[[id_col]].copy()
        if time_col in df.columns:
            result[time_col] = pd.to_datetime(df[time_col])
        if segment_col and segment_col in df.columns:
            result[segment_col] = df[segment_col].values

        result["anomaly_score"] = component["ensemble_score"].round(1)
        result["alert_band"] = self._assign_band(component["ensemble_score"])
        result["uni_flag_count"] = uni_flag_count
        result["neden"] = reasons
        result["detay"] = details
        result["full_detay"] = full_details
        result["ae_raw"] = np.round(component["ae_raw"], 6)
        result["if_raw"] = np.round(component["if_raw"], 6)
        result["md_raw"] = np.round(component["md_raw"], 6)
        result["ae_cal"] = np.round(component["ae_cal"], 2)
        result["if_cal"] = np.round(component["if_cal"], 2)
        result["md_cal"] = np.round(component["md_cal"], 2)
        result["ae_score"] = result["ae_cal"]
        result["if_score"] = result["if_cal"]
        result["md_score"] = result["md_cal"]
        for index in range(1, self.top_n + 1):
            column_name = f"reason_{index}"
            result[column_name] = result["neden"].apply(
                lambda value, position=index - 1: self._reason_at_position(value, position)
            )

        for key, value in self.metadata.items():
            # Preserve row-level columns (for example entity/segment identifiers)
            # that already came from the input frame. Metadata should only fill
            # absent columns, not overwrite per-record context.
            if key in result.columns and not result[key].isna().all():
                continue
            result[key] = value

        result = result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
        result["rank_in_run"] = np.arange(1, len(result) + 1)

        band_counts = result["alert_band"].value_counts().to_dict()
        logger.info("Scored %s customers: %s", n_rows, band_counts)
        return result

    def component_scores(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        self.features = self._resolve_features(df)
        X = self.models.transform(df)
        raw_scores = {
            "ae_raw": self.models.raw_ae_scores(X),
            "if_raw": self.models.raw_if_scores(X),
            "md_raw": self.models.raw_md_scores(X),
        }
        if self.calibration_artifact is not None:
            calibrated_scores = self.calibrator.apply(raw_scores, self.calibration_artifact)
        else:
            calibrated_scores = {
                "ae_cal": self.models.ae_scores(X),
                "if_cal": self.models.if_scores(X),
                "md_cal": self.models.md_scores(X),
            }

        ensemble = np.clip(
            self.weights["autoencoder"] * calibrated_scores["ae_cal"]
            + self.weights["isolation_forest"] * calibrated_scores["if_cal"]
            + self.weights["mahalanobis"] * calibrated_scores["md_cal"],
            self.score_min,
            self.score_max,
        )
        return {
            **raw_scores,
            **calibrated_scores,
            "ensemble_score": ensemble,
        }

    def _resolve_features(self, df: pd.DataFrame) -> list[str]:
        if getattr(self.models, "feature_names", None):
            return list(self.models.feature_names)
        if self.features:
            return list(self.features)
        self.features = resolve_feature_list(self.config, df)
        return list(self.features)

    @staticmethod
    def _reason_at_position(value: str, position: int):
        if not isinstance(value, str):
            return None
        parts = [part.strip() for part in value.split("|") if part.strip()]
        return parts[position] if position < len(parts) else None

    def _assign_band(self, scores):
        bands = []
        for score in scores:
            assigned = "NORMAL"
            for band_name, (lower, upper) in self.bands.items():
                if lower <= score < upper or (band_name == "KIRMIZI" and score >= lower):
                    assigned = band_name
            bands.append(assigned)
        return bands
