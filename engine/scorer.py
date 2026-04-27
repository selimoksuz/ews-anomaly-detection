"""Anomaly scoring and explanation generation."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

from engine.calibration import ScoreCalibrator
from engine.config_loader import (
    get_alert_bands,
    get_directionality,
    get_ensemble_weights,
    get_label,
    get_reasoning_hint,
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
        actual_values = self.models.actual_values(df)
        expected_values = self.models.inverse_transform(expected_X)

        ae_c = self.models.ae_contribution(X)
        if_c = self.models.if_contribution(X)
        md_c = self.models.md_contribution(X)
        ae_weighted = self.weights["autoencoder"] * ae_c * 100.0
        if_weighted = self.weights["isolation_forest"] * if_c * 100.0
        md_weighted = self.weights["mahalanobis"] * md_c * 100.0

        unified = (
            ae_weighted / 100.0
            + if_weighted / 100.0
            + md_weighted / 100.0
        )
        unified_pct = ae_weighted + if_weighted + md_weighted
        feature_index_map = {feature: index for index, feature in enumerate(self.features)}
        family_indices = defaultdict(list)
        base_feature_indices = {}
        for feature_index, feature_name in enumerate(self.features):
            family_name = self._base_feature_name(feature_name)
            family_indices[family_name].append(feature_index)
            if feature_name == family_name and family_name not in base_feature_indices:
                base_feature_indices[family_name] = feature_index

        population_reference_map = self._build_population_reference_map(
            df,
            actual_values,
            time_column=time_col,
            base_feature_indices=base_feature_indices,
        )
        feature_population_reference_map = self._build_feature_population_reference_map(
            df,
            actual_values,
            time_column=time_col,
        )

        z_abs = np.abs(X)
        uni_flag_count = (z_abs > self.z_threshold).sum(axis=1)

        reasons = []
        details = []
        full_details = []
        for row_index in range(n_rows):
            ordered_idx = np.argsort(unified[row_index])[::-1]
            parts = []
            detail_payload = {}
            full_detail_payload = {}
            for rank, feature_index in enumerate(ordered_idx, 1):
                feature_name = self.features[feature_index]
                detail_record = self._build_feature_level_detail(
                    row_index=row_index,
                    feature_index=feature_index,
                    feature_name=feature_name,
                    feature_index_map=feature_index_map,
                    actual_values=actual_values,
                    expected_values=expected_values,
                    feature_population_reference_map=feature_population_reference_map,
                    ae_weighted=ae_weighted,
                    if_weighted=if_weighted,
                    md_weighted=md_weighted,
                    unified_pct=unified_pct,
                    rank=rank,
                )
                full_detail_payload[feature_name] = detail_record

            family_order = sorted(
                family_indices.keys(),
                key=lambda family_name: float(np.sum(unified_pct[row_index, family_indices[family_name]])),
                reverse=True,
            )[: self.top_n]
            for rank, family_name in enumerate(family_order, 1):
                family_detail = self._build_family_detail(
                    row_index=row_index,
                    family_name=family_name,
                    family_indices=family_indices,
                    base_feature_indices=base_feature_indices,
                    feature_index_map=feature_index_map,
                    actual_values=actual_values,
                    expected_values=expected_values,
                    population_reference_map=population_reference_map,
                    ae_weighted=ae_weighted,
                    if_weighted=if_weighted,
                    md_weighted=md_weighted,
                    unified_pct=unified_pct,
                    rank=rank,
                )
                detail_payload[family_name] = family_detail
                parts.append(self._format_reason_block(family_detail))

            reasons.append(parts)
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
        if value is None:
            return None
        if isinstance(value, Sequence) and not isinstance(value, str):
            return value[position] if position < len(value) else None
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

    @staticmethod
    def _base_feature_name(feature_name: str) -> str:
        return str(feature_name).split("__", 1)[0].strip().lower()

    @staticmethod
    def _round_optional(value, digits: int):
        if pd.isna(value):
            return None
        return round(float(value), digits)

    @staticmethod
    def _compute_pct_change(actual, reference) -> float:
        if pd.isna(actual) or pd.isna(reference):
            return 0.0
        if abs(reference) > 1e-6:
            return float(((actual - reference) / abs(reference)) * 100.0)
        return 0.0 if abs(actual) < 1e-6 else 999.0

    def _build_population_reference_map(
        self,
        df: pd.DataFrame,
        actual_values: np.ndarray,
        *,
        time_column: str,
        base_feature_indices: dict[str, int],
    ) -> dict[str, np.ndarray]:
        reference_map: dict[str, np.ndarray] = {}
        time_values = None
        if time_column in df.columns:
            time_values = pd.to_datetime(df[time_column], errors="coerce")

        for family_name, feature_index in base_feature_indices.items():
            series = pd.Series(actual_values[:, feature_index])
            if time_values is not None and not time_values.isna().all():
                reference_map[family_name] = series.groupby(time_values, sort=False).transform("median").to_numpy()
            else:
                fallback = np.nanmedian(actual_values[:, feature_index])
                reference_map[family_name] = np.full(len(actual_values), fallback, dtype=float)
        return reference_map

    def _build_family_detail(
        self,
        *,
        row_index: int,
        family_name: str,
        family_indices: dict[str, list[int]],
        base_feature_indices: dict[str, int],
        feature_index_map: dict[str, int],
        actual_values: np.ndarray,
        expected_values: np.ndarray,
        population_reference_map: dict[str, np.ndarray],
        ae_weighted: np.ndarray,
        if_weighted: np.ndarray,
        md_weighted: np.ndarray,
        unified_pct: np.ndarray,
        rank: int,
    ) -> dict:
        family_feature_indices = family_indices[family_name]
        base_index = base_feature_indices.get(family_name, family_feature_indices[0])
        actual = actual_values[row_index, base_index]
        ae_reference = expected_values[row_index, base_index]

        customer_history_reference = np.nan
        delta_feature_name = f"{family_name}__delta_1"
        delta_index = feature_index_map.get(delta_feature_name)
        if delta_index is not None:
            delta_value = actual_values[row_index, delta_index]
            if not pd.isna(delta_value) and not pd.isna(actual):
                customer_history_reference = actual - delta_value

        population_reference = population_reference_map.get(family_name, np.full(len(actual_values), np.nan))[row_index]
        total_contribution = float(np.sum(unified_pct[row_index, family_feature_indices]))
        ae_contribution = float(np.sum(ae_weighted[row_index, family_feature_indices]))
        if_contribution = float(np.sum(if_weighted[row_index, family_feature_indices]))
        md_contribution = float(np.sum(md_weighted[row_index, family_feature_indices]))
        directionality = get_directionality(self.config, family_name)
        direction_hint = get_reasoning_hint(self.config, family_name)
        direction_comment = self._build_direction_comment(
            actual=actual,
            customer_history_reference=customer_history_reference,
            ae_reference=ae_reference,
            population_reference=population_reference,
            directionality=directionality,
        )

        return {
            "label": get_label(self.config, family_name),
            "gerceklesen": self._round_optional(actual, 2),
            "musteri_gecmis_referansi": self._round_optional(customer_history_reference, 2),
            "populasyon_referansi": self._round_optional(population_reference, 2),
            "ae_referansi": self._round_optional(ae_reference, 2),
            "expected_value": self._round_optional(ae_reference, 6),
            "actual_value": self._round_optional(actual, 6),
            "delta_pct": self._round_optional(self._compute_pct_change(actual, ae_reference), 1),
            "contribution_pct": self._round_optional(total_contribution, 1),
            "ensemble_katki_pct": self._round_optional(total_contribution, 1),
            "ae_katki_pct": self._round_optional(ae_contribution, 1),
            "if_katki_pct": self._round_optional(if_contribution, 1),
            "md_katki_pct": self._round_optional(md_contribution, 1),
            "directionality": directionality,
            "yon": direction_hint,
            "yon_yorumu": direction_comment,
            "rank": rank,
            "is_top_reason": True,
        }

    def _build_feature_population_reference_map(
        self,
        df: pd.DataFrame,
        actual_values: np.ndarray,
        *,
        time_column: str,
    ) -> dict[str, np.ndarray]:
        reference_map: dict[str, np.ndarray] = {}
        time_values = None
        if time_column in df.columns:
            time_values = pd.to_datetime(df[time_column], errors="coerce")

        for feature_index, feature_name in enumerate(self.features):
            if feature_name.endswith("__population_percentile"):
                reference_map[feature_name] = np.full(len(actual_values), 0.5, dtype=float)
                continue
            if feature_name.endswith("__delta_1"):
                reference_map[feature_name] = np.zeros(len(actual_values), dtype=float)
                continue
            if feature_name.endswith("__self_zscore_6"):
                reference_map[feature_name] = np.zeros(len(actual_values), dtype=float)
                continue
            if feature_name.endswith("__trend_slope_6"):
                reference_map[feature_name] = np.zeros(len(actual_values), dtype=float)
                continue
            if feature_name.endswith("__vs_population_median_delta"):
                reference_map[feature_name] = np.zeros(len(actual_values), dtype=float)
                continue

            series = pd.Series(actual_values[:, feature_index])
            if time_values is not None and not time_values.isna().all():
                reference_map[feature_name] = series.groupby(time_values, sort=False).transform("median").to_numpy()
            else:
                fallback = np.nanmedian(actual_values[:, feature_index])
                reference_map[feature_name] = np.full(len(actual_values), fallback, dtype=float)
        return reference_map

    def _build_feature_level_detail(
        self,
        *,
        row_index: int,
        feature_index: int,
        feature_name: str,
        feature_index_map: dict[str, int],
        actual_values: np.ndarray,
        expected_values: np.ndarray,
        feature_population_reference_map: dict[str, np.ndarray],
        ae_weighted: np.ndarray,
        if_weighted: np.ndarray,
        md_weighted: np.ndarray,
        unified_pct: np.ndarray,
        rank: int,
    ) -> dict:
        actual = actual_values[row_index, feature_index]
        ae_reference = expected_values[row_index, feature_index]
        customer_history_reference = self._feature_customer_history_reference(
            row_index=row_index,
            feature_name=feature_name,
            feature_index_map=feature_index_map,
            actual_values=actual_values,
        )
        population_reference = feature_population_reference_map.get(
            feature_name,
            np.full(len(actual_values), np.nan, dtype=float),
        )[row_index]
        directionality = get_directionality(self.config, feature_name)
        direction_hint = get_reasoning_hint(self.config, feature_name)
        direction_label, direction_value = self._feature_direction_reference(
            feature_name=feature_name,
            customer_history_reference=customer_history_reference,
            population_reference=population_reference,
            ae_reference=ae_reference,
        )
        direction_comment = self._compose_direction_comment(
            actual=actual,
            reference_label=direction_label,
            reference_value=direction_value,
            directionality=directionality,
        )
        contribution_pct = unified_pct[row_index, feature_index]

        return {
            "label": get_label(self.config, feature_name),
            "gerceklesen": self._round_optional(actual, 2),
            "musteri_gecmis_referansi": self._round_optional(customer_history_reference, 2),
            "populasyon_referansi": self._round_optional(population_reference, 2),
            "ae_referansi": self._round_optional(ae_reference, 2),
            "expected_value": self._round_optional(ae_reference, 6),
            "actual_value": self._round_optional(actual, 6),
            "delta_pct": self._round_optional(self._compute_pct_change(actual, ae_reference), 1),
            "contribution_pct": self._round_optional(contribution_pct, 1),
            "ensemble_katki_pct": self._round_optional(contribution_pct, 1),
            "ae_katki_pct": self._round_optional(ae_weighted[row_index, feature_index], 1),
            "if_katki_pct": self._round_optional(if_weighted[row_index, feature_index], 1),
            "md_katki_pct": self._round_optional(md_weighted[row_index, feature_index], 1),
            "directionality": directionality,
            "yon": direction_hint,
            "yon_yorumu": direction_comment,
            "rank": rank,
            "is_top_reason": rank <= self.top_n,
        }

    def _feature_customer_history_reference(
        self,
        *,
        row_index: int,
        feature_name: str,
        feature_index_map: dict[str, int],
        actual_values: np.ndarray,
    ) -> float:
        normalized = str(feature_name).strip().lower()
        actual = actual_values[row_index, feature_index_map[normalized]]

        if normalized.endswith("__population_percentile"):
            return 0.5
        if normalized.endswith("__delta_1"):
            return 0.0
        if normalized.endswith("__self_zscore_6"):
            return 0.0
        if normalized.endswith("__trend_slope_6"):
            return 0.0
        if normalized.endswith("__vs_population_median_delta"):
            return 0.0

        family_name = self._base_feature_name(normalized)
        delta_feature_name = f"{family_name}__delta_1"
        delta_index = feature_index_map.get(delta_feature_name)
        if delta_index is not None and not pd.isna(actual):
            delta_value = actual_values[row_index, delta_index]
            if not pd.isna(delta_value):
                return float(actual - delta_value)
        return np.nan

    @staticmethod
    def _direction_reference(
        customer_history_reference,
        ae_reference,
        population_reference,
    ) -> tuple[str | None, float | None]:
        for label, value in (
            ("musteri gecmis referansina", customer_history_reference),
            ("ae referansina", ae_reference),
            ("populasyon referansina", population_reference),
        ):
            if value is not None and not pd.isna(value):
                return label, float(value)
        return None, None

    @staticmethod
    def _compose_direction_comment(
        *,
        actual,
        reference_label: str | None,
        reference_value: float | None,
        directionality: str | None,
    ) -> str:
        if actual is None or pd.isna(actual):
            return "gerceklesen degeri olmadigi icin yon yorumu uretilemedi"

        if reference_label is None or reference_value is None or pd.isna(reference_value):
            return "referans deger olmadigi icin yon yorumu uretilemedi"

        actual_value = float(actual)
        delta = actual_value - float(reference_value)
        if abs(delta) <= 1e-9:
            return f"{reference_label} ile ayni seviyede"

        movement = "artmis" if delta > 0 else "azalmis"
        if directionality == "increase_is_risk":
            outcome = "kotulesme yonunde" if delta > 0 else "iyilesme yonunde"
        elif directionality == "decrease_is_risk":
            outcome = "kotulesme yonunde" if delta < 0 else "iyilesme yonunde"
        else:
            outcome = "yon etkisi tanimsiz"
        return f"{reference_label} gore {movement} ve {outcome}"

    def _feature_direction_reference(
        self,
        *,
        feature_name: str,
        customer_history_reference,
        population_reference,
        ae_reference,
    ) -> tuple[str | None, float | None]:
        normalized = str(feature_name).strip().lower()
        if normalized.endswith("__population_percentile"):
            return "genel percentile referansina", 0.5
        if normalized.endswith("__delta_1"):
            return "notr degisim referansina", 0.0
        if normalized.endswith("__self_zscore_6"):
            return "notr self-z referansina", 0.0
        if normalized.endswith("__trend_slope_6"):
            return "notr trend referansina", 0.0
        if normalized.endswith("__vs_population_median_delta"):
            return "genel median referansina", 0.0
        return self._direction_reference(
            customer_history_reference=customer_history_reference,
            ae_reference=ae_reference,
            population_reference=population_reference,
        )

    @staticmethod
    def _build_direction_comment(
        *,
        actual,
        customer_history_reference,
        ae_reference,
        population_reference,
        directionality: str | None,
    ) -> str:
        if actual is None or pd.isna(actual):
            return "gerceklesen degeri olmadigi icin yon yorumu uretilemedi"

        reference_label, reference_value = AnomalyScorer._direction_reference(
            customer_history_reference=customer_history_reference,
            ae_reference=ae_reference,
            population_reference=population_reference,
        )
        return AnomalyScorer._compose_direction_comment(
            actual=actual,
            reference_label=reference_label,
            reference_value=reference_value,
            directionality=directionality,
        )

    @staticmethod
    def _format_reason_block(detail: dict) -> str:
        lines = [
            f"{detail['label']}",
            f"gerceklesen: {AnomalyScorer._display_value(detail.get('gerceklesen'))}",
            f"musteri_gecmis_referansi: {AnomalyScorer._display_value(detail.get('musteri_gecmis_referansi'))}",
            f"populasyon_referansi: {AnomalyScorer._display_value(detail.get('populasyon_referansi'))}",
            f"ae_referansi: {AnomalyScorer._display_value(detail.get('ae_referansi'))}",
        ]
        if detail.get("yon"):
            lines.append(f"yon: {detail.get('yon')}")
        if detail.get("yon_yorumu"):
            lines.append(f"yon_yorumu: {detail.get('yon_yorumu')}")
        lines.append(
            f"ensemble_katki: %{AnomalyScorer._display_pct(detail.get('ensemble_katki_pct'))} "
            f"(AE %{AnomalyScorer._display_pct(detail.get('ae_katki_pct'))}, "
            f"IF %{AnomalyScorer._display_pct(detail.get('if_katki_pct'))}, "
            f"MD %{AnomalyScorer._display_pct(detail.get('md_katki_pct'))})"
        )
        return "\n".join(lines)

    @staticmethod
    def _display_value(value) -> str:
        return "NA" if value is None or pd.isna(value) else f"{float(value):.2f}"

    @staticmethod
    def _display_pct(value) -> str:
        return "0" if value is None or pd.isna(value) else f"{float(value):.1f}".rstrip("0").rstrip(".")
