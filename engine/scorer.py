"""Anomaly scorer — ensemble scoring + human-readable explanations."""

import logging
import numpy as np
import pandas as pd
from engine.config_loader import get_feature_list, get_label, get_ensemble_weights, get_alert_bands

logger = logging.getLogger(__name__)


class AnomalyScorer:
    """Ensemble scoring + explanation generation."""

    def __init__(self, config, models):
        self.config = config
        self.models = models
        self.features = get_feature_list(config)
        self.weights = get_ensemble_weights(config)
        self.top_n = config.get("scoring", {}).get("top_n_reasons", 3)
        self.z_threshold = config.get("scoring", {}).get("univariate_z_threshold", 2.5)
        self.bands = get_alert_bands(config)

    def score(self, df):
        """Score DataFrame, return results with explanations."""
        id_col = self.config["pipeline"]["id_column"]
        X_raw = df[self.features].fillna(0).values
        X = self.models.transform(X_raw)
        n = len(X)

        # Per-model contributions
        ae_c = self.models.ae_contribution(X)
        if_c = self.models.if_contribution(X)
        md_c = self.models.md_contribution(X)

        w = self.weights
        unified = w["autoencoder"] * ae_c + w["isolation_forest"] * if_c + w["mahalanobis"] * md_c

        # Ensemble score
        ae_s = self.models.ae_scores(X)
        if_s = self.models.if_scores(X)
        md_s = self.models.md_scores(X)

        ensemble = np.clip(
            w["autoencoder"] * ae_s + w["isolation_forest"] * if_s + w["mahalanobis"] * md_s,
            0, 100,
        ).round(1)

        # Expected values from AE
        expected_X = self.models.ae_reconstruct(X)

        # Univariate flags
        z_abs = np.abs(X)
        uni_flag_count = (z_abs > self.z_threshold).sum(axis=1)

        # Build results
        reasons = []
        details = []
        for i in range(n):
            top_idx = np.argsort(unified[i])[::-1][: self.top_n]
            parts = []
            det = {}
            for rank, j in enumerate(top_idx):
                feat = self.features[j]
                label = get_label(self.config, feat)
                contrib_pct = unified[i, j] * 100

                actual = X[i, j] * self.models.scaler.scale_[j] + self.models.scaler.mean_[j]
                expected = expected_X[i, j] * self.models.scaler.scale_[j] + self.models.scaler.mean_[j]

                if abs(expected) > 1e-6:
                    pct_chg = ((actual - expected) / abs(expected)) * 100
                else:
                    pct_chg = 0.0 if abs(actual) < 1e-6 else 999.0

                role = "ana etken" if rank == 0 else f"katki %{contrib_pct:.0f}"
                parts.append(f"{label}: {expected:.2f}->{actual:.2f} ({'UP' if pct_chg > 0 else 'DN'}%{abs(pct_chg):.0f}, {role})")

                det[feat] = {
                    "label": label,
                    "beklenen": round(expected, 2),
                    "gerceklesen": round(actual, 2),
                    "degisim_pct": round(pct_chg, 1),
                    "katki_pct": round(contrib_pct, 1),
                }

            reasons.append(" | ".join(parts))
            details.append(det)

        result = df[[id_col]].copy()
        result["anomaly_score"] = ensemble
        result["alert_band"] = self._assign_band(ensemble)
        result["uni_flag_count"] = uni_flag_count
        result["neden"] = reasons
        result["detay"] = details
        result["ae_score"] = ae_s.round(1)
        result["if_score"] = if_s.round(1)
        result["md_score"] = md_s.round(1)

        result = result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

        band_counts = result["alert_band"].value_counts().to_dict()
        logger.info(f"Scored {n} customers: {band_counts}")
        return result

    def _assign_band(self, scores):
        bands = []
        for s in scores:
            assigned = "NORMAL"
            for band_name, (lo, hi) in self.bands.items():
                if lo <= s < hi or (band_name == "KIRMIZI" and s >= lo):
                    assigned = band_name
            bands.append(assigned)
        return bands
