"""
Explainable Ensemble Anomaly Detector.

3 models:
  1. Autoencoder       - learns normal inter-feature relationships
  2. Isolation Forest  - tree-based anomaly isolation
  3. Mahalanobis Dist  - covariance-aware multivariate distance

Each model produces per-feature contribution scores.
Contributions are merged into a unified explanation matrix.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

from config import (
    ALL_FEATURES, FEATURE_LABELS,
    AE_LATENT_DIM, AE_HIDDEN_LAYERS, AE_EPOCHS, AE_LEARNING_RATE,
    ISO_N_ESTIMATORS, ISO_CONTAMINATION,
    WEIGHT_AE, WEIGHT_IF, WEIGHT_MD,
)


class _Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim):
        super().__init__()
        # Encoder
        enc = []
        prev = input_dim
        for h in hidden_layers:
            enc.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        enc.extend([nn.Linear(prev, latent_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*enc)

        # Decoder (mirror)
        dec = []
        prev = latent_dim
        for h in reversed(hidden_layers):
            dec.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ExplainableEnsemble:

    def __init__(self):
        self.feature_cols = ALL_FEATURES
        self.n_features = len(ALL_FEATURES)
        self.scaler = StandardScaler()
        self.is_fitted = False

    # ═══════════════════════════════════════════════════════════
    # FIT
    # ═══════════════════════════════════════════════════════════

    def fit(self, df):
        X = self.scaler.fit_transform(df[self.feature_cols].fillna(0))

        # 1. Autoencoder
        self.ae = _Autoencoder(self.n_features, AE_HIDDEN_LAYERS, AE_LATENT_DIM)
        self._train_ae(X)

        # 2. Isolation Forest
        self.iso = IsolationForest(
            n_estimators=ISO_N_ESTIMATORS,
            contamination=ISO_CONTAMINATION,
            random_state=42,
        )
        self.iso.fit(X)

        # 3. Mahalanobis (Robust Covariance)
        self.mcd = MinCovDet(random_state=42, support_fraction=0.85).fit(X)
        self.cov_inv = np.linalg.inv(self.mcd.covariance_)
        self.center = self.mcd.location_

        # Reference quantiles from training data (for 0-100 normalization)
        ae_err = self._ae_total_error(X)
        self.ae_ref = np.percentile(ae_err, 99.5)

        md_dist = self._mahal_distances(X)
        self.md_ref = np.percentile(md_dist, 99.5)

        self.is_fitted = True
        print(f"[FIT] Trained on {len(df)} customers, {self.n_features} features")
        print(f"  AE ref (p99.5): {self.ae_ref:.4f}")
        print(f"  MD ref (p99.5): {self.md_ref:.4f}")
        return self

    # ═══════════════════════════════════════════════════════════
    # PREDICT
    # ═══════════════════════════════════════════════════════════

    def predict(self, df, top_n=3):
        assert self.is_fitted, "Call fit() first"
        X = self.scaler.transform(df[self.feature_cols].fillna(0))
        n = len(X)

        # --- Per-model feature contributions (each: n x 35, row sums = 1) ---
        ae_contrib = self._ae_contribution(X)
        if_contrib = self._if_contribution(X)
        md_contrib = self._md_contribution(X)

        # --- Unified contribution ---
        unified = (WEIGHT_AE * ae_contrib + WEIGHT_IF * if_contrib + WEIGHT_MD * md_contrib)

        # --- Ensemble score (0-100) ---
        ae_scores = np.clip(self._ae_total_error(X) / self.ae_ref * 100, 0, 120)
        if_scores = rankdata(-self.iso.decision_function(X)) / n * 100
        md_scores = np.clip(self._mahal_distances(X) / self.md_ref * 100, 0, 120)

        ensemble = np.clip(
            WEIGHT_AE * ae_scores + WEIGHT_IF * if_scores + WEIGHT_MD * md_scores,
            0, 100,
        ).round(1)

        # --- Expected values from AE ---
        expected_X = self._ae_reconstruct(X)

        # --- Build results ---
        reasons = []
        detail_list = []
        for i in range(n):
            top_idx = np.argsort(unified[i])[::-1][:top_n]
            parts = []
            details = {}
            for rank, j in enumerate(top_idx):
                feat = self.feature_cols[j]
                label = FEATURE_LABELS.get(feat, feat)
                contrib_pct = unified[i, j] * 100

                actual = X[i, j] * self.scaler.scale_[j] + self.scaler.mean_[j]
                expected = expected_X[i, j] * self.scaler.scale_[j] + self.scaler.mean_[j]

                if abs(expected) > 1e-6:
                    pct_chg = ((actual - expected) / abs(expected)) * 100
                else:
                    pct_chg = 0.0 if abs(actual) < 1e-6 else 999.0

                icon = "\u2191" if pct_chg > 0 else "\u2193"
                role = "ana etken" if rank == 0 else f"katki %{contrib_pct:.0f}"

                parts.append(
                    f"{label}: {expected:.2f} -> {actual:.2f} "
                    f"({icon}%{abs(pct_chg):.0f}, {role})"
                )
                details[feat] = {
                    "label": label,
                    "beklenen": round(expected, 2),
                    "gerceklesen": round(actual, 2),
                    "degisim_pct": round(pct_chg, 1),
                    "katki_pct": round(contrib_pct, 1),
                }
            reasons.append(" | ".join(parts))
            detail_list.append(details)

        result = df[["customer_id"]].copy()
        result["anomaly_score"] = ensemble
        result["alert_band"] = pd.cut(
            ensemble,
            bins=[-0.1, 60, 75, 90, 100.1],
            labels=["NORMAL", "SARI", "TURUNCU", "KIRMIZI"],
        )
        result["neden"] = reasons
        result["detay"] = detail_list
        result["ae_score"] = ae_scores.round(1)
        result["if_score"] = if_scores.round(1)
        result["md_score"] = md_scores.round(1)

        return result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════
    # FEATURE CONTRIBUTIONS
    # ═══════════════════════════════════════════════════════════

    def _ae_contribution(self, X):
        """Per-feature reconstruction error (natural decomposition)."""
        X_t = torch.FloatTensor(X)
        self.ae.eval()
        with torch.no_grad():
            recon = self.ae(X_t)
        errors = (X_t - recon).numpy() ** 2
        row_sum = errors.sum(axis=1, keepdims=True)
        row_sum = np.clip(row_sum, 1e-10, None)
        return errors / row_sum

    def _if_contribution(self, X):
        """Perturbation-based: zero-out each feature, measure score drop."""
        base = -self.iso.decision_function(X)
        contrib = np.zeros_like(X)
        for j in range(self.n_features):
            X_pert = X.copy()
            X_pert[:, j] = 0.0  # mean in scaled space
            contrib[:, j] = base - (-self.iso.decision_function(X_pert))
        contrib = np.clip(contrib, 0, None)
        row_sum = contrib.sum(axis=1, keepdims=True)
        row_sum = np.clip(row_sum, 1e-10, None)
        return contrib / row_sum

    def _md_contribution(self, X):
        """Analytic Mahalanobis decomposition per feature."""
        diff = X - self.center
        contrib = np.abs(diff * (diff @ self.cov_inv))
        row_sum = contrib.sum(axis=1, keepdims=True)
        row_sum = np.clip(row_sum, 1e-10, None)
        return contrib / row_sum

    # ═══════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════

    def _train_ae(self, X):
        X_t = torch.FloatTensor(X)
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=AE_LEARNING_RATE)
        self.ae.train()
        for epoch in range(AE_EPOCHS):
            recon = self.ae(X_t)
            loss = ((recon - X_t) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0:
                print(f"  AE epoch {epoch+1}/{AE_EPOCHS}, loss: {loss.item():.6f}")

    def _ae_total_error(self, X):
        X_t = torch.FloatTensor(X)
        self.ae.eval()
        with torch.no_grad():
            return ((self.ae(X_t) - X_t) ** 2).sum(dim=1).numpy()

    def _ae_reconstruct(self, X):
        X_t = torch.FloatTensor(X)
        self.ae.eval()
        with torch.no_grad():
            return self.ae(X_t).numpy()

    def _mahal_distances(self, X):
        diff = X - self.center
        return np.sqrt((diff @ self.cov_inv * diff).sum(axis=1))
