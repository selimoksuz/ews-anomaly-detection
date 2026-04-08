"""Anomaly detection models — AE, IF, Mahalanobis."""

import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


class _Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_layers:
            enc.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        enc.extend([nn.Linear(prev, latent_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim
        for h in reversed(hidden_layers):
            dec.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AnomalyModels:
    """3 model: Autoencoder, Isolation Forest, Mahalanobis (LedoitWolf)."""

    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.ae = None
        self.iso = None
        self.lw = None
        self.lw_inv = None
        self.lw_center = None
        self.ae_ref = None
        self.md_ref = None
        self.n_features = None
        self.is_fitted = False

    def fit(self, X_raw):
        """X_raw: numpy array (n_samples, n_features)."""
        X = self.scaler.fit_transform(X_raw)
        self.n_features = X.shape[1]

        mc = self.config.get("models", self.config.get("model", {}))

        # Autoencoder
        ae_cfg = mc["autoencoder"]
        self.ae = _Autoencoder(self.n_features, ae_cfg["hidden_layers"], ae_cfg["latent_dim"])
        self._train_ae(X, ae_cfg["epochs"], ae_cfg["learning_rate"])

        # Isolation Forest
        if_cfg = mc["isolation_forest"]
        self.iso = IsolationForest(
            n_estimators=if_cfg["n_estimators"],
            contamination=if_cfg["contamination"],
            random_state=if_cfg["random_state"],
        )
        self.iso.fit(X)

        # Mahalanobis (LedoitWolf shrinkage)
        self.lw = LedoitWolf().fit(X)
        self.lw_inv = np.linalg.inv(self.lw.covariance_)
        self.lw_center = self.lw.location_

        # Reference quantiles
        self.ae_ref = np.percentile(self._ae_total_error(X), 99.5)
        self.md_ref = np.percentile(self._mahal_distances(X), 99.5)

        self.is_fitted = True
        logger.info(f"Models fitted: {X.shape[0]} samples, {self.n_features} features")
        logger.info(f"  AE ref (p99.5): {self.ae_ref:.4f}, MD ref: {self.md_ref:.4f}")

    def transform(self, X_raw):
        """Scale raw data."""
        return self.scaler.transform(X_raw)

    # ── Per-model scores ──

    def ae_scores(self, X):
        return np.clip(self._ae_total_error(X) / self.ae_ref * 100, 0, 120)

    def if_scores(self, X):
        return rankdata(-self.iso.decision_function(X)) / len(X) * 100

    def md_scores(self, X):
        return np.clip(self._mahal_distances(X) / self.md_ref * 100, 0, 120)

    # ── Per-model feature contributions ──

    def ae_contribution(self, X):
        X_t = torch.FloatTensor(X)
        self.ae.eval()
        with torch.no_grad():
            recon = self.ae(X_t)
        errors = (X_t - recon).numpy() ** 2
        row_sum = np.clip(errors.sum(axis=1, keepdims=True), 1e-10, None)
        return errors / row_sum

    def if_contribution(self, X):
        base = -self.iso.decision_function(X)
        contrib = np.zeros_like(X)
        for j in range(self.n_features):
            X_p = X.copy()
            X_p[:, j] = 0.0
            contrib[:, j] = base - (-self.iso.decision_function(X_p))
        contrib = np.clip(contrib, 0, None)
        row_sum = np.clip(contrib.sum(axis=1, keepdims=True), 1e-10, None)
        return contrib / row_sum

    def md_contribution(self, X):
        diff = X - self.lw_center
        contrib = np.abs(diff * (diff @ self.lw_inv))
        row_sum = np.clip(contrib.sum(axis=1, keepdims=True), 1e-10, None)
        return contrib / row_sum

    def ae_reconstruct(self, X):
        X_t = torch.FloatTensor(X)
        self.ae.eval()
        with torch.no_grad():
            return self.ae(X_t).numpy()

    # ── Helpers ──

    def _train_ae(self, X, epochs, lr):
        X_t = torch.FloatTensor(X)
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
        self.ae.train()
        for epoch in range(epochs):
            recon = self.ae(X_t)
            loss = ((recon - X_t) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0:
                logger.info(f"  AE epoch {epoch + 1}/{epochs}, loss: {loss.item():.6f}")

    def _ae_total_error(self, X):
        X_t = torch.FloatTensor(X)
        self.ae.eval()
        with torch.no_grad():
            return ((self.ae(X_t) - X_t) ** 2).sum(dim=1).numpy()

    def _mahal_distances(self, X):
        diff = X - self.lw_center
        return np.sqrt(np.clip((diff @ self.lw_inv * diff).sum(axis=1), 0, None))
