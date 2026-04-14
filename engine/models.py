"""Anomaly detection models - AE, IF, Mahalanobis."""

import copy
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest

from engine.config_loader import get_feature_list
from engine.preprocessing import FeaturePreprocessor

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
        self.training_cfg = config.get("training", {})
        self.random_seed = int(self.training_cfg.get("random_seed", 42))
        self.feature_names = get_feature_list(config)
        self.preprocessor = FeaturePreprocessor(config, self.feature_names)
        self.scaler = self.preprocessor.scaler
        self.ae = None
        self.iso = None
        self.lw = None
        self.lw_inv = None
        self.lw_center = None
        self.ae_ref = None
        self.md_ref = None
        self.n_features = None
        self.is_fitted = False
        self.ae_history = []

    def fit(self, X_raw):
        """X_raw: numpy array (n_samples, n_features)."""
        self._set_random_seed()

        X = self.preprocessor.fit_transform(X_raw).astype(np.float32, copy=False)
        self.scaler = self.preprocessor.scaler
        self.n_features = X.shape[1]

        mc = self.config.get("models", self.config.get("model", {}))

        ae_cfg = mc["autoencoder"]
        self.ae = _Autoencoder(self.n_features, ae_cfg["hidden_layers"], ae_cfg["latent_dim"])
        self._train_ae(
            X,
            ae_cfg["epochs"],
            ae_cfg["learning_rate"],
            ae_cfg.get("batch_size", len(X)),
            ae_cfg.get("validation_fraction", 0.1),
            ae_cfg.get("early_stopping_patience", 20),
            ae_cfg.get("min_improvement", 1e-5),
            ae_cfg.get("min_epochs", 50),
        )

        if_cfg = mc["isolation_forest"]
        self.iso = IsolationForest(
            n_estimators=if_cfg["n_estimators"],
            contamination=if_cfg["contamination"],
            random_state=if_cfg["random_state"],
        )
        self.iso.fit(X)

        self.lw = LedoitWolf().fit(X)
        self.lw_inv = np.linalg.inv(self.lw.covariance_)
        self.lw_center = self.lw.location_

        self.ae_ref = np.percentile(self.raw_ae_scores(X), 99.5)
        self.md_ref = np.percentile(self.raw_md_scores(X), 99.5)

        self.is_fitted = True
        logger.info("Models fitted: %s samples, %s features", X.shape[0], self.n_features)
        logger.info("  AE ref (p99.5): %.4f, MD ref: %.4f", self.ae_ref, self.md_ref)

    def transform(self, X_raw):
        """Scale raw data."""
        return self.preprocessor.transform(X_raw).astype(np.float32, copy=False)

    def inverse_transform(self, X_scaled):
        """Back-project scaled features into display space."""
        return self.preprocessor.inverse_transform(X_scaled)

    def actual_values(self, X_raw):
        """Return bounded/imputed raw values used for display."""
        return self.preprocessor.prepare_actual_values(X_raw)

    def preprocessing_summary(self):
        return self.preprocessor.summarize()

    # Per-model raw scores

    def raw_ae_scores(self, X):
        return self._ae_total_error(X)

    def raw_if_scores(self, X):
        return -self.iso.decision_function(X)

    def raw_md_scores(self, X):
        return self._mahal_distances(X)

    # Per-model normalized scores

    def ae_scores(self, X):
        return np.clip(self.raw_ae_scores(X) / self.ae_ref * 100, 0, 120)

    def if_scores(self, X):
        return rankdata(self.raw_if_scores(X)) / len(X) * 100

    def md_scores(self, X):
        return np.clip(self.raw_md_scores(X) / self.md_ref * 100, 0, 120)

    # Per-model feature contributions

    def ae_contribution(self, X):
        X_t = torch.FloatTensor(X)
        self.ae.eval()
        with torch.no_grad():
            recon = self.ae(X_t)
        errors = (X_t - recon).numpy() ** 2
        row_sum = np.clip(errors.sum(axis=1, keepdims=True), 1e-10, None)
        return errors / row_sum

    def if_contribution(self, X):
        base = self.raw_if_scores(X)
        contrib = np.zeros_like(X)
        for j in range(self.n_features):
            X_p = X.copy()
            X_p[:, j] = 0.0
            contrib[:, j] = base - self.raw_if_scores(X_p)
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

    # Helpers

    def _set_random_seed(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _train_val_split(self, X, validation_fraction):
        if validation_fraction <= 0 or len(X) < 20:
            return X, np.empty((0, X.shape[1]), dtype=X.dtype)

        val_size = int(round(len(X) * validation_fraction))
        val_size = max(1, min(val_size, len(X) - 1))

        rng = np.random.default_rng(self.random_seed)
        order = rng.permutation(len(X))
        val_idx = order[:val_size]
        train_idx = order[val_size:]
        return X[train_idx], X[val_idx]

    def _train_ae(
        self,
        X,
        epochs,
        lr,
        batch_size,
        validation_fraction,
        early_stopping_patience,
        min_improvement,
        min_epochs,
    ):
        train_X, val_X = self._train_val_split(X, validation_fraction)
        train_t = torch.FloatTensor(train_X)
        val_t = torch.FloatTensor(val_X) if len(val_X) > 0 else None
        batch_size = max(1, min(int(batch_size), len(train_X)))

        optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
        rng = np.random.default_rng(self.random_seed)
        best_state = copy.deepcopy(self.ae.state_dict())
        best_val_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        self.ae_history = []

        for epoch in range(epochs):
            self.ae.train()
            batch_losses = []
            order = rng.permutation(len(train_X))

            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                batch = train_t[idx]
                recon = self.ae(batch)
                loss = ((recon - batch) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            val_loss = train_loss
            if val_t is not None:
                self.ae.eval()
                with torch.no_grad():
                    val_loss = float(((self.ae(val_t) - val_t) ** 2).mean().item())

            self.ae_history.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            )

            improved = val_loss < (best_val_loss - float(min_improvement))
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                stale_epochs = 0
                best_state = copy.deepcopy(self.ae.state_dict())
            else:
                stale_epochs += 1

            if (epoch + 1) % 25 == 0 or epoch == 0:
                logger.info(
                    "  AE epoch %s/%s, train_loss: %.6f, val_loss: %.6f",
                    epoch + 1,
                    epochs,
                    train_loss,
                    val_loss,
                )

            if (
                val_t is not None
                and (epoch + 1) >= int(min_epochs)
                and stale_epochs >= int(early_stopping_patience)
            ):
                logger.info(
                    "  AE early stopping at epoch %s/%s (best epoch: %s, best val_loss: %.6f)",
                    epoch + 1,
                    epochs,
                    best_epoch,
                    best_val_loss,
                )
                break

        self.ae.load_state_dict(best_state)
        if val_t is not None and best_epoch:
            logger.info("  AE restored best weights from epoch %s", best_epoch)

    def _ae_total_error(self, X):
        X_t = torch.FloatTensor(X)
        self.ae.eval()
        with torch.no_grad():
            return ((self.ae(X_t) - X_t) ** 2).sum(dim=1).numpy()

    def _mahal_distances(self, X):
        diff = X - self.lw_center
        return np.sqrt(np.clip((diff @ self.lw_inv * diff).sum(axis=1), 0, None))
