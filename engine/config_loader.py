"""Config and secrets loader for the anomaly pipeline."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path=None):
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "pipeline_config.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_secrets(secrets_path=None):
    path = Path(secrets_path) if secrets_path else PROJECT_ROOT / "config" / "secrets.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_feature_list(config):
    """Return the flattened feature list from the grouped config."""
    features = config["features"]
    result = []
    for group in ("instant", "rolling_4w", "trend"):
        for item in features.get(group, []):
            result.append(item["name"])
    return result


def get_label(config, feature_name):
    """Return the Turkish label for a configured feature."""
    features = config["features"]
    for group in ("instant", "rolling_4w", "trend"):
        for item in features.get(group, []):
            if item["name"] == feature_name:
                return item.get("label_tr", feature_name)
    return feature_name


def get_ensemble_weights(config):
    """Return normalized ensemble weights from config."""
    return normalize_ensemble_weights(config["ensemble"]["weights"])


def normalize_ensemble_weights(weights):
    """Normalize weight mappings to the expected model keys and unit sum."""
    resolved = {
        "autoencoder": weights.get("ae", weights.get("autoencoder", 0.5)),
        "isolation_forest": weights.get("if", weights.get("isolation_forest", 0.3)),
        "mahalanobis": weights.get("md", weights.get("mahalanobis", 0.2)),
    }
    total = sum(float(value) for value in resolved.values())
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    return {
        key: float(value) / total
        for key, value in resolved.items()
    }


def get_alert_bands(config):
    """Return alert bands as {band_name: (min_score, max_score)}."""
    bands_raw = config.get("alert_bands", config.get("scoring", {}).get("bands", {}))
    bands = {}
    for name, value in bands_raw.items():
        if isinstance(value, dict):
            bands[name.upper()] = (value.get("min_score", 0), value.get("max_score", 100))
        elif isinstance(value, list):
            bands[name.upper()] = (value[0], value[1])
    return bands
