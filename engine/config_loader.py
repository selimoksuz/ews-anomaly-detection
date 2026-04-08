"""Config and secrets loader — reads YAML pipeline config."""

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path=None):
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "pipeline_config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_secrets(secrets_path=None):
    path = Path(secrets_path) if secrets_path else PROJECT_ROOT / "config" / "secrets.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_list(config):
    """Config'den tum feature isimlerini duz liste olarak dondur."""
    features = config["features"]
    result = []
    for group in ("instant", "rolling_4w", "trend"):
        for item in features.get(group, []):
            result.append(item["name"])
    return result


def get_label(config, feature_name):
    """Feature'in Turkce etiketini dondur."""
    features = config["features"]
    for group in ("instant", "rolling_4w", "trend"):
        for item in features.get(group, []):
            if item["name"] == feature_name:
                return item.get("label_tr", feature_name)
    return feature_name


def get_ensemble_weights(config):
    """Ensemble agirliklarini dondur — key mapping."""
    w = config["ensemble"]["weights"]
    return {
        "autoencoder": w.get("ae", w.get("autoencoder", 0.5)),
        "isolation_forest": w.get("if", w.get("isolation_forest", 0.3)),
        "mahalanobis": w.get("md", w.get("mahalanobis", 0.2)),
    }


def get_alert_bands(config):
    """Alert bantlarini dondur — {band_name: (min, max)} formatinda."""
    bands_raw = config.get("alert_bands", config.get("scoring", {}).get("bands", {}))
    bands = {}
    for name, val in bands_raw.items():
        if isinstance(val, dict):
            bands[name.upper()] = (val.get("min_score", 0), val.get("max_score", 100))
        elif isinstance(val, list):
            bands[name.upper()] = (val[0], val[1])
    return bands
