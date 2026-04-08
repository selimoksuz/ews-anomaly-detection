"""Config and secrets loader."""

import yaml
from pathlib import Path


def load_config(config_path="config/pipeline_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_secrets(secrets_path="config/secrets.yaml"):
    with open(secrets_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_list(config):
    """Config'den tum feature listesini duz liste olarak dondur."""
    features = config["features"]
    return features["instant"] + features["rolling_4w"] + features["trend"]


def get_label(config, feature_name):
    """Feature'in Turkce etiketini dondur."""
    return config.get("labels", {}).get(feature_name, feature_name)
