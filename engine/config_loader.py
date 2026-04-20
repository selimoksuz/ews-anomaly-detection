"""Config and secrets loader for the anomaly pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml_mapping(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping at the root: {path}")
    return data


def _resolve_config_refs(config: dict, *, config_path: Path) -> dict:
    merged = dict(config)
    refs = config.get("config_refs", {}) or {}
    if not isinstance(refs, dict):
        raise ValueError("config_refs must be a mapping of section names to yaml file paths.")
    for section_name, relative_path in refs.items():
        resolved_path = (config_path.parent / str(relative_path)).resolve()
        section_payload = _load_yaml_mapping(resolved_path)
        merged[str(section_name).strip()] = section_payload
    return merged


def load_config(config_path=None):
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "pipeline_config.yaml"
    raw = _load_yaml_mapping(path)
    return _resolve_config_refs(raw, config_path=path)


def save_config(config: dict, config_path=None):
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "pipeline_config.yaml"
    raw_root = _load_yaml_mapping(path)
    refs = raw_root.get("config_refs", {}) or {}
    external_sections = {str(name).strip() for name in refs.keys()}

    for section_name, relative_path in refs.items():
        section_key = str(section_name).strip()
        if section_key not in config:
            continue
        resolved_path = (path.parent / str(relative_path)).resolve()
        with open(resolved_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config[section_key], handle, sort_keys=False, allow_unicode=True)

    saved_root: dict = {}
    for key, value in raw_root.items():
        if key == "config_refs":
            saved_root[key] = config.get(key, value)
            continue
        if key in external_sections:
            continue
        saved_root[key] = config.get(key, value)

    for key, value in config.items():
        if key in saved_root or key in external_sections:
            continue
        saved_root[key] = value

    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(saved_root, handle, sort_keys=False, allow_unicode=True)


def load_secrets(secrets_path=None):
    path = Path(secrets_path) if secrets_path else PROJECT_ROOT / "config" / "secrets.yaml"
    return _load_yaml_mapping(path)


def resolve_project_path(path_like) -> Path:
    """Resolve a filesystem path against the repository root when relative."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_registry_dir(config: dict) -> Path:
    registry_cfg = config.get("registry", {}) or {}
    return resolve_project_path(
        registry_cfg.get("registry_dir", registry_cfg.get("meta_dir", "runtime/registry"))
    )


def resolve_runs_dir(config: dict) -> Path:
    registry_cfg = config.get("registry", {}) or {}
    return resolve_project_path(registry_cfg.get("runs_dir", resolve_registry_dir(config) / "runs"))


def resolve_models_dir(config: dict) -> Path:
    registry_cfg = config.get("registry", {}) or {}
    return resolve_project_path(
        registry_cfg.get("models_dir", registry_cfg.get("artifacts_dir", "runtime/models"))
    )


def resolve_logs_dir(config: dict) -> Path:
    registry_cfg = config.get("registry", {}) or {}
    return resolve_project_path(registry_cfg.get("logs_dir", "runtime/logs"))


def resolve_monitoring_dir(config: dict) -> Path:
    return resolve_project_path(config.get("monitoring", {}).get("directory", "runtime/monitoring"))


def _normalize_columns(columns: Iterable[str]) -> list[str]:
    return [str(column).strip().lower() for column in columns if str(column).strip()]


def get_feature_list(config):
    """Return explicitly configured features, if any."""
    features = config.get("features", {})
    result = []
    for group in ("instant", "rolling_4w", "trend"):
        for item in features.get(group, []):
            result.append(str(item["name"]).strip().lower())

    result.extend(_normalize_columns(features.get("include_columns", [])))
    return list(dict.fromkeys(result))


def get_categorical_feature_settings(config) -> dict[str, dict]:
    """Return normalized categorical feature settings keyed by raw feature name."""
    categorical_cfg = (config.get("features", {}) or {}).get("categorical", {}) or {}
    per_feature = categorical_cfg.get("per_feature", {}) or {}
    result = {}
    for feature_name, payload in per_feature.items():
        name = str(feature_name).strip().lower()
        if not name:
            continue

        if payload is None:
            payload = {}
        include = bool(payload.get("include", False))
        transforms = payload.get("transforms", [])
        if isinstance(transforms, str):
            transforms = [transforms]
        transforms = [str(item).strip().lower() for item in transforms if str(item).strip()]

        order = [str(item).strip() for item in payload.get("order", []) if str(item).strip()]
        result[name] = {
            "include": include,
            "transforms": transforms,
            "order": order,
        }
    return result


def get_included_categorical_features(config) -> set[str]:
    return {
        name
        for name, payload in get_categorical_feature_settings(config).items()
        if payload.get("include")
    }


def get_non_feature_columns(config) -> set[str]:
    """Return columns that should never be treated as model features."""
    pipeline = config.get("pipeline", {})
    features_cfg = config.get("features", {})
    development_cfg = config.get("development", {})

    reserved = set()
    reserved.update(_normalize_columns([pipeline.get("id_column"), pipeline.get("time_column")]))
    reserved.update(_normalize_columns([pipeline.get("split_column")]))
    reserved.update(_normalize_columns([development_cfg.get("segment_column")]))
    reserved.update(_normalize_columns(pipeline.get("non_feature_columns", [])))
    reserved.update(_normalize_columns(features_cfg.get("exclude_columns", [])))
    return reserved


def _is_feature_candidate(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return False
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    converted = pd.to_numeric(non_null, errors="coerce")
    return bool(converted.notnull().all())


def infer_feature_list(config, frame: pd.DataFrame) -> list[str]:
    """Infer numeric feature columns from a frame after excluding reserved columns."""
    normalized = frame.copy()
    normalized.columns = _normalize_columns(normalized.columns)

    reserved = get_non_feature_columns(config)
    inferred = []
    for column in normalized.columns:
        if column in reserved:
            continue
        if _is_feature_candidate(normalized[column]):
            inferred.append(column)

    include_columns = _normalize_columns(config.get("features", {}).get("include_columns", []))
    for column in include_columns:
        if column in normalized.columns and column not in inferred:
            inferred.append(column)

    for column in sorted(get_included_categorical_features(config)):
        if column in normalized.columns and column not in reserved and column not in inferred:
            inferred.append(column)

    return inferred


def resolve_feature_list(config, frame: pd.DataFrame | None = None) -> list[str]:
    """Resolve active feature columns from config or by inferring them from data."""
    configured = get_feature_list(config)
    if frame is None:
        return configured

    mode = str(config.get("features", {}).get("mode", "infer")).strip().lower()
    inferred = infer_feature_list(config, frame)

    if mode == "explicit":
        return configured or inferred
    if mode == "hybrid":
        return list(dict.fromkeys([*configured, *inferred]))
    if mode == "infer":
        return inferred or configured
    raise ValueError(f"Unsupported features.mode: {mode}")


def get_label(config, feature_name):
    """Return the display label for a feature, falling back to its raw name."""
    feature_name = str(feature_name).strip().lower()
    features = config.get("features", {})
    overrides = {
        str(name).strip().lower(): value
        for name, value in (features.get("label_overrides", {}) or {}).items()
    }
    if feature_name in overrides:
        return overrides[feature_name]
    for group in ("instant", "rolling_4w", "trend"):
        for item in features.get(group, []):
            if str(item["name"]).strip().lower() == feature_name:
                return item.get("label_tr", feature_name)
    if "__" in feature_name:
        base_name, suffix = feature_name.split("__", 1)
        base_label = get_label(config, base_name)
        suffix_label = suffix.replace("__", " ").replace("_", " ").strip()
        if suffix_label.startswith("oh "):
            suffix_label = suffix_label[3:].strip().upper()
        return f"{base_label} [{suffix_label}]"
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
