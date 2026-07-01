"""Config and secrets loader for the anomaly pipeline."""

from __future__ import annotations

import os
from copy import deepcopy
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECRETS_PATH = PROJECT_ROOT / "secret" / "secrets.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline_config.yaml"
REQUIRED_PIPELINE_CONFIG_KEYS = ("pipeline", "oracle")
BUILTIN_PIPELINE_CONFIG = {
    "pipeline": {
        "name": "anomaly_multivar_detection",
        "version": "2.0.0",
        "id_column": "mono_id",
        "time_column": "cohort_dt",
        "non_feature_columns": [
            "financial_term_l1y",
            "bilanco_flg",
            "financial_term_q",
            "annualization_q",
            "ref_donem_id",
            "kkbguncelsorgu_no",
            "yukleme_zmn",
        ],
    },
    "sources": {
        "multivar_input": {
            "backend": "oracle",
            "oracle": {"table": "multivar_input"},
        }
    },
    "multivar_anomaly": {
        "source_name": "multivar_input",
        "time_column": "cohort_dt",
        "id_column": "mono_id",
        "default_train_rows": None,
        "outputs": {
            "backend": "oracle",
            "oracle": {
                "results_table_key": "multivar_results",
                "details_table_key": "multivar_details",
            },
        },
    },
    "oracle": {
        "section": "ORA_PRD_ZTUSER",
        "tables": {
            "multivar_input": {
                "owner": "ZT_VAR2",
                "table": "EWS_ANOMALY_MULTIVAR_INPUT",
            },
            "multivar_results": {
                "owner": "ZT_VAR2",
                "table": "EWS_ANOMALY_MULTIVAR_RESULTS",
            },
            "multivar_details": {
                "owner": "ZT_VAR2",
                "table": "EWS_ANOMALY_MULTIVAR_DETAILS",
            },
            "llm_results": {
                "owner": "ZT_VAR2",
                "table": "EWS_ANOMALY_LLM_RESULTS",
            },
            "llm_reason_details": {
                "owner": "ZT_VAR2",
                "table": "EWS_ANOMALY_LLM_REASONS",
            },
            "llm_feature_details": {
                "owner": "ZT_VAR2",
                "table": "EWS_ANOMALY_LLM_FEATURES",
            },
        },
    },
    "llm": {
        "outputs": {
            "oracle": {
                "write_mode": "replace",
            },
        },
    },
    "logging": {
        "logger_name": "ews.multivar",
        "level": "INFO",
        "directory": "runtime/logs/cli",
        "file_name": "multivar.log",
        "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "console": True,
    },
}


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _path_variants(path_like) -> list[Path]:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return [path]
    variants: list[Path] = []
    for root in _search_roots():
        variants.append(root / path)
        if len(path.parts) == 1:
            variants.append(root / "secret" / path)
            variants.append(root / "config" / path)
    return _unique_paths(variants)


def _search_roots() -> list[Path]:
    roots: list[Path] = []
    for start in (Path.cwd(), PROJECT_ROOT):
        try:
            resolved = start.resolve()
        except OSError:
            resolved = start
        roots.append(resolved)
        roots.extend(resolved.parents)
    return _unique_paths(roots)


def _candidate_config_paths(root: Path) -> list[Path]:
    return [
        root / "config" / "pipeline_config.yaml",
        root / "pipeline_config.yaml",
        root / "config" / "config.yaml",
        root / "config.yaml",
    ]


def _candidate_secret_paths(root: Path) -> list[Path]:
    return [
        root / "secret" / "secrets.yaml",
        root / "secret" / "secrets.yml",
        root / "secret" / "secret.yaml",
        root / "secret" / "secret.yml",
        root / "secrets.yaml",
        root / "secrets.yml",
        # Some notebook workspaces accidentally keep this file with a space before
        # the extension. Keep it as a low-priority compatibility candidate.
        root / "secrets .yaml",
    ]


def _project_root_candidates() -> list[Path]:
    roots: list[Path] = []
    for candidate in _search_roots():
        if any(path.exists() for path in _candidate_config_paths(candidate)):
            roots.append(candidate)
            continue
        if any(path.exists() for path in _candidate_secret_paths(candidate)):
            roots.append(candidate)
            continue
        if (candidate / ".git").exists() and (candidate / "engine" / "config_loader.py").exists():
            roots.append(candidate)
    return _unique_paths(roots)


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
        section_key = str(section_name).strip()
        existing_value = merged.get(section_key)
        if isinstance(existing_value, dict) and existing_value:
            continue
        resolved_path = (config_path.parent / str(relative_path)).resolve()
        section_payload = _load_yaml_mapping(resolved_path)
        merged[section_key] = section_payload
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_pipeline_config(config: dict) -> dict:
    """Merge user config with the minimal runtime contract needed by this repo."""
    if not isinstance(config, dict):
        return deepcopy(BUILTIN_PIPELINE_CONFIG)
    return _deep_merge(BUILTIN_PIPELINE_CONFIG, config)


def _valid_pipeline_config(path: Path) -> bool:
    try:
        config = _resolve_config_refs(_load_yaml_mapping(path), config_path=path)
    except (OSError, ValueError, yaml.YAMLError):
        return False
    return all(isinstance(config.get(key), dict) for key in REQUIRED_PIPELINE_CONFIG_KEYS)


def load_config(config_path=None):
    path = resolve_config_path(config_path)
    raw = _load_yaml_mapping(path)
    return normalize_pipeline_config(_resolve_config_refs(raw, config_path=path))


def save_config(config: dict, config_path=None):
    path = resolve_config_path(config_path)
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
    path = resolve_secrets_path(secrets_path)
    return _load_yaml_mapping(path)


def resolve_config_path(config_path=None) -> Path:
    candidates: list[Path] = []
    explicit_path = config_path is not None
    if config_path:
        candidates.extend(_path_variants(config_path))
    else:
        env_candidates: list[Path] = []
        for env_name in ("EWS_ANOMALY_CONFIG_PATH", "RISK_PIPELINE_CONFIG_PATH"):
            env_value = os.getenv(env_name)
            if env_value:
                env_candidates.extend(_path_variants(env_value))

        # If the command is run from the cloned repo, that repo's config must win.
        # Parent workspace files such as /opt/app/config/config.yaml may belong to
        # another project and are only fallbacks.
        repo_candidates = [
            Path.cwd() / "config" / "pipeline_config.yaml",
            DEFAULT_CONFIG_PATH,
            Path.cwd() / "pipeline_config.yaml",
            PROJECT_ROOT / "pipeline_config.yaml",
        ]
        parent_candidates: list[Path] = []
        for root in _project_root_candidates():
            parent_candidates.extend(_candidate_config_paths(root))

        candidates.extend(env_candidates)
        candidates.extend(repo_candidates)
        candidates.extend(parent_candidates)

    candidates = _unique_paths(candidates)
    invalid_env_candidates: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        # Env config only overrides when it is clearly this pipeline's config.
        # Otherwise keep searching for the repo config and merge defaults later.
        if candidate in locals().get("env_candidates", []) and not _valid_pipeline_config(candidate):
            invalid_env_candidates.append(candidate)
            continue
        return candidate.resolve()

    if explicit_path:
        checked = ", ".join(str(path) for path in candidates)
        raise ValueError(
            "Explicit pipeline config file was not found. "
            + f"Checked: {checked}. "
            + "Use an existing file or omit the argument to use repo config."
        )

    checked = ", ".join(str(path) for path in candidates) or str(DEFAULT_CONFIG_PATH)
    invalid_note = ""
    if invalid_env_candidates:
        invalid_note = (
            " Env config files ignored because they do not look like anomaly pipeline config: "
            + ", ".join(str(path) for path in invalid_env_candidates)
            + "."
        )
    raise FileNotFoundError(
        "Pipeline config file not found. Checked: "
        + checked
        + invalid_note
        + ". Put config under <repo>/config/pipeline_config.yaml or set EWS_ANOMALY_CONFIG_PATH."
    )


def resolve_secrets_path(secrets_path=None) -> Path:
    candidates: list[Path] = []
    if secrets_path:
        candidates.extend(_path_variants(secrets_path))
    else:
        for env_name in ("EWS_ANOMALY_SECRETS_PATH", "RISK_PIPELINE_SECRETS_PATH"):
            env_value = os.getenv(env_name)
            if env_value:
                candidates.extend(_path_variants(env_value))
        for root in _project_root_candidates():
            candidates.extend(_candidate_secret_paths(root))
        candidates.append(DEFAULT_SECRETS_PATH)

    candidates = _unique_paths(candidates)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = ", ".join(str(path) for path in candidates) or str(DEFAULT_SECRETS_PATH)
    raise FileNotFoundError(
        "Secrets file not found. Checked: "
        + checked
        + ". Put credentials under <repo>/secret/secrets.yaml or set EWS_ANOMALY_SECRETS_PATH."
    )


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


def get_feature_metadata(config, feature_name: str) -> dict:
    """Return configured metadata for a feature, inheriting from base feature when needed."""
    normalized = str(feature_name).strip().lower()
    base_name = normalized.split("__", 1)[0]
    features = config.get("features", {}) or {}
    base_metadata: dict = {}

    for group in ("instant", "rolling_4w", "trend"):
        for item in features.get(group, []):
            if str(item.get("name", "")).strip().lower() == base_name:
                base_metadata = dict(item)
                break
        if base_metadata:
            break

    for group in ("instant", "rolling_4w", "trend"):
        for item in features.get(group, []):
            if str(item.get("name", "")).strip().lower() == normalized:
                resolved = dict(base_metadata)
                resolved.update(dict(item))
                return resolved
    return dict(base_metadata)


def get_directionality(config, feature_name: str) -> str | None:
    metadata = get_feature_metadata(config, feature_name)
    value = metadata.get("directionality")
    if value is None:
        return None
    return str(value).strip().lower() or None


def get_reasoning_hint(config, feature_name: str) -> str | None:
    metadata = get_feature_metadata(config, feature_name)
    value = metadata.get("reasoning_hint_tr")
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


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
