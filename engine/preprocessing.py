"""Feature preprocessing utilities for robust anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from engine.config_loader import get_categorical_feature_settings


@dataclass
class PreprocessingSummary:
    scaler_type: str
    missing_strategy: str
    missing_feature_strategies: dict[str, dict[str, float | str]]
    hard_bounds_enabled: bool
    winsorization_enabled: bool
    log1p_features: list[str]
    hard_bounds_rules: dict[str, dict[str, float | None]]
    categorical_features: dict[str, dict[str, Any]]
    feature_registry: dict[str, dict[str, Any]]
    raw_feature_names: list[str]
    output_feature_names: list[str]
    fit_statistics: dict[str, int | float | dict]

    def to_dict(self) -> dict:
        return {
            "scaler_type": self.scaler_type,
            "missing_strategy": self.missing_strategy,
            "missing_feature_strategies": self.missing_feature_strategies,
            "hard_bounds_enabled": self.hard_bounds_enabled,
            "winsorization_enabled": self.winsorization_enabled,
            "log1p_features": self.log1p_features,
            "hard_bounds_rules": self.hard_bounds_rules,
            "categorical_features": self.categorical_features,
            "feature_registry": self.feature_registry,
            "raw_feature_names": self.raw_feature_names,
            "output_feature_names": self.output_feature_names,
            "fit_statistics": self.fit_statistics,
        }


class FeaturePreprocessor:
    """Config-driven preprocessing with optional robust scaling and categorical transforms."""

    def __init__(self, config: dict, feature_names: list[str]):
        self.config = config
        self.raw_feature_names = [str(name).strip().lower() for name in feature_names if str(name).strip()]
        self.feature_names = list(self.raw_feature_names)
        self.preprocessing_cfg = config.get("preprocessing", {})
        self.enabled = bool(self.preprocessing_cfg.get("enabled", False))
        self.missing_cfg = self.preprocessing_cfg.get("missing", {})
        self.hard_bounds_cfg = self.preprocessing_cfg.get("hard_bounds", {})
        self.winsor_cfg = self.preprocessing_cfg.get("winsorization", {})
        self.log_cfg = self.preprocessing_cfg.get("log1p", {})
        self.scaler_cfg = self.preprocessing_cfg.get("scaler", {})
        self.pipeline_cfg = config.get("pipeline", {})
        self.categorical_cfg = (config.get("features", {}) or {}).get("categorical", {}) or {}
        self.raw_type_overrides = {
            str(feature).strip().lower(): str(raw_type).strip().lower()
            for feature, raw_type in ((config.get("features", {}) or {}).get("raw_type_overrides", {}) or {}).items()
            if str(feature).strip() and str(raw_type).strip()
        }

        self.id_column = str(self.pipeline_cfg.get("id_column", "customer_id")).strip().lower()
        self.time_column = str(self.pipeline_cfg.get("time_column", "snapshot_date")).strip().lower()
        self.low_cardinality_threshold = int(self.categorical_cfg.get("low_cardinality_threshold", 8))

        self.scaler_type = str(self.scaler_cfg.get("type", "standard")).lower()
        self.missing_strategy = str(
            self.missing_cfg.get(
                "default_strategy",
                self.missing_cfg.get("strategy", "median"),
            )
        ).lower()
        self.log1p_features = (
            {str(name).strip().lower() for name in self.log_cfg.get("features", [])}
            if self.log_cfg.get("enabled", False)
            else set()
        )
        self.hard_bounds_enabled = bool(self.hard_bounds_cfg.get("enabled", False))
        self.winsorization_enabled = bool(self.winsor_cfg.get("enabled", False))

        self.categorical_settings = self._resolve_categorical_settings()
        self.numeric_raw_features = [
            feature for feature in self.raw_feature_names
            if feature not in self.categorical_settings
        ]

        self.hard_bounds_rules = self._parse_hard_bounds_rules()
        self.missing_feature_strategies = self._parse_missing_feature_strategies()
        self.fill_values_: dict[str, float] = {}
        self.winsor_bounds_: dict[str, tuple[float, float]] = {}
        self.scaler = self._build_scaler()
        self.is_fitted = False
        self.fit_statistics_: dict[str, int | float | dict] = {}

        self.output_feature_names: list[str] = list(self.raw_feature_names)
        self.category_levels_: dict[str, list[str]] = {}
        self.category_frequency_: dict[str, dict[str, float]] = {}
        self.category_rarity_cap_: dict[str, float] = {}
        self.ordinal_mapping_: dict[str, dict[str, int]] = {}
        self._one_hot_feature_names: dict[str, list[str]] = {}
        self.raw_feature_types_: dict[str, str] = {}
        self.feature_registry_: dict[str, dict[str, Any]] = {}

    def fit(self, X_raw) -> "FeaturePreprocessor":
        raw_frame = self._to_input_frame(X_raw)
        numeric_frame = self._prepare_numeric_raw_frame(raw_frame)

        missing_counts = numeric_frame[self.numeric_raw_features].isnull().sum().to_dict() if self.numeric_raw_features else {}
        hard_hits = self._count_hard_bound_hits(self._coerce_numeric_features(raw_frame)) if self.numeric_raw_features else {}

        model_frame = self._build_model_frame(raw_frame, fit=True)
        self.fill_values_ = self._compute_fill_values(model_frame)
        model_frame = model_frame.fillna(self.fill_values_)
        model_frame = self._apply_log1p(model_frame)
        model_frame, winsor_stats = self._fit_apply_winsorization(model_frame)
        self.scaler.fit(model_frame[self.feature_names].values)

        self.fit_statistics_ = {
            "rows": int(len(model_frame)),
            "missing_values": int(sum(missing_counts.values())),
            "missing_by_feature": {key: int(value) for key, value in missing_counts.items() if int(value) > 0},
            "hard_bounds_hits": hard_hits,
            "winsorized_values": winsor_stats,
            "generated_features": int(len(self.feature_names)),
        }
        self.is_fitted = True
        return self

    def fit_transform(self, X_raw) -> np.ndarray:
        self.fit(X_raw)
        return self.transform(X_raw)

    def transform(self, X_raw) -> np.ndarray:
        self._ensure_fitted()
        raw_frame = self._to_input_frame(X_raw)
        frame = self._build_model_frame(raw_frame, fit=False)
        frame = frame.fillna(self.fill_values_)
        frame = self._apply_log1p(frame)
        frame = self._apply_winsorization(frame)
        return self.scaler.transform(frame[self.feature_names].values)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        restored = self.scaler.inverse_transform(np.asarray(X_scaled, dtype=float))
        frame = pd.DataFrame(restored, columns=self.feature_names)
        for feature in self.feature_names:
            if feature in self.log1p_features:
                frame[feature] = np.expm1(frame[feature]).clip(lower=0.0)
        return frame[self.feature_names].values

    def prepare_actual_values(self, X_raw) -> np.ndarray:
        """Return display-ready values after categorical expansion and bounds/imputation."""
        self._ensure_fitted()
        raw_frame = self._to_input_frame(X_raw)
        frame = self._build_model_frame(raw_frame, fit=False)
        frame = frame.fillna(self.fill_values_)
        return frame[self.feature_names].values

    def summarize(self) -> dict:
        return PreprocessingSummary(
            scaler_type=self.scaler_type,
            missing_strategy=self.missing_strategy,
            missing_feature_strategies=self.missing_feature_strategies,
            hard_bounds_enabled=self.hard_bounds_enabled,
            winsorization_enabled=self.winsorization_enabled,
            log1p_features=sorted(self.log1p_features),
            hard_bounds_rules=self.hard_bounds_rules,
            categorical_features=self.categorical_settings,
            feature_registry=self.feature_registry_,
            raw_feature_names=self.raw_feature_names,
            output_feature_names=self.feature_names,
            fit_statistics=self.fit_statistics_,
        ).to_dict()

    def inspect_frame(self, X_raw) -> dict:
        raw_frame = self._to_input_frame(X_raw)
        bounded_numeric = self._prepare_numeric_raw_frame(raw_frame)
        hard_hits = self._count_hard_bound_hits(self._coerce_numeric_features(raw_frame))
        missing = int(bounded_numeric[self.numeric_raw_features].isnull().sum().sum()) if self.numeric_raw_features else 0

        model_frame = self._build_model_frame(raw_frame, fit=False)
        filled = model_frame.fillna(self.fill_values_ if self.fill_values_ else 0.0)
        transformed = self._apply_log1p(filled)
        winsor_hits = self._count_winsor_hits(transformed)
        return {
            "rows": int(len(raw_frame)),
            "missing_values": missing,
            "hard_bounds_hits": hard_hits,
            "winsorized_values": winsor_hits,
        }

    def _build_scaler(self):
        if not self.enabled:
            return StandardScaler()
        if self.scaler_type == "robust":
            return RobustScaler()
        if self.scaler_type == "standard":
            return StandardScaler()
        raise ValueError(f"Unsupported preprocessing scaler type: {self.scaler_type}")

    def _resolve_categorical_settings(self) -> dict[str, dict[str, Any]]:
        settings = {}
        for feature, payload in get_categorical_feature_settings(self.config).items():
            if feature not in self.raw_feature_names or not payload.get("include"):
                continue
            transforms = [item for item in payload.get("transforms", []) if item]
            if not transforms:
                continue
            if "ordinal" in transforms and not payload.get("order"):
                raise ValueError(f"Categorical feature '{feature}' uses 'ordinal' transform but defines no order.")
            settings[feature] = {
                "transforms": transforms,
                "order": list(payload.get("order", [])),
            }
        return settings

    def _parse_hard_bounds_rules(self) -> dict[str, dict[str, float | None]]:
        rules = {}
        for rule in self.hard_bounds_cfg.get("rules", []):
            feature = str(rule.get("feature", "")).strip().lower()
            if not feature:
                continue
            if feature not in self.numeric_raw_features:
                continue
            rules[feature] = {
                "lower": None if rule.get("lower") is None else float(rule.get("lower")),
                "upper": None if rule.get("upper") is None else float(rule.get("upper")),
            }
        return rules

    def _parse_missing_feature_strategies(self) -> dict[str, dict[str, float | str]]:
        rules = {}
        feature_strategies = self.missing_cfg.get("feature_strategies", {}) or {}
        for feature, payload in feature_strategies.items():
            feature_name = str(feature).strip().lower()
            if feature_name not in self.raw_feature_names:
                continue

            if isinstance(payload, str):
                strategy = payload.strip().lower()
                config = {"strategy": strategy}
            elif isinstance(payload, dict):
                strategy = str(payload.get("strategy", "")).strip().lower()
                if not strategy:
                    raise ValueError(f"Missing strategy for feature '{feature_name}' must define 'strategy'.")
                config = {"strategy": strategy}
                if payload.get("value") is not None:
                    config["value"] = float(payload.get("value"))
            else:
                raise ValueError(
                    f"Missing strategy for feature '{feature_name}' must be a string or mapping."
                )

            if strategy not in {"median", "mean", "min", "max", "constant"}:
                raise ValueError(
                    f"Unsupported missing strategy '{strategy}' for feature '{feature_name}'."
                )
            if strategy == "constant" and "value" not in config:
                raise ValueError(
                    f"Missing strategy 'constant' for feature '{feature_name}' requires a numeric 'value'."
                )
            rules[feature_name] = config
        return rules

    def _compute_fill_values(self, frame: pd.DataFrame) -> dict[str, float]:
        values = {}
        for feature in frame.columns:
            strategy_cfg = self.missing_feature_strategies.get(
                feature,
                {"strategy": self.missing_strategy},
            )
            values[feature] = self._resolve_fill_value(frame[feature], strategy_cfg)
        return values

    @staticmethod
    def _resolve_fill_value(series: pd.Series, strategy_cfg: dict[str, float | str]) -> float:
        strategy = str(strategy_cfg["strategy"]).lower()
        if strategy == "constant":
            return float(strategy_cfg.get("value", 0.0))
        if not series.notnull().any():
            return 0.0
        if strategy == "median":
            return float(series.median())
        if strategy == "mean":
            return float(series.mean())
        if strategy == "min":
            return float(series.min())
        if strategy == "max":
            return float(series.max())
        raise ValueError(f"Unsupported missing strategy: {strategy}")

    def _to_input_frame(self, X_raw) -> pd.DataFrame:
        if isinstance(X_raw, pd.DataFrame):
            frame = X_raw.copy()
            frame.columns = [str(column).strip().lower() for column in frame.columns]
            missing_columns = [column for column in self.raw_feature_names if column not in frame.columns]
            if missing_columns:
                raise ValueError(
                    "Input frame is missing required feature columns: "
                    + ", ".join(missing_columns)
                )
            return frame
        frame = pd.DataFrame(X_raw, columns=self.raw_feature_names)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        return frame

    def _coerce_numeric_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = pd.DataFrame(index=frame.index)
        for feature in self.numeric_raw_features:
            numeric[feature] = pd.to_numeric(frame[feature], errors="coerce").astype(float)
        return numeric

    def _prepare_numeric_raw_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = self._coerce_numeric_features(frame)
        return self._apply_hard_bounds(numeric)

    def _build_model_frame(self, raw_frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        numeric_frame = self._prepare_numeric_raw_frame(raw_frame)
        if fit:
            self.raw_feature_types_ = self._infer_raw_feature_types(numeric_frame)
        categorical_frame = self._build_categorical_frame(raw_frame, fit=fit)
        model_frame = pd.concat([numeric_frame, categorical_frame], axis=1)
        if fit:
            self.output_feature_names = [str(column).strip().lower() for column in model_frame.columns]
            self.feature_names = list(self.output_feature_names)
            self.feature_registry_ = self._build_feature_registry()
        else:
            for feature in self.feature_names:
                if feature not in model_frame.columns:
                    model_frame[feature] = np.nan
            model_frame = model_frame[self.feature_names]
        return model_frame.astype(float)

    def _build_categorical_frame(self, raw_frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        result = pd.DataFrame(index=raw_frame.index)
        if not self.categorical_settings:
            return result

        for feature, settings in self.categorical_settings.items():
            series = self._normalize_categorical_series(raw_frame[feature])
            if fit:
                observed = series.dropna()
                counts = observed.value_counts(normalize=True)
                if "one_hot" in settings["transforms"] and self.low_cardinality_threshold > 0:
                    unique_count = int(counts.shape[0])
                    if unique_count > self.low_cardinality_threshold:
                        raise ValueError(
                            f"Categorical feature '{feature}' has {unique_count} distinct values, "
                            f"which exceeds features.categorical.low_cardinality_threshold="
                            f"{self.low_cardinality_threshold} for one_hot encoding."
                        )
                self.category_frequency_[feature] = {
                    str(level): float(freq)
                    for level, freq in counts.items()
                }
                self.category_levels_[feature] = list(self.category_frequency_[feature].keys())
                non_null_rows = int(observed.shape[0])
                self.category_rarity_cap_[feature] = float(np.log(max(non_null_rows + 1, 2)))
                if "ordinal" in settings["transforms"]:
                    self.ordinal_mapping_[feature] = {
                        value: index
                        for index, value in enumerate(settings.get("order", []))
                    }

            if "one_hot" in settings["transforms"]:
                levels = self.category_levels_.get(feature, [])
                feature_names = []
                for level in levels:
                    output_name = f"{feature}__oh__{self._slugify(level)}"
                    feature_names.append(output_name)
                    result[output_name] = (series == level).astype(float)
                if fit:
                    self._one_hot_feature_names[feature] = feature_names

            if "freq" in settings["transforms"]:
                result[f"{feature}__freq"] = self._map_category_frequency(feature, series)

            if "rarity" in settings["transforms"]:
                result[f"{feature}__rarity"] = self._map_category_rarity(feature, series)

            if "is_unseen" in settings["transforms"]:
                known = set(self.category_levels_.get(feature, []))
                result[f"{feature}__is_unseen"] = (
                    series.notna() & ~series.isin(known)
                ).astype(float)

            if "ordinal" in settings["transforms"]:
                mapping = self.ordinal_mapping_.get(feature, {})
                result[f"{feature}__ordinal"] = series.map(mapping).astype(float)

            if "changed_from_prev" in settings["transforms"]:
                result[f"{feature}__changed_from_prev"] = self._compute_changed_from_prev(raw_frame, feature, series)

        return result

    @staticmethod
    def _normalize_categorical_series(series: pd.Series) -> pd.Series:
        normalized = series.copy()
        normalized = normalized.where(normalized.isnull(), normalized.astype(str).str.strip())
        normalized = normalized.replace("", pd.NA)
        return normalized

    def _map_category_frequency(self, feature: str, series: pd.Series) -> pd.Series:
        frequency = self.category_frequency_.get(feature, {})
        return series.map(lambda value: float(frequency.get(str(value), 0.0)) if pd.notna(value) else np.nan)

    def _map_category_rarity(self, feature: str, series: pd.Series) -> pd.Series:
        frequency = self.category_frequency_.get(feature, {})
        rarity_cap = float(self.category_rarity_cap_.get(feature, np.log(2.0)))

        def _encode(value):
            if pd.isna(value):
                return np.nan
            freq = float(frequency.get(str(value), 0.0))
            if freq <= 0:
                return rarity_cap
            return float(-np.log(freq))

        return series.map(_encode)

    def _compute_changed_from_prev(self, raw_frame: pd.DataFrame, feature: str, series: pd.Series) -> pd.Series:
        if self.id_column not in raw_frame.columns or self.time_column not in raw_frame.columns:
            return pd.Series(np.zeros(len(raw_frame), dtype=float), index=raw_frame.index)

        helper = pd.DataFrame(
            {
                "_row_id": raw_frame.index,
                "_entity_id": raw_frame[self.id_column].astype(str),
                "_snapshot_time": pd.to_datetime(raw_frame[self.time_column]),
                "_value": series,
            }
        )
        helper = helper.sort_values(["_entity_id", "_snapshot_time", "_row_id"])
        previous = helper.groupby("_entity_id")["_value"].shift(1)
        changed = (
            helper["_value"].notna()
            & previous.notna()
            & (helper["_value"] != previous)
        ).astype(float)
        return changed.sort_index()

    def _apply_hard_bounds(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not (self.enabled and self.hard_bounds_enabled and not frame.empty):
            return frame.copy()
        bounded = frame.copy()
        for feature, bounds in self.hard_bounds_rules.items():
            lower = bounds.get("lower")
            upper = bounds.get("upper")
            if lower is not None:
                bounded[feature] = bounded[feature].clip(lower=lower)
            if upper is not None:
                bounded[feature] = bounded[feature].clip(upper=upper)
        return bounded

    def _count_hard_bound_hits(self, frame: pd.DataFrame) -> dict[str, int]:
        if not (self.enabled and self.hard_bounds_enabled) or frame.empty:
            return {}
        hits = {}
        for feature, bounds in self.hard_bounds_rules.items():
            mask = pd.Series(False, index=frame.index)
            lower = bounds.get("lower")
            upper = bounds.get("upper")
            if lower is not None:
                mask |= frame[feature] < lower
            if upper is not None:
                mask |= frame[feature] > upper
            count = int(mask.fillna(False).sum())
            if count > 0:
                hits[feature] = count
        return hits

    def _apply_log1p(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not (self.enabled and self.log1p_features):
            return frame.copy()
        transformed = frame.copy()
        for feature in self.log1p_features:
            if feature not in transformed.columns:
                continue
            transformed[feature] = np.log1p(np.clip(transformed[feature], a_min=0.0, a_max=None))
        return transformed

    def _fit_apply_winsorization(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        if not (self.enabled and self.winsorization_enabled):
            self.winsor_bounds_ = {}
            return frame.copy(), {}

        lower_q = float(self.winsor_cfg.get("lower_quantile", 0.005))
        upper_q = float(self.winsor_cfg.get("upper_quantile", 0.995))
        clipped = frame.copy()
        winsor_stats = {}
        self.winsor_bounds_ = {}
        for feature in self.feature_names:
            lower = float(clipped[feature].quantile(lower_q))
            upper = float(clipped[feature].quantile(upper_q))
            self.winsor_bounds_[feature] = (lower, upper)
            mask = (clipped[feature] < lower) | (clipped[feature] > upper)
            count = int(mask.sum())
            if count > 0:
                winsor_stats[feature] = count
            clipped[feature] = clipped[feature].clip(lower=lower, upper=upper)
        return clipped, winsor_stats

    def _apply_winsorization(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not (self.enabled and self.winsorization_enabled and self.winsor_bounds_):
            return frame.copy()
        clipped = frame.copy()
        for feature, (lower, upper) in self.winsor_bounds_.items():
            clipped[feature] = clipped[feature].clip(lower=lower, upper=upper)
        return clipped

    def _count_winsor_hits(self, frame: pd.DataFrame) -> dict[str, int]:
        if not (self.enabled and self.winsorization_enabled and self.winsor_bounds_):
            return {}
        hits = {}
        for feature, (lower, upper) in self.winsor_bounds_.items():
            mask = (frame[feature] < lower) | (frame[feature] > upper)
            count = int(mask.sum())
            if count > 0:
                hits[feature] = count
        return hits

    @staticmethod
    def _slugify(value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            return "empty"
        normalized = []
        for char in text:
            if char.isalnum():
                normalized.append(char)
            else:
                normalized.append("_")
        slug = "".join(normalized)
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug.strip("_") or "value"

    @staticmethod
    def _is_binary_numeric(series: pd.Series) -> bool:
        non_null = pd.Series(series).dropna()
        if non_null.empty:
            return False
        unique_values = set(pd.to_numeric(non_null, errors="coerce").dropna().tolist())
        return unique_values.issubset({0.0, 1.0}) and len(unique_values) <= 2

    def _infer_raw_feature_types(self, numeric_frame: pd.DataFrame) -> dict[str, str]:
        result = {}
        for feature in self.numeric_raw_features:
            if feature in self.raw_type_overrides:
                result[feature] = self.raw_type_overrides[feature]
                continue
            result[feature] = "binary" if self._is_binary_numeric(numeric_frame[feature]) else "continuous"
        for feature in self.categorical_settings:
            result[feature] = "categorical"
        return result

    def _build_feature_registry(self) -> dict[str, dict[str, Any]]:
        registry: dict[str, dict[str, Any]] = {}
        for feature in self.numeric_raw_features:
            registry[feature] = {
                "source_feature": feature,
                "origin": "raw_numeric",
                "raw_type": self.raw_feature_types_.get(feature, "continuous"),
                "transform": "raw",
            }
        for feature, settings in self.categorical_settings.items():
            transforms = list(settings.get("transforms", []))
            if "one_hot" in transforms:
                for generated_name in self._one_hot_feature_names.get(feature, []):
                    registry[generated_name] = {
                        "source_feature": feature,
                        "origin": "generated_categorical",
                        "raw_type": "binary",
                        "transform": "one_hot",
                    }
            if "freq" in transforms:
                registry[f"{feature}__freq"] = {
                    "source_feature": feature,
                    "origin": "generated_categorical",
                    "raw_type": "continuous",
                    "transform": "freq",
                }
            if "rarity" in transforms:
                registry[f"{feature}__rarity"] = {
                    "source_feature": feature,
                    "origin": "generated_categorical",
                    "raw_type": "continuous",
                    "transform": "rarity",
                }
            if "is_unseen" in transforms:
                registry[f"{feature}__is_unseen"] = {
                    "source_feature": feature,
                    "origin": "generated_categorical",
                    "raw_type": "binary",
                    "transform": "is_unseen",
                }
            if "ordinal" in transforms:
                registry[f"{feature}__ordinal"] = {
                    "source_feature": feature,
                    "origin": "generated_categorical",
                    "raw_type": "ordinal",
                    "transform": "ordinal",
                }
            if "changed_from_prev" in transforms:
                registry[f"{feature}__changed_from_prev"] = {
                    "source_feature": feature,
                    "origin": "generated_categorical",
                    "raw_type": "binary",
                    "transform": "changed_from_prev",
                }
        return registry

    def _ensure_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("FeaturePreprocessor must be fitted before use.")
