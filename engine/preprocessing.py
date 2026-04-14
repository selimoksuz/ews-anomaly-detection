"""Feature preprocessing utilities for robust anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class PreprocessingSummary:
    scaler_type: str
    missing_strategy: str
    missing_feature_strategies: dict[str, dict[str, float | str]]
    hard_bounds_enabled: bool
    winsorization_enabled: bool
    log1p_features: list[str]
    hard_bounds_rules: dict[str, dict[str, float | None]]
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
            "fit_statistics": self.fit_statistics,
        }


class FeaturePreprocessor:
    """Config-driven preprocessing with optional robust scaling."""

    def __init__(self, config: dict, feature_names: list[str]):
        self.config = config
        self.feature_names = list(feature_names)
        self.preprocessing_cfg = config.get("preprocessing", {})
        self.enabled = bool(self.preprocessing_cfg.get("enabled", False))
        self.missing_cfg = self.preprocessing_cfg.get("missing", {})
        self.hard_bounds_cfg = self.preprocessing_cfg.get("hard_bounds", {})
        self.winsor_cfg = self.preprocessing_cfg.get("winsorization", {})
        self.log_cfg = self.preprocessing_cfg.get("log1p", {})
        self.scaler_cfg = self.preprocessing_cfg.get("scaler", {})

        self.scaler_type = str(self.scaler_cfg.get("type", "standard")).lower()
        self.missing_strategy = str(
            self.missing_cfg.get(
                "default_strategy",
                self.missing_cfg.get("strategy", "median"),
            )
        ).lower()
        self.log1p_features = set(self.log_cfg.get("features", [])) if self.log_cfg.get("enabled", False) else set()
        self.hard_bounds_enabled = bool(self.hard_bounds_cfg.get("enabled", False))
        self.winsorization_enabled = bool(self.winsor_cfg.get("enabled", False))

        self.hard_bounds_rules = self._parse_hard_bounds_rules()
        self.missing_feature_strategies = self._parse_missing_feature_strategies()
        self.fill_values_: dict[str, float] = {}
        self.winsor_bounds_: dict[str, tuple[float, float]] = {}
        self.scaler = self._build_scaler()
        self.is_fitted = False
        self.fit_statistics_: dict[str, int | float | dict] = {}

    def fit(self, X_raw) -> "FeaturePreprocessor":
        frame = self._to_numeric_frame(X_raw)
        frame = self._apply_hard_bounds(frame)

        missing_counts = frame[self.feature_names].isnull().sum().to_dict()
        self.fill_values_ = self._compute_fill_values(frame)
        frame = frame.fillna(self.fill_values_)
        frame = self._apply_log1p(frame)
        frame, winsor_stats = self._fit_apply_winsorization(frame)
        self.scaler.fit(frame[self.feature_names].values)

        self.fit_statistics_ = {
            "rows": int(len(frame)),
            "missing_values": int(sum(missing_counts.values())),
            "missing_by_feature": {key: int(value) for key, value in missing_counts.items() if int(value) > 0},
            "hard_bounds_hits": self._count_hard_bound_hits(self._to_numeric_frame(X_raw)),
            "winsorized_values": winsor_stats,
        }
        self.is_fitted = True
        return self

    def fit_transform(self, X_raw) -> np.ndarray:
        self.fit(X_raw)
        return self.transform(X_raw)

    def transform(self, X_raw) -> np.ndarray:
        self._ensure_fitted()
        frame = self._to_numeric_frame(X_raw)
        frame = self._apply_hard_bounds(frame)
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
        """Return display-ready actual values after bounds/imputation, before winsor/log scaling."""
        self._ensure_fitted()
        frame = self._to_numeric_frame(X_raw)
        frame = self._apply_hard_bounds(frame)
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
            fit_statistics=self.fit_statistics_,
        ).to_dict()

    def inspect_frame(self, X_raw) -> dict:
        frame = self._to_numeric_frame(X_raw)
        bounded = self._apply_hard_bounds(frame)
        hard_hits = self._count_hard_bound_hits(frame)
        missing = int(bounded[self.feature_names].isnull().sum().sum())

        filled = bounded.fillna(self.fill_values_ if self.fill_values_ else 0.0)
        transformed = self._apply_log1p(filled)
        winsor_hits = self._count_winsor_hits(transformed)
        return {
            "rows": int(len(frame)),
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

    def _parse_hard_bounds_rules(self) -> dict[str, dict[str, float | None]]:
        rules = {}
        for rule in self.hard_bounds_cfg.get("rules", []):
            feature = str(rule.get("feature", "")).strip()
            if not feature:
                continue
            if feature not in self.feature_names:
                raise ValueError(f"Hard bound rule references unknown feature: {feature}")
            rules[feature] = {
                "lower": None if rule.get("lower") is None else float(rule.get("lower")),
                "upper": None if rule.get("upper") is None else float(rule.get("upper")),
            }
        return rules

    def _parse_missing_feature_strategies(self) -> dict[str, dict[str, float | str]]:
        rules = {}
        feature_strategies = self.missing_cfg.get("feature_strategies", {}) or {}
        for feature, payload in feature_strategies.items():
            feature_name = str(feature).strip()
            if feature_name not in self.feature_names:
                raise ValueError(f"Missing strategy references unknown feature: {feature_name}")

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
        for feature in self.feature_names:
            series = frame[feature]
            strategy_cfg = self.missing_feature_strategies.get(
                feature,
                {"strategy": self.missing_strategy},
            )
            values[feature] = self._resolve_fill_value(series, strategy_cfg)
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

    def _to_numeric_frame(self, X_raw) -> pd.DataFrame:
        if isinstance(X_raw, pd.DataFrame):
            frame = X_raw.copy()
        else:
            frame = pd.DataFrame(X_raw, columns=self.feature_names)
        frame = frame[self.feature_names].copy()
        for feature in self.feature_names:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        return frame.astype(float)

    def _apply_hard_bounds(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not (self.enabled and self.hard_bounds_enabled):
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
        if not (self.enabled and self.hard_bounds_enabled):
            return {}
        hits = {}
        for feature, bounds in self.hard_bounds_rules.items():
            lower = bounds.get("lower")
            upper = bounds.get("upper")
            mask = pd.Series(False, index=frame.index)
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

    def _ensure_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("FeaturePreprocessor must be fitted before use.")
