"""Unsupervised feature elimination and branch routing for the Faz 1 longlist."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FeatureSelectionSummary:
    enabled: bool
    kept_features: list[str]
    dropped_features: dict[str, str]
    branch_features: dict[str, list[str]]
    branch_counts: dict[str, int]
    generated_feature_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "kept_features": self.kept_features,
            "dropped_features": self.dropped_features,
            "branch_features": self.branch_features,
            "branch_counts": self.branch_counts,
            "generated_feature_count": self.generated_feature_count,
        }


class FeatureSelector:
    """Apply unsupervised longlist-to-shortlist elimination and branch-specific routing."""

    def __init__(self, config: dict):
        self.config = config
        cfg = config.get("feature_selection", {}) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.drop_zero_variance_continuous = bool(cfg.get("drop_zero_variance_continuous", True))
        self.drop_exact_duplicates = bool(cfg.get("drop_exact_duplicates", True))
        self.drop_low_coverage_cfg = cfg.get("drop_low_coverage", {}) or {}
        self.drop_high_corr_cfg = cfg.get("drop_high_correlation", {}) or {}
        self.branch_cfg = cfg.get("branch_routing", {}) or {}
        self.priority_order = [
            str(item).strip().lower()
            for item in self.drop_high_corr_cfg.get(
                "priority_order",
                ["base", "delta_1", "self_zscore_6", "trend_slope_6", "population_percentile", "vs_population_median_delta"],
            )
            if str(item).strip()
        ]
        self.priority_rank = {name: index for index, name in enumerate(self.priority_order)}

    def select(self, frame: pd.DataFrame, feature_registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
        feature_names = [str(column).strip().lower() for column in frame.columns]
        kept_features = list(feature_names)
        dropped_features: dict[str, str] = {}
        coverage_map = {
            feature: float(pd.Series(frame[feature]).notna().mean())
            for feature in feature_names
        }

        if self.enabled:
            if bool(self.drop_low_coverage_cfg.get("enabled", False)):
                min_coverage = float(self.drop_low_coverage_cfg.get("min_coverage", 0.0))
                for feature in list(kept_features):
                    if coverage_map.get(feature, 0.0) < min_coverage:
                        kept_features.remove(feature)
                        dropped_features[feature] = f"low_coverage<{min_coverage}"

            if self.drop_zero_variance_continuous:
                for feature in list(kept_features):
                    meta = feature_registry.get(feature, {})
                    if meta.get("raw_type") != "continuous":
                        continue
                    values = pd.Series(frame[feature], dtype=float)
                    filled = values.fillna(values.median())
                    if float(filled.var()) <= 1e-12:
                        kept_features.remove(feature)
                        dropped_features[feature] = "zero_variance_continuous"

            if self.drop_exact_duplicates:
                kept_features, duplicate_drops = self._drop_exact_duplicates(frame, kept_features)
                dropped_features.update(duplicate_drops)

            if bool(self.drop_high_corr_cfg.get("enabled", False)):
                kept_features, corr_drops = self._drop_high_correlation(
                    frame,
                    kept_features,
                    feature_registry,
                    coverage_map,
                )
                dropped_features.update(corr_drops)

        branch_features = self._route_branches(kept_features, feature_registry)
        summary = FeatureSelectionSummary(
            enabled=self.enabled,
            kept_features=kept_features,
            dropped_features=dropped_features,
            branch_features=branch_features,
            branch_counts={name: len(columns) for name, columns in branch_features.items()},
            generated_feature_count=len(feature_names),
        )
        return summary.to_dict()

    @staticmethod
    def _drop_exact_duplicates(frame: pd.DataFrame, feature_names: list[str]) -> tuple[list[str], dict[str, str]]:
        kept = []
        dropped: dict[str, str] = {}
        seen_signatures: dict[tuple, str] = {}
        for feature in feature_names:
            values = tuple(pd.Series(frame[feature]).fillna("__nan__").tolist())
            original = seen_signatures.get(values)
            if original is not None:
                dropped[feature] = f"exact_duplicate_of:{original}"
                continue
            seen_signatures[values] = feature
            kept.append(feature)
        return kept, dropped

    def _drop_high_correlation(
        self,
        frame: pd.DataFrame,
        kept_features: list[str],
        feature_registry: dict[str, dict[str, Any]],
        coverage_map: dict[str, float],
    ) -> tuple[list[str], dict[str, str]]:
        threshold = float(self.drop_high_corr_cfg.get("threshold", 0.999))
        continuous_features = [
            feature
            for feature in kept_features
            if feature_registry.get(feature, {}).get("raw_type", "continuous") == "continuous"
        ]
        if len(continuous_features) < 2:
            return list(kept_features), {}

        numeric = frame[continuous_features].apply(pd.to_numeric, errors="coerce")
        medians = numeric.median(axis=0, skipna=True).fillna(0.0)
        filled = numeric.fillna(medians)
        corr = filled.corr().abs().fillna(0.0)
        variances = filled.var(axis=0).fillna(0.0).to_dict()

        ordered = sorted(
            continuous_features,
            key=lambda feature: (
                self._priority_rank(feature, feature_registry.get(feature, {})),
                -coverage_map.get(feature, 0.0),
                -float(variances.get(feature, 0.0)),
                feature,
            ),
        )

        corr_keep: list[str] = []
        dropped: dict[str, str] = {}
        for feature in ordered:
            conflicting = next(
                (
                    kept
                    for kept in corr_keep
                    if float(corr.loc[feature, kept]) >= threshold
                ),
                None,
            )
            if conflicting is not None:
                dropped[feature] = f"high_correlation_with:{conflicting}"
                continue
            corr_keep.append(feature)

        selected_set = set(corr_keep)
        final_kept = []
        for feature in kept_features:
            meta = feature_registry.get(feature, {})
            if meta.get("raw_type", "continuous") != "continuous":
                final_kept.append(feature)
                continue
            if feature in selected_set:
                final_kept.append(feature)
        return final_kept, dropped

    def _priority_rank(self, feature: str, meta: dict[str, Any]) -> int:
        variant = self._feature_variant(feature, meta)
        return self.priority_rank.get(variant, len(self.priority_rank) + 10)

    @staticmethod
    def _feature_variant(feature: str, meta: dict[str, Any]) -> str:
        feature = str(feature).strip().lower()
        transform = str(meta.get("transform", "raw")).strip().lower()
        if feature.endswith("__delta_1"):
            return "delta_1"
        if feature.endswith("__self_zscore_6"):
            return "self_zscore_6"
        if feature.endswith("__trend_slope_6"):
            return "trend_slope_6"
        if feature.endswith("__population_percentile"):
            return "population_percentile"
        if feature.endswith("__vs_population_median_delta"):
            return "vs_population_median_delta"
        if transform and transform != "raw":
            return transform
        return "base"

    def _route_branches(self, kept_features: list[str], feature_registry: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
        branches = {
            "autoencoder": [],
            "isolation_forest": [],
            "mahalanobis": [],
        }
        for branch_name in branches:
            branches[branch_name] = self._filter_branch_features(
                branch_name,
                kept_features,
                feature_registry,
            )
            if not branches[branch_name]:
                branches[branch_name] = list(kept_features)
        return branches

    def _filter_branch_features(
        self,
        branch_name: str,
        kept_features: list[str],
        feature_registry: dict[str, dict[str, Any]],
    ) -> list[str]:
        if not self.enabled or not self.branch_cfg.get("enabled", False):
            return list(kept_features)

        cfg = self.branch_cfg.get(branch_name, {}) or {}
        exclude_raw_types = {
            str(item).strip().lower()
            for item in cfg.get("exclude_raw_types", [])
            if str(item).strip()
        }
        exclude_transforms = {
            str(item).strip().lower()
            for item in cfg.get("exclude_transforms", [])
            if str(item).strip()
        }

        selected = []
        for feature in kept_features:
            meta = feature_registry.get(feature, {})
            if meta.get("raw_type") in exclude_raw_types:
                continue
            if meta.get("transform") in exclude_transforms:
                continue
            selected.append(feature)
        return selected
