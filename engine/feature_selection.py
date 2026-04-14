"""Lightweight unsupervised feature selection and branch routing."""

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
    """Apply minimal unsupervised filtering and branch-specific routing."""

    def __init__(self, config: dict):
        self.config = config
        cfg = config.get("feature_selection", {}) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.drop_zero_variance_continuous = bool(cfg.get("drop_zero_variance_continuous", True))
        self.drop_exact_duplicates = bool(cfg.get("drop_exact_duplicates", True))
        self.branch_cfg = cfg.get("branch_routing", {}) or {}

    def select(self, frame: pd.DataFrame, feature_registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
        feature_names = [str(column).strip().lower() for column in frame.columns]
        kept_features = list(feature_names)
        dropped_features: dict[str, str] = {}

        if self.enabled:
            if self.drop_zero_variance_continuous:
                for feature in list(kept_features):
                    meta = feature_registry.get(feature, {})
                    if meta.get("raw_type") != "continuous":
                        continue
                    values = pd.Series(frame[feature])
                    if float(values.fillna(values.median()).var()) <= 1e-12:
                        kept_features.remove(feature)
                        dropped_features[feature] = "zero_variance_continuous"

            if self.drop_exact_duplicates:
                kept_features, duplicate_drops = self._drop_exact_duplicates(frame, kept_features)
                dropped_features.update(duplicate_drops)

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
