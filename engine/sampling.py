"""Train-only sampling with representativeness validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class SamplingResult:
    frame: pd.DataFrame
    report: dict[str, Any]


class TrainSampler:
    """Time-stratified sampler with tail and missing preservation."""

    def __init__(self, config: dict, *, id_column: str, time_column: str):
        self.config = config
        self.id_column = id_column
        self.time_column = time_column
        self.sampling_cfg = (config.get("development", {}) or {}).get("sampling", {}) or {}
        self.validation_cfg = (self.sampling_cfg.get("validation", {}) or {})
        self.sampled_windows = {
            str(item).strip().lower()
            for item in self.sampling_cfg.get("windows", ["train"]) or ["train"]
            if str(item).strip()
        }

    def sample(
        self,
        frame: pd.DataFrame,
        *,
        feature_names: list[str],
        window_name: str,
        seed_offset: int = 0,
    ) -> SamplingResult:
        if frame.empty:
            return SamplingResult(frame=frame.copy(), report={"window": window_name, "status": "empty"})
        if str(window_name).strip().lower() not in self.sampled_windows:
            return SamplingResult(
                frame=frame.reset_index(drop=True),
                report={"window": window_name, "status": "skipped_window"},
            )
        if not bool(self.sampling_cfg.get("enabled", False)):
            return SamplingResult(frame=frame.reset_index(drop=True), report={"window": window_name, "status": "disabled"})

        activate_if_rows_gt = self.sampling_cfg.get("activate_if_rows_gt")
        max_rows = self.sampling_cfg.get("max_rows")
        if max_rows in (None, "", 0):
            return SamplingResult(frame=frame.reset_index(drop=True), report={"window": window_name, "status": "max_rows_not_set"})
        max_rows = int(max_rows)
        if activate_if_rows_gt not in (None, "") and len(frame) <= int(activate_if_rows_gt):
            return SamplingResult(
                frame=frame.reset_index(drop=True),
                report={
                    "window": window_name,
                    "status": "below_activation_threshold",
                    "original_rows": int(len(frame)),
                    "activation_threshold": int(activate_if_rows_gt),
                },
            )
        if len(frame) <= max_rows:
            return SamplingResult(
                frame=frame.reset_index(drop=True),
                report={
                    "window": window_name,
                    "status": "below_max_rows",
                    "original_rows": int(len(frame)),
                    "max_rows": max_rows,
                },
            )

        rng = np.random.default_rng(int(self.sampling_cfg.get("random_seed", 42)) + int(seed_offset))
        working = frame.copy()
        working[self.time_column] = pd.to_datetime(working[self.time_column])

        strata = self._build_strata(working, feature_names)
        sample_index = self._sample_index(strata, max_rows=max_rows, rng=rng)
        sampled = working.loc[sample_index].copy()
        report = self._validate_sample(working, sampled, strata, feature_names, max_rows=max_rows)
        sampled = sampled.sort_values([self.time_column, self.id_column]).reset_index(drop=True)
        fallback_to_full = bool(self.validation_cfg.get("fallback_to_full_on_fail", True))
        if report["status"] == "validation_failed" and fallback_to_full:
            report["status"] = "validation_failed_fallback"
            report["used_rows"] = int(len(working))
            return SamplingResult(frame=working.reset_index(drop=True), report=report)

        report["used_rows"] = int(len(sampled))
        return SamplingResult(frame=sampled, report=report)

    def _build_strata(self, frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        work = frame[[self.id_column, self.time_column, *feature_names]].copy()
        snapshot = pd.to_datetime(work[self.time_column]).dt.date.astype(str)
        missing_count = work[feature_names].isnull().sum(axis=1)
        tail_count = self._numeric_tail_count(work[feature_names])
        strata = pd.DataFrame(index=frame.index)
        strata["snapshot_bucket"] = snapshot
        strata["missing_bucket"] = np.where(missing_count > 0, "missing", "complete")
        strata["tail_bucket"] = np.where(tail_count > 0, "tail", "normal")
        strata["stratum"] = (
            strata["snapshot_bucket"].astype(str)
            + "|"
            + strata["missing_bucket"].astype(str)
            + "|"
            + strata["tail_bucket"].astype(str)
        )
        return strata

    def _numeric_tail_count(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.zeros(len(frame), dtype=int)
        threshold = float(self.sampling_cfg.get("tail_z_threshold", 3.5))
        tail_hits = np.zeros((len(frame), len(frame.columns)), dtype=bool)
        for idx, column in enumerate(frame.columns):
            series = pd.to_numeric(frame[column], errors="coerce")
            non_null = series.dropna()
            if non_null.empty:
                continue
            median = float(non_null.median())
            mad = float((non_null - median).abs().median())
            if mad <= 1e-12:
                continue
            robust_z = 0.6745 * (series - median) / mad
            tail_hits[:, idx] = robust_z.abs().fillna(0.0).to_numpy() >= threshold
        return tail_hits.sum(axis=1)

    @staticmethod
    def _sample_index(strata: pd.DataFrame, *, max_rows: int, rng: np.random.Generator) -> np.ndarray:
        counts = strata["stratum"].value_counts().sort_index()
        total = int(counts.sum())
        base = (counts / total * max_rows).to_numpy(dtype=float)
        quotas = np.floor(base).astype(int)
        non_empty = counts.to_numpy() > 0
        quotas = np.where((quotas == 0) & non_empty, 1, quotas)
        quotas = np.minimum(quotas, counts.to_numpy())

        current = int(quotas.sum())
        if current > max_rows:
            overflow = current - max_rows
            order = np.argsort((base - np.floor(base)))
            for idx in order:
                if overflow <= 0:
                    break
                if quotas[idx] > 1:
                    quotas[idx] -= 1
                    overflow -= 1
        elif current < max_rows:
            remaining = max_rows - current
            order = np.argsort(-(base - np.floor(base)))
            for idx in order:
                if remaining <= 0:
                    break
                available = counts.to_numpy()[idx] - quotas[idx]
                if available <= 0:
                    continue
                take = min(available, remaining)
                quotas[idx] += take
                remaining -= take

        selected = []
        for stratum_name, quota in zip(counts.index.tolist(), quotas.tolist()):
            group_index = strata.index[strata["stratum"] == stratum_name].to_numpy()
            if quota >= len(group_index):
                selected.extend(group_index.tolist())
            elif quota > 0:
                chosen = rng.choice(group_index, size=quota, replace=False)
                selected.extend(chosen.tolist())
        return np.asarray(selected, dtype=int)

    def _validate_sample(
        self,
        full_frame: pd.DataFrame,
        sample_frame: pd.DataFrame,
        strata: pd.DataFrame,
        feature_names: list[str],
        *,
        max_rows: int,
    ) -> dict[str, Any]:
        report = {
            "window": "train",
            "status": "applied",
            "original_rows": int(len(full_frame)),
            "sampled_rows": int(len(sample_frame)),
            "target_rows": int(max_rows),
            "sample_rate": round(float(len(sample_frame) / len(full_frame)), 6),
        }

        sample_index = sample_frame.index
        sample_strata = strata.loc[sample_index].copy()
        snapshot_delta = self._share_delta(strata["snapshot_bucket"], sample_strata["snapshot_bucket"])
        tail_delta = self._share_delta(strata["tail_bucket"], sample_strata["tail_bucket"])
        missing_delta = self._share_delta(strata["missing_bucket"], sample_strata["missing_bucket"])
        feature_missing_delta = self._feature_missing_delta(full_frame, sample_frame, feature_names)
        ks_summary = self._ks_summary(full_frame, sample_frame, feature_names)

        report["validation"] = {
            "max_snapshot_share_delta": snapshot_delta,
            "max_tail_share_delta": tail_delta,
            "max_missing_share_delta": missing_delta,
            "max_feature_missing_delta": feature_missing_delta,
            "max_feature_ks": ks_summary["max_feature_ks"],
            "median_feature_ks": ks_summary["median_feature_ks"],
        }

        max_snapshot_share_delta = float(self.validation_cfg.get("max_snapshot_share_delta", 0.02))
        max_tail_share_delta = float(self.validation_cfg.get("max_tail_share_delta", 0.02))
        max_missing_share_delta = float(self.validation_cfg.get("max_missing_share_delta", 0.02))
        max_feature_missing_delta = float(self.validation_cfg.get("max_feature_missing_delta", 0.01))
        max_feature_ks = float(self.validation_cfg.get("max_feature_ks", 0.10))

        checks = {
            "snapshot_share": snapshot_delta <= max_snapshot_share_delta,
            "tail_share": tail_delta <= max_tail_share_delta,
            "missing_share": missing_delta <= max_missing_share_delta,
            "feature_missing": feature_missing_delta <= max_feature_missing_delta,
            "feature_ks": ks_summary["max_feature_ks"] <= max_feature_ks,
        }
        report["validation"]["checks"] = checks
        if not all(checks.values()):
            report["status"] = "validation_failed"
        return report

    @staticmethod
    def _share_delta(full_series: pd.Series, sample_series: pd.Series) -> float:
        full_share = full_series.value_counts(normalize=True)
        sample_share = sample_series.value_counts(normalize=True)
        keys = set(full_share.index).union(sample_share.index)
        if not keys:
            return 0.0
        deltas = [abs(float(full_share.get(key, 0.0)) - float(sample_share.get(key, 0.0))) for key in keys]
        return round(float(max(deltas)), 6)

    @staticmethod
    def _feature_missing_delta(full_frame: pd.DataFrame, sample_frame: pd.DataFrame, feature_names: list[str]) -> float:
        if not feature_names:
            return 0.0
        full_missing = full_frame[feature_names].isnull().mean()
        sample_missing = sample_frame[feature_names].isnull().mean()
        delta = (full_missing - sample_missing).abs().max()
        return round(float(delta), 6)

    @staticmethod
    def _ks_summary(full_frame: pd.DataFrame, sample_frame: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
        stats = []
        for feature in feature_names:
            full_values = pd.to_numeric(full_frame[feature], errors="coerce").dropna()
            sample_values = pd.to_numeric(sample_frame[feature], errors="coerce").dropna()
            if full_values.empty or sample_values.empty:
                continue
            statistic = float(ks_2samp(full_values.to_numpy(), sample_values.to_numpy()).statistic)
            stats.append(statistic)
        if not stats:
            return {"max_feature_ks": 0.0, "median_feature_ks": 0.0}
        stats_arr = np.asarray(stats, dtype=float)
        return {
            "max_feature_ks": round(float(stats_arr.max()), 6),
            "median_feature_ks": round(float(np.median(stats_arr)), 6),
        }
