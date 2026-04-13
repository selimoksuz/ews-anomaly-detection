"""Calibration utilities for raw anomaly model scores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


RAW_COMPONENT_COLUMNS = ("ae_raw", "if_raw", "md_raw")
CAL_COMPONENT_COLUMNS = ("ae_cal", "if_cal", "md_cal")


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_monotonic(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([0.0], dtype=float)
    return np.maximum.accumulate(values)


@dataclass
class CalibrationArtifact:
    version: str
    method: str
    model_version: str
    segment: str
    created_at: str
    window: dict
    n_rows: int
    quantile_levels: list[float]
    grids: dict[str, list[float]]

    def to_dict(self):
        return {
            "version": self.version,
            "method": self.method,
            "model_version": self.model_version,
            "segment": self.segment,
            "created_at": self.created_at,
            "window": self.window,
            "n_rows": self.n_rows,
            "quantile_levels": self.quantile_levels,
            "grids": self.grids,
        }


class ScoreCalibrator:
    """Fit and apply empirical score calibration artifacts."""

    def __init__(self, config: dict):
        self.config = config
        calibration_cfg = config.get("calibration", {})
        scoring_cfg = config.get("scoring", {})
        quantiles = int(calibration_cfg.get("quantiles", 101) or 101)
        quantiles = max(11, quantiles)
        self.method = calibration_cfg.get("method", "empirical_cdf")
        self.quantile_levels = np.linspace(0.0, 1.0, quantiles)
        self.score_min = float(scoring_cfg.get("score_min", 0))
        self.score_max = float(scoring_cfg.get("score_max", 100))

    def fit(self, raw_scores: dict[str, np.ndarray], *, model_version: str, segment: str, window: dict):
        grids = {}
        for column in RAW_COMPONENT_COLUMNS:
            values = np.asarray(raw_scores[column], dtype=float)
            if values.size == 0:
                raise ValueError(f"Calibration raw scores are empty for '{column}'.")
            grids[column] = _ensure_monotonic(
                np.quantile(values, self.quantile_levels)
            ).tolist()

        return CalibrationArtifact(
            version=f"{model_version}-cal",
            method=self.method,
            model_version=model_version,
            segment=segment,
            created_at=_utc_now(),
            window=window,
            n_rows=int(len(raw_scores[RAW_COMPONENT_COLUMNS[0]])),
            quantile_levels=self.quantile_levels.tolist(),
            grids=grids,
        )

    def apply(self, raw_scores: dict[str, np.ndarray], artifact) -> dict[str, np.ndarray]:
        payload = artifact.to_dict() if isinstance(artifact, CalibrationArtifact) else dict(artifact)
        quantile_levels = np.asarray(payload["quantile_levels"], dtype=float)
        calibrated = {}
        for raw_column, cal_column in zip(RAW_COMPONENT_COLUMNS, CAL_COMPONENT_COLUMNS):
            grid = _ensure_monotonic(payload["grids"][raw_column])
            values = np.asarray(raw_scores[raw_column], dtype=float)
            percentile = np.interp(values, grid, quantile_levels, left=0.0, right=1.0)
            calibrated[cal_column] = np.clip(percentile * self.score_max, self.score_min, self.score_max)
        return calibrated

    @staticmethod
    def save(path: Path, artifact):
        payload = artifact.to_dict() if isinstance(artifact, CalibrationArtifact) else artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: Path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
