"""Monitoring summaries for input populations, scores, and outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class MonitoringManager:
    """Build compact monitoring payloads for run-level artifacts."""

    def __init__(self, config: dict):
        self.config = config
        self.id_column = config["pipeline"]["id_column"]
        self.time_column = config["pipeline"]["time_column"]

    def summarize_input(self, frame: pd.DataFrame, feature_names: list[str]) -> dict:
        if frame.empty:
            return {"rows": 0, "unique_customers": 0, "snapshots": 0}

        dates = pd.to_datetime(frame[self.time_column])
        missing = frame[feature_names].isnull().mean().sort_values(ascending=False)
        return {
            "rows": int(len(frame)),
            "unique_customers": int(frame[self.id_column].nunique()),
            "snapshots": int(dates.nunique()),
            "start": dates.min().date().isoformat(),
            "end": dates.max().date().isoformat(),
            "avg_feature_missing_ratio": round(float(frame[feature_names].isnull().mean().mean()), 6),
            "top_missing_features": {
                column: round(float(value), 6)
                for column, value in missing.head(5).items()
                if value > 0
            },
        }

    def summarize_scores(self, results: pd.DataFrame) -> dict:
        if results.empty:
            return {"rows": 0}

        summary = {
            "rows": int(len(results)),
            "band_share": {
                band: round(float((results["alert_band"] == band).mean()), 4)
                for band in ("NORMAL", "SARI", "TURUNCU", "KIRMIZI")
            },
        }
        for column in (
            "anomaly_score",
            "ae_score",
            "if_score",
            "md_score",
            "ae_raw",
            "if_raw",
            "md_raw",
        ):
            if column not in results.columns:
                continue
            values = np.asarray(results[column], dtype=float)
            summary[column] = {
                "mean": round(float(values.mean()), 4),
                "median": round(float(np.median(values)), 4),
                "p95": round(float(np.percentile(values, 95)), 4),
                "p99": round(float(np.percentile(values, 99)), 4),
            }
        return summary

    def summarize_outcomes(self, frame: pd.DataFrame, target_columns: list[str]) -> dict:
        if frame.empty:
            return {"rows": 0}
        return {
            column: {
                "positive_rows": int(frame[column].sum()),
                "positive_rate": round(float(frame[column].mean()), 4),
            }
            for column in target_columns
            if column in frame.columns
        }

    @staticmethod
    def write_json(path: Path, payload: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
