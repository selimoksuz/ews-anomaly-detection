"""Monitoring summaries for input populations, scores, and outcomes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config_loader import resolve_project_path
from engine.run_logging import get_run_log_path


class MonitoringManager:
    """Build compact monitoring payloads for run-level artifacts."""

    def __init__(self, config: dict):
        self.config = config
        self.id_column = config["pipeline"]["id_column"]
        self.time_column = config["pipeline"]["time_column"]
        self.monitoring_dir = resolve_project_path(
            config.get("monitoring", {}).get("directory", "runtime/monitoring")
        )

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

    def write_bundle(self, *, segment: str, run_id: str, payload: dict) -> dict:
        bundle_dir = self.monitoring_dir / str(segment).strip() / str(run_id).strip()
        bundle_dir.mkdir(parents=True, exist_ok=True)

        summary_path = bundle_dir / "monitoring.json"
        self.write_json(summary_path, payload)

        written_files = {"monitoring_path": str(summary_path), "directory": str(bundle_dir)}
        section_frames = {
            "input_summary.csv": self._section_to_frame(payload.get("input")),
            "score_summary.csv": self._section_to_frame(payload.get("scores")),
            "outcome_summary.csv": self._section_to_frame(payload.get("outcomes")),
        }
        for file_name, frame in section_frames.items():
            if frame.empty:
                continue
            csv_path = bundle_dir / file_name
            frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
            written_files[file_name.replace(".csv", "_path")] = str(csv_path)

        if payload.get("sampling"):
            sampling_path = bundle_dir / "sampling.json"
            self.write_json(sampling_path, payload["sampling"])
            written_files["sampling_path"] = str(sampling_path)

        self._write_monitoring_log(segment=segment, run_id=run_id, payload=payload, written_files=written_files)
        return written_files

    @staticmethod
    def write_json(path: Path, payload: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _section_to_frame(payload) -> pd.DataFrame:
        if not payload:
            return pd.DataFrame()
        if isinstance(payload, dict):
            if payload and all(isinstance(value, dict) for value in payload.values()):
                rows = []
                for scope, content in payload.items():
                    row = {"scope": scope}
                    row.update(pd.json_normalize(content, sep=".").to_dict(orient="records")[0])
                    rows.append(row)
                return pd.DataFrame(rows)
            return pd.json_normalize(payload, sep=".")
        return pd.DataFrame([{"value": payload}])

    def _write_monitoring_log(self, *, segment: str, run_id: str, payload: dict, written_files: dict):
        log_path = get_run_log_path(self.config, category="monitoring", run_id=run_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} | INFO | monitoring | segment={segment} run_id={run_id}",
            f"written_files={json.dumps(written_files, ensure_ascii=False)}",
        ]
        for section_name in ("input", "scores", "outcomes", "sampling"):
            if section_name in payload and payload[section_name]:
                lines.append(f"{section_name}_keys={list(payload[section_name].keys()) if isinstance(payload[section_name], dict) else section_name}")
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
