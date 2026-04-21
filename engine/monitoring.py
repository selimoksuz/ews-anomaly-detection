"""Monitoring summaries for input populations, scores, and outcomes."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from engine.run_logging import get_run_directory, get_run_log_path


logger = logging.getLogger(__name__)


class MonitoringManager:
    """Build compact monitoring payloads for run-level artifacts."""

    def __init__(self, config: dict):
        self.config = config
        self.id_column = config["pipeline"]["id_column"]
        self.time_column = config["pipeline"]["time_column"]
        self.history_cfg = (config.get("monitoring", {}) or {}).get("history", {}) or {}

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

    SCORE_BUCKET_EDGES: tuple[float, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

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
            values = values[~np.isnan(values)]
            if values.size == 0:
                continue
            summary[column] = {
                "mean": round(float(values.mean()), 4),
                "median": round(float(np.median(values)), 4),
                "p95": round(float(np.percentile(values, 95)), 4),
                "p99": round(float(np.percentile(values, 99)), 4),
                "skew": round(float(self._series_skew(values)), 4),
                "kurtosis": round(float(self._series_kurtosis(values)), 4),
            }
            if column == "anomaly_score":
                summary["score_buckets"] = self._bucket_counts(values)
        return summary

    @classmethod
    def _bucket_counts(cls, values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {}
        edges = np.array(cls.SCORE_BUCKET_EDGES, dtype=float)
        counts, _ = np.histogram(values, bins=edges)
        total = float(counts.sum()) or 1.0
        buckets = {}
        for idx in range(len(counts)):
            low = int(edges[idx])
            high = int(edges[idx + 1])
            buckets[f"{low:03d}_{high:03d}"] = round(float(counts[idx] / total), 6)
        return buckets

    @staticmethod
    def _series_skew(values: np.ndarray) -> float:
        if values.size < 3:
            return 0.0
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std <= 1e-12:
            return 0.0
        return float(((values - mean) ** 3).mean() / (std ** 3))

    @staticmethod
    def _series_kurtosis(values: np.ndarray) -> float:
        if values.size < 4:
            return 0.0
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std <= 1e-12:
            return 0.0
        return float(((values - mean) ** 4).mean() / (std ** 4) - 3.0)

    @staticmethod
    def compute_psi(current: dict, previous: dict) -> Optional[float]:
        """Population Stability Index between two bucket share dicts."""
        if not current or not previous:
            return None
        keys = sorted(set(current) | set(previous))
        eps = 1e-6
        psi = 0.0
        for key in keys:
            p_curr = max(float(current.get(key, 0.0)), eps)
            p_prev = max(float(previous.get(key, 0.0)), eps)
            psi += (p_curr - p_prev) * np.log(p_curr / p_prev)
        return round(float(psi), 6)

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
        bundle_dir = get_run_directory(self.config, run_id) / "monitoring"
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

        if payload.get("quality"):
            quality_path = bundle_dir / "quality.json"
            self.write_json(quality_path, payload["quality"])
            written_files["quality_path"] = str(quality_path)

            quality_summary = self._quality_summary_frame(payload["quality"])
            if not quality_summary.empty:
                quality_summary_path = bundle_dir / "quality_summary.csv"
                quality_summary.to_csv(quality_summary_path, index=False, encoding="utf-8-sig")
                written_files["quality_summary_path"] = str(quality_summary_path)

            quality_checks = self._quality_checks_frame(payload["quality"])
            if not quality_checks.empty:
                quality_checks_path = bundle_dir / "quality_checks.csv"
                quality_checks.to_csv(quality_checks_path, index=False, encoding="utf-8-sig")
                written_files["quality_checks_path"] = str(quality_checks_path)

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

    @staticmethod
    def _quality_reports(payload, prefix: str = ""):
        if isinstance(payload, dict):
            if "status" in payload and "dataset_name" in payload:
                report = dict(payload)
                report["_quality_key"] = prefix.rstrip(".")
                yield report
            else:
                for key, value in payload.items():
                    next_prefix = f"{prefix}.{key}" if prefix else str(key)
                    yield from MonitoringManager._quality_reports(value, next_prefix)

    @staticmethod
    def _quality_summary_frame(payload) -> pd.DataFrame:
        rows = []
        for report in MonitoringManager._quality_reports(payload):
            rows.append(
                {
                    "quality_key": report.get("_quality_key"),
                    "dataset_name": report.get("dataset_name"),
                    "stage": report.get("stage"),
                    "rule_key": report.get("rule_key"),
                    "status": report.get("status"),
                    "rows": report.get("rows"),
                    "unique_customers": report.get("unique_customers"),
                    "snapshots": report.get("snapshots"),
                    "feature_count": report.get("feature_count"),
                    "avg_feature_coverage": report.get("avg_feature_coverage"),
                    "min_feature_coverage": report.get("min_feature_coverage"),
                    "avg_missing_ratio": report.get("avg_missing_ratio"),
                    "max_feature_missing_ratio": report.get("max_feature_missing_ratio"),
                    "max_outlier_share": report.get("max_outlier_share"),
                    "duplicate_key_count": report.get("duplicate_key_count"),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _quality_checks_frame(payload) -> pd.DataFrame:
        rows = []
        for report in MonitoringManager._quality_reports(payload):
            for check in report.get("checks", []) or []:
                rows.append(
                    {
                        "quality_key": report.get("_quality_key"),
                        "dataset_name": report.get("dataset_name"),
                        "rule_key": report.get("rule_key"),
                        "report_status": report.get("status"),
                        "check": check.get("check"),
                        "check_status": check.get("status"),
                        "observed": check.get("observed"),
                        "warn_threshold": check.get("warn_threshold"),
                        "fail_threshold": check.get("fail_threshold"),
                        "message": check.get("message"),
                    }
                )
        return pd.DataFrame(rows)

    def _write_monitoring_log(self, *, segment: str, run_id: str, payload: dict, written_files: dict):
        log_path = get_run_log_path(self.config, category="monitoring", run_id=run_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} | INFO | monitoring | segment={segment} run_id={run_id}",
            f"written_files={json.dumps(written_files, ensure_ascii=False)}",
        ]
        for section_name in ("input", "scores", "outcomes", "sampling", "quality"):
            if section_name in payload and payload[section_name]:
                lines.append(f"{section_name}_keys={list(payload[section_name].keys()) if isinstance(payload[section_name], dict) else section_name}")
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    # ---------------------------------------------------------------------
    # Monitor history (Oracle-backed run-level trend table)
    # ---------------------------------------------------------------------

    def build_history_row(
        self,
        *,
        run_info: dict,
        payload: Optional[dict] = None,
        extras: Optional[dict] = None,
    ) -> dict:
        """Flatten a monitoring payload into a single history row."""
        payload = payload or {}
        extras = extras or {}
        scores = payload.get("scores") or {}
        bands = scores.get("band_share") or {}
        input_section = payload.get("input") or {}
        materialization_quality = (payload.get("quality") or {}).get("materialization") or {}
        stability = extras.get("stability") or {}

        def _stability_metrics(window_key):
            window = stability.get(window_key) if stability else None
            if not isinstance(window, dict):
                return {}
            return (window.get("metrics") or {}).get("ensemble_score", {}) or {}

        stability_test = _stability_metrics("test")
        stability_cal = _stability_metrics("calibration")
        stability_oot = _stability_metrics("oot")

        supervised = extras.get("supervised") or {}
        weights = extras.get("weights") or {}
        calibration = extras.get("calibration") or {}
        freshness_detail = extras.get("freshness") or {}

        def _num(value):
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        def _score_stat(section, stat):
            inner = scores.get(section)
            if not isinstance(inner, dict):
                return None
            return _num(inner.get(stat))

        def _quality_field(key, attr="status"):
            report = materialization_quality.get(key) or {}
            value = report.get(attr)
            if attr == "status":
                return str(value).upper() if value is not None else None
            return _num(value)

        def _as_date(value):
            if value in (None, "", "null"):
                return None
            try:
                return pd.Timestamp(value).to_pydatetime()
            except Exception:  # noqa: BLE001
                return None

        started_at = _as_date(run_info.get("started_at"))
        finished_at = _as_date(
            run_info.get("finished_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        duration_seconds = None
        if started_at and finished_at:
            try:
                duration_seconds = round((finished_at - started_at).total_seconds(), 3)
            except Exception:  # noqa: BLE001
                duration_seconds = None

        freshness_age = _num(freshness_detail.get("max_age_days"))
        if freshness_age is None:
            native_full = materialization_quality.get("native_full") or {}
            fs_report = (native_full.get("freshness") or {}).get("fs_last_update_date") or {}
            freshness_age = _num(fs_report.get("max_age_days_observed"))

        score_buckets = scores.get("score_buckets") or {}
        score_buckets_json = json.dumps(score_buckets, ensure_ascii=False) if score_buckets else None

        dominant = extras.get("dominant_reason") or {}

        row = {
            "run_id": run_info.get("run_id"),
            "run_type": run_info.get("run_type"),
            "segment": run_info.get("segment"),
            "status": run_info.get("status"),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
            "model_version": run_info.get("model_version"),
            "scope_snapshot": _as_date(run_info.get("scope_snapshot")),
            "scope_start": _as_date(run_info.get("scope_start")),
            "scope_end": _as_date(run_info.get("scope_end")),
            "input_rows": int(input_section.get("rows", 0)) if input_section.get("rows") is not None else None,
            "input_customers": int(input_section.get("unique_customers", 0)) if input_section.get("unique_customers") is not None else None,
            "input_snapshots": int(input_section.get("snapshots", 0)) if input_section.get("snapshots") is not None else None,
            "avg_missing_ratio": _num(input_section.get("avg_feature_missing_ratio")),
            "band_normal": _num(bands.get("NORMAL")),
            "band_sari": _num(bands.get("SARI")),
            "band_turuncu": _num(bands.get("TURUNCU")),
            "band_kirmizi": _num(bands.get("KIRMIZI")),
            "band_persistence_kirmizi": _num(extras.get("band_persistence_kirmizi")),
            "score_mean": _score_stat("anomaly_score", "mean"),
            "score_median": _score_stat("anomaly_score", "median"),
            "score_p95": _score_stat("anomaly_score", "p95"),
            "score_p99": _score_stat("anomaly_score", "p99"),
            "score_skew": _score_stat("anomaly_score", "skew"),
            "score_kurtosis": _score_stat("anomaly_score", "kurtosis"),
            "score_psi_vs_prev": _num(extras.get("score_psi_vs_prev")),
            "score_buckets": score_buckets_json,
            "ae_score_mean": _score_stat("ae_score", "mean"),
            "if_score_mean": _score_stat("if_score", "mean"),
            "md_score_mean": _score_stat("md_score", "mean"),
            "quality_native_full": _quality_field("native_full"),
            "quality_native_scope": _quality_field("native_scope"),
            "quality_derived_full": _quality_field("derived_full"),
            "quality_derived_scope": _quality_field("derived_scope"),
            "native_avg_coverage": _quality_field("native_full", "avg_feature_coverage"),
            "derived_avg_coverage": _quality_field("derived_full", "avg_feature_coverage"),
            "native_max_outlier": _quality_field("native_full", "max_outlier_share"),
            "derived_max_outlier": _quality_field("derived_full", "max_outlier_share"),
            "freshness_max_age_days": int(freshness_age) if freshness_age is not None else None,
            "stability_test_ks": _num(stability_test.get("ks_stat")),
            "stability_test_mean_ratio": _num(stability_test.get("mean_ratio")),
            "stability_cal_ks": _num(stability_cal.get("ks_stat")),
            "stability_cal_mean_ratio": _num(stability_cal.get("mean_ratio")),
            "stability_oot_ks": _num(stability_oot.get("ks_stat")),
            "stability_oot_mean_ratio": _num(stability_oot.get("mean_ratio")),
            "calibration_rows": int(calibration["rows"]) if calibration.get("rows") is not None else None,
            "calibration_monotonic": int(bool(calibration.get("monotonic"))) if calibration.get("monotonic") is not None else None,
            "supervised_precision": _num(supervised.get("precision_at_top_percent")),
            "supervised_recall": _num(supervised.get("recall_at_top_percent")),
            "supervised_f1": _num(supervised.get("f1_at_top_percent")),
            "supervised_lift": _num(supervised.get("lift_at_top_percent")),
            "weight_ae": _num(weights.get("autoencoder") if weights else None),
            "weight_if": _num(weights.get("isolation_forest") if weights else None),
            "weight_md": _num(weights.get("mahalanobis") if weights else None),
            "dominant_reason_feature": dominant.get("feature"),
            "dominant_reason_share": _num(dominant.get("share")),
            "result_row_count": int(extras["result_row_count"]) if extras.get("result_row_count") is not None else None,
            "monitoring_path": run_info.get("monitoring_path"),
        }
        return row

    def write_history_row(self, oracle_connector, row: dict) -> int:
        """Persist a history row to the Oracle monitor history table."""
        if not row.get("run_id"):
            return 0
        if str(self.history_cfg.get("backend", "oracle")).lower() != "oracle":
            return 0
        try:
            return oracle_connector.write_monitor_history_row(row)
        except Exception:  # noqa: BLE001 - history is best-effort, never blocks a run
            logger.exception("Failed to persist monitor history row for run_id=%s", row.get("run_id"))
            return 0
