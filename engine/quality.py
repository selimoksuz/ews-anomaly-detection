"""Data quality controls for native and derived anomaly inputs."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from engine.config_loader import get_non_feature_columns


class QualityGateError(ValueError):
    """Raised when a blocking quality gate fails."""


class QualityManager:
    """Evaluate row, coverage, outlier, duplicate, and freshness controls."""

    _STATUS_ORDER = {"pass": 0, "warn": 1, "fail": 2}

    def __init__(self, config: dict):
        self.config = config
        self.quality_cfg = config.get("quality", {}) or {}
        self.pipeline_cfg = config.get("pipeline", {}) or {}
        self.development_cfg = config.get("development", {}) or {}
        self.id_column = str(self.pipeline_cfg.get("id_column", "customer_id")).strip().lower()
        self.time_column = str(self.pipeline_cfg.get("time_column", "snapshot_date")).strip().lower()
        self.segment_column = str(self.development_cfg.get("segment_column", "segment")).strip().lower()
        self.non_feature_columns = get_non_feature_columns(config)

    @property
    def enabled(self) -> bool:
        return bool(self.quality_cfg.get("enabled", False))

    def evaluate(
        self,
        frame: pd.DataFrame | None,
        *,
        dataset_name: str,
        stage: str,
        rule_key: str,
        feature_columns: Iterable[str] | None = None,
    ) -> dict:
        report = {
            "dataset_name": dataset_name,
            "stage": str(stage).strip().lower(),
            "rule_key": str(rule_key).strip().lower(),
            "status": "disabled" if not self.enabled else "pass",
            "rows": 0,
            "unique_customers": 0,
            "snapshots": 0,
            "feature_count": 0,
            "avg_feature_coverage": 1.0,
            "min_feature_coverage": 1.0,
            "avg_missing_ratio": 0.0,
            "max_feature_missing_ratio": 0.0,
            "max_outlier_share": 0.0,
            "duplicate_key_count": 0,
            "top_missing_features": {},
            "top_outlier_features": {},
            "freshness": {},
            "checks": [],
        }
        if not self.enabled:
            return report

        if frame is None:
            report["status"] = "empty"
            report["checks"].append(
                {"check": "frame_presence", "status": "warn", "observed": 0, "message": "Frame is None."}
            )
            return report

        working = frame.copy()
        working.columns = [str(column).strip().lower() for column in working.columns]
        if working.empty:
            report["status"] = "empty"
            report["checks"].append(
                {"check": "row_count", "status": "warn", "observed": 0, "message": "Frame is empty."}
            )
            return report

        rules = self.quality_cfg.get(report["rule_key"], {}) or {}
        feature_list = self._resolve_feature_columns(working, feature_columns)
        report["rows"] = int(len(working))
        report["unique_customers"] = int(working[self.id_column].astype(str).nunique()) if self.id_column in working.columns else 0
        report["snapshots"] = int(pd.to_datetime(working[self.time_column], errors="coerce").nunique()) if self.time_column in working.columns else 0
        report["feature_count"] = int(len(feature_list))

        checks: list[dict] = []
        checks.append(self._min_threshold_check("row_count", report["rows"], rules.get("min_rows_warn"), rules.get("min_rows_fail")))
        checks.append(
            self._min_threshold_check(
                "unique_customers",
                report["unique_customers"],
                rules.get("min_unique_customers_warn"),
                rules.get("min_unique_customers_fail"),
            )
        )

        if self.id_column in working.columns and self.time_column in working.columns:
            duplicate_count = int(working.duplicated(subset=[self.id_column, self.time_column], keep=False).sum())
            report["duplicate_key_count"] = duplicate_count
            duplicate_fail = bool((rules.get("duplicate_keys", {}) or {}).get("fail_on_any", True))
            duplicate_status = "fail" if duplicate_fail and duplicate_count > 0 else "pass"
            checks.append(
                {
                    "check": "duplicate_keys",
                    "status": duplicate_status,
                    "observed": duplicate_count,
                    "warn_threshold": 0,
                    "fail_threshold": 0 if duplicate_fail else None,
                    "message": "Duplicate customer/snapshot keys detected." if duplicate_count > 0 else "No duplicate keys.",
                }
            )

        if feature_list:
            missing_ratio = working[feature_list].isnull().mean().sort_values(ascending=False)
            coverage = 1.0 - missing_ratio
            report["avg_feature_coverage"] = round(float(coverage.mean()), 6)
            report["min_feature_coverage"] = round(float(coverage.min()), 6)
            report["avg_missing_ratio"] = round(float(missing_ratio.mean()), 6)
            report["max_feature_missing_ratio"] = round(float(missing_ratio.max()), 6)
            report["top_missing_features"] = {
                column: round(float(value), 6)
                for column, value in missing_ratio.head(5).items()
                if value > 0
            }
            checks.append(
                self._min_threshold_check(
                    "avg_feature_coverage",
                    report["avg_feature_coverage"],
                    rules.get("min_avg_coverage_warn"),
                    rules.get("min_avg_coverage_fail"),
                )
            )
            checks.append(
                self._min_threshold_check(
                    "min_feature_coverage",
                    report["min_feature_coverage"],
                    rules.get("min_feature_coverage_warn"),
                    rules.get("min_feature_coverage_fail"),
                )
            )

            outlier_summary = self._outlier_summary(
                working[feature_list],
                z_threshold=float(rules.get("outlier_z_threshold", 6.0)),
            )
            report["max_outlier_share"] = outlier_summary["max_outlier_share"]
            report["top_outlier_features"] = outlier_summary["top_outlier_features"]
            checks.append(
                self._max_threshold_check(
                    "max_outlier_share",
                    report["max_outlier_share"],
                    rules.get("max_outlier_share_warn"),
                    rules.get("max_outlier_share_fail"),
                )
            )

        freshness_summary = self._freshness_summary(working, rules)
        if freshness_summary:
            report["freshness"] = freshness_summary
            for column_name, column_report in freshness_summary.items():
                checks.append(
                    {
                        "check": f"freshness::{column_name}",
                        "status": column_report["status"],
                        "observed": column_report["stale_fail_share"],
                        "warn_threshold": column_report["max_stale_share_warn"],
                        "fail_threshold": column_report["max_stale_share_fail"],
                        "message": (
                            f"stale_warn_share={column_report['stale_warn_share']}, "
                            f"stale_fail_share={column_report['stale_fail_share']}, "
                            f"max_age_days={column_report['max_age_days_observed']}"
                        ),
                    }
                )

        report["checks"] = checks
        report["status"] = self._collapse_status(check["status"] for check in checks)
        return report

    def enforce(self, report: dict, *, stage: str) -> None:
        if not self.enabled:
            return
        stage_key = str(stage).strip().lower()
        if report.get("status") != "fail":
            return
        block_cfg = self.quality_cfg.get("block_on", {}) or {}
        if bool(block_cfg.get(stage_key, False)):
            raise QualityGateError(self.format_failure_message(report, stage=stage_key))

    @staticmethod
    def _format_check(check: dict) -> str:
        name = check.get("check", "unknown_check")
        observed = check.get("observed")
        warn = check.get("warn_threshold")
        fail = check.get("fail_threshold")
        message = check.get("message")
        parts = [f"{name}: observed={observed}"]
        if fail is not None:
            parts.append(f"fail<{fail}>")
        if warn is not None:
            parts.append(f"warn<{warn}>")
        if message:
            parts.append(str(message))
        return " ".join(parts)

    @classmethod
    def format_failure_message(cls, report: dict, *, stage: str) -> str:
        """Human readable one-line summary for QualityGateError."""
        failing_checks = [
            check for check in (report.get("checks") or []) if check.get("status") == "fail"
        ]
        details = "; ".join(cls._format_check(check) for check in failing_checks) or (
            "no per-check details available"
        )
        return (
            f"Quality gate failed for {report.get('dataset_name', 'dataset')} "
            f"(stage={stage}, rule_key={report.get('rule_key')}): {details}"
        )

    @classmethod
    def format_report_lines(cls, report: dict) -> list[str]:
        """Return multi-line human readable summary of a quality report."""
        status = str(report.get("status", "unknown")).upper()
        lines = [
            f"[{status}] {report.get('dataset_name', 'dataset')} "
            f"(stage={report.get('stage')}, rule_key={report.get('rule_key')})",
            (
                f"  rows={report.get('rows')} unique_customers={report.get('unique_customers')} "
                f"snapshots={report.get('snapshots')} feature_count={report.get('feature_count')}"
            ),
            (
                f"  avg_coverage={report.get('avg_feature_coverage')} "
                f"min_coverage={report.get('min_feature_coverage')} "
                f"max_outlier_share={report.get('max_outlier_share')} "
                f"duplicate_keys={report.get('duplicate_key_count')}"
            ),
        ]
        top_missing = report.get("top_missing_features") or {}
        if top_missing:
            missing_fmt = ", ".join(f"{name}={share}" for name, share in top_missing.items())
            lines.append(f"  top_missing: {missing_fmt}")
        top_outliers = report.get("top_outlier_features") or {}
        if top_outliers:
            outlier_fmt = ", ".join(f"{name}={share}" for name, share in top_outliers.items())
            lines.append(f"  top_outliers: {outlier_fmt}")
        freshness = report.get("freshness") or {}
        for column_name, column_report in freshness.items():
            lines.append(
                f"  freshness::{column_name} status={column_report.get('status')} "
                f"stale_fail_share={column_report.get('stale_fail_share')} "
                f"future_date_share={column_report.get('future_date_share')} "
                f"max_age_days={column_report.get('max_age_days_observed')}"
            )
        failing_checks = [
            check for check in (report.get("checks") or []) if check.get("status") == "fail"
        ]
        warn_checks = [
            check for check in (report.get("checks") or []) if check.get("status") == "warn"
        ]
        if failing_checks:
            lines.append("  failing checks:")
            for check in failing_checks:
                lines.append(f"    - {cls._format_check(check)}")
        if warn_checks:
            lines.append("  warn checks:")
            for check in warn_checks:
                lines.append(f"    - {cls._format_check(check)}")
        return lines

    def enforce_many(self, reports: Iterable[dict], *, stage: str) -> None:
        for report in reports:
            self.enforce(report, stage=stage)

    def _resolve_feature_columns(self, frame: pd.DataFrame, feature_columns: Iterable[str] | None) -> list[str]:
        if feature_columns is not None:
            candidates = [str(column).strip().lower() for column in feature_columns if str(column).strip()]
            return [column for column in candidates if column in frame.columns]

        feature_list = []
        for column in frame.columns:
            if column in self.non_feature_columns:
                continue
            if column in {self.id_column, self.time_column, self.segment_column}:
                continue
            feature_list.append(column)
        return feature_list

    @classmethod
    def _collapse_status(cls, statuses: Iterable[str]) -> str:
        highest = "pass"
        for status in statuses:
            normalized = str(status).strip().lower()
            if normalized not in cls._STATUS_ORDER:
                continue
            if cls._STATUS_ORDER[normalized] > cls._STATUS_ORDER[highest]:
                highest = normalized
        return highest

    @staticmethod
    def _min_threshold_check(check_name: str, observed, warn_threshold, fail_threshold) -> dict:
        status = "pass"
        if fail_threshold is not None and observed < fail_threshold:
            status = "fail"
        elif warn_threshold is not None and observed < warn_threshold:
            status = "warn"
        return {
            "check": check_name,
            "status": status,
            "observed": observed,
            "warn_threshold": warn_threshold,
            "fail_threshold": fail_threshold,
        }

    @staticmethod
    def _max_threshold_check(check_name: str, observed, warn_threshold, fail_threshold) -> dict:
        status = "pass"
        if fail_threshold is not None and observed > fail_threshold:
            status = "fail"
        elif warn_threshold is not None and observed > warn_threshold:
            status = "warn"
        return {
            "check": check_name,
            "status": status,
            "observed": observed,
            "warn_threshold": warn_threshold,
            "fail_threshold": fail_threshold,
        }

    @staticmethod
    def _outlier_summary(frame: pd.DataFrame, *, z_threshold: float) -> dict:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        per_feature_share: dict[str, float] = {}
        for column in numeric.columns:
            series = numeric[column].dropna().astype(float)
            if series.empty:
                continue
            median = float(series.median())
            mad = float(np.median(np.abs(series - median)))
            if mad <= 1e-12:
                per_feature_share[column] = 0.0
                continue
            robust_z = 0.6745 * (series - median) / mad
            share = float((np.abs(robust_z) >= z_threshold).mean())
            per_feature_share[column] = share

        if not per_feature_share:
            return {"max_outlier_share": 0.0, "top_outlier_features": {}}
        max_share = max(per_feature_share.values())
        top_outliers = {
            column: round(float(value), 6)
            for column, value in sorted(per_feature_share.items(), key=lambda item: item[1], reverse=True)[:5]
            if value > 0
        }
        return {
            "max_outlier_share": round(float(max_share), 6),
            "top_outlier_features": top_outliers,
        }

    def _freshness_summary(self, frame: pd.DataFrame, rules: dict) -> dict:
        freshness_cfg = rules.get("freshness", {}) or {}
        if not bool(freshness_cfg.get("enabled", False)):
            return {}
        if self.time_column not in frame.columns:
            return {}

        snapshot_dates = pd.to_datetime(frame[self.time_column], errors="coerce")
        summary = {}
        warn_age = freshness_cfg.get("max_age_days_warn")
        fail_age = freshness_cfg.get("max_age_days_fail")
        warn_share_threshold = freshness_cfg.get("max_stale_share_warn")
        fail_share_threshold = freshness_cfg.get("max_stale_share_fail")
        future_warn_share = freshness_cfg.get("max_future_share_warn")
        future_fail_share = freshness_cfg.get("max_future_share_fail")
        for raw_column in freshness_cfg.get("date_columns", []) or []:
            column_name = str(raw_column).strip().lower()
            if column_name not in frame.columns:
                continue

            source_dates = pd.to_datetime(frame[column_name], errors="coerce")
            valid_mask = snapshot_dates.notna() & source_dates.notna()
            if not valid_mask.any():
                summary[column_name] = {
                    "status": "warn",
                    "valid_rows": 0,
                    "stale_warn_share": None,
                    "stale_fail_share": None,
                    "future_date_share": None,
                    "max_age_days_observed": None,
                    "max_stale_share_warn": warn_share_threshold,
                    "max_stale_share_fail": fail_share_threshold,
                    "max_future_share_warn": future_warn_share,
                    "max_future_share_fail": future_fail_share,
                }
                continue

            ages = (snapshot_dates.loc[valid_mask].dt.normalize() - source_dates.loc[valid_mask].dt.normalize()).dt.days.astype(float)
            future_share = float((ages < 0).mean())
            # Future dates (fs issued after snapshot) are expected for recent filings; only stale side drives fail.
            # Oldness checks run on non-negative ages so future records are ignored when measuring staleness.
            past_ages = ages[ages >= 0]
            if past_ages.empty:
                stale_warn_share = 0.0
                stale_fail_share = 0.0
                max_age_days = 0
            else:
                stale_warn_share = float((past_ages > float(warn_age)).mean()) if warn_age is not None else 0.0
                stale_fail_share = float((past_ages > float(fail_age)).mean()) if fail_age is not None else 0.0
                max_age_days = int(past_ages.max())

            status = "pass"
            if fail_share_threshold is not None and stale_fail_share > float(fail_share_threshold):
                status = "fail"
            elif future_fail_share is not None and future_share > float(future_fail_share):
                status = "fail"
            elif warn_share_threshold is not None and stale_warn_share > float(warn_share_threshold):
                status = "warn"
            elif future_warn_share is not None and future_share > float(future_warn_share):
                status = "warn"

            summary[column_name] = {
                "status": status,
                "valid_rows": int(valid_mask.sum()),
                "stale_warn_share": round(stale_warn_share, 6),
                "stale_fail_share": round(stale_fail_share, 6),
                "future_date_share": round(future_share, 6),
                "max_age_days_observed": max_age_days,
                "max_stale_share_warn": warn_share_threshold,
                "max_stale_share_fail": fail_share_threshold,
                "max_future_share_warn": future_warn_share,
                "max_future_share_fail": future_fail_share,
            }
        return summary
