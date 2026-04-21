"""Config-driven native-to-derived materialization helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from engine.business_features import safe_divide
from engine.config_loader import resolve_project_path
from engine.history_features import (
    add_population_reference_features,
    add_self_history_features,
    add_trend_slope_features,
)
from engine.oracle_io import OracleConnector
from engine.quality import QualityGateError, QualityManager
from engine.source_loader import SourceLoader


logger = logging.getLogger(__name__)


class NativeMaterializer:
    """Build and persist derived input features from native Oracle inputs."""

    def __init__(self, config: dict, secrets: Optional[dict] = None) -> None:
        self.config = config
        self.secrets = secrets
        self.materialization_cfg = config.get("materialization", {}) or {}
        self.pipeline_cfg = config.get("pipeline", {}) or {}
        self.development_cfg = config.get("development", {}) or {}
        self.id_column = str(self.pipeline_cfg.get("id_column", "customer_id")).lower()
        self.time_column = str(self.pipeline_cfg.get("time_column", "snapshot_date")).lower()
        self.segment_column = str(self.development_cfg.get("segment_column", "segment")).lower()
        self.source_loader = SourceLoader(config, secrets)
        self.quality = QualityManager(config)

    @property
    def enabled(self) -> bool:
        return bool(self.materialization_cfg.get("enabled", False))

    @property
    def source_name(self) -> str:
        return str(self.materialization_cfg.get("source_name", "native_features"))

    @property
    def target_name(self) -> str:
        return str(self.materialization_cfg.get("target_name", "input_features"))

    @property
    def persist_target(self) -> bool:
        return bool(self.materialization_cfg.get("persist_target", True))

    @property
    def trim_warmup_snapshots(self) -> int:
        return int(self.materialization_cfg.get("trim_warmup_snapshots", 0))

    @property
    def base_feature_specs(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.materialization_cfg.get("base_features", [])]

    @property
    def base_feature_names(self) -> list[str]:
        return [str(item["name"]).strip().lower() for item in self.base_feature_specs]

    @property
    def trend_features(self) -> list[str]:
        return [
            str(item).strip().lower()
            for item in ((self.materialization_cfg.get("trend", {}) or {}).get("features", []) or [])
        ]

    @property
    def population_features(self) -> list[str]:
        return [
            str(item).strip().lower()
            for item in ((self.materialization_cfg.get("population_reference", {}) or {}).get("features", []) or [])
        ]

    def build_derived_frame(self, native_frame: pd.DataFrame) -> pd.DataFrame:
        """Build the final derived input frame from native atomic fields."""
        frame = native_frame.copy()
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if frame.empty:
            return frame

        frame[self.time_column] = pd.to_datetime(frame[self.time_column], errors="raise")
        frame = frame.sort_values([self.id_column, self.time_column]).reset_index(drop=True)

        passthrough_columns = [self.id_column, self.time_column]
        if self.segment_column in frame.columns:
            passthrough_columns.append(self.segment_column)
        derived = frame[passthrough_columns].copy()

        for feature_spec in self.base_feature_specs:
            feature_name = str(feature_spec["name"]).strip().lower()
            derived[feature_name] = self._evaluate_feature(frame, feature_spec)

        history_cfg = self.materialization_cfg.get("history", {}) or {}
        delta_cfg = history_cfg.get("delta", {}) or {}
        self_z_cfg = history_cfg.get("self_zscore", {}) or {}
        if delta_cfg.get("enabled") or self_z_cfg.get("enabled"):
            derived = add_self_history_features(
                derived,
                base_features=self.base_feature_names,
                id_column=self.id_column,
                time_column=self.time_column,
                delta_lag=int(delta_cfg.get("lag", 1)),
                zscore_window=int(self_z_cfg.get("window", 6)),
                zscore_min_periods=int(self_z_cfg.get("min_periods", 3)),
            )

        trend_cfg = self.materialization_cfg.get("trend", {}) or {}
        if trend_cfg.get("enabled", False) and self.trend_features:
            derived = add_trend_slope_features(
                derived,
                trend_features=self.trend_features,
                id_column=self.id_column,
                time_column=self.time_column,
                window=int(trend_cfg.get("window", 6)),
                min_periods=int(trend_cfg.get("min_periods", 4)),
            )

        population_cfg = self.materialization_cfg.get("population_reference", {}) or {}
        if population_cfg.get("enabled", False) and self.population_features:
            derived = add_population_reference_features(
                derived,
                population_features=self.population_features,
                time_column=self.time_column,
                include_percentile=bool(population_cfg.get("include_percentile", True)),
                include_median_delta=bool(population_cfg.get("include_median_delta", True)),
            )

        return self._trim_warmup_rows(derived)

    def materialize_development(self, segment_value: str) -> dict[str, Any]:
        """Rebuild the full derived history for a segment from native inputs."""
        return self._materialize(
            segment_value=segment_value,
            full_refresh=True,
        )

    def materialize_live(
        self,
        segment_value: str,
        *,
        snapshot_date=None,
        start_date=None,
        end_date=None,
        current_day: bool = False,
        latest_snapshot: bool = False,
    ) -> dict[str, Any]:
        """Refresh only the requested live-scoring scope in the derived table."""
        return self._materialize(
            segment_value=segment_value,
            snapshot_date=snapshot_date,
            start_date=start_date,
            end_date=end_date,
            current_day=current_day,
            latest_snapshot=latest_snapshot,
            full_refresh=False,
        )

    def _materialize(
        self,
        *,
        segment_value: str,
        snapshot_date=None,
        start_date=None,
        end_date=None,
        current_day: bool = False,
        latest_snapshot: bool = False,
        full_refresh: bool,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "persisted_rows": 0, "frame": pd.DataFrame()}

        native_frame = self.source_loader.load_frame(
            self.source_name,
            segment_column=self.segment_column,
            segment_value=None if segment_value == "ALL" else segment_value,
        )
        if native_frame.empty:
            derived_full = pd.DataFrame(columns=[self.id_column, self.time_column, self.segment_column])
        else:
            derived_full = self.build_derived_frame(native_frame)
        scoped_native = self._filter_scope(
            native_frame,
            snapshot_date=snapshot_date,
            start_date=start_date,
            end_date=end_date,
            current_day=current_day,
            latest_snapshot=latest_snapshot,
        )
        scoped_frame = self._filter_scope(
            derived_full,
            snapshot_date=snapshot_date,
            start_date=start_date,
            end_date=end_date,
            current_day=current_day,
            latest_snapshot=latest_snapshot,
        )
        quality_summary = self._build_quality_summary(
            native_frame=native_frame,
            scoped_native=scoped_native,
            derived_full=derived_full,
            scoped_derived=scoped_frame,
            stage="development" if full_refresh else "live_scoring",
        )
        self._enforce_quality(quality_summary, stage="development" if full_refresh else "live_scoring")

        persisted_rows = 0
        if self.persist_target:
            with OracleConnector(self.config, self.secrets) as ora:
                ora.setup_tables(drop_existing=False)
                if full_refresh:
                    persisted_rows = ora.replace_source_scope(
                        self.target_name,
                        scoped_frame,
                        segment=None if segment_value == "ALL" else segment_value,
                        all_rows=True,
                    )
                else:
                    scope = self._resolve_scope(
                        scoped_frame,
                        snapshot_date=snapshot_date,
                        start_date=start_date,
                        end_date=end_date,
                        current_day=current_day,
                        latest_snapshot=latest_snapshot,
                    )
                    persisted_rows = ora.replace_source_scope(
                        self.target_name,
                        scoped_frame,
                        snapshot_date=scope["snapshot_date"],
                        start_date=scope["start_date"],
                        end_date=scope["end_date"],
                        segment=None if segment_value == "ALL" else segment_value,
                    )

        return {
            "enabled": True,
            "segment": segment_value,
            "native_rows": int(len(native_frame)),
            "native_scope_rows": int(len(scoped_native)),
            "derived_rows": int(len(derived_full)),
            "scoped_rows": int(len(scoped_frame)),
            "persisted_rows": int(persisted_rows),
            "frame": scoped_frame,
            "quality": quality_summary,
            "snapshot_min": self._date_or_none(scoped_frame[self.time_column].min()) if not scoped_frame.empty else None,
            "snapshot_max": self._date_or_none(scoped_frame[self.time_column].max()) if not scoped_frame.empty else None,
        }

    def _build_quality_summary(
        self,
        *,
        native_frame: pd.DataFrame,
        scoped_native: pd.DataFrame,
        derived_full: pd.DataFrame,
        scoped_derived: pd.DataFrame,
        stage: str,
    ) -> dict[str, Any]:
        quality = {
            "native_full": self.quality.evaluate(
                native_frame,
                dataset_name="native_full",
                stage=stage,
                rule_key="native",
            ),
            "native_scope": self.quality.evaluate(
                scoped_native,
                dataset_name="native_scope",
                stage=stage,
                rule_key="native",
            ),
            "derived_full": self.quality.evaluate(
                derived_full,
                dataset_name="derived_full",
                stage=stage,
                rule_key="derived",
            ),
            "derived_scope": self.quality.evaluate(
                scoped_derived,
                dataset_name="derived_scope",
                stage=stage,
                rule_key="derived",
            ),
        }
        logger.info(
            "Materialization quality stage=%s native_full=%s native_scope=%s derived_full=%s derived_scope=%s",
            stage,
            quality["native_full"]["status"],
            quality["native_scope"]["status"],
            quality["derived_full"]["status"],
            quality["derived_scope"]["status"],
        )
        self._log_report_details(quality, stage=stage)
        return quality

    def _log_report_details(self, quality_summary: dict[str, Any], *, stage: str) -> None:
        """Emit warning-level human-readable summary for warn/fail reports."""
        for report in quality_summary.values():
            status = str(report.get("status", "")).lower()
            if status not in {"warn", "fail"}:
                continue
            lines = QualityManager.format_report_lines(report)
            level = logging.ERROR if status == "fail" else logging.WARNING
            logger.log(level, "Quality detail stage=%s\n%s", stage, "\n".join(lines))

    def _enforce_quality(self, quality_summary: dict[str, Any], *, stage: str) -> None:
        if stage == "development":
            reports = [quality_summary["native_full"], quality_summary["derived_full"]]
        else:
            reports = []
            if quality_summary["native_scope"].get("status") != "empty":
                reports.append(quality_summary["native_scope"])
            if quality_summary["derived_scope"].get("status") != "empty":
                reports.append(quality_summary["derived_scope"])
        try:
            self.quality.enforce_many(reports, stage=stage)
        except QualityGateError as exc:
            dump_path = self._dump_quality_debug(quality_summary, stage=stage)
            if dump_path is not None:
                message = f"{exc} | See debug report: {dump_path}"
                raise QualityGateError(message) from exc
            raise

    def _dump_quality_debug(self, quality_summary: dict[str, Any], *, stage: str) -> Optional[Path]:
        """Persist the full quality payload so operators can investigate failures."""
        try:
            dump_root = resolve_project_path("runtime/logs/quality")
            dump_root.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dump_path = dump_root / f"{stage}_{timestamp}.json"
            with open(dump_path, "w", encoding="utf-8") as handle:
                json.dump(quality_summary, handle, indent=2, ensure_ascii=False, default=str)
            logger.error("Quality debug report written to %s", dump_path)
            return dump_path
        except Exception:  # noqa: BLE001 - diagnostic path, never mask original error
            logger.exception("Failed to write quality debug report to runtime/logs/quality")
            return None

    def _trim_warmup_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        warmup = self.trim_warmup_snapshots
        if warmup <= 0 or frame.empty:
            return frame

        working = frame.copy()
        working[self.time_column] = pd.to_datetime(working[self.time_column], errors="raise")
        working = working.sort_values([self.id_column, self.time_column]).reset_index(drop=True)
        row_position = working.groupby(self.id_column, sort=False).cumcount()
        return working.loc[row_position >= warmup].reset_index(drop=True)

    def _filter_scope(
        self,
        frame: pd.DataFrame,
        *,
        snapshot_date=None,
        start_date=None,
        end_date=None,
        current_day: bool,
        latest_snapshot: bool,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        working = frame.copy()
        working[self.time_column] = pd.to_datetime(working[self.time_column], errors="raise")
        if snapshot_date is not None:
            target = pd.Timestamp(snapshot_date).normalize()
            return working.loc[working[self.time_column].dt.normalize() == target].reset_index(drop=True)
        if start_date is not None or end_date is not None:
            mask = pd.Series(True, index=working.index)
            if start_date is not None:
                mask &= working[self.time_column].dt.normalize() >= pd.Timestamp(start_date).normalize()
            if end_date is not None:
                mask &= working[self.time_column].dt.normalize() <= pd.Timestamp(end_date).normalize()
            return working.loc[mask].reset_index(drop=True)
        if current_day:
            target = pd.Timestamp.today().normalize()
            return working.loc[working[self.time_column].dt.normalize() == target].reset_index(drop=True)
        if latest_snapshot:
            latest = working[self.time_column].max()
            return working.loc[working[self.time_column] == latest].reset_index(drop=True)
        return working.reset_index(drop=True)

    def _resolve_scope(
        self,
        frame: pd.DataFrame,
        *,
        snapshot_date=None,
        start_date=None,
        end_date=None,
        current_day: bool,
        latest_snapshot: bool,
    ) -> dict[str, Any]:
        if snapshot_date is not None:
            return {"snapshot_date": pd.Timestamp(snapshot_date), "start_date": None, "end_date": None}
        if start_date is not None or end_date is not None:
            return {
                "snapshot_date": None,
                "start_date": pd.Timestamp(start_date) if start_date is not None else None,
                "end_date": pd.Timestamp(end_date) if end_date is not None else None,
            }
        if current_day:
            return {
                "snapshot_date": pd.Timestamp.today().normalize(),
                "start_date": None,
                "end_date": None,
            }
        if latest_snapshot:
            latest = pd.to_datetime(frame[self.time_column]).max() if not frame.empty else None
            return {"snapshot_date": latest, "start_date": None, "end_date": None}
        if frame.empty:
            return {"snapshot_date": None, "start_date": None, "end_date": None}
        return {
            "snapshot_date": None,
            "start_date": pd.to_datetime(frame[self.time_column]).min(),
            "end_date": pd.to_datetime(frame[self.time_column]).max(),
        }

    def _evaluate_feature(self, frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
        method = str(spec.get("method", "")).strip().lower()
        if method == "ratio":
            numerator = self._evaluate_value(frame, spec.get("numerator", {}))
            denominator = self._evaluate_value(frame, spec.get("denominator", {}))
            return safe_divide(numerator, denominator)
        if method == "pct_change_abs":
            current = self._evaluate_value(frame, spec.get("current", {}))
            reference = self._evaluate_value(frame, spec.get("reference", {}))
            return safe_divide(current - reference, np.abs(reference))
        if method == "passthrough":
            return self._evaluate_value(frame, spec.get("source", {})).astype(float)
        raise ValueError(f"Unsupported materialization base feature method: {method}")

    def _evaluate_value(self, frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
        method = str((spec or {}).get("method", "")).strip().lower()
        if method == "column":
            column_name = str(spec["column"]).strip().lower()
            return pd.to_numeric(frame[column_name], errors="coerce")
        if method == "annualized":
            column_name = str(spec["column"]).strip().lower()
            return pd.to_numeric(frame[column_name], errors="coerce") * self._annualization_factor(frame)
        if method == "add":
            values = spec.get("values", []) or []
            if not values:
                return pd.Series(np.nan, index=frame.index, dtype=float)
            result = pd.Series(0.0, index=frame.index, dtype=float)
            for item in values:
                result = result + self._evaluate_value(frame, item)
            return result
        if method == "multiply":
            values = spec.get("values", []) or []
            if not values:
                return pd.Series(np.nan, index=frame.index, dtype=float)
            result = pd.Series(1.0, index=frame.index, dtype=float)
            for item in values:
                result = result * self._evaluate_value(frame, item)
            return result
        if method == "lag":
            periods = int(spec.get("periods", 1))
            inner = self._evaluate_value(frame, spec.get("value", {}))
            return inner.groupby(frame[self.id_column], sort=False).shift(periods)
        if method == "constant":
            return pd.Series(float(spec.get("value", 0.0)), index=frame.index, dtype=float)
        raise ValueError(f"Unsupported materialization value method: {method}")

    def _annualization_factor(self, frame: pd.DataFrame) -> pd.Series:
        annualization_cfg = self.materialization_cfg.get("annualization", {}) or {}
        period_column = str(annualization_cfg.get("period_column", "fs_period_code")).strip().lower()
        factors = {
            str(key).strip().upper(): float(value)
            for key, value in (annualization_cfg.get("factors", {}) or {}).items()
        }
        normalized = frame[period_column].astype(str).str.upper().str.strip()
        normalized = normalized.str.replace(r"[^A-Z0-9]", "", regex=True)
        normalized = normalized.str.extract(r"(Q1|Q2|Q3|Q4|YE)", expand=False).fillna("YE")
        return normalized.map(factors).fillna(1.0).astype(float)

    @staticmethod
    def _date_or_none(value) -> Optional[str]:
        if pd.isna(value):
            return None
        return pd.Timestamp(value).date().isoformat()
