"""Oracle input and output utilities for the EWS anomaly detection pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union
import logging

import pandas as pd
import yaml

try:
    import oracledb
except ImportError as exc:  # pragma: no cover
    oracledb = None
    _ORACLEDB_IMPORT_ERROR = exc
else:
    _ORACLEDB_IMPORT_ERROR = None


ConfigSource = Optional[Union[str, Path, Mapping[str, Any]]]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline_config.yaml"
DEFAULT_SECRETS_CONFIG = PROJECT_ROOT / "config" / "secrets.yaml"


def load_yaml_config(source: ConfigSource, default_path: Optional[Path] = None) -> dict[str, Any]:
    """Load YAML configuration from a mapping or file path."""
    if isinstance(source, Mapping):
        data = dict(source)
        refs = data.get("config_refs", {}) or {}
        if not refs:
            return data
        base_path = default_path or DEFAULT_PIPELINE_CONFIG
        return _resolve_config_refs(data, Path(base_path))

    path = Path(source) if source is not None else default_path
    if path is None:
        raise ValueError("A configuration mapping or file path is required.")
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, Mapping):
        raise ValueError(f"Configuration file must contain a mapping at the root: {path}")

    return _resolve_config_refs(dict(data), path)


def _resolve_config_refs(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    merged = dict(config)
    refs = config.get("config_refs", {}) or {}
    if not isinstance(refs, Mapping):
        raise ValueError("config_refs must be a mapping of section names to yaml file paths.")
    for section_name, relative_path in refs.items():
        section_key = str(section_name).strip()
        existing_value = merged.get(section_key)
        if isinstance(existing_value, Mapping) and existing_value:
            continue
        resolved_path = (config_path.parent / str(relative_path)).resolve()
        with resolved_path.open("r", encoding="utf-8") as handle:
            section_payload = yaml.safe_load(handle) or {}
        if not isinstance(section_payload, Mapping):
            raise ValueError(f"Referenced config file must contain a mapping at the root: {resolved_path}")
        merged[section_key] = dict(section_payload)
    return merged


class OracleConnector:
    """Read pipeline configuration and manage Oracle data exchange."""

    def __init__(
        self,
        pipeline_config: ConfigSource = None,
        secrets: ConfigSource = None,
    ) -> None:
        self.pipeline_config = load_yaml_config(pipeline_config, DEFAULT_PIPELINE_CONFIG)
        self.secrets = load_yaml_config(secrets, DEFAULT_SECRETS_CONFIG)

        self.pipeline_settings = self.pipeline_config["pipeline"]
        self.oracle_settings = self.pipeline_config["oracle"]
        self.connection_settings = self.secrets["oracle"]
        self.logging_settings = self.pipeline_config.get("logging", {})

        logger_name = self.logging_settings.get("logger_name", "ews.oracle")
        self.logger = logging.getLogger(logger_name)

        self.id_column = str(self.pipeline_settings["id_column"])
        self.time_column = str(self.pipeline_settings["time_column"])
        self.split_column = str(self.pipeline_settings.get("split_column", "split_flag"))

        self.connection = None

    def __enter__(self) -> "OracleConnector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def schema(self) -> str:
        schema = self.oracle_settings.get("schema") or self.connection_settings.get("schema")
        if not schema:
            raise ValueError("Oracle schema must be defined in pipeline_config.yaml or secrets.yaml.")
        return str(schema).upper()

    @property
    def feature_definitions(self) -> list[dict[str, Any]]:
        feature_groups = self.pipeline_config.get("features", {})
        flattened: list[dict[str, Any]] = []
        for group_name in ("instant", "rolling_4w", "trend"):
            for item in feature_groups.get(group_name, []):
                record = dict(item)
                record["group"] = group_name
                flattened.append(record)
        return flattened

    @property
    def feature_names(self) -> list[str]:
        return [item["name"] for item in self.feature_definitions]

    @property
    def feature_labels(self) -> dict[str, str]:
        labels = {
            item["name"]: item.get("label_tr", item["name"])
            for item in self.feature_definitions
        }
        labels.update(
            {
                str(name).strip().lower(): value
                for name, value in (self.pipeline_config.get("features", {}).get("label_overrides", {}) or {}).items()
            }
        )
        return labels

    @property
    def top_n_reasons(self) -> int:
        return int(self.pipeline_config.get("scoring", {}).get("top_n_reasons", 3))

    @property
    def segment_column(self) -> str:
        return str(
            self.pipeline_config.get("development", {}).get("segment_column", "segment")
        )

    def connect(self):
        """Create and cache an Oracle database connection."""
        if self.connection is not None:
            return self.connection

        if oracledb is None:
            raise RuntimeError(
                "The 'oracledb' package is required to use OracleConnector."
            ) from _ORACLEDB_IMPORT_ERROR

        host = self.connection_settings["host"]
        port = self.connection_settings["port"]
        service_name = self.connection_settings["service_name"]
        proxy_user = self.connection_settings["proxy_user"]
        user = self.connection_settings["user"]
        password = self.connection_settings["password"]

        dsn = f"{host}:{port}/{service_name}"
        proxy_identity = f"{proxy_user}[{user}]"
        self.logger.info("Opening Oracle connection to %s", dsn)
        self.connection = oracledb.connect(user=proxy_identity, password=password, dsn=dsn)
        return self.connection

    def close(self) -> None:
        """Close the cached Oracle connection."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def read_training_data(self, split: str = "TRAIN") -> pd.DataFrame:
        """Read training data from the configured Oracle table."""
        column_names = [self.id_column, self.time_column, self.split_column]
        available_columns = self._table_columns("training")
        if self.segment_column.upper() in available_columns:
            column_names.append(self.segment_column)
        column_names.extend(self.feature_names)
        select_clause = ", ".join(column.upper() for column in column_names)
        table_name = self._qualified_table_name("training")
        sql = (
            f"SELECT {select_clause} "
            f"FROM {table_name} "
            f"WHERE UPPER({self.split_column.upper()}) = UPPER(:split_value)"
        )
        return self._read_query(sql, {"split_value": split})

    def read_scoring_data(self) -> pd.DataFrame:
        """Read scoring data from the configured Oracle table."""
        column_names = [self.id_column, self.time_column]
        available_columns = self._table_columns("scoring")
        if self.segment_column.upper() in available_columns:
            column_names.append(self.segment_column)
        column_names.extend(self.feature_names)
        select_clause = ", ".join(column.upper() for column in column_names)
        table_name = self._qualified_table_name("scoring")
        sql = f"SELECT {select_clause} FROM {table_name}"
        return self._read_query(sql)

    def replace_rows(self, table_key: str, frame: pd.DataFrame, batch_size: int = 1000) -> int:
        """Replace table contents with the provided frame for controlled test loads."""
        normalized = self._normalize_columns(frame)
        with self.connect().cursor() as cursor:
            cursor.execute(f"TRUNCATE TABLE {self._qualified_table_name(table_key)}")
        self.connection.commit()
        return self.write_source_rows(table_key, normalized, batch_size=batch_size)

    def replace_source_scope(
        self,
        table_key: str,
        frame: pd.DataFrame,
        *,
        snapshot_date=None,
        start_date=None,
        end_date=None,
        segment: Optional[str] = None,
        all_rows: bool = False,
        batch_size: int = 1000,
    ) -> int:
        """Delete a scoped slice of a source table and insert the replacement rows."""
        normalized = self._normalize_columns(frame)
        self.delete_source_rows(
            table_key,
            snapshot_date=snapshot_date,
            start_date=start_date,
            end_date=end_date,
            segment=segment,
            all_rows=all_rows,
        )
        if normalized.empty:
            return 0
        return self.write_source_rows(table_key, normalized, batch_size=batch_size)

    def delete_source_rows(
        self,
        table_key: str,
        *,
        snapshot_date=None,
        start_date=None,
        end_date=None,
        segment: Optional[str] = None,
        all_rows: bool = False,
    ) -> int:
        """Delete rows from a source table by scope."""
        active_keys = self._active_table_keys()
        if table_key not in active_keys:
            return 0

        available_columns = self._table_columns(table_key)
        clauses: list[str] = []
        params: dict[str, Any] = {}

        if all_rows:
            clauses.append("1 = 1")
        elif snapshot_date is not None:
            clauses.append(f"TRUNC({self.time_column.upper()}) = TRUNC(:snapshot_date)")
            params["snapshot_date"] = pd.Timestamp(snapshot_date).to_pydatetime()
        else:
            if start_date is not None:
                clauses.append(f"TRUNC({self.time_column.upper()}) >= TRUNC(:start_date)")
                params["start_date"] = pd.Timestamp(start_date).to_pydatetime()
            if end_date is not None:
                clauses.append(f"TRUNC({self.time_column.upper()}) <= TRUNC(:end_date)")
                params["end_date"] = pd.Timestamp(end_date).to_pydatetime()

        if segment and segment != "ALL" and self.segment_column.upper() in available_columns:
            clauses.append(f"{self.segment_column.upper()} = :segment_value")
            params["segment_value"] = segment

        if not clauses:
            raise ValueError(
                f"delete_source_rows for {table_key} requires snapshot_date, date range, or all_rows=True."
            )

        connection = self.connect()
        with connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {self._qualified_table_name(table_key)} WHERE {' AND '.join(clauses)}",
                params,
            )
            deleted = int(cursor.rowcount or 0)
        connection.commit()
        return deleted

    def delete_scored_snapshot(self, snapshot_date, *, segment: Optional[str] = None) -> dict[str, int]:
        """Delete previously written scored outputs for the same snapshot before re-inserting."""
        return self.delete_scored_scope(snapshot_date=snapshot_date, segment=segment)

    def delete_scored_scope(
        self,
        *,
        snapshot_date=None,
        start_date=None,
        end_date=None,
        segment: Optional[str] = None,
    ) -> dict[str, int]:
        """Delete previously written scored outputs for the same scoped period before re-inserting."""
        deleted = {"results": 0, "details": 0, "full_effects": 0}
        deleted["full_effects"] = self.delete_source_rows(
            "full_effects",
            snapshot_date=snapshot_date,
            start_date=start_date,
            end_date=end_date,
        )
        deleted["details"] = self.delete_source_rows(
            "details",
            snapshot_date=snapshot_date,
            start_date=start_date,
            end_date=end_date,
        )
        deleted["results"] = self.delete_source_rows(
            "results",
            snapshot_date=snapshot_date,
            start_date=start_date,
            end_date=end_date,
            segment=segment,
        )
        return deleted

    def write_source_rows(self, table_key: str, frame: pd.DataFrame, batch_size: int = 1000) -> int:
        """Write training/scoring/outcome source frames according to table schema."""
        if frame.empty:
            return 0

        normalized = self._normalize_columns(frame)
        available_columns = self._table_columns(table_key)

        if table_key == "outcomes":
            ordered_columns = [
                self.id_column,
                self.time_column,
                "label_30dpd_8w",
                "label_default_12m",
            ]
        else:
            ordered_columns = [self.id_column, self.time_column]
            if self.split_column.upper() in available_columns and self.split_column in normalized.columns:
                ordered_columns.append(self.split_column)
            if self.segment_column.upper() in available_columns and self.segment_column in normalized.columns:
                ordered_columns.append(self.segment_column)
            ordered_columns.extend(self.feature_names)

        missing_columns = [column for column in ordered_columns if column not in normalized.columns]
        if missing_columns:
            raise ValueError(
                f"Source DataFrame is missing required columns for {table_key}: {', '.join(missing_columns)}"
            )

        normalized[self.time_column] = pd.to_datetime(normalized[self.time_column], errors="raise")
        insert_sql = f"""
            INSERT INTO {self._qualified_table_name(table_key)} (
                {", ".join(column.upper() for column in ordered_columns)}
            ) VALUES (
                {", ".join(f":{index}" for index in range(1, len(ordered_columns) + 1))}
            )
        """
        rows = [
            self._coerce_scalar_sequence(record)
            for record in normalized[ordered_columns].itertuples(index=False, name=None)
        ]
        return self._executemany(insert_sql, rows, batch_size=batch_size)

    def write_results(self, results_df: pd.DataFrame, batch_size: int = 1000) -> int:
        """Write aggregated anomaly results into the configured Oracle table."""
        if results_df.empty:
            return 0

        frame = self._normalize_columns(results_df)
        required_columns = [self.id_column, self.time_column, "anomaly_score", "alert_band"]
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Results DataFrame is missing required columns: {', '.join(missing_columns)}"
            )

        frame[self.time_column] = pd.to_datetime(frame[self.time_column], errors="raise")
        frame = self._ensure_reason_columns(frame)

        for score_column in ("ae_score", "if_score", "md_score"):
            if score_column not in frame.columns:
                frame[score_column] = None

        available_columns = self._table_columns("results")
        optional_score_columns = [
            column
            for column in (
                "raw_shadow_score",
                "raw_shadow_alert_band",
                "raw_shadow_ae_score",
                "raw_shadow_if_score",
                "raw_shadow_md_score",
                "score_delta",
            )
            if column in frame.columns and column.upper() in available_columns
        ]
        optional_metadata_columns = [
            column
            for column in ("segment", "run_id", "model_version", "calibration_version", "weight_version")
            if column in frame.columns and column.upper() in available_columns
        ]
        ordered_columns = [
            self.id_column,
            self.time_column,
            "anomaly_score",
            "alert_band",
            "ae_score",
            "if_score",
            "md_score",
            *optional_score_columns,
            *optional_metadata_columns,
            *self._reason_column_names(),
        ]

        insert_sql = f"""
            INSERT INTO {self._qualified_table_name("results")} (
                {", ".join(column.upper() for column in ordered_columns)}
            ) VALUES (
                {", ".join(f":{index}" for index in range(1, len(ordered_columns) + 1))}
            )
        """

        rows = [
            self._coerce_scalar_sequence(record)
            for record in frame[ordered_columns].itertuples(index=False, name=None)
        ]

        return self._executemany(insert_sql, rows, batch_size=batch_size)

    def write_details(self, details_df: pd.DataFrame, batch_size: int = 1000) -> int:
        """Write per-feature explanation details into the configured Oracle table."""
        if details_df.empty:
            return 0

        frame = self._normalize_columns(details_df)
        required_columns = [
            self.id_column,
            self.time_column,
            "feature_name",
            "expected_value",
            "actual_value",
            "contribution_pct",
            "rank",
        ]
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Details DataFrame is missing required columns: {', '.join(missing_columns)}"
            )

        frame[self.time_column] = pd.to_datetime(frame[self.time_column], errors="raise")
        frame["feature_name"] = frame["feature_name"].astype(str).str.strip().str.lower()

        if "feature_label" not in frame.columns:
            frame["feature_label"] = frame["feature_name"].map(self.feature_labels).fillna(frame["feature_name"])

        if "delta_pct" not in frame.columns:
            frame["delta_pct"] = frame.apply(self._compute_delta_pct, axis=1)

        available_columns = self._table_columns("details")
        ordered_columns = [
            self.id_column,
            self.time_column,
            "feature_name",
            "feature_label",
            "expected_value",
            "actual_value",
            "delta_pct",
            "contribution_pct",
            "rank",
        ]
        for optional in (
            "customer_history_reference",
            "population_reference",
            "ae_reference",
            "ae_contribution_pct",
            "if_contribution_pct",
            "md_contribution_pct",
            "directionality",
            "direction_hint",
            "direction_comment",
        ):
            if optional in frame.columns and optional.upper() in available_columns:
                ordered_columns.append(optional)

        insert_columns = []
        for column in ordered_columns:
            if column == "rank":
                insert_columns.append("FEATURE_RANK")
            else:
                insert_columns.append(column.upper())
        insert_sql = f"""
            INSERT INTO {self._qualified_table_name("details")} (
                {", ".join(insert_columns)}
            ) VALUES (
                {", ".join(f":{index}" for index in range(1, len(ordered_columns) + 1))}
            )
        """

        rows = [
            self._coerce_scalar_sequence(record)
            for record in frame[ordered_columns].itertuples(index=False, name=None)
        ]

        return self._executemany(insert_sql, rows, batch_size=batch_size)

    def write_full_effects(self, effects_df: pd.DataFrame, batch_size: int = 1000) -> int:
        """Write full feature-effect rows into the configured Oracle table."""
        if effects_df.empty:
            return 0

        frame = self._normalize_columns(effects_df)
        required_columns = [
            self.id_column,
            self.time_column,
            "feature_name",
            "expected_value",
            "actual_value",
            "contribution_pct",
            "rank",
        ]
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Full effects DataFrame is missing required columns: {', '.join(missing_columns)}"
            )

        frame[self.time_column] = pd.to_datetime(frame[self.time_column], errors="raise")
        frame["feature_name"] = frame["feature_name"].astype(str).str.strip().str.lower()

        if "feature_label" not in frame.columns:
            frame["feature_label"] = frame["feature_name"].map(self.feature_labels).fillna(frame["feature_name"])
        if "delta_pct" not in frame.columns:
            frame["delta_pct"] = frame.apply(self._compute_delta_pct, axis=1)
        if "is_top_reason" not in frame.columns:
            frame["is_top_reason"] = 0

        available_columns = self._table_columns("full_effects")
        ordered_columns = [
            self.id_column,
            self.time_column,
            "feature_name",
            "feature_label",
            "expected_value",
            "actual_value",
            "delta_pct",
            "contribution_pct",
            "rank",
            "is_top_reason",
        ]
        for optional in (
            "customer_history_reference",
            "population_reference",
            "ae_reference",
            "ae_contribution_pct",
            "if_contribution_pct",
            "md_contribution_pct",
            "directionality",
            "direction_hint",
            "direction_comment",
        ):
            if optional in frame.columns and optional.upper() in available_columns:
                ordered_columns.append(optional)
        for optional in (
            "alert_band",
            "run_id",
            "model_version",
            "calibration_version",
            "weight_version",
        ):
            if optional in frame.columns and optional.upper() in available_columns:
                ordered_columns.append(optional)

        insert_columns = []
        for column in ordered_columns:
            if column == "rank":
                insert_columns.append("FEATURE_RANK")
            else:
                insert_columns.append(column.upper())

        insert_sql = f"""
            INSERT INTO {self._qualified_table_name("full_effects")} (
                {", ".join(insert_columns)}
            ) VALUES (
                {", ".join(f":{index}" for index in range(1, len(ordered_columns) + 1))}
            )
        """
        rows = [
            self._coerce_scalar_sequence(record)
            for record in frame[ordered_columns].itertuples(index=False, name=None)
        ]
        return self._executemany(insert_sql, rows, batch_size=batch_size)

    def setup_tables(self, drop_existing: bool = False) -> None:
        """Create the Oracle tables required by the pipeline."""
        table_ddls = {
            "input_features": self._input_features_table_ddl() if "input_features" in self.oracle_settings["tables"] else None,
            "training": self._training_table_ddl(),
            "scoring": self._scoring_table_ddl(),
            "outcomes": self._outcomes_table_ddl() if "outcomes" in self.oracle_settings["tables"] else None,
            "results": self._results_table_ddl(),
            "details": self._details_table_ddl(),
            "full_effects": self._full_effects_table_ddl() if "full_effects" in self.oracle_settings["tables"] else None,
            "monitor_history": self._monitor_history_table_ddl() if "monitor_history" in self.oracle_settings["tables"] else None,
        }
        active_keys = self._active_table_keys()

        connection = self.connect()
        with connection.cursor() as cursor:
            if drop_existing:
                for table_key in ("full_effects", "details", "results", "outcomes", "input_features", "scoring", "training", "monitor_history"):
                    if table_key not in active_keys:
                        continue
                    if self._table_exists(table_key):
                        cursor.execute(f"DROP TABLE {self._qualified_table_name(table_key)} PURGE")
                        self.logger.info("Dropped table %s", self._qualified_table_name(table_key))

            for table_key in ("input_features", "training", "scoring", "outcomes", "results", "details", "full_effects", "monitor_history"):
                if table_key not in active_keys or table_ddls.get(table_key) is None:
                    continue
                if self._table_exists(table_key):
                    self._ensure_managed_columns(cursor, table_key)
                    self.logger.info(
                        "Table %s already exists; skipping creation.",
                        self._qualified_table_name(table_key),
                    )
                    continue

                cursor.execute(table_ddls[table_key])
                self.logger.info("Created table %s", self._qualified_table_name(table_key))

        connection.commit()

    def _active_table_keys(self) -> set[str]:
        active = set()
        for source_cfg in self.pipeline_config.get("sources", {}).values():
            if source_cfg.get("backend") != "oracle":
                continue
            table_key = source_cfg.get("oracle", {}).get("table")
            if table_key in self.oracle_settings.get("tables", {}):
                active.add(table_key)

        outputs_cfg = self.pipeline_config.get("sources", {}).get("outputs", {})
        if outputs_cfg.get("backend") == "oracle":
            oracle_cfg = outputs_cfg.get("oracle", {})
            for key_name in ("results_table_key", "details_table_key", "full_effects_table_key"):
                table_key = oracle_cfg.get(key_name)
                if table_key in self.oracle_settings.get("tables", {}):
                    active.add(table_key)

        monitoring_cfg = self.pipeline_config.get("monitoring", {}) or {}
        history_cfg = monitoring_cfg.get("history", {}) or {}
        if history_cfg.get("backend", "oracle") == "oracle":
            history_key = history_cfg.get("table_key", "monitor_history")
            if history_key in self.oracle_settings.get("tables", {}):
                active.add(history_key)
        return active

    def _table_exists(self, table_key: str) -> bool:
        table_name = self._table_name(table_key)
        connection = self.connect()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(1)
                FROM ALL_TABLES
                WHERE OWNER = :owner
                  AND TABLE_NAME = :table_name
                """,
                owner=self.schema,
                table_name=table_name,
            )
            return bool(cursor.fetchone()[0])

    def _table_columns(self, table_key: str) -> set[str]:
        table_name = self._table_name(table_key)
        connection = self.connect()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COLUMN_NAME
                FROM ALL_TAB_COLUMNS
                WHERE OWNER = :owner
                  AND TABLE_NAME = :table_name
                """,
                owner=self.schema,
                table_name=table_name,
            )
            return {row[0] for row in cursor.fetchall()}

    def _read_query(
        self,
        sql: str,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        connection = self.connect()
        with connection.cursor() as cursor:
            cursor.execute(sql, parameters or {})
            rows = cursor.fetchall()
            columns = [description[0].lower() for description in cursor.description]
        return pd.DataFrame(rows, columns=columns)

    def _executemany(self, sql: str, rows: Sequence[Sequence[Any]], batch_size: int) -> int:
        connection = self.connect()
        inserted = 0
        with connection.cursor() as cursor:
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset : offset + batch_size]
                cursor.executemany(sql, batch)
                inserted += len(batch)
        connection.commit()
        return inserted

    def _training_table_ddl(self) -> str:
        feature_columns = self._feature_columns_ddl()
        primary_key = f"PK_{self._table_name('training')}"
        return f"""
            CREATE TABLE {self._qualified_table_name("training")} (
                {self.id_column.upper()} VARCHAR2(128) NOT NULL,
                {self.time_column.upper()} DATE NOT NULL,
                {self.split_column.upper()} VARCHAR2(32) NOT NULL,
                {self.segment_column.upper()} VARCHAR2(64),
                {feature_columns},
                CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT {primary_key} PRIMARY KEY ({self.id_column.upper()}, {self.time_column.upper()})
            )
        """

    def _input_features_table_ddl(self) -> str:
        feature_columns = self._feature_columns_ddl()
        primary_key = f"PK_{self._table_name('input_features')}"
        return f"""
            CREATE TABLE {self._qualified_table_name("input_features")} (
                {self.id_column.upper()} VARCHAR2(128) NOT NULL,
                {self.time_column.upper()} DATE NOT NULL,
                {self.segment_column.upper()} VARCHAR2(64),
                {feature_columns},
                DATA_TIME TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
                CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT {primary_key} PRIMARY KEY ({self.id_column.upper()}, {self.time_column.upper()})
            )
        """

    def _scoring_table_ddl(self) -> str:
        feature_columns = self._feature_columns_ddl()
        primary_key = f"PK_{self._table_name('scoring')}"
        return f"""
            CREATE TABLE {self._qualified_table_name("scoring")} (
                {self.id_column.upper()} VARCHAR2(128) NOT NULL,
                {self.time_column.upper()} DATE NOT NULL,
                {self.segment_column.upper()} VARCHAR2(64),
                {feature_columns},
                CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT {primary_key} PRIMARY KEY ({self.id_column.upper()}, {self.time_column.upper()})
            )
        """

    def _results_table_ddl(self) -> str:
        primary_key = f"PK_{self._table_name('results')}"
        reason_columns = ",\n                ".join(
            f"{column.upper()} VARCHAR2(512)" for column in self._reason_column_names()
        )
        return f"""
            CREATE TABLE {self._qualified_table_name("results")} (
                {self.id_column.upper()} VARCHAR2(128) NOT NULL,
                {self.time_column.upper()} DATE NOT NULL,
                ANOMALY_SCORE NUMBER(6,2) NOT NULL,
                ALERT_BAND VARCHAR2(32) NOT NULL,
                AE_SCORE NUMBER(6,2),
                IF_SCORE NUMBER(6,2),
                MD_SCORE NUMBER(6,2),
                RAW_SHADOW_SCORE NUMBER(6,2),
                RAW_SHADOW_ALERT_BAND VARCHAR2(32),
                RAW_SHADOW_AE_SCORE NUMBER(6,2),
                RAW_SHADOW_IF_SCORE NUMBER(6,2),
                RAW_SHADOW_MD_SCORE NUMBER(6,2),
                SCORE_DELTA NUMBER(6,2),
                SEGMENT VARCHAR2(64),
                RUN_ID VARCHAR2(128),
                MODEL_VERSION VARCHAR2(128),
                CALIBRATION_VERSION VARCHAR2(128),
                WEIGHT_VERSION VARCHAR2(128),
                {reason_columns},
                CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT {primary_key} PRIMARY KEY ({self.id_column.upper()}, {self.time_column.upper()})
            )
        """

    def _outcomes_table_ddl(self) -> str:
        primary_key = f"PK_{self._table_name('outcomes')}"
        return f"""
            CREATE TABLE {self._qualified_table_name("outcomes")} (
                {self.id_column.upper()} VARCHAR2(128) NOT NULL,
                {self.time_column.upper()} DATE NOT NULL,
                LABEL_30DPD_8W NUMBER(1) DEFAULT 0 NOT NULL,
                LABEL_DEFAULT_12M NUMBER(1) DEFAULT 0 NOT NULL,
                CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT {primary_key} PRIMARY KEY ({self.id_column.upper()}, {self.time_column.upper()})
            )
        """

    def _details_table_ddl(self) -> str:
        primary_key = f"PK_{self._table_name('details')}"
        return f"""
            CREATE TABLE {self._qualified_table_name("details")} (
                {self.id_column.upper()} VARCHAR2(128) NOT NULL,
                {self.time_column.upper()} DATE NOT NULL,
                FEATURE_NAME VARCHAR2(128) NOT NULL,
                FEATURE_LABEL VARCHAR2(256) NOT NULL,
                EXPECTED_VALUE NUMBER(18,6),
                ACTUAL_VALUE NUMBER(18,6),
                DELTA_PCT NUMBER(18,6),
                CONTRIBUTION_PCT NUMBER(18,6),
                CUSTOMER_HISTORY_REFERENCE NUMBER(18,6),
                POPULATION_REFERENCE NUMBER(18,6),
                AE_REFERENCE NUMBER(18,6),
                AE_CONTRIBUTION_PCT NUMBER(18,6),
                IF_CONTRIBUTION_PCT NUMBER(18,6),
                MD_CONTRIBUTION_PCT NUMBER(18,6),
                DIRECTIONALITY VARCHAR2(64),
                DIRECTION_HINT VARCHAR2(128),
                DIRECTION_COMMENT VARCHAR2(500),
                FEATURE_RANK NUMBER(4) NOT NULL,
                CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT {primary_key} PRIMARY KEY ({self.id_column.upper()}, {self.time_column.upper()}, FEATURE_NAME)
            )
        """

    def _full_effects_table_ddl(self) -> str:
        primary_key = f"PK_{self._table_name('full_effects')}"
        return f"""
            CREATE TABLE {self._qualified_table_name("full_effects")} (
                {self.id_column.upper()} VARCHAR2(128) NOT NULL,
                {self.time_column.upper()} DATE NOT NULL,
                FEATURE_NAME VARCHAR2(128) NOT NULL,
                FEATURE_LABEL VARCHAR2(256) NOT NULL,
                EXPECTED_VALUE NUMBER(18,6),
                ACTUAL_VALUE NUMBER(18,6),
                DELTA_PCT NUMBER(18,6),
                CONTRIBUTION_PCT NUMBER(18,6),
                CUSTOMER_HISTORY_REFERENCE NUMBER(18,6),
                POPULATION_REFERENCE NUMBER(18,6),
                AE_REFERENCE NUMBER(18,6),
                AE_CONTRIBUTION_PCT NUMBER(18,6),
                IF_CONTRIBUTION_PCT NUMBER(18,6),
                MD_CONTRIBUTION_PCT NUMBER(18,6),
                DIRECTIONALITY VARCHAR2(64),
                DIRECTION_HINT VARCHAR2(128),
                DIRECTION_COMMENT VARCHAR2(500),
                FEATURE_RANK NUMBER(4) NOT NULL,
                IS_TOP_REASON NUMBER(1) DEFAULT 0 NOT NULL,
                ALERT_BAND VARCHAR2(32),
                RUN_ID VARCHAR2(128),
                MODEL_VERSION VARCHAR2(128),
                CALIBRATION_VERSION VARCHAR2(128),
                WEIGHT_VERSION VARCHAR2(128),
                CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT {primary_key} PRIMARY KEY ({self.id_column.upper()}, {self.time_column.upper()}, FEATURE_NAME)
            )
        """

    def _feature_columns_ddl(self) -> str:
        return ",\n                ".join(
            f"{feature_name.upper()} NUMBER(18,6)" for feature_name in self.feature_names
        )

    def _monitor_history_table_ddl(self) -> str:
        table_name = self._table_name("monitor_history")
        pk_name = f"PK_{table_name}"[:30]
        return f"""
            CREATE TABLE {self._qualified_table_name("monitor_history")} (
                RUN_ID                 VARCHAR2(200) NOT NULL,
                RUN_TYPE               VARCHAR2(64) NOT NULL,
                SEGMENT                VARCHAR2(128) NOT NULL,
                STATUS                 VARCHAR2(32),
                STARTED_AT             TIMESTAMP,
                FINISHED_AT            TIMESTAMP,
                DURATION_SECONDS       NUMBER(12,3),
                MODEL_VERSION          VARCHAR2(200),
                SCOPE_SNAPSHOT         DATE,
                SCOPE_START            DATE,
                SCOPE_END              DATE,
                INPUT_ROWS             NUMBER,
                INPUT_CUSTOMERS        NUMBER,
                INPUT_SNAPSHOTS        NUMBER,
                AVG_MISSING_RATIO      NUMBER(10,6),
                BAND_NORMAL            NUMBER(7,4),
                BAND_SARI              NUMBER(7,4),
                BAND_TURUNCU           NUMBER(7,4),
                BAND_KIRMIZI           NUMBER(7,4),
                BAND_PERSISTENCE_KIRMIZI NUMBER(7,4),
                SCORE_MEAN             NUMBER(10,4),
                SCORE_MEDIAN           NUMBER(10,4),
                SCORE_P95              NUMBER(10,4),
                SCORE_P99              NUMBER(10,4),
                SCORE_SKEW             NUMBER(10,4),
                SCORE_KURTOSIS         NUMBER(10,4),
                SCORE_PSI_VS_PREV      NUMBER(10,6),
                SCORE_BUCKETS          VARCHAR2(1000),
                AE_SCORE_MEAN          NUMBER(10,4),
                IF_SCORE_MEAN          NUMBER(10,4),
                MD_SCORE_MEAN          NUMBER(10,4),
                QUALITY_NATIVE_FULL    VARCHAR2(16),
                QUALITY_NATIVE_SCOPE   VARCHAR2(16),
                QUALITY_DERIVED_FULL   VARCHAR2(16),
                QUALITY_DERIVED_SCOPE  VARCHAR2(16),
                NATIVE_AVG_COVERAGE    NUMBER(8,6),
                DERIVED_AVG_COVERAGE   NUMBER(8,6),
                NATIVE_MAX_OUTLIER     NUMBER(8,6),
                DERIVED_MAX_OUTLIER    NUMBER(8,6),
                FRESHNESS_MAX_AGE_DAYS NUMBER,
                STABILITY_TEST_KS      NUMBER(8,6),
                STABILITY_TEST_MEAN_RATIO NUMBER(10,6),
                STABILITY_CAL_KS       NUMBER(8,6),
                STABILITY_CAL_MEAN_RATIO NUMBER(10,6),
                STABILITY_OOT_KS       NUMBER(8,6),
                STABILITY_OOT_MEAN_RATIO NUMBER(10,6),
                CALIBRATION_ROWS       NUMBER,
                CALIBRATION_MONOTONIC  NUMBER(1),
                SUPERVISED_PRECISION   NUMBER(8,6),
                SUPERVISED_RECALL      NUMBER(8,6),
                SUPERVISED_F1          NUMBER(8,6),
                SUPERVISED_LIFT        NUMBER(10,6),
                WEIGHT_AE              NUMBER(6,4),
                WEIGHT_IF              NUMBER(6,4),
                WEIGHT_MD              NUMBER(6,4),
                DOMINANT_REASON_FEATURE VARCHAR2(128),
                DOMINANT_REASON_SHARE  NUMBER(7,4),
                HEALTH_OVERALL         VARCHAR2(16),
                HEALTH_GREEN_COUNT     NUMBER,
                HEALTH_YELLOW_COUNT    NUMBER,
                HEALTH_RED_COUNT       NUMBER,
                HEALTH_SKIPPED_COUNT   NUMBER,
                RESULT_ROW_COUNT       NUMBER,
                MONITORING_PATH        VARCHAR2(500),
                CREATED_AT             TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT {pk_name} PRIMARY KEY (RUN_ID)
            )
        """

    MONITOR_HISTORY_COLUMNS: tuple[str, ...] = (
        "RUN_ID", "RUN_TYPE", "SEGMENT", "STATUS", "STARTED_AT", "FINISHED_AT",
        "DURATION_SECONDS", "MODEL_VERSION", "SCOPE_SNAPSHOT", "SCOPE_START", "SCOPE_END",
        "INPUT_ROWS", "INPUT_CUSTOMERS", "INPUT_SNAPSHOTS", "AVG_MISSING_RATIO",
        "BAND_NORMAL", "BAND_SARI", "BAND_TURUNCU", "BAND_KIRMIZI", "BAND_PERSISTENCE_KIRMIZI",
        "SCORE_MEAN", "SCORE_MEDIAN", "SCORE_P95", "SCORE_P99",
        "SCORE_SKEW", "SCORE_KURTOSIS", "SCORE_PSI_VS_PREV", "SCORE_BUCKETS",
        "AE_SCORE_MEAN", "IF_SCORE_MEAN", "MD_SCORE_MEAN",
        "QUALITY_NATIVE_FULL", "QUALITY_NATIVE_SCOPE", "QUALITY_DERIVED_FULL", "QUALITY_DERIVED_SCOPE",
        "NATIVE_AVG_COVERAGE", "DERIVED_AVG_COVERAGE", "NATIVE_MAX_OUTLIER", "DERIVED_MAX_OUTLIER",
        "FRESHNESS_MAX_AGE_DAYS",
        "STABILITY_TEST_KS", "STABILITY_TEST_MEAN_RATIO",
        "STABILITY_CAL_KS", "STABILITY_CAL_MEAN_RATIO",
        "STABILITY_OOT_KS", "STABILITY_OOT_MEAN_RATIO",
        "CALIBRATION_ROWS", "CALIBRATION_MONOTONIC",
        "SUPERVISED_PRECISION", "SUPERVISED_RECALL", "SUPERVISED_F1", "SUPERVISED_LIFT",
        "WEIGHT_AE", "WEIGHT_IF", "WEIGHT_MD",
        "DOMINANT_REASON_FEATURE", "DOMINANT_REASON_SHARE",
        "HEALTH_OVERALL", "HEALTH_GREEN_COUNT", "HEALTH_YELLOW_COUNT", "HEALTH_RED_COUNT", "HEALTH_SKIPPED_COUNT",
        "RESULT_ROW_COUNT",
        "MONITORING_PATH",
    )

    def write_monitor_history_row(self, row: dict) -> int:
        """Insert or replace a run-level monitoring history row."""
        if "monitor_history" not in self.oracle_settings.get("tables", {}):
            return 0

        columns = list(self.MONITOR_HISTORY_COLUMNS)
        values = [row.get(col.lower()) for col in columns]
        placeholders = ", ".join(f":{index + 1}" for index in range(len(columns)))
        column_clause = ", ".join(columns)
        table_full = self._qualified_table_name("monitor_history")

        connection = self.connect()
        with connection.cursor() as cursor:
            cursor.execute(f"DELETE FROM {table_full} WHERE RUN_ID = :1", [row.get("run_id")])
            cursor.execute(
                f"INSERT INTO {table_full} ({column_clause}) VALUES ({placeholders})",
                values,
            )
        connection.commit()
        return 1

    def read_previous_monitor_row(
        self,
        *,
        segment: str,
        run_type: str,
        before: Optional[str] = None,
    ) -> Optional[dict]:
        """Fetch the most recent completed monitor history row before a given timestamp."""
        if "monitor_history" not in self.oracle_settings.get("tables", {}):
            return None

        table_full = self._qualified_table_name("monitor_history")
        params: dict[str, Any] = {"segment": segment, "run_type": run_type}
        filter_clause = ""
        if before is not None:
            filter_clause = "AND FINISHED_AT < :before"
            params["before"] = pd.Timestamp(before).to_pydatetime()

        query = f"""
            SELECT *
            FROM {table_full}
            WHERE SEGMENT = :segment
              AND RUN_TYPE = :run_type
              AND STATUS = 'completed'
              {filter_clause}
            ORDER BY FINISHED_AT DESC
            FETCH FIRST 1 ROWS ONLY
        """
        frame = self._read_query(query, params)
        if frame.empty:
            return None
        row = frame.iloc[0].to_dict()
        return {str(key).lower(): value for key, value in row.items()}

    def _ensure_managed_columns(self, cursor, table_key: str) -> None:
        available_columns = self._table_columns(table_key)
        if table_key == "input_features" and "DATA_TIME" not in available_columns:
            cursor.execute(
                f"ALTER TABLE {self._qualified_table_name(table_key)} "
                "ADD (DATA_TIME TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL)"
            )
            self.logger.info(
                "Added DATA_TIME column to existing table %s",
                self._qualified_table_name(table_key),
            )
        if table_key == "monitor_history":
            migration = {
                "DURATION_SECONDS": "NUMBER(12,3)",
                "BAND_PERSISTENCE_KIRMIZI": "NUMBER(7,4)",
                "SCORE_SKEW": "NUMBER(10,4)",
                "SCORE_KURTOSIS": "NUMBER(10,4)",
                "SCORE_PSI_VS_PREV": "NUMBER(10,6)",
                "SCORE_BUCKETS": "VARCHAR2(1000)",
                "FRESHNESS_MAX_AGE_DAYS": "NUMBER",
                "STABILITY_TEST_KS": "NUMBER(8,6)",
                "STABILITY_TEST_MEAN_RATIO": "NUMBER(10,6)",
                "STABILITY_CAL_KS": "NUMBER(8,6)",
                "STABILITY_CAL_MEAN_RATIO": "NUMBER(10,6)",
                "CALIBRATION_ROWS": "NUMBER",
                "CALIBRATION_MONOTONIC": "NUMBER(1)",
                "DOMINANT_REASON_FEATURE": "VARCHAR2(128)",
                "DOMINANT_REASON_SHARE": "NUMBER(7,4)",
                "HEALTH_OVERALL": "VARCHAR2(16)",
                "HEALTH_GREEN_COUNT": "NUMBER",
                "HEALTH_YELLOW_COUNT": "NUMBER",
                "HEALTH_RED_COUNT": "NUMBER",
                "HEALTH_SKIPPED_COUNT": "NUMBER",
                "RESULT_ROW_COUNT": "NUMBER",
            }
            missing = [(name, ddl) for name, ddl in migration.items() if name not in available_columns]
            if missing:
                add_clause = ", ".join(f"{name} {ddl}" for name, ddl in missing)
                cursor.execute(
                    f"ALTER TABLE {self._qualified_table_name(table_key)} ADD ({add_clause})"
                )
                self.logger.info(
                    "Added %d new columns to %s: %s",
                    len(missing),
                    self._qualified_table_name(table_key),
                    ", ".join(name for name, _ in missing),
                )
        if table_key in {"details", "full_effects"}:
            migration = {
                "DIRECTIONALITY": "VARCHAR2(64)",
                "DIRECTION_HINT": "VARCHAR2(128)",
                "DIRECTION_COMMENT": "VARCHAR2(500)",
            }
            missing = [(name, ddl) for name, ddl in migration.items() if name not in available_columns]
            if missing:
                add_clause = ", ".join(f"{name} {ddl}" for name, ddl in missing)
                cursor.execute(
                    f"ALTER TABLE {self._qualified_table_name(table_key)} ADD ({add_clause})"
                )
                self.logger.info(
                    "Added %d new columns to %s: %s",
                    len(missing),
                    self._qualified_table_name(table_key),
                    ", ".join(name for name, _ in missing),
                )

    def _table_name(self, table_key: str) -> str:
        tables = self.oracle_settings["tables"]
        if table_key not in tables:
            raise KeyError(f"Oracle table '{table_key}' is not defined in pipeline_config.yaml.")
        return str(tables[table_key]).upper()

    def _qualified_table_name(self, table_key: str) -> str:
        return f"{self.schema}.{self._table_name(table_key)}"

    def _normalize_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized.columns = [str(column).strip().lower() for column in normalized.columns]
        return normalized

    def _reason_column_names(self) -> list[str]:
        return [f"reason_{index}" for index in range(1, self.top_n_reasons + 1)]

    def _ensure_reason_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "reasons" in frame.columns:
            for index, column_name in enumerate(self._reason_column_names()):
                frame[column_name] = frame["reasons"].apply(
                    lambda value: self._reason_at_position(value, index)
                )

        for column_name in self._reason_column_names():
            if column_name not in frame.columns:
                frame[column_name] = None

        return frame

    @staticmethod
    def _reason_at_position(value: Any, index: int) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            parts = [part.strip() for part in value.split("|") if part.strip()]
            return parts[index] if index < len(parts) else None
        if isinstance(value, Sequence):
            return str(value[index]) if index < len(value) else None
        return None

    @staticmethod
    def _compute_delta_pct(row: pd.Series) -> float:
        expected_value = row["expected_value"]
        actual_value = row["actual_value"]
        if pd.isna(expected_value) or pd.isna(actual_value):
            return 0.0
        if abs(expected_value) < 1e-12:
            return 0.0 if abs(actual_value) < 1e-12 else 100.0
        return float(((actual_value - expected_value) / abs(expected_value)) * 100.0)

    @staticmethod
    def _coerce_scalar_sequence(values: Sequence[Any]) -> tuple[Any, ...]:
        coerced: list[Any] = []
        for value in values:
            if isinstance(value, pd.Timestamp):
                coerced.append(value.to_pydatetime())
            elif pd.isna(value):
                coerced.append(None)
            else:
                coerced.append(value)
        return tuple(coerced)
