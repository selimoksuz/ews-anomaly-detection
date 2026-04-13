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
        return dict(source)

    path = Path(source) if source is not None else default_path
    if path is None:
        raise ValueError("A configuration mapping or file path is required.")
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, Mapping):
        raise ValueError(f"Configuration file must contain a mapping at the root: {path}")

    return dict(data)


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
        return {
            item["name"]: item.get("label_tr", item["name"])
            for item in self.feature_definitions
        }

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
        unknown_features = sorted(set(frame["feature_name"]) - set(self.feature_names))
        if unknown_features:
            raise ValueError(
                f"Details DataFrame contains features not defined in config: {', '.join(unknown_features)}"
            )

        if "feature_label" not in frame.columns:
            frame["feature_label"] = frame["feature_name"].map(self.feature_labels)

        if "delta_pct" not in frame.columns:
            frame["delta_pct"] = frame.apply(self._compute_delta_pct, axis=1)

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

        insert_sql = f"""
            INSERT INTO {self._qualified_table_name("details")} (
                {self.id_column.upper()},
                {self.time_column.upper()},
                FEATURE_NAME,
                FEATURE_LABEL,
                EXPECTED_VALUE,
                ACTUAL_VALUE,
                DELTA_PCT,
                CONTRIBUTION_PCT,
                FEATURE_RANK
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
            frame["feature_label"] = frame["feature_name"].map(self.feature_labels)
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
        }
        active_keys = self._active_table_keys()

        connection = self.connect()
        with connection.cursor() as cursor:
            if drop_existing:
                for table_key in ("full_effects", "details", "results", "outcomes", "input_features", "scoring", "training"):
                    if table_key not in active_keys:
                        continue
                    if self._table_exists(table_key):
                        cursor.execute(f"DROP TABLE {self._qualified_table_name(table_key)} PURGE")
                        self.logger.info("Dropped table %s", self._qualified_table_name(table_key))

            for table_key in ("input_features", "training", "scoring", "outcomes", "results", "details", "full_effects"):
                if table_key not in active_keys or table_ddls.get(table_key) is None:
                    continue
                if self._table_exists(table_key):
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
