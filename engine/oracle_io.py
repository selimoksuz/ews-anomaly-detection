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
        column_names = [self.id_column, self.time_column, self.split_column, *self.feature_names]
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
        column_names = [self.id_column, self.time_column, *self.feature_names]
        select_clause = ", ".join(column.upper() for column in column_names)
        table_name = self._qualified_table_name("scoring")
        sql = f"SELECT {select_clause} FROM {table_name}"
        return self._read_query(sql)

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

        ordered_columns = [
            self.id_column,
            self.time_column,
            "anomaly_score",
            "alert_band",
            "ae_score",
            "if_score",
            "md_score",
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

    def setup_tables(self, drop_existing: bool = False) -> None:
        """Create the Oracle tables required by the pipeline."""
        table_ddls = {
            "training": self._training_table_ddl(),
            "scoring": self._scoring_table_ddl(),
            "results": self._results_table_ddl(),
            "details": self._details_table_ddl(),
        }

        connection = self.connect()
        with connection.cursor() as cursor:
            if drop_existing:
                for table_key in ("details", "results", "scoring", "training"):
                    if self._table_exists(table_key):
                        cursor.execute(f"DROP TABLE {self._qualified_table_name(table_key)} PURGE")
                        self.logger.info("Dropped table %s", self._qualified_table_name(table_key))

            for table_key in ("training", "scoring", "results", "details"):
                if self._table_exists(table_key):
                    self.logger.info(
                        "Table %s already exists; skipping creation.",
                        self._qualified_table_name(table_key),
                    )
                    continue

                cursor.execute(table_ddls[table_key])
                self.logger.info("Created table %s", self._qualified_table_name(table_key))

        connection.commit()

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
                {reason_columns},
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
