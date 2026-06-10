"""Oracle utilities used by the anomaly_multivar pipeline."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    import oracledb
except ImportError as exc:  # pragma: no cover - dependency is environment-specific
    oracledb = None
    _ORACLEDB_IMPORT_ERROR = exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline_config.yaml"
DEFAULT_SECRETS_CONFIG = PROJECT_ROOT / "config" / "secrets.yaml"


def load_yaml_config(config_source=None, default_path: Path | None = None) -> dict:
    if isinstance(config_source, Mapping):
        return dict(config_source)

    path = Path(config_source) if config_source is not None else default_path
    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return payload


class OracleConnector:
    """Thin Oracle connector for multivar input/output tables."""

    def __init__(self, pipeline_config=None, secrets=None) -> None:
        self.pipeline_config = load_yaml_config(pipeline_config, DEFAULT_PIPELINE_CONFIG)
        self.secrets = load_yaml_config(secrets, DEFAULT_SECRETS_CONFIG)
        self.pipeline_settings = self.pipeline_config["pipeline"]
        self.oracle_settings = self.pipeline_config["oracle"]
        self.connection_settings = self.secrets["oracle"]
        logger_name = self.pipeline_config.get("logging", {}).get("logger_name", "ews.multivar")
        self.logger = logging.getLogger(logger_name)
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

    def connect(self):
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
        if self.connection is not None:
            self.connection.close()
            self.connection = None

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

    def _read_query(self, sql: str, parameters: Mapping[str, Any] | None = None) -> pd.DataFrame:
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

    def _table_name(self, table_key: str) -> str:
        tables = self.oracle_settings["tables"]
        if table_key not in tables:
            raise KeyError(f"Oracle table '{table_key}' is not defined in pipeline_config.yaml.")
        return str(tables[table_key]).upper()

    def _qualified_table_name(self, table_key: str) -> str:
        return f"{self.schema}.{self._table_name(table_key)}"

    @staticmethod
    def _coerce_scalar_sequence(values: Sequence[Any]) -> tuple[Any, ...]:
        coerced: list[Any] = []
        for value in values:
            if isinstance(value, pd.Timestamp):
                coerced.append(value.to_pydatetime())
                continue
            try:
                is_null = pd.isna(value)
            except (TypeError, ValueError):
                is_null = False
            if isinstance(is_null, bool) and is_null:
                coerced.append(None)
            else:
                coerced.append(value)
        return tuple(coerced)
