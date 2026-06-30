"""Oracle utilities used by the anomaly_multivar pipeline."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from engine.config_loader import (
    REQUIRED_PIPELINE_CONFIG_KEYS,
    load_config,
    normalize_pipeline_config,
    resolve_secrets_path,
)

try:
    import oracledb
except ImportError as exc:  # pragma: no cover - dependency is environment-specific
    oracledb = None
    _ORACLEDB_IMPORT_ERROR = exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline_config.yaml"
_ORACLE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$.#]*$")


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


def resolve_default_secrets_path() -> Path:
    return resolve_secrets_path()


def case_get(mapping: Mapping[str, Any], key: str) -> Any:
    key_lower = str(key).lower()
    for current_key, value in mapping.items():
        if str(current_key).lower() == key_lower:
            return value
    return None


def merge_non_empty(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        target[key] = value


def parse_proxy_user(user: str) -> tuple[str, str | None]:
    match = re.match(r"^([^[\]]+)\[([^[\]]+)\]$", str(user).strip())
    if not match:
        return str(user), None
    return match.group(1).strip(), match.group(2).strip()


def validate_oracle_identifier(value: str, label: str) -> str:
    text = str(value).strip()
    for part in text.split("."):
        if not _ORACLE_IDENT_RE.match(part):
            raise ValueError(f"Unsafe Oracle {label}: {value!r}")
    return text.upper()


class OracleConnector:
    """Thin Oracle connector for multivar input/output tables."""

    def __init__(self, pipeline_config=None, secrets=None) -> None:
        self.pipeline_config = (
            load_config()
            if pipeline_config is None
            else normalize_pipeline_config(load_yaml_config(pipeline_config, DEFAULT_PIPELINE_CONFIG))
        )
        missing_config_keys = [
            key
            for key in REQUIRED_PIPELINE_CONFIG_KEYS
            if not isinstance(self.pipeline_config.get(key), Mapping)
        ]
        if missing_config_keys:
            available = ", ".join(str(key) for key in sorted(self.pipeline_config.keys()))
            raise ValueError(
                "Pipeline config missing required root mapping(s): "
                + ", ".join(missing_config_keys)
                + f". Available root keys: {available or '<empty>'}. "
                + "Use the repo config/pipeline_config.yaml or set EWS_ANOMALY_CONFIG_PATH to the anomaly config."
            )
        self.secrets = load_yaml_config(secrets, resolve_default_secrets_path())
        self.pipeline_settings = self.pipeline_config["pipeline"]
        self.oracle_settings = self.pipeline_config["oracle"]
        self.oracle_section = str(
            self.oracle_settings.get("section")
            or os.getenv("EWS_ANOMALY_ORACLE_SECTION")
            or os.getenv("ORACLE_SECTION")
            or "ORA_PRD_ZTUSER"
        )
        self.connection_settings = self._oracle_connection_settings(self.oracle_section)
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
        schema = (
            self.oracle_settings.get("default_owner")
            or self.oracle_settings.get("schema")
            or self.connection_settings.get("schema")
        )
        if not schema:
            raise ValueError("Oracle schema must be defined in pipeline_config.yaml or secrets.yaml.")
        return validate_oracle_identifier(str(schema), "schema")

    def _oracle_connection_settings(self, section: str) -> dict[str, Any]:
        oracle = self.secrets.get("oracle") if isinstance(self.secrets, dict) else None
        if not isinstance(oracle, dict):
            raise ValueError("secret/secrets.yaml must contain an oracle mapping.")

        values: dict[str, Any] = {}
        connection = oracle.get("connection")
        if isinstance(connection, Mapping):
            merge_non_empty(values, connection)

        sections = oracle.get("sections") or oracle.get("connections")
        if isinstance(sections, Mapping):
            section_payload = case_get(sections, section)
            if isinstance(section_payload, Mapping):
                merge_non_empty(values, section_payload)

        # Backward compatibility for the older flat oracle secret shape.
        flat_keys = {"user", "password", "host", "port", "service_name", "service", "sid", "proxy_user"}
        if not values and any(key in oracle for key in flat_keys):
            merge_non_empty(values, oracle)

        missing = [key for key in ("user", "password", "host", "port") if not values.get(key)]
        if missing:
            raise ValueError(
                f"Oracle credential section {section!r} missing required keys: {', '.join(missing)}"
            )
        if not (values.get("service_name") or values.get("service") or values.get("sid")):
            raise ValueError(f"Oracle credential section {section!r} must define service_name, service, or sid.")
        return values

    def connect(self):
        if self.connection is not None:
            return self.connection

        if oracledb is None:
            raise RuntimeError(
                "The 'oracledb' package is required to use OracleConnector."
            ) from _ORACLEDB_IMPORT_ERROR

        host = self.connection_settings["host"]
        port = self.connection_settings["port"]
        service_name = (
            self.connection_settings.get("service_name")
            or self.connection_settings.get("service")
        )
        sid = self.connection_settings.get("sid")
        user = self.connection_settings["user"]
        password = self.connection_settings["password"]

        if service_name:
            dsn = f"{host}:{port}/{service_name}"
        else:
            dsn = oracledb.makedsn(str(host), int(port), sid=str(sid))
        connect_user, proxy_user = parse_proxy_user(str(user))
        self.logger.info("Opening Oracle connection to %s", dsn)
        kwargs = {"user": connect_user, "password": password, "dsn": dsn}
        if proxy_user:
            kwargs["proxy_user"] = proxy_user
        self.connection = oracledb.connect(**kwargs)
        return self.connection

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _table_exists(self, table_key: str) -> bool:
        table_name = self._table_name(table_key)
        owner = self._table_owner(table_key)
        connection = self.connect()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(1)
                FROM ALL_TABLES
                WHERE OWNER = :owner
                  AND TABLE_NAME = :table_name
                """,
                owner=owner,
                table_name=table_name,
            )
            return bool(cursor.fetchone()[0])

    def _table_columns(self, table_key: str) -> set[str]:
        table_name = self._table_name(table_key)
        owner = self._table_owner(table_key)
        connection = self.connect()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COLUMN_NAME
                FROM ALL_TAB_COLUMNS
                WHERE OWNER = :owner
                  AND TABLE_NAME = :table_name
                """,
                owner=owner,
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
        payload = tables[table_key]
        if isinstance(payload, Mapping):
            table = payload.get("table") or payload.get("name")
        else:
            table = payload
        if not table:
            raise ValueError(f"Oracle table '{table_key}' must define table/name.")
        return validate_oracle_identifier(str(table), "table")

    def _table_owner(self, table_key: str) -> str:
        tables = self.oracle_settings["tables"]
        if table_key not in tables:
            raise KeyError(f"Oracle table '{table_key}' is not defined in pipeline_config.yaml.")
        payload = tables[table_key]
        owner = None
        if isinstance(payload, Mapping):
            owner = payload.get("owner") or payload.get("schema")
        owner = owner or self.oracle_settings.get("default_owner") or self.oracle_settings.get("schema")
        if not owner:
            raise ValueError(f"Oracle table '{table_key}' must define owner/schema.")
        return validate_oracle_identifier(str(owner), "owner")

    def _qualified_table_name(self, table_key: str) -> str:
        return f"{self._table_owner(table_key)}.{self._table_name(table_key)}"

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
