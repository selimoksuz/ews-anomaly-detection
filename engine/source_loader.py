"""Oracle-only source loading for development and live scoring datasets."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from engine.oracle_io import OracleConnector


class SourceLoader:
    """Load data only from configured Oracle sources."""

    def __init__(self, config: dict, secrets: Optional[dict] = None):
        self.config = config
        self.secrets = secrets
        self.sources_cfg = config.get("sources", {})
        self.id_column = config["pipeline"]["id_column"]
        self.time_column = config["pipeline"]["time_column"]

    def list_snapshots(self, source_name: str, segment_column: Optional[str] = None, segment_value=None):
        source_cfg = self._get_source_config(source_name)
        self._assert_oracle_backend(source_cfg, source_name)
        with OracleConnector(self.config, self.secrets) as ora:
            table = self._resolve_table_name(ora, source_cfg.get("oracle", {}).get("table"))
            sql = f"SELECT DISTINCT {self.time_column.upper()} FROM {table}"
            params = {}
            if segment_column and segment_value is not None:
                sql += f" WHERE {segment_column.upper()} = :segment_value"
                params["segment_value"] = segment_value
            sql += f" ORDER BY {self.time_column.upper()}"
            frame = ora._read_query(sql, params)
        return sorted(pd.to_datetime(frame[self.time_column]).unique())

    def load_frame(
        self,
        source_name: str,
        *,
        start_date=None,
        end_date=None,
        snapshot_date=None,
        latest_snapshot: bool = False,
        segment_column: Optional[str] = None,
        segment_value=None,
    ) -> pd.DataFrame:
        source_cfg = self._get_source_config(source_name)
        self._assert_oracle_backend(source_cfg, source_name)
        return self._load_oracle(
            source_cfg,
            start_date=start_date,
            end_date=end_date,
            snapshot_date=snapshot_date,
            latest_snapshot=latest_snapshot,
            segment_column=segment_column,
            segment_value=segment_value,
        )

    def _load_oracle(
        self,
        source_cfg: dict,
        *,
        start_date=None,
        end_date=None,
        snapshot_date=None,
        latest_snapshot: bool = False,
        segment_column: Optional[str] = None,
        segment_value=None,
    ) -> pd.DataFrame:
        with OracleConnector(self.config, self.secrets) as ora:
            table = self._resolve_table_name(ora, source_cfg.get("oracle", {}).get("table"))
            clauses = []
            params = {}

            if latest_snapshot:
                clauses.append(
                    f"{self.time_column.upper()} = (SELECT MAX({self.time_column.upper()}) FROM {table})"
                )
            elif snapshot_date is not None:
                clauses.append(f"{self.time_column.upper()} = :snapshot_date")
                params["snapshot_date"] = pd.Timestamp(snapshot_date).to_pydatetime()
            else:
                if start_date is not None:
                    clauses.append(f"{self.time_column.upper()} >= :start_date")
                    params["start_date"] = pd.Timestamp(start_date).to_pydatetime()
                if end_date is not None:
                    clauses.append(f"{self.time_column.upper()} <= :end_date")
                    params["end_date"] = pd.Timestamp(end_date).to_pydatetime()

            if segment_column and segment_value is not None:
                clauses.append(f"{segment_column.upper()} = :segment_value")
                params["segment_value"] = segment_value

            sql = f"SELECT * FROM {table}"
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            frame = ora._read_query(sql, params)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if self.time_column in frame.columns:
            frame[self.time_column] = pd.to_datetime(frame[self.time_column])
        return frame

    def _resolve_table_name(self, ora: OracleConnector, table_name: Optional[str]) -> str:
        if not table_name:
            raise ValueError("Oracle source table must be configured.")

        tables = self.config.get("oracle", {}).get("tables", {})
        if table_name in tables:
            return ora._qualified_table_name(table_name)
        if "." in table_name:
            return table_name
        return f"{ora.schema}.{table_name.upper()}"

    def _get_source_config(self, source_name: str) -> dict:
        if source_name not in self.sources_cfg:
            raise KeyError(f"Unknown source '{source_name}' in config.sources.")
        return self.sources_cfg[source_name]

    @staticmethod
    def _assert_oracle_backend(source_cfg: dict, source_name: str) -> None:
        backend = source_cfg.get("backend", "oracle")
        if backend != "oracle":
            raise ValueError(
                f"Source '{source_name}' is configured with backend '{backend}', but this project now supports Oracle only."
            )
