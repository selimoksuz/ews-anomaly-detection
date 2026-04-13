"""Config-driven source loading for development and live scoring datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from engine.oracle_io import OracleConnector


class SourceLoader:
    """Load data from configured CSV or Oracle sources."""

    def __init__(self, config: dict, secrets: Optional[dict] = None):
        self.config = config
        self.secrets = secrets
        self.sources_cfg = config.get("sources", {})
        self.id_column = config["pipeline"]["id_column"]
        self.time_column = config["pipeline"]["time_column"]

    def list_snapshots(self, source_name: str, segment_column: Optional[str] = None, segment_value=None):
        source_cfg = self._get_source_config(source_name)
        backend = source_cfg.get("backend", "csv")
        if backend == "csv":
            frame = self._load_csv(source_cfg)
            frame = self._apply_filters(frame, segment_column=segment_column, segment_value=segment_value)
            return sorted(pd.to_datetime(frame[self.time_column]).unique())

        if backend == "oracle":
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

        raise ValueError(f"Unsupported source backend: {backend}")

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
        backend = source_cfg.get("backend", "csv")

        if backend == "csv":
            frame = self._load_csv(source_cfg)
            return self._apply_filters(
                frame,
                start_date=start_date,
                end_date=end_date,
                snapshot_date=snapshot_date,
                latest_snapshot=latest_snapshot,
                segment_column=segment_column,
                segment_value=segment_value,
            )

        if backend == "oracle":
            return self._load_oracle(
                source_cfg,
                start_date=start_date,
                end_date=end_date,
                snapshot_date=snapshot_date,
                latest_snapshot=latest_snapshot,
                segment_column=segment_column,
                segment_value=segment_value,
            )

        raise ValueError(f"Unsupported source backend: {backend}")

    def _load_csv(self, source_cfg: dict) -> pd.DataFrame:
        path = Path(source_cfg.get("csv", {}).get("path", ""))
        if not path.exists():
            raise FileNotFoundError(f"Configured CSV source not found: {path}")
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if self.time_column in frame.columns:
            frame[self.time_column] = pd.to_datetime(frame[self.time_column])
        return frame

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

    def _apply_filters(
        self,
        frame: pd.DataFrame,
        *,
        start_date=None,
        end_date=None,
        snapshot_date=None,
        latest_snapshot: bool = False,
        segment_column: Optional[str] = None,
        segment_value=None,
    ) -> pd.DataFrame:
        result = frame.copy()
        if self.time_column in result.columns:
            result[self.time_column] = pd.to_datetime(result[self.time_column])

        if segment_column and segment_value is not None and segment_column in result.columns:
            result = result[result[segment_column] == segment_value]

        if latest_snapshot:
            latest = result[self.time_column].max()
            result = result[result[self.time_column] == latest]
        elif snapshot_date is not None:
            target = pd.Timestamp(snapshot_date)
            result = result[result[self.time_column] == target]
        else:
            if start_date is not None:
                result = result[result[self.time_column] >= pd.Timestamp(start_date)]
            if end_date is not None:
                result = result[result[self.time_column] <= pd.Timestamp(end_date)]

        return result.reset_index(drop=True)

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
