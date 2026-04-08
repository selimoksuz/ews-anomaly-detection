"""Dataset loading and validation utilities for the EWS anomaly detection pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from .oracle_io import DEFAULT_PIPELINE_CONFIG, OracleConnector, load_yaml_config


ConfigSource = Optional[Union[str, Path, Mapping[str, Any]]]


class DataLoader:
    """Load data from Oracle or in-memory DataFrames and enforce schema validation."""

    def __init__(self, config: ConfigSource = None) -> None:
        self.pipeline_config = load_yaml_config(config, DEFAULT_PIPELINE_CONFIG)
        self._apply_pipeline_config(self.pipeline_config)

    def load_from_oracle(
        self,
        config: ConfigSource = None,
        secrets: ConfigSource = None,
        split: str = "TRAIN",
    ) -> pd.DataFrame:
        """Load a dataset from Oracle and validate it before returning."""
        if config is not None:
            self.pipeline_config = load_yaml_config(config, DEFAULT_PIPELINE_CONFIG)
            self._apply_pipeline_config(self.pipeline_config)

        connector = OracleConnector(pipeline_config=self.pipeline_config, secrets=secrets)
        split_name = split.upper()
        if split_name in {"SCORE", "SCORING", "INFERENCE", "PREDICT"}:
            frame = connector.read_scoring_data()
        else:
            frame = connector.read_training_data(split=split_name)

        return self.validate_data(frame)

    def load_from_dataframe(self, df: pd.DataFrame, config: ConfigSource = None) -> pd.DataFrame:
        """Validate and normalize an in-memory DataFrame."""
        if config is not None:
            self.pipeline_config = load_yaml_config(config, DEFAULT_PIPELINE_CONFIG)
            self._apply_pipeline_config(self.pipeline_config)
        return self.validate_data(df)

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns, nulls, duplicate keys, and numeric typing."""
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        frame = df.copy()
        frame.columns = [str(column).strip().lower() for column in frame.columns]

        required_columns = [self.id_column, self.time_column, *self.feature_names]
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Input DataFrame is missing required columns: {', '.join(missing_columns)}"
            )

        required_nulls = frame[required_columns].isnull().sum()
        columns_with_nulls = required_nulls[required_nulls > 0]
        if not columns_with_nulls.empty:
            details = ", ".join(
                f"{column}={count}" for column, count in columns_with_nulls.items()
            )
            raise ValueError(f"Null values detected in required columns: {details}")

        frame[self.id_column] = frame[self.id_column].astype(str).str.strip()
        if (frame[self.id_column] == "").any():
            raise ValueError(f"Column '{self.id_column}' contains blank identifiers.")

        try:
            frame[self.time_column] = pd.to_datetime(frame[self.time_column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Column '{self.time_column}' must contain valid datetime values."
            ) from exc

        invalid_numeric_columns: list[str] = []
        for feature_name in self.feature_names:
            converted = pd.to_numeric(frame[feature_name], errors="coerce")
            if converted.isnull().any():
                invalid_numeric_columns.append(feature_name)
                continue
            frame[feature_name] = converted.astype(float)

        if invalid_numeric_columns:
            raise TypeError(
                "Feature columns must be numeric: "
                + ", ".join(invalid_numeric_columns)
            )

        duplicate_keys = frame.duplicated(subset=[self.id_column, self.time_column], keep=False)
        if duplicate_keys.any():
            duplicate_rows = frame.loc[duplicate_keys, [self.id_column, self.time_column]]
            duplicates = ", ".join(
                f"{row[self.id_column]}@{row[self.time_column].date()}"
                for _, row in duplicate_rows.head(5).iterrows()
            )
            raise ValueError(
                "Duplicate customer/time keys detected. Examples: "
                f"{duplicates}"
            )

        if self.split_column in frame.columns:
            split_nulls = frame[self.split_column].isnull().sum()
            if split_nulls > 0:
                raise ValueError(
                    f"Column '{self.split_column}' contains {split_nulls} null values."
                )
            frame[self.split_column] = frame[self.split_column].astype(str).str.upper().str.strip()

        ordered_columns = [self.id_column, self.time_column]
        if self.split_column in frame.columns:
            ordered_columns.append(self.split_column)
        ordered_columns.extend(self.feature_names)
        remaining_columns = [column for column in frame.columns if column not in ordered_columns]
        return frame[ordered_columns + remaining_columns]

    def _apply_pipeline_config(self, config: Mapping[str, Any]) -> None:
        pipeline_settings = config["pipeline"]
        feature_settings = config.get("features", {})

        self.id_column = str(pipeline_settings["id_column"]).lower()
        self.time_column = str(pipeline_settings["time_column"]).lower()
        self.split_column = str(pipeline_settings.get("split_column", "split_flag")).lower()

        self.feature_names = [
            str(item["name"]).lower()
            for group_name in ("instant", "rolling_4w", "trend")
            for item in feature_settings.get(group_name, [])
        ]
