"""Self-history and population reference feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.business_features import safe_divide


def add_self_history_features(
    feature_frame: pd.DataFrame,
    *,
    base_features: list[str],
    id_column: str = "customer_id",
    time_column: str = "snapshot_date",
    delta_lag: int = 1,
    zscore_window: int = 6,
    zscore_min_periods: int = 3,
) -> pd.DataFrame:
    """Append delta and self-zscore features for each base feature."""
    frame = feature_frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    id_column = id_column.lower()
    time_column = time_column.lower()
    base_features = [str(feature).strip().lower() for feature in base_features]

    frame[time_column] = pd.to_datetime(frame[time_column], errors="raise")
    frame = frame.sort_values([id_column, time_column]).reset_index(drop=True)

    for feature in base_features:
        grouped = frame.groupby(id_column, sort=False)[feature]
        prior = grouped.shift(delta_lag)
        frame[f"{feature}__delta_1"] = frame[feature] - prior

        mean = (
            grouped.apply(lambda series: series.shift(1).rolling(zscore_window, min_periods=zscore_min_periods).mean())
            .reset_index(level=0, drop=True)
        )
        std = (
            grouped.apply(lambda series: series.shift(1).rolling(zscore_window, min_periods=zscore_min_periods).std())
            .reset_index(level=0, drop=True)
        )
        zscore = safe_divide(frame[feature] - mean, std)
        frame[f"{feature}__self_zscore_6"] = zscore

    return frame


def add_trend_slope_features(
    feature_frame: pd.DataFrame,
    *,
    trend_features: list[str],
    id_column: str = "customer_id",
    time_column: str = "snapshot_date",
    window: int = 6,
    min_periods: int = 4,
) -> pd.DataFrame:
    """Append trailing trend slopes for selected features."""
    frame = feature_frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    id_column = id_column.lower()
    time_column = time_column.lower()
    trend_features = [str(feature).strip().lower() for feature in trend_features]

    frame[time_column] = pd.to_datetime(frame[time_column], errors="raise")
    frame = frame.sort_values([id_column, time_column]).reset_index(drop=True)

    for feature in trend_features:
        slope = (
            frame.groupby(id_column, sort=False)[feature]
            .apply(
                lambda series: series.rolling(window, min_periods=min_periods).apply(
                    _rolling_slope,
                    raw=True,
                )
            )
            .reset_index(level=0, drop=True)
        )
        frame[f"{feature}__trend_slope_6"] = slope

    return frame


def add_population_reference_features(
    feature_frame: pd.DataFrame,
    *,
    population_features: list[str],
    time_column: str = "snapshot_date",
    include_percentile: bool = True,
    include_median_delta: bool = True,
) -> pd.DataFrame:
    """Append within-snapshot percentiles and median ratios for selected features."""
    frame = feature_frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    time_column = time_column.lower()
    population_features = [str(feature).strip().lower() for feature in population_features]

    frame[time_column] = pd.to_datetime(frame[time_column], errors="raise")
    for feature in population_features:
        grouped = frame.groupby(time_column, sort=False)[feature]
        if include_percentile:
            frame[f"{feature}__population_percentile"] = grouped.rank(pct=True, method="average")
        if include_median_delta:
            median = grouped.transform("median")
            frame[f"{feature}__vs_population_median_delta"] = frame[feature] - median
    return frame


def _rolling_slope(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    mask = ~np.isnan(clean)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(clean.size, dtype=float)[mask]
    y = clean[mask]
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 0:
        return np.nan
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)
