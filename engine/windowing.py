"""Window resolution utilities for time-based development and retraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def is_empty(self) -> bool:
        return self.start is pd.NaT or self.end is pd.NaT


class WindowResolver:
    """Resolve configured train/test/calibration/OOT windows from available snapshots."""

    WINDOW_ORDER = ("train", "test", "calibration", "oot")

    def __init__(self, config: dict):
        self.config = config
        self.development_cfg = config.get("development", {})
        self.windows_cfg = self.development_cfg.get("windows", {})

    def resolve(self, available_snapshots) -> dict[str, WindowSpec]:
        snapshots = sorted(pd.to_datetime(pd.Index(available_snapshots)).unique())
        if not snapshots:
            raise ValueError("No snapshot dates available to resolve development windows.")

        mode = self.windows_cfg.get("mode", "relative_periods")
        if mode == "relative_periods":
            return self._resolve_relative(snapshots)
        if mode == "fixed":
            return self._resolve_fixed()
        raise ValueError(f"Unsupported development window mode: {mode}")

    def _resolve_relative(self, snapshots) -> dict[str, WindowSpec]:
        relative_cfg = self.windows_cfg.get("relative", {})
        anchor_date = relative_cfg.get("anchor_date")
        if anchor_date:
            anchor_ts = pd.Timestamp(anchor_date)
            snapshots = [snapshot for snapshot in snapshots if snapshot <= anchor_ts]
            if not snapshots:
                raise ValueError(f"No snapshots available on or before anchor_date={anchor_date}.")

        windows: dict[str, WindowSpec] = {}

        oot_count = int(relative_cfg.get("oot_periods", 0) or 0)
        calibration_count = int(relative_cfg.get("calibration_periods", 0) or 0)

        if oot_count > 0:
            if len(snapshots) < oot_count:
                raise ValueError(f"Not enough snapshots to allocate {oot_count} periods for window 'oot'.")
            oot_snapshots = snapshots[-oot_count:]
            windows["oot"] = WindowSpec(
                name="oot",
                start=pd.Timestamp(oot_snapshots[0]),
                end=pd.Timestamp(oot_snapshots[-1]),
            )
            history_snapshots = [snapshot for snapshot in snapshots if snapshot < pd.Timestamp(oot_snapshots[0])]
        else:
            history_snapshots = list(snapshots)

        if calibration_count > 0:
            if len(snapshots) < calibration_count:
                raise ValueError(
                    f"Not enough snapshots to allocate {calibration_count} periods for window 'calibration'."
                )
            calibration_snapshots = snapshots[-calibration_count:]
            windows["calibration"] = WindowSpec(
                name="calibration",
                start=pd.Timestamp(calibration_snapshots[0]),
                end=pd.Timestamp(calibration_snapshots[-1]),
            )

        if not history_snapshots:
            raise ValueError("No historical snapshots remain before OOT to allocate train/test windows.")

        history_start = pd.Timestamp(history_snapshots[0])
        history_end = pd.Timestamp(history_snapshots[-1])
        windows["train"] = WindowSpec(name="train", start=history_start, end=history_end)
        windows["test"] = WindowSpec(name="test", start=history_start, end=history_end)

        return {name: windows[name] for name in self.WINDOW_ORDER if name in windows}

    def _resolve_fixed(self) -> dict[str, WindowSpec]:
        fixed_cfg = self.windows_cfg.get("fixed", {})
        windows: dict[str, WindowSpec] = {}

        for name in self.WINDOW_ORDER:
            item = fixed_cfg.get(name)
            if not item:
                continue
            start = item.get("start")
            end = item.get("end")
            if not start or not end:
                continue
            windows[name] = WindowSpec(
                name=name,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
            )

        if not windows:
            raise ValueError("No fixed development windows configured.")
        return windows


def summarize_window(frame: pd.DataFrame, time_column: str) -> dict:
    """Return row count and date boundaries for a sliced development window."""
    if frame.empty:
        return {"rows": 0, "start": None, "end": None}

    dates = pd.to_datetime(frame[time_column])
    return {
        "rows": int(len(frame)),
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
    }
