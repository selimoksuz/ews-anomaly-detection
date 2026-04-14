from __future__ import annotations

import numpy as np
import pandas as pd

from engine.config_loader import get_feature_list, load_config


def make_feature_frame(n: int, *, seed: int = 42, include_split: bool = False) -> pd.DataFrame:
    config = load_config()
    features = get_feature_list(config)
    rng = np.random.default_rng(seed)

    data = {}
    for index, feature in enumerate(features):
        scale = 1.0 + (index % 5) * 0.5
        loc = float(index % 7)
        values = rng.normal(loc=loc, scale=scale, size=n)
        if any(token in feature for token in ("amount", "balance", "limit")):
            values = np.abs(values) * 1000
        if any(token in feature for token in ("ratio", "util")):
            values = np.clip(rng.normal(loc=0.5, scale=0.2, size=n), 0, 1.5)
        if any(token in feature for token in ("count", "days", "dpd")):
            values = np.abs(np.round(values))
        data[feature] = values.astype(float)

    frame = pd.DataFrame(data)
    frame["customer_id"] = [f"CUST_{i:05d}" for i in range(n)]
    frame["snapshot_date"] = pd.date_range("2026-01-01", periods=n, freq="D")
    if include_split:
        split_point = int(n * 0.8)
        frame["split_flag"] = ["TRAIN" if i < split_point else "TEST" for i in range(n)]
    return frame


def make_scoring_frame(n: int, *, seed: int = 99) -> pd.DataFrame:
    return make_feature_frame(n, seed=seed, include_split=False)
