"""Outcome-based weight tuning for calibrated component scores."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config_loader import normalize_ensemble_weights


COMPONENT_COLUMNS = ("ae_cal", "if_cal", "md_cal")


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WeightOptimizer:
    """Search simple ensemble weights against a forward outcome target."""

    def __init__(self, config: dict):
        self.config = config
        cfg = config.get("weight_optimization", {})
        self.grid_step = float(cfg.get("grid_step", 0.1))
        self.objective = cfg.get("objective", "precision_at_top_percent")
        self.top_percent = float(cfg.get("top_percent", 0.1))

    def optimize(
        self,
        tuning_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        *,
        target_column: str,
        monitoring_columns: list[str] | None = None,
        model_version: str,
        segment: str,
    ) -> dict:
        monitoring_columns = monitoring_columns or []
        candidates = []
        for weights in self._weight_grid(self.grid_step):
            tuning_metrics = self.evaluate(
                tuning_frame,
                weights,
                target_column=target_column,
                monitoring_columns=monitoring_columns,
            )
            candidates.append(
                {
                    "weights": weights,
                    "objective_value": tuning_metrics["primary"][self.objective],
                    "tuning_metrics": tuning_metrics,
                }
            )

        if not candidates:
            raise ValueError("No weight candidates were generated for optimization.")

        candidates.sort(
            key=lambda item: (
                item["objective_value"],
                item["tuning_metrics"]["primary"]["recall_at_top_percent"],
                item["tuning_metrics"]["primary"]["lift_at_top_percent"],
            ),
            reverse=True,
        )
        best = candidates[0]
        validation_metrics = self.evaluate(
            validation_frame,
            best["weights"],
            target_column=target_column,
            monitoring_columns=monitoring_columns,
        )

        return {
            "weight_version": f"{model_version}-wgt",
            "created_at": _utc_now(),
            "model_version": model_version,
            "segment": segment,
            "target_column": target_column,
            "monitoring_columns": monitoring_columns,
            "objective": self.objective,
            "top_percent": self.top_percent,
            "weights": best["weights"],
            "candidate_count": len(candidates),
            "tuning_metrics": best["tuning_metrics"],
            "validation_metrics": validation_metrics,
            "leaderboard": candidates[:10],
        }

    def evaluate(
        self,
        frame: pd.DataFrame,
        weights: dict,
        *,
        target_column: str,
        monitoring_columns: list[str] | None = None,
    ) -> dict:
        monitoring_columns = monitoring_columns or []
        if frame.empty:
            raise ValueError("Weight evaluation frame is empty.")
        if target_column not in frame.columns:
            raise KeyError(f"Target column '{target_column}' is missing from the weight dataset.")

        score = self._ensemble_score(frame, weights)
        top_n = max(1, int(np.ceil(len(frame) * self.top_percent)))
        order = np.argsort(score)[::-1]
        selected = order[:top_n]

        payload = {
            "rows": int(len(frame)),
            "top_n": int(top_n),
            "weights": weights,
            "primary": self._label_metrics(frame[target_column], selected, top_n),
            "monitoring": {},
        }
        for column in monitoring_columns:
            if column in frame.columns:
                payload["monitoring"][column] = self._label_metrics(frame[column], selected, top_n)
        return payload

    def _ensemble_score(self, frame: pd.DataFrame, weights: dict) -> np.ndarray:
        resolved = normalize_ensemble_weights(weights)
        return (
            resolved["autoencoder"] * frame["ae_cal"].to_numpy(dtype=float)
            + resolved["isolation_forest"] * frame["if_cal"].to_numpy(dtype=float)
            + resolved["mahalanobis"] * frame["md_cal"].to_numpy(dtype=float)
        )

    def _label_metrics(self, labels, selected: np.ndarray, top_n: int) -> dict:
        values = pd.Series(labels).fillna(0).astype(int).to_numpy()
        positives = int(values.sum())
        true_positives = int(values[selected].sum())
        baseline = float(values.mean()) if len(values) else 0.0
        precision = true_positives / top_n if top_n else 0.0
        recall = true_positives / positives if positives else 0.0
        lift = precision / baseline if baseline > 0 else None
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0

        return {
            "positive_rows": positives,
            "baseline_rate": round(baseline, 6),
            "precision_at_top_percent": round(float(precision), 6),
            "recall_at_top_percent": round(float(recall), 6),
            "lift_at_top_percent": round(float(lift), 6) if lift is not None else None,
            "f1_at_top_percent": round(float(f1_score), 6),
        }

    @staticmethod
    def save(path: Path, payload: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _weight_grid(step: float):
        units = max(1, int(round(1.0 / step)))
        for ae_units in range(units + 1):
            for if_units in range(units - ae_units + 1):
                md_units = units - ae_units - if_units
                weights = normalize_ensemble_weights(
                    {
                        "autoencoder": ae_units / units,
                        "isolation_forest": if_units / units,
                        "mahalanobis": md_units / units,
                    }
                )
                yield weights
