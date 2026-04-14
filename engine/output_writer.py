"""Oracle-only output writer for scored results."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from engine.oracle_io import OracleConnector


class OutputWriter:
    """Write scored outputs only to Oracle."""

    def __init__(self, config: dict, secrets: Optional[dict] = None):
        self.config = config
        self.secrets = secrets
        self.output_cfg = config.get("sources", {}).get("outputs", {})
        self.id_column = config["pipeline"]["id_column"]
        self.time_column = config["pipeline"]["time_column"]
        self.scoring_cfg = config.get("scoring", {})

    def write(self, results: pd.DataFrame, snapshot_date, *, run_id: str, segment: str):
        backend = self.output_cfg.get("backend", "oracle")
        if backend != "oracle":
            raise ValueError(
                f"Output backend '{backend}' is configured, but this project now supports Oracle only."
            )
        return self._write_oracle(results, snapshot_date)

    def _write_oracle(self, results: pd.DataFrame, snapshot_date):
        results_frame = results.copy()
        results_frame[self.time_column] = pd.Timestamp(snapshot_date)
        results_frame["reasons"] = self._build_reasons(results_frame)
        detail_rows = self._build_effect_rows(
            results,
            snapshot_date,
            detail_column="detay",
            alerts_only=True,
        )
        full_effect_rows = []
        if self.scoring_cfg.get("persist_full_feature_effects", False):
            full_effect_rows = self._build_effect_rows(
                results,
                snapshot_date,
                detail_column="full_detay",
                alerts_only=self.scoring_cfg.get("full_effect_scope", "all") == "alerts_only",
            )

        with OracleConnector(self.config, self.secrets) as ora:
            inserted_results = ora.write_results(results_frame)
            inserted_details = 0
            inserted_full_effects = 0
            if detail_rows:
                inserted_details = ora.write_details(pd.DataFrame(detail_rows))
            if full_effect_rows and self.output_cfg.get("oracle", {}).get("full_effects_table_key"):
                inserted_full_effects = ora.write_full_effects(pd.DataFrame(full_effect_rows))
        return {
            "backend": "oracle",
            "inserted_results": inserted_results,
            "inserted_details": inserted_details,
            "inserted_full_effects": inserted_full_effects,
        }

    def _build_reasons(self, frame: pd.DataFrame):
        reasons = []
        for _, row in frame.iterrows():
            parts = []
            if isinstance(row.get("detay"), dict):
                for _, detail in row["detay"].items():
                    direction = "UP" if detail["degisim_pct"] > 0 else "DN"
                    parts.append(
                        f"{detail['label']}: {detail['beklenen']}->{detail['gerceklesen']}"
                        f" ({direction}%{abs(detail['degisim_pct']):.0f})"
                    )
            reasons.append(parts)
        return reasons

    def _build_effect_rows(self, results: pd.DataFrame, snapshot_date, *, detail_column: str, alerts_only: bool):
        scoped = results
        if alerts_only:
            scoped = results[results["alert_band"].isin(["KIRMIZI", "TURUNCU", "SARI"])]

        effect_rows = []
        for _, row in scoped.iterrows():
            if not isinstance(row.get(detail_column), dict):
                continue
            for rank, (feature_name, detail) in enumerate(row[detail_column].items(), 1):
                effect_rows.append(
                    {
                        self.id_column: row[self.id_column],
                        self.time_column: pd.Timestamp(snapshot_date),
                        "feature_name": feature_name,
                        "feature_label": detail["label"],
                        "expected_value": detail["beklenen"],
                        "actual_value": detail["gerceklesen"],
                        "delta_pct": detail["degisim_pct"],
                        "contribution_pct": detail["katki_pct"],
                        "rank": int(detail.get("rank", rank)),
                        "is_top_reason": 1 if detail.get("is_top_reason", rank <= self.scoring_cfg.get("top_n_reasons", 3)) else 0,
                        "alert_band": row.get("alert_band"),
                        "run_id": row.get("run_id"),
                        "model_version": row.get("model_version"),
                        "calibration_version": row.get("calibration_version"),
                        "weight_version": row.get("weight_version"),
                    }
                )
        return effect_rows
