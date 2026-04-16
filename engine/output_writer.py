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
        return self._write_oracle(results, snapshot_date, segment=segment)

    def _write_oracle(self, results: pd.DataFrame, snapshot_date, *, segment: str):
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
            deleted = ora.delete_scored_snapshot(snapshot_date, segment=segment)
            inserted_results = ora.write_results(results_frame)
            inserted_details = 0
            inserted_full_effects = 0
            if detail_rows:
                inserted_details = ora.write_details(pd.DataFrame(detail_rows))
            if full_effect_rows and self.output_cfg.get("oracle", {}).get("full_effects_table_key"):
                inserted_full_effects = ora.write_full_effects(pd.DataFrame(full_effect_rows))
        return {
            "backend": "oracle",
            "deleted_results": deleted.get("results", 0),
            "deleted_details": deleted.get("details", 0),
            "deleted_full_effects": deleted.get("full_effects", 0),
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
                    parts.append(self._format_reason_block(detail))
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
                        "expected_value": detail.get("expected_value", detail.get("ae_referansi", detail.get("beklenen"))),
                        "actual_value": detail.get("actual_value", detail.get("gerceklesen")),
                        "delta_pct": detail.get("delta_pct", detail.get("degisim_pct")),
                        "contribution_pct": detail.get("contribution_pct", detail.get("ensemble_katki_pct", detail.get("katki_pct"))),
                        "customer_history_reference": detail.get("musteri_gecmis_referansi"),
                        "population_reference": detail.get("populasyon_referansi"),
                        "ae_reference": detail.get("ae_referansi", detail.get("beklenen")),
                        "ae_contribution_pct": detail.get("ae_katki_pct"),
                        "if_contribution_pct": detail.get("if_katki_pct"),
                        "md_contribution_pct": detail.get("md_katki_pct"),
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

    @staticmethod
    def _format_reason_block(detail: dict) -> str:
        return (
            f"{detail['label']}\n"
            f"gerceklesen: {OutputWriter._display_value(detail.get('gerceklesen'))}\n"
            f"musteri_gecmis_referansi: {OutputWriter._display_value(detail.get('musteri_gecmis_referansi'))}\n"
            f"populasyon_referansi: {OutputWriter._display_value(detail.get('populasyon_referansi'))}\n"
            f"ae_referansi: {OutputWriter._display_value(detail.get('ae_referansi', detail.get('beklenen')))}\n"
            f"ensemble_katki: %{OutputWriter._display_pct(detail.get('ensemble_katki_pct', detail.get('katki_pct')))} "
            f"(AE %{OutputWriter._display_pct(detail.get('ae_katki_pct'))}, "
            f"IF %{OutputWriter._display_pct(detail.get('if_katki_pct'))}, "
            f"MD %{OutputWriter._display_pct(detail.get('md_katki_pct'))})"
        )

    @staticmethod
    def _display_value(value) -> str:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.2f}"

    @staticmethod
    def _display_pct(value) -> str:
        if value is None or pd.isna(value):
            return "0"
        return f"{float(value):.1f}".rstrip("0").rstrip(".")
