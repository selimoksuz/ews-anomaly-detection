"""Synthetic Ticari Orta Faz 1 data preparation and demo run helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine.business_features import FAZ1_BASE_FEATURES, build_ticari_orta_faz1_business_features
from engine.config_loader import load_config, load_secrets
from engine.history_features import (
    add_population_reference_features,
    add_self_history_features,
    add_trend_slope_features,
)
from engine.lifecycle import LifecycleManager
from engine.oracle_io import OracleConnector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline_config.yaml"
DEFAULT_NUM_SNAPSHOTS = 36
# Strict warm-up for Faz 1:
# - monthly year-over-year change features need 12 prior snapshots
# - self_zscore_6 needs 6 prior base observations
# Earliest fully mature row is therefore the 19th snapshot per customer.
STRICT_HISTORY_WARMUP_SNAPSHOTS = 18

NATIVE_COLUMNS = [
    "customer_id",
    "snapshot_date",
    "segment",
    "is_balance_sheet_customer",
    "has_pos",
    "nace_section",
    "nace_main",
    "fs_period_code",
    "fs_last_update_date",
    "memzuc_total_cash_risk_0_24m",
    "tlref_factor",
    "fs_net_sales_cumulative",
    "fs_ebitda_cumulative",
    "fs_trade_receivables",
    "fs_notes_receivable",
    "fs_net_profit_cumulative",
    "fs_equity",
    "pos_monthly_volume",
    "ifrs9_behavioral_pd",
    "kkb_commercial_score",
    "kkb_indebtedness_index",
    "memzuc_total_limit",
    "memzuc_total_risk",
]

TREND_FEATURES = ["pos_volume_change", "net_sales_change"]
POPULATION_FEATURES = ["pos_volume_change", "net_sales_change"]


@dataclass
class DemoPreparationSummary:
    segment: str
    native_rows: int
    derived_rows: int
    outcome_rows: int
    native_table: str
    input_table: str
    outcomes_table: str
    native_snapshot_start: str
    native_snapshot_end: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment": self.segment,
            "native_rows": self.native_rows,
            "derived_rows": self.derived_rows,
            "outcome_rows": self.outcome_rows,
            "native_table": self.native_table,
            "input_table": self.input_table,
            "outcomes_table": self.outcomes_table,
            "native_snapshot_start": self.native_snapshot_start,
            "native_snapshot_end": self.native_snapshot_end,
        }


class TicariOrtaFaz1DemoBuilder:
    """Create synthetic native data, derived features, and an end-to-end demo run."""

    def __init__(self, config_path=None, secrets_path=None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = load_config(self.config_path)
        self.secrets = load_secrets(secrets_path)
        self.id_column = self.config["pipeline"]["id_column"]
        self.time_column = self.config["pipeline"]["time_column"]
        self.segment_column = self.config["development"]["segment_column"]
        self.default_segment = self.config["development"].get("segment_value", "TICARI_ORTA_FAZ1")

    def prepare(
        self,
        *,
        segment: str | None = None,
        num_customers: int = 180,
        num_snapshots: int = DEFAULT_NUM_SNAPSHOTS,
        end_date=None,
        seed: int = 20260415,
        drop_existing: bool = True,
    ) -> dict[str, Any]:
        segment_value = segment or self.default_segment
        native_df = self.build_native_frame(
            num_customers=num_customers,
            num_snapshots=num_snapshots,
            end_date=end_date,
            segment=segment_value,
            seed=seed,
        )
        derived_df = self.build_derived_frame(native_df)
        outcomes_df = self.build_outcomes_frame(derived_df, seed=seed + 17)

        with OracleConnector(self.config, self.secrets) as ora:
            ora.setup_tables(drop_existing=drop_existing)
            self._create_native_table(ora, drop_existing=drop_existing)
            native_rows = self._write_native_rows(ora, native_df)
            derived_rows = ora.replace_rows("input_features", derived_df)
            outcome_rows = ora.replace_rows("outcomes", outcomes_df)

            summary = DemoPreparationSummary(
                segment=segment_value,
                native_rows=native_rows,
                derived_rows=derived_rows,
                outcome_rows=outcome_rows,
                native_table=ora._qualified_table_name("native_features"),
                input_table=ora._qualified_table_name("input_features"),
                outcomes_table=ora._qualified_table_name("outcomes"),
                native_snapshot_start=pd.to_datetime(native_df[self.time_column]).min().date().isoformat(),
                native_snapshot_end=pd.to_datetime(native_df[self.time_column]).max().date().isoformat(),
            )
            return summary.to_dict()

    def run(
        self,
        *,
        segment: str | None = None,
        num_customers: int = 180,
        num_snapshots: int = DEFAULT_NUM_SNAPSHOTS,
        end_date=None,
        seed: int = 20260415,
        drop_existing: bool = True,
    ) -> dict[str, Any]:
        segment_value = segment or self.default_segment
        preparation = self.prepare(
            segment=segment_value,
            num_customers=num_customers,
            num_snapshots=num_snapshots,
            end_date=end_date,
            seed=seed,
            drop_existing=drop_existing,
        )
        manager = LifecycleManager(config_path=self.config_path)
        develop_summary = manager.develop(segment=segment_value)
        promote_summary = manager.promote(segment=segment_value, model_version=develop_summary["model_version"])
        manager.live_scoring_cfg.setdefault("snapshot", {})["explicit_date"] = preparation["native_snapshot_end"]
        manager.config.setdefault("live_scoring", {}).setdefault("snapshot", {})["explicit_date"] = preparation["native_snapshot_end"]
        live_summary = manager.score_live(segment=segment_value)
        return {
            "prepare": preparation,
            "develop": develop_summary,
            "promote": promote_summary,
            "score_live": live_summary,
        }

    def build_native_frame(
        self,
        *,
        num_customers: int = 180,
        num_snapshots: int = DEFAULT_NUM_SNAPSHOTS,
        end_date=None,
        segment: str | None = None,
        seed: int = 20260415,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        segment_value = segment or self.default_segment
        snapshot_end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.today().normalize()
        snapshot_end = snapshot_end + pd.offsets.MonthEnd(0)
        snapshots = pd.date_range(end=snapshot_end, periods=num_snapshots, freq="ME")

        profiles = self._build_customer_profiles(num_customers, rng, segment_value, num_snapshots)
        records: list[dict[str, Any]] = []
        for snapshot_index, snapshot_date in enumerate(snapshots):
            quarter_code = _quarter_code(snapshot_date)
            fs_last_update = _quarter_reference_date(snapshot_date)
            seasonal_phase = 2.0 * np.pi * snapshot_index / 12.0
            for profile in profiles.itertuples(index=False):
                stress_multiplier = self._stress_multiplier(profile, snapshot_index, num_snapshots)
                seasonal = 1.0 + profile.season_amp * np.sin(seasonal_phase + profile.phase_shift)
                mild_growth = 1.0 + 0.003 * snapshot_index

                annual_sales = max(
                    50.0,
                    profile.base_turnover
                    * seasonal
                    * mild_growth
                    * stress_multiplier["turnover"]
                    * rng.normal(1.0, 0.03),
                )
                ebitda_margin = np.clip(
                    profile.base_ebitda_margin + stress_multiplier["ebitda_margin_shift"] + rng.normal(0.0, 0.01),
                    -0.10,
                    0.30,
                )
                profit_margin = np.clip(
                    profile.base_profit_margin + stress_multiplier["profit_margin_shift"] + rng.normal(0.0, 0.01),
                    -0.12,
                    0.20,
                )
                receivables_ratio = np.clip(
                    profile.base_receivables_ratio + stress_multiplier["receivables_ratio_shift"] + rng.normal(0.0, 0.02),
                    0.05,
                    1.60,
                )
                factor = _annualization_factor_from_code(quarter_code)
                annual_ebitda = annual_sales * ebitda_margin
                annual_profit = annual_sales * profit_margin
                trade_receivables = max(1.0, annual_sales * receivables_ratio * rng.normal(1.0, 0.03))
                notes_receivables = max(0.5, trade_receivables * profile.notes_share * rng.normal(1.0, 0.05))
                equity_value = (
                    profile.base_equity
                    * mild_growth
                    * stress_multiplier["equity"]
                    * rng.normal(1.0, 0.04)
                )
                pos_volume = max(
                    10.0,
                    profile.base_pos_volume
                    * seasonal
                    * mild_growth
                    * stress_multiplier["pos"]
                    * rng.normal(1.0, 0.05),
                )
                total_cash_risk = max(
                    20.0,
                    annual_sales
                    * profile.base_debt_ratio
                    * stress_multiplier["debt"]
                    * rng.normal(1.0, 0.05),
                )
                total_limit = max(
                    25.0,
                    profile.base_limit
                    * stress_multiplier["limit"]
                    * rng.normal(1.0, 0.04),
                )
                utilization = np.clip(
                    profile.base_utilization + stress_multiplier["utilization_shift"] + rng.normal(0.0, 0.03),
                    0.05,
                    1.60,
                )
                total_risk = total_limit * utilization
                pd_value = np.clip(
                    profile.base_pd + stress_multiplier["pd_shift"] + rng.normal(0.0, 0.01),
                    0.002,
                    0.90,
                )
                kkb_score = np.clip(
                    profile.base_kkb_score + stress_multiplier["kkb_score_shift"] + rng.normal(0.0, 20.0),
                    600.0,
                    1900.0,
                )
                indebtedness_index = np.clip(
                    profile.base_indebtedness_index + stress_multiplier["indebtedness_shift"] + rng.normal(0.0, 0.03),
                    0.05,
                    2.50,
                )

                records.append(
                    {
                        "customer_id": profile.customer_id,
                        "snapshot_date": snapshot_date,
                        "segment": profile.segment,
                        "is_balance_sheet_customer": 1,
                        "has_pos": 1,
                        "nace_section": profile.nace_section,
                        "nace_main": profile.nace_main,
                        "fs_period_code": quarter_code,
                        "fs_last_update_date": fs_last_update,
                        "memzuc_total_cash_risk_0_24m": total_cash_risk,
                        "tlref_factor": profile.tlref_factor,
                        "fs_net_sales_cumulative": annual_sales / factor,
                        "fs_ebitda_cumulative": annual_ebitda / factor,
                        "fs_trade_receivables": trade_receivables,
                        "fs_notes_receivable": notes_receivables,
                        "fs_net_profit_cumulative": annual_profit / factor,
                        "fs_equity": equity_value,
                        "pos_monthly_volume": pos_volume,
                        "ifrs9_behavioral_pd": pd_value,
                        "kkb_commercial_score": kkb_score,
                        "kkb_indebtedness_index": indebtedness_index,
                        "memzuc_total_limit": total_limit,
                        "memzuc_total_risk": total_risk,
                    }
                )

        native = pd.DataFrame.from_records(records, columns=NATIVE_COLUMNS)
        native = self._inject_missing_and_outliers(native, rng)
        native = native.sort_values([self.id_column, self.time_column]).reset_index(drop=True)
        return native

    def build_derived_frame(self, native_df: pd.DataFrame) -> pd.DataFrame:
        base = build_ticari_orta_faz1_business_features(
            native_df,
            id_column=self.id_column,
            time_column=self.time_column,
            segment_column=self.segment_column,
        )
        derived = add_self_history_features(
            base,
            base_features=FAZ1_BASE_FEATURES,
            id_column=self.id_column,
            time_column=self.time_column,
            zscore_min_periods=6,
        )
        derived = add_trend_slope_features(
            derived,
            trend_features=TREND_FEATURES,
            id_column=self.id_column,
            time_column=self.time_column,
            min_periods=6,
        )
        derived = add_population_reference_features(
            derived,
            population_features=POPULATION_FEATURES,
            time_column=self.time_column,
        )
        derived = self._trim_history_warmup_rows(derived)
        derived = derived.sort_values([self.id_column, self.time_column]).reset_index(drop=True)
        return derived

    def _trim_history_warmup_rows(self, derived_df: pd.DataFrame) -> pd.DataFrame:
        frame = derived_df.copy()
        frame[self.time_column] = pd.to_datetime(frame[self.time_column], errors="raise")
        frame = frame.sort_values([self.id_column, self.time_column]).reset_index(drop=True)
        position = frame.groupby(self.id_column, sort=False).cumcount()
        frame = frame.loc[position >= STRICT_HISTORY_WARMUP_SNAPSHOTS].reset_index(drop=True)
        return frame

    def build_outcomes_frame(self, derived_df: pd.DataFrame, *, seed: int = 20260432) -> pd.DataFrame:
        frame = derived_df.copy()
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        rng = np.random.default_rng(seed)

        score_frame = frame[FAZ1_BASE_FEATURES].copy()
        score_frame = score_frame.apply(lambda series: series.fillna(series.median()), axis=0).fillna(0.0)

        risk_signal = (
            1.35 * np.clip(score_frame["bank_debt_to_turnover"], 0.0, 4.0)
            + 0.70 * np.clip(score_frame["bank_debt_to_ebitda"], 0.0, 8.0)
            + 0.75 * np.clip(score_frame["trade_receivables_to_turnover"], 0.0, 2.5)
            + 0.95 * np.clip(-score_frame["pos_volume_change"], 0.0, 2.0)
            + 0.85 * np.clip(-score_frame["net_sales_change"], 0.0, 2.0)
            + 0.70 * np.clip(-score_frame["profitability_to_turnover"], 0.0, 1.0) * 5.0
            + 0.55 * np.clip(-score_frame["equity_change"], 0.0, 1.0) * 4.0
            + 6.0 * np.clip(score_frame["ifrs9_behavioral_pd"], 0.0, 1.0)
            + 0.75 * np.clip((1500.0 - score_frame["kkb_commercial_score"]) / 250.0, 0.0, 4.0)
            + 1.10 * np.clip(score_frame["kkb_indebtedness_index"], 0.0, 2.5)
            + 0.90 * np.clip(score_frame["memzuc_limit_utilization_increase"], 0.0, 1.8)
        )

        probability_30 = _sigmoid((risk_signal - 4.7) / 0.95)
        probability_default = _sigmoid((risk_signal - 5.6) / 1.05) * 0.8
        labels_30 = rng.binomial(1, np.clip(probability_30, 0.01, 0.95))
        labels_default = rng.binomial(1, np.clip(probability_default, 0.005, 0.80))

        return pd.DataFrame(
            {
                self.id_column: frame[self.id_column].astype(str),
                self.time_column: pd.to_datetime(frame[self.time_column]),
                "label_30dpd_8w": labels_30.astype(int),
                "label_default_12m": labels_default.astype(int),
            }
        )

    def _build_customer_profiles(
        self,
        num_customers: int,
        rng: np.random.Generator,
        segment_value: str,
        num_snapshots: int,
    ) -> pd.DataFrame:
        sectors = [
            ("C", "IMALAT"),
            ("G", "TICARET"),
            ("F", "INSAAT"),
            ("N", "HIZMET"),
        ]
        sector_idx = rng.integers(0, len(sectors), size=num_customers)
        severe_count = min(num_customers, max(4, num_customers // 10))
        severe_ids = set(rng.choice(num_customers, size=severe_count, replace=False).tolist())
        remaining = [index for index in range(num_customers) if index not in severe_ids]
        moderate_count = min(len(remaining), max(6, num_customers // 8))
        moderate_ids = set(rng.choice(remaining, size=moderate_count, replace=False).tolist()) if remaining else set()

        profiles = []
        stress_low = max(2, min(8, num_snapshots - 4))
        stress_high = max(stress_low + 1, num_snapshots - 1)
        for index in range(num_customers):
            nace_section, nace_main = sectors[sector_idx[index]]
            profile_kind = "stable"
            if index in severe_ids:
                profile_kind = "severe"
            elif index in moderate_ids:
                profile_kind = "moderate"
            phase_shift = float(rng.uniform(0.0, 2.0 * np.pi))
            profiles.append(
                {
                    "customer_id": f"TOF{index + 1:04d}",
                    "segment": segment_value,
                    "nace_section": nace_section,
                    "nace_main": nace_main,
                    "base_turnover": float(np.exp(rng.normal(7.2, 0.35))),
                    "base_pos_volume": float(np.exp(rng.normal(5.6, 0.30))),
                    "base_debt_ratio": float(rng.uniform(0.30, 1.40)),
                    "base_ebitda_margin": float(rng.uniform(0.05, 0.18)),
                    "base_profit_margin": float(rng.uniform(0.02, 0.09)),
                    "base_receivables_ratio": float(rng.uniform(0.18, 0.58)),
                    "base_equity": float(np.exp(rng.normal(6.0, 0.35))),
                    "base_pd": float(rng.uniform(0.01, 0.07)),
                    "base_kkb_score": float(rng.normal(1450.0, 110.0)),
                    "base_indebtedness_index": float(rng.uniform(0.35, 0.90)),
                    "base_limit": float(np.exp(rng.normal(6.5, 0.35))),
                    "base_utilization": float(rng.uniform(0.35, 0.72)),
                    "notes_share": float(rng.uniform(0.08, 0.22)),
                    "tlref_factor": float(rng.uniform(1.02, 1.25)),
                    "season_amp": float(rng.uniform(0.03, 0.12)),
                    "phase_shift": phase_shift,
                    "stress_start": int(rng.integers(stress_low, stress_high)),
                    "profile_kind": profile_kind,
                }
            )
        return pd.DataFrame.from_records(profiles)

    @staticmethod
    def _stress_multiplier(profile, snapshot_index: int, num_snapshots: int) -> dict[str, float]:
        if profile.profile_kind == "stable" or snapshot_index < int(profile.stress_start):
            return {
                "turnover": 1.0,
                "debt": 1.0,
                "pos": 1.0,
                "equity": 1.0,
                "limit": 1.0,
                "utilization_shift": 0.0,
                "pd_shift": 0.0,
                "kkb_score_shift": 0.0,
                "indebtedness_shift": 0.0,
                "ebitda_margin_shift": 0.0,
                "profit_margin_shift": 0.0,
                "receivables_ratio_shift": 0.0,
            }

        progress = (snapshot_index - int(profile.stress_start) + 1) / max(1, num_snapshots - int(profile.stress_start))
        if profile.profile_kind == "severe":
            return {
                "turnover": 1.0 - 0.28 * progress,
                "debt": 1.0 + 0.42 * progress,
                "pos": 1.0 - 0.36 * progress,
                "equity": 1.0 - 0.38 * progress,
                "limit": 1.0 - 0.12 * progress,
                "utilization_shift": 0.26 * progress,
                "pd_shift": 0.18 * progress,
                "kkb_score_shift": -260.0 * progress,
                "indebtedness_shift": 0.40 * progress,
                "ebitda_margin_shift": -0.08 * progress,
                "profit_margin_shift": -0.07 * progress,
                "receivables_ratio_shift": 0.22 * progress,
            }

        return {
            "turnover": 1.0 - 0.14 * progress,
            "debt": 1.0 + 0.20 * progress,
            "pos": 1.0 - 0.18 * progress,
            "equity": 1.0 - 0.16 * progress,
            "limit": 1.0 - 0.04 * progress,
            "utilization_shift": 0.10 * progress,
            "pd_shift": 0.06 * progress,
            "kkb_score_shift": -90.0 * progress,
            "indebtedness_shift": 0.12 * progress,
            "ebitda_margin_shift": -0.03 * progress,
            "profit_margin_shift": -0.02 * progress,
            "receivables_ratio_shift": 0.09 * progress,
        }

    @staticmethod
    def _inject_missing_and_outliers(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        adjusted = frame.copy()
        total_rows = len(adjusted)

        def choose_rows(rate: float) -> np.ndarray:
            count = max(1, int(total_rows * rate))
            return rng.choice(total_rows, size=count, replace=False)

        missing_rates = {
            "pos_monthly_volume": 0.02,
            "fs_ebitda_cumulative": 0.03,
            "fs_trade_receivables": 0.02,
            "ifrs9_behavioral_pd": 0.015,
            "kkb_commercial_score": 0.015,
            "memzuc_total_limit": 0.01,
        }
        for column, rate in missing_rates.items():
            adjusted.loc[choose_rows(rate), column] = np.nan

        adjusted.loc[choose_rows(0.01), "memzuc_total_cash_risk_0_24m"] *= 5.0
        adjusted.loc[choose_rows(0.01), "pos_monthly_volume"] *= 8.0
        adjusted.loc[choose_rows(0.01), "fs_trade_receivables"] *= 4.0
        adjusted.loc[choose_rows(0.005), "ifrs9_behavioral_pd"] = 0.75
        adjusted.loc[choose_rows(0.005), "kkb_commercial_score"] = 650.0
        return adjusted

    def _create_native_table(self, ora: OracleConnector, *, drop_existing: bool) -> None:
        connection = ora.connect()
        with connection.cursor() as cursor:
            if drop_existing and ora._table_exists("native_features"):
                cursor.execute(f"DROP TABLE {ora._qualified_table_name('native_features')} PURGE")
            if not ora._table_exists("native_features"):
                cursor.execute(self._native_table_ddl(ora))
        connection.commit()

    def _native_table_ddl(self, ora: OracleConnector) -> str:
        primary_key = f"PK_{ora._table_name('native_features')}"
        return f"""
            CREATE TABLE {ora._qualified_table_name("native_features")} (
                {self.id_column.upper()} VARCHAR2(128) NOT NULL,
                {self.time_column.upper()} DATE NOT NULL,
                {self.segment_column.upper()} VARCHAR2(64) NOT NULL,
                IS_BALANCE_SHEET_CUSTOMER NUMBER(1) NOT NULL,
                HAS_POS NUMBER(1) NOT NULL,
                NACE_SECTION VARCHAR2(16),
                NACE_MAIN VARCHAR2(64),
                FS_PERIOD_CODE VARCHAR2(16),
                FS_LAST_UPDATE_DATE DATE,
                MEMZUC_TOTAL_CASH_RISK_0_24M NUMBER(18,6),
                TLREF_FACTOR NUMBER(18,6),
                FS_NET_SALES_CUMULATIVE NUMBER(18,6),
                FS_EBITDA_CUMULATIVE NUMBER(18,6),
                FS_TRADE_RECEIVABLES NUMBER(18,6),
                FS_NOTES_RECEIVABLE NUMBER(18,6),
                FS_NET_PROFIT_CUMULATIVE NUMBER(18,6),
                FS_EQUITY NUMBER(18,6),
                POS_MONTHLY_VOLUME NUMBER(18,6),
                IFRS9_BEHAVIORAL_PD NUMBER(18,6),
                KKB_COMMERCIAL_SCORE NUMBER(18,6),
                KKB_INDEBTEDNESS_INDEX NUMBER(18,6),
                MEMZUC_TOTAL_LIMIT NUMBER(18,6),
                MEMZUC_TOTAL_RISK NUMBER(18,6),
                CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT {primary_key} PRIMARY KEY ({self.id_column.upper()}, {self.time_column.upper()})
            )
        """

    def _write_native_rows(self, ora: OracleConnector, native_df: pd.DataFrame, batch_size: int = 1000) -> int:
        normalized = ora._normalize_columns(native_df)
        ordered_columns = list(NATIVE_COLUMNS)
        normalized[self.time_column] = pd.to_datetime(normalized[self.time_column], errors="raise")
        normalized["fs_last_update_date"] = pd.to_datetime(normalized["fs_last_update_date"], errors="raise")

        insert_sql = f"""
            INSERT INTO {ora._qualified_table_name("native_features")} (
                {", ".join(column.upper() for column in ordered_columns)}
            ) VALUES (
                {", ".join(f":{index}" for index in range(1, len(ordered_columns) + 1))}
            )
        """
        rows = [
            ora._coerce_scalar_sequence(record)
            for record in normalized[ordered_columns].itertuples(index=False, name=None)
        ]
        return ora._executemany(insert_sql, rows, batch_size=batch_size)


def prepare_ticari_orta_faz1_demo(**kwargs) -> dict[str, Any]:
    return TicariOrtaFaz1DemoBuilder(
        config_path=kwargs.pop("config_path", None),
        secrets_path=kwargs.pop("secrets_path", None),
    ).prepare(**kwargs)


def run_ticari_orta_faz1_demo(**kwargs) -> dict[str, Any]:
    return TicariOrtaFaz1DemoBuilder(
        config_path=kwargs.pop("config_path", None),
        secrets_path=kwargs.pop("secrets_path", None),
    ).run(**kwargs)


def _quarter_code(snapshot_date: pd.Timestamp) -> str:
    quarter = int(pd.Timestamp(snapshot_date).quarter)
    return "YE" if quarter == 4 else f"Q{quarter}"


def _annualization_factor_from_code(code: str) -> float:
    if code == "Q1":
        return 4.0
    if code == "Q2":
        return 2.0
    if code == "Q3":
        return 4.0 / 3.0
    return 1.0


def _quarter_reference_date(snapshot_date: pd.Timestamp) -> pd.Timestamp:
    snapshot = pd.Timestamp(snapshot_date)
    quarter_period = snapshot.to_period("Q")
    return quarter_period.end_time.normalize()


def _sigmoid(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))
