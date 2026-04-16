"""Business feature builders for the Ticari Orta Faz 1 demo."""

from __future__ import annotations

import numpy as np
import pandas as pd


FAZ1_BASE_FEATURES = [
    "bank_debt_to_turnover",
    "pos_volume_change",
    "bank_debt_to_ebitda",
    "trade_receivables_to_turnover",
    "profitability_to_turnover",
    "equity_change",
    "ifrs9_behavioral_pd",
    "kkb_commercial_score",
    "kkb_indebtedness_index",
    "net_sales_change",
    "memzuc_limit_utilization_increase",
]

YEARLY_SEASONAL_LAG = 12


ANNUALIZATION_FACTORS = {
    "Q1": 4.0,
    "Q2": 2.0,
    "Q3": 4.0 / 3.0,
    "Q4": 1.0,
    "YE": 1.0,
}


def safe_divide(numerator, denominator) -> pd.Series:
    """Safely divide two vectors and return NaN for missing/zero denominators."""
    num = pd.Series(numerator, copy=False, dtype=float)
    den = pd.Series(denominator, copy=False, dtype=float)
    ratio = num / den.replace(0.0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    return ratio.astype(float)


def annualization_factor(period_codes) -> pd.Series:
    """Resolve quarter-to-annual multipliers from a period code column."""
    raw = pd.Series(period_codes, copy=False)
    normalized = raw.astype(str).str.upper().str.strip()
    normalized = normalized.str.replace(r"[^A-Z0-9]", "", regex=True)
    normalized = normalized.str.extract(r"(Q1|Q2|Q3|Q4|YE)", expand=False).fillna("YE")
    return normalized.map(ANNUALIZATION_FACTORS).fillna(1.0).astype(float)


def build_ticari_orta_faz1_business_features(
    native_frame: pd.DataFrame,
    *,
    id_column: str = "customer_id",
    time_column: str = "snapshot_date",
    segment_column: str = "segment",
) -> pd.DataFrame:
    """Build phase-1 business features from native atomic inputs."""
    frame = native_frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    id_column = id_column.lower()
    time_column = time_column.lower()
    segment_column = segment_column.lower()

    frame[time_column] = pd.to_datetime(frame[time_column], errors="raise")
    frame = frame.sort_values([id_column, time_column]).reset_index(drop=True)

    annual_factor = annualization_factor(frame["fs_period_code"])
    annualized_sales = frame["fs_net_sales_cumulative"].astype(float) * annual_factor
    annualized_ebitda = frame["fs_ebitda_cumulative"].astype(float) * annual_factor
    annualized_profit = frame["fs_net_profit_cumulative"].astype(float) * annual_factor
    total_trade_receivables = (
        frame["fs_trade_receivables"].astype(float) + frame["fs_notes_receivable"].astype(float)
    )

    prior_pos = frame.groupby(id_column)["pos_monthly_volume"].shift(YEARLY_SEASONAL_LAG)
    prior_sales = pd.Series(annualized_sales).groupby(frame[id_column]).shift(YEARLY_SEASONAL_LAG)
    prior_equity = frame.groupby(id_column)["fs_equity"].shift(YEARLY_SEASONAL_LAG)

    derived = frame[[id_column, time_column, segment_column]].copy()
    derived["bank_debt_to_turnover"] = safe_divide(frame["memzuc_total_cash_risk_0_24m"], annualized_sales)
    derived["pos_volume_change"] = safe_divide(frame["pos_monthly_volume"] - prior_pos, prior_pos.abs())
    derived["bank_debt_to_ebitda"] = safe_divide(
        frame["memzuc_total_cash_risk_0_24m"] * frame["tlref_factor"],
        annualized_ebitda,
    )
    derived["trade_receivables_to_turnover"] = safe_divide(total_trade_receivables, annualized_sales)
    derived["profitability_to_turnover"] = safe_divide(annualized_profit, annualized_sales)
    derived["equity_change"] = safe_divide(frame["fs_equity"] - prior_equity, prior_equity.abs())
    derived["ifrs9_behavioral_pd"] = frame["ifrs9_behavioral_pd"].astype(float)
    derived["kkb_commercial_score"] = frame["kkb_commercial_score"].astype(float)
    derived["kkb_indebtedness_index"] = frame["kkb_indebtedness_index"].astype(float)
    derived["net_sales_change"] = safe_divide(annualized_sales - prior_sales, prior_sales.abs())
    derived["memzuc_limit_utilization_increase"] = safe_divide(
        frame["memzuc_total_risk"],
        frame["memzuc_total_limit"],
    )

    return derived
