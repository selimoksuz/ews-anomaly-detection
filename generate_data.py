"""
Synthetic data generator for EWS Anomaly Detection.

Generates:
  1. Training data  - normal customer behavior (5000 customers)
  2. Inference data  - current week with injected anomalies

Anomaly types:
  A: Single feature extreme shift (univariate)
  B: Correlated features breaking normal relationship (multivariate)
  C: Gradual drift across multiple features (subtle)
"""

import numpy as np
import pandas as pd
from config import ALL_FEATURES

np.random.seed(42)

N_CUSTOMERS = 5000
N_ANOMALY_A = 50
N_ANOMALY_B = 80
N_ANOMALY_C = 70
N_ANOMALIES = N_ANOMALY_A + N_ANOMALY_B + N_ANOMALY_C


def _base_params():
    return {
        "days_past_due": (2.0, 3.0),
        "worst_dpd_last_6m": (5.0, 6.0),
        "min_payment_only_count_6m": (0.5, 0.8),
        "payment_to_minimum_ratio": (3.0, 1.5),
        "avg_days_to_payment": (12.0, 5.0),
        "payment_amount_cv": (0.3, 0.15),
        "payment_reversal_count": (0.2, 0.4),
        "number_of_30dpd_last_12m": (0.3, 0.6),
        "utilization_ratio": (0.40, 0.18),
        "utilization_delta_3m": (0.0, 0.05),
        "credit_limit_change_pct": (0.0, 0.02),
        "overlimit_frequency_6m": (0.1, 0.3),
        "cash_advance_to_spending_ratio": (0.05, 0.06),
        "draw_ratio": (0.35, 0.15),
        "txn_count_monthly": (25.0, 12.0),
        "total_amount_monthly": (15000.0, 8000.0),
        "avg_txn_amount": (600.0, 350.0),
        "txn_count_ratio_vs_3m": (1.0, 0.15),
        "credit_turnover_trend": (0.0, 0.05),
        "debit_turnover_trend": (0.0, 0.05),
        "channel_diversity_score": (3.0, 1.2),
        "new_merchant_category_count": (1.0, 1.5),
        "outstanding_balance": (25000.0, 15000.0),
        "balance_to_income_ratio": (0.35, 0.15),
        "avg_checking_balance_3m": (8000.0, 5000.0),
        "min_checking_balance_3m": (3000.0, 3000.0),
        "deposit_volatility_6m": (0.25, 0.12),
        "nsf_event_count_12m": (0.1, 0.3),
        "util_ratio_slope_3m": (0.0, 0.02),
        "payment_ratio_slope_3m": (0.0, 0.03),
        "balance_change_pct_1m": (0.0, 0.08),
        "dpd_trend_3m": (0.0, 0.5),
        "spending_to_limit_ratio_delta": (0.0, 0.03),
        "amount_volatility_change": (0.0, 0.1),
        "avg_txn_amount_delta_3m": (0.0, 0.08),
    }


def _build_corr(n_features):
    A = np.random.RandomState(42).randn(n_features, n_features) * 0.05
    cov = A @ A.T + np.eye(n_features)

    idx = {f: i for i, f in enumerate(ALL_FEATURES)}
    pairs = [
        ("utilization_ratio", "outstanding_balance", 0.6),
        ("utilization_ratio", "draw_ratio", 0.7),
        ("utilization_ratio", "txn_count_monthly", 0.3),
        ("days_past_due", "worst_dpd_last_6m", 0.8),
        ("days_past_due", "min_payment_only_count_6m", 0.5),
        ("days_past_due", "number_of_30dpd_last_12m", 0.6),
        ("payment_to_minimum_ratio", "days_past_due", -0.4),
        ("avg_checking_balance_3m", "min_checking_balance_3m", 0.7),
        ("avg_checking_balance_3m", "balance_to_income_ratio", -0.3),
        ("total_amount_monthly", "txn_count_monthly", 0.6),
        ("total_amount_monthly", "avg_txn_amount", 0.4),
        ("outstanding_balance", "balance_to_income_ratio", 0.5),
        ("util_ratio_slope_3m", "utilization_delta_3m", 0.8),
        ("dpd_trend_3m", "days_past_due", 0.4),
        ("nsf_event_count_12m", "min_checking_balance_3m", -0.3),
        ("cash_advance_to_spending_ratio", "utilization_ratio", 0.3),
        ("payment_reversal_count", "nsf_event_count_12m", 0.4),
    ]
    for f1, f2, c in pairs:
        i, j = idx[f1], idx[f2]
        cov[i, j] = c
        cov[j, i] = c

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.01, None)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def _constrain(data):
    idx = {f: i for i, f in enumerate(ALL_FEATURES)}
    non_neg = [
        "days_past_due", "worst_dpd_last_6m", "min_payment_only_count_6m",
        "avg_days_to_payment", "payment_amount_cv", "payment_reversal_count",
        "number_of_30dpd_last_12m", "overlimit_frequency_6m",
        "cash_advance_to_spending_ratio", "txn_count_monthly",
        "total_amount_monthly", "avg_txn_amount", "outstanding_balance",
        "avg_checking_balance_3m", "min_checking_balance_3m",
        "deposit_volatility_6m", "nsf_event_count_12m",
        "new_merchant_category_count",
    ]
    for f in non_neg:
        data[:, idx[f]] = np.clip(data[:, idx[f]], 0, None)

    for f in ["utilization_ratio", "draw_ratio", "cash_advance_to_spending_ratio"]:
        data[:, idx[f]] = np.clip(data[:, idx[f]], 0, 1)

    data[:, idx["payment_to_minimum_ratio"]] = np.clip(
        data[:, idx["payment_to_minimum_ratio"]], 0, None
    )

    counts = [
        "days_past_due", "worst_dpd_last_6m", "min_payment_only_count_6m",
        "payment_reversal_count", "number_of_30dpd_last_12m",
        "overlimit_frequency_6m", "txn_count_monthly",
        "nsf_event_count_12m", "new_merchant_category_count",
    ]
    for f in counts:
        data[:, idx[f]] = np.round(data[:, idx[f]])
    return data


def generate_normal_data(n=N_CUSTOMERS):
    params = _base_params()
    means = np.array([params[f][0] for f in ALL_FEATURES])
    stds = np.array([params[f][1] for f in ALL_FEATURES])
    corr = _build_corr(len(ALL_FEATURES))

    raw = np.random.multivariate_normal(np.zeros(len(ALL_FEATURES)), corr, size=n)
    data = _constrain(raw * stds + means)

    df = pd.DataFrame(data, columns=ALL_FEATURES)
    df.insert(0, "customer_id", [f"CUST_{i:05d}" for i in range(n)])
    return df


def generate_inference_data(n=N_CUSTOMERS):
    df = generate_normal_data(n)
    data = df[ALL_FEATURES].values.copy()

    params = _base_params()
    means = np.array([params[f][0] for f in ALL_FEATURES])
    stds = np.array([params[f][1] for f in ALL_FEATURES])
    idx = {f: i for i, f in enumerate(ALL_FEATURES)}

    rng = np.random.RandomState(123)
    anom_idx = rng.choice(n, N_ANOMALIES, replace=False)
    a_idx = anom_idx[:N_ANOMALY_A]
    b_idx = anom_idx[N_ANOMALY_A:N_ANOMALY_A + N_ANOMALY_B]
    c_idx = anom_idx[N_ANOMALY_A + N_ANOMALY_B:]

    # Type A: single feature extreme
    ext_feats = ["utilization_ratio", "days_past_due", "outstanding_balance",
                 "payment_reversal_count", "nsf_event_count_12m"]
    for i in a_idx:
        f = rng.choice(ext_feats)
        j = idx[f]
        data[i, j] = means[j] + stds[j] * rng.uniform(4, 7)

    # Type B: break correlation structure
    scenarios = ["high_util_low_txn", "stress_no_dpd", "payment_break"]
    for i in b_idx:
        sc = rng.choice(scenarios)
        if sc == "high_util_low_txn":
            data[i, idx["utilization_ratio"]] = rng.uniform(0.85, 0.98)
            data[i, idx["draw_ratio"]] = rng.uniform(0.80, 0.95)
            data[i, idx["txn_count_monthly"]] = rng.uniform(1, 5)
            data[i, idx["total_amount_monthly"]] = rng.uniform(500, 2000)
            data[i, idx["util_ratio_slope_3m"]] = rng.uniform(0.05, 0.15)
        elif sc == "stress_no_dpd":
            data[i, idx["days_past_due"]] = 0
            data[i, idx["worst_dpd_last_6m"]] = 0
            data[i, idx["min_payment_only_count_6m"]] = rng.randint(3, 6)
            data[i, idx["payment_to_minimum_ratio"]] = rng.uniform(1.0, 1.2)
            data[i, idx["cash_advance_to_spending_ratio"]] = rng.uniform(0.3, 0.6)
            data[i, idx["deposit_volatility_6m"]] = rng.uniform(0.6, 0.9)
            data[i, idx["avg_checking_balance_3m"]] = rng.uniform(200, 800)
        else:
            data[i, idx["avg_days_to_payment"]] = rng.uniform(25, 35)
            data[i, idx["payment_amount_cv"]] = rng.uniform(0.8, 1.5)
            data[i, idx["payment_reversal_count"]] = rng.randint(2, 5)
            data[i, idx["channel_diversity_score"]] = rng.uniform(0.5, 1.0)
            data[i, idx["new_merchant_category_count"]] = rng.randint(5, 10)

    # Type C: subtle drift
    stress_decrease = {
        "payment_to_minimum_ratio", "avg_checking_balance_3m",
        "min_checking_balance_3m", "txn_count_monthly",
        "txn_count_ratio_vs_3m", "channel_diversity_score",
    }
    for i in c_idx:
        n_drift = rng.randint(8, 13)
        feats = rng.choice(len(ALL_FEATURES), n_drift, replace=False)
        for j in feats:
            d = -1 if ALL_FEATURES[j] in stress_decrease else 1
            data[i, j] += d * stds[j] * rng.uniform(1.5, 2.5)

    data = _constrain(data)
    df_inf = pd.DataFrame(data, columns=ALL_FEATURES)
    df_inf.insert(0, "customer_id", [f"CUST_{i:05d}" for i in range(n)])

    labels = pd.DataFrame({
        "customer_id": df_inf["customer_id"],
        "is_anomaly": False,
        "anomaly_type": "NORMAL",
    })
    labels.loc[a_idx, ["is_anomaly", "anomaly_type"]] = [True, "A_UNIVARIATE"]
    labels.loc[b_idx, ["is_anomaly", "anomaly_type"]] = [True, "B_MULTIVARIATE"]
    labels.loc[c_idx, ["is_anomaly", "anomaly_type"]] = [True, "C_SUBTLE_DRIFT"]

    return df_inf, labels


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    print("Training data (normal)...")
    train = generate_normal_data()
    train.to_csv("data/training_data.csv", index=False)
    print(f"  {len(train)} customers x {len(ALL_FEATURES)} features -> data/training_data.csv")

    print("Inference data (with anomalies)...")
    inf, lab = generate_inference_data()
    inf.to_csv("data/inference_data.csv", index=False)
    lab.to_csv("data/anomaly_labels.csv", index=False)
    print(f"  {len(inf)} customers, {lab['is_anomaly'].sum()} anomalies")
    print(f"  Type A: {N_ANOMALY_A} | Type B: {N_ANOMALY_B} | Type C: {N_ANOMALY_C}")
