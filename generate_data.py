"""
Synthetic data generator for EWS Anomaly Detection.

4 musteri segmenti ile gercekci veri uretir:
  PREMIUM:  Yuksek gelir, dusuk risk, yuksek islem hacmi
  STANDARD: Orta gelir, normal davranis
  RISKY:    Dusuk gelir, yuksek kullanim, gecikme egilimi
  NEW:      Yeni musteriler, dusuk islem gecmisi

32 degisken, 4 katman:
  Katman 1: Anlik (8)
  Katman 2: Rolling 4W (11)
  Katman 3: Trend (9)
  Katman 4: Interaction (4)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import ALL_FEATURES, INSTANT_FEATURES, ROLLING_4W_FEATURES, TREND_FEATURES, INTERACTION_FEATURES

N_CUSTOMERS = 5000
N_ANOMALY_A = 50
N_ANOMALY_B = 80
N_ANOMALY_C = 70
N_ANOMALIES = N_ANOMALY_A + N_ANOMALY_B + N_ANOMALY_C

# Interaction feature'lar haric base feature'lar (model bunlari ogrenir, interaction turetilir)
BASE_FEATURES = INSTANT_FEATURES + ROLLING_4W_FEATURES + TREND_FEATURES
N_BASE = len(BASE_FEATURES)  # 28

# ═══════════════════════════════════════════════════════════════
# SEGMENT PARAMETRELERI — (mean, std) her feature icin
# ═══════════════════════════════════════════════════════════════

_SEGMENTS = {
    "PREMIUM": {
        "weight": 0.20,
        "params": {
            # Katman 1 — Anlik
            "dpd_current": (0.0, 0.5),
            "utilization_ratio": (0.22, 0.10),
            "outstanding_balance": (12000, 8000),
            "checking_balance": (28000, 12000),
            "txn_count_weekly": (12, 4),
            "txn_amount_weekly": (8000, 3500),
            "avg_txn_amount_weekly": (680, 300),
            "payment_amount_this_week": (5000, 3000),
            # Katman 2 — Rolling 4W
            "dpd_max_4w": (0.0, 1.0),
            "min_payment_only_count_4w": (0.0, 0.1),
            "payment_to_min_ratio_4w": (8.0, 3.0),
            "avg_days_to_payment_4w": (4.0, 2.0),
            "payment_reversal_count_4w": (0.0, 0.05),
            "nsf_count_4w": (0.0, 0.05),
            "overlimit_count_4w": (0.0, 0.05),
            "cash_advance_ratio_4w": (0.01, 0.015),
            "checking_balance_min_4w": (18000, 10000),
            "deposit_amount_avg_4w": (12000, 5000),
            "channel_count_4w": (4.5, 0.8),
            # Katman 3 — Trend
            "util_slope_4w": (0.0, 0.008),
            "balance_slope_4w": (0, 500),
            "checking_slope_4w": (200, 600),
            "payment_ratio_slope_4w": (0.01, 0.02),
            "txn_count_change_pct": (0, 8),
            "txn_amount_change_pct": (0, 10),
            "deposit_change_pct": (2, 8),
            "util_acceleration": (0.0, 0.004),
            "dpd_direction_4w": (0.0, 0.2),
        },
    },
    "STANDARD": {
        "weight": 0.45,
        "params": {
            "dpd_current": (1.5, 2.5),
            "utilization_ratio": (0.40, 0.14),
            "outstanding_balance": (22000, 11000),
            "checking_balance": (6500, 3500),
            "txn_count_weekly": (6, 3),
            "txn_amount_weekly": (3500, 1800),
            "avg_txn_amount_weekly": (580, 250),
            "payment_amount_this_week": (2500, 1500),
            "dpd_max_4w": (3.0, 4.0),
            "min_payment_only_count_4w": (0.3, 0.5),
            "payment_to_min_ratio_4w": (3.8, 1.5),
            "avg_days_to_payment_4w": (11.0, 4.0),
            "payment_reversal_count_4w": (0.05, 0.2),
            "nsf_count_4w": (0.05, 0.15),
            "overlimit_count_4w": (0.05, 0.2),
            "cash_advance_ratio_4w": (0.03, 0.03),
            "checking_balance_min_4w": (2500, 2000),
            "deposit_amount_avg_4w": (5500, 2500),
            "channel_count_4w": (3.0, 1.0),
            "util_slope_4w": (0.0, 0.015),
            "balance_slope_4w": (0, 800),
            "checking_slope_4w": (0, 400),
            "payment_ratio_slope_4w": (0.0, 0.03),
            "txn_count_change_pct": (0, 12),
            "txn_amount_change_pct": (0, 15),
            "deposit_change_pct": (0, 12),
            "util_acceleration": (0.0, 0.006),
            "dpd_direction_4w": (0.3, 0.6),
        },
    },
    "RISKY": {
        "weight": 0.25,
        "params": {
            "dpd_current": (7.0, 6.0),
            "utilization_ratio": (0.72, 0.13),
            "outstanding_balance": (42000, 16000),
            "checking_balance": (1800, 1200),
            "txn_count_weekly": (4, 2.5),
            "txn_amount_weekly": (2000, 1200),
            "avg_txn_amount_weekly": (520, 280),
            "payment_amount_this_week": (800, 600),
            "dpd_max_4w": (15.0, 10.0),
            "min_payment_only_count_4w": (2.0, 1.2),
            "payment_to_min_ratio_4w": (1.4, 0.6),
            "avg_days_to_payment_4w": (22.0, 6.0),
            "payment_reversal_count_4w": (0.4, 0.5),
            "nsf_count_4w": (0.3, 0.4),
            "overlimit_count_4w": (0.4, 0.5),
            "cash_advance_ratio_4w": (0.14, 0.10),
            "checking_balance_min_4w": (250, 300),
            "deposit_amount_avg_4w": (2500, 1500),
            "channel_count_4w": (2.0, 0.8),
            "util_slope_4w": (0.015, 0.02),
            "balance_slope_4w": (800, 1200),
            "checking_slope_4w": (-200, 500),
            "payment_ratio_slope_4w": (-0.02, 0.03),
            "txn_count_change_pct": (-5, 15),
            "txn_amount_change_pct": (-3, 18),
            "deposit_change_pct": (-5, 15),
            "util_acceleration": (0.003, 0.008),
            "dpd_direction_4w": (1.5, 1.0),
        },
    },
    "NEW": {
        "weight": 0.10,
        "params": {
            "dpd_current": (0.3, 1.0),
            "utilization_ratio": (0.28, 0.16),
            "outstanding_balance": (7000, 4500),
            "checking_balance": (4500, 2500),
            "txn_count_weekly": (3, 2),
            "txn_amount_weekly": (1500, 1000),
            "avg_txn_amount_weekly": (500, 300),
            "payment_amount_this_week": (1200, 800),
            "dpd_max_4w": (1.0, 2.0),
            "min_payment_only_count_4w": (0.1, 0.3),
            "payment_to_min_ratio_4w": (4.5, 2.0),
            "avg_days_to_payment_4w": (9.0, 4.0),
            "payment_reversal_count_4w": (0.05, 0.15),
            "nsf_count_4w": (0.02, 0.1),
            "overlimit_count_4w": (0.02, 0.1),
            "cash_advance_ratio_4w": (0.02, 0.025),
            "checking_balance_min_4w": (2000, 1500),
            "deposit_amount_avg_4w": (3500, 2000),
            "channel_count_4w": (2.2, 1.0),
            "util_slope_4w": (0.005, 0.015),
            "balance_slope_4w": (200, 600),
            "checking_slope_4w": (100, 400),
            "payment_ratio_slope_4w": (0.0, 0.02),
            "txn_count_change_pct": (5, 15),
            "txn_amount_change_pct": (5, 18),
            "deposit_change_pct": (3, 12),
            "util_acceleration": (0.001, 0.005),
            "dpd_direction_4w": (0.1, 0.3),
        },
    },
}

# ═══════════════════════════════════════════════════════════════
# KORELASYON CIFTLERI
# ═══════════════════════════════════════════════════════════════

_CORR_PAIRS = [
    ("dpd_current", "dpd_max_4w", 0.75),
    ("dpd_current", "dpd_direction_4w", 0.50),
    ("dpd_current", "payment_to_min_ratio_4w", -0.40),
    ("utilization_ratio", "outstanding_balance", 0.55),
    ("utilization_ratio", "txn_count_weekly", 0.30),
    ("utilization_ratio", "util_slope_4w", 0.35),
    ("outstanding_balance", "balance_slope_4w", 0.40),
    ("checking_balance", "checking_balance_min_4w", 0.80),
    ("checking_balance", "checking_slope_4w", 0.35),
    ("checking_balance", "nsf_count_4w", -0.30),
    ("txn_count_weekly", "txn_amount_weekly", 0.60),
    ("txn_amount_weekly", "avg_txn_amount_weekly", 0.40),
    ("min_payment_only_count_4w", "payment_to_min_ratio_4w", -0.55),
    ("min_payment_only_count_4w", "cash_advance_ratio_4w", 0.30),
    ("payment_reversal_count_4w", "nsf_count_4w", 0.45),
    ("deposit_amount_avg_4w", "deposit_change_pct", 0.30),
    ("util_slope_4w", "util_acceleration", 0.40),
    ("checking_slope_4w", "deposit_change_pct", 0.30),
    ("balance_slope_4w", "checking_slope_4w", -0.25),
]


def _build_corr():
    rng = np.random.RandomState(42)
    A = rng.randn(N_BASE, N_BASE) * 0.03
    cov = A @ A.T + np.eye(N_BASE)
    idx = {f: i for i, f in enumerate(BASE_FEATURES)}
    for f1, f2, c in _CORR_PAIRS:
        i, j = idx[f1], idx[f2]
        cov[i, j] = c
        cov[j, i] = c
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.01, None)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def _constrain(data):
    idx = {f: i for i, f in enumerate(BASE_FEATURES)}
    non_neg = [
        "dpd_current", "dpd_max_4w", "outstanding_balance", "checking_balance",
        "txn_count_weekly", "txn_amount_weekly", "avg_txn_amount_weekly",
        "payment_amount_this_week", "min_payment_only_count_4w",
        "payment_reversal_count_4w", "nsf_count_4w", "overlimit_count_4w",
        "cash_advance_ratio_4w", "checking_balance_min_4w",
        "deposit_amount_avg_4w", "avg_days_to_payment_4w", "dpd_direction_4w",
    ]
    for f in non_neg:
        data[:, idx[f]] = np.clip(data[:, idx[f]], 0, None)

    data[:, idx["utilization_ratio"]] = np.clip(data[:, idx["utilization_ratio"]], 0, 1)
    data[:, idx["cash_advance_ratio_4w"]] = np.clip(data[:, idx["cash_advance_ratio_4w"]], 0, 1)
    data[:, idx["payment_to_min_ratio_4w"]] = np.clip(data[:, idx["payment_to_min_ratio_4w"]], 0.5, 20)
    data[:, idx["channel_count_4w"]] = np.clip(data[:, idx["channel_count_4w"]], 1, 6)
    data[:, idx["avg_days_to_payment_4w"]] = np.clip(data[:, idx["avg_days_to_payment_4w"]], 1, 40)
    data[:, idx["dpd_direction_4w"]] = np.clip(data[:, idx["dpd_direction_4w"]], 0, 4)
    data[:, idx["min_payment_only_count_4w"]] = np.clip(data[:, idx["min_payment_only_count_4w"]], 0, 4)
    data[:, idx["overlimit_count_4w"]] = np.clip(data[:, idx["overlimit_count_4w"]], 0, 4)

    # dpd_max >= dpd_current
    data[:, idx["dpd_max_4w"]] = np.maximum(data[:, idx["dpd_max_4w"]], data[:, idx["dpd_current"]])
    # min_balance <= balance
    data[:, idx["checking_balance_min_4w"]] = np.minimum(
        data[:, idx["checking_balance_min_4w"]], data[:, idx["checking_balance"]]
    )
    # avg_txn tutarlilik
    safe_txn = np.clip(data[:, idx["txn_count_weekly"]], 1, None)
    implied = data[:, idx["txn_amount_weekly"]] / safe_txn
    data[:, idx["avg_txn_amount_weekly"]] = 0.6 * data[:, idx["avg_txn_amount_weekly"]] + 0.4 * implied

    # Tamsayilar
    for f in ["dpd_current", "dpd_max_4w", "txn_count_weekly",
              "min_payment_only_count_4w", "payment_reversal_count_4w",
              "nsf_count_4w", "overlimit_count_4w", "dpd_direction_4w"]:
        data[:, idx[f]] = np.round(data[:, idx[f]])

    return data


def _compute_interactions(df):
    """Katman 4 interaction feature'lari hesapla."""
    eps = 1e-6

    # Likidite sikismasi: util yuksek * bakiye dusuk * cash_advance yuksek
    df["liquidity_squeeze_score"] = (
        df["utilization_ratio"].clip(lower=0) *
        (1.0 / df["checking_balance"].clip(lower=100)) * 1000 *
        (1 + df["cash_advance_ratio_4w"] * 10)
    )

    # Gizli stres: DPD dusuk AMA min_pay yuksek, odeme orani dusuk, mevduat dusuyor
    dpd_low_factor = 1.0 / (1 + df["dpd_current"])  # DPD sifira yakinsa yuksek
    df["hidden_stress_score"] = (
        dpd_low_factor *
        df["min_payment_only_count_4w"] *
        (1.0 / df["payment_to_min_ratio_4w"].clip(lower=0.5)) *
        np.clip(-df["deposit_change_pct"], 0, 100) / 10
    )

    # Gelir erozyonu: mevduat dusuyor + borc artiyor + bakiye eriyor
    df["income_erosion_score"] = (
        np.clip(-df["deposit_change_pct"], 0, 100) / 10 *
        np.clip(df["balance_slope_4w"], 0, None) / 500 *
        np.clip(-df["checking_slope_4w"], 0, None) / 200
    )

    # Odeme kirilmasi: gec odeme + iade + islem dususu + kanal dususu
    df["payment_breakdown_score"] = (
        df["avg_days_to_payment_4w"] / 30 *
        (1 + df["payment_reversal_count_4w"]) *
        np.clip(-df["txn_count_change_pct"], 0, 100) / 20 *
        (1.0 / df["channel_count_4w"].clip(lower=1))
    )

    # NaN/inf temizle + log normalizasyon (olcek farki gidermek icin)
    for f in INTERACTION_FEATURES:
        df[f] = df[f].replace([np.inf, -np.inf], 0).fillna(0).clip(lower=0)
        # log(1+x) transform — buyuk degerleri sıkistir, kucukleri koru
        df[f] = np.log1p(df[f])

    return df


def _generate_segment_data(n, rng):
    corr = _build_corr()
    all_data, all_segments = [], []

    for seg_name, seg_info in _SEGMENTS.items():
        seg_n = int(n * seg_info["weight"])
        params = seg_info["params"]
        means = np.array([params[f][0] for f in BASE_FEATURES])
        stds = np.array([params[f][1] for f in BASE_FEATURES])
        raw = rng.multivariate_normal(np.zeros(N_BASE), corr, size=seg_n)
        all_data.append(raw * stds + means)
        all_segments.extend([seg_name] * seg_n)

    remaining = n - len(all_segments)
    if remaining > 0:
        params = _SEGMENTS["STANDARD"]["params"]
        means = np.array([params[f][0] for f in BASE_FEATURES])
        stds = np.array([params[f][1] for f in BASE_FEATURES])
        raw = rng.multivariate_normal(np.zeros(N_BASE), corr, size=remaining)
        all_data.append(raw * stds + means)
        all_segments.extend(["STANDARD"] * remaining)

    data = np.vstack(all_data)
    order = rng.permutation(len(data))
    data = data[order]
    segments = [all_segments[i] for i in order]
    data = _constrain(data)
    return data, segments


# ═══════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════

def generate_normal_data(n=N_CUSTOMERS, seed=42):
    rng = np.random.RandomState(seed)
    data, segments = _generate_segment_data(n, rng)
    df = pd.DataFrame(data, columns=BASE_FEATURES)
    df.insert(0, "customer_id", [f"CUST_{i:05d}" for i in range(n)])
    df = _compute_interactions(df)
    return df


def generate_training_data(n=N_CUSTOMERS, seed=42):
    rng = np.random.RandomState(seed)
    data, segments = _generate_segment_data(n, rng)
    df = pd.DataFrame(data, columns=BASE_FEATURES)
    df.insert(0, "customer_id", [f"CUST_{i:05d}" for i in range(n)])
    df["segment"] = segments
    df = _compute_interactions(df)

    test_idx = rng.choice(n, int(n * 0.2), replace=False)
    df["split_flag"] = "TRAIN"
    df.loc[test_idx, "split_flag"] = "TEST"

    base = datetime(2026, 4, 7)
    df["snapshot_date"] = [base - timedelta(days=int(rng.randint(7, 180))) for _ in range(n)]
    return df


def generate_scoring_data(n=N_CUSTOMERS, seed=99):
    rng = np.random.RandomState(seed)
    data, segments = _generate_segment_data(n, rng)
    idx = {f: i for i, f in enumerate(BASE_FEATURES)}

    anom_idx = rng.choice(n, N_ANOMALIES, replace=False)
    a_idx = anom_idx[:N_ANOMALY_A]
    b_idx = anom_idx[N_ANOMALY_A:N_ANOMALY_A + N_ANOMALY_B]
    c_idx = anom_idx[N_ANOMALY_A + N_ANOMALY_B:]

    # Type A: tek degisken asiri sapma
    for i in a_idx:
        sc = rng.choice(["extreme_dpd", "extreme_util", "extreme_balance", "extreme_nsf"])
        if sc == "extreme_dpd":
            data[i, idx["dpd_current"]] = rng.randint(25, 55)
            data[i, idx["dpd_max_4w"]] = data[i, idx["dpd_current"]] + rng.randint(0, 10)
            data[i, idx["dpd_direction_4w"]] = rng.randint(3, 5)
        elif sc == "extreme_util":
            data[i, idx["utilization_ratio"]] = rng.uniform(0.95, 1.0)
            data[i, idx["overlimit_count_4w"]] = rng.randint(2, 4)
        elif sc == "extreme_balance":
            data[i, idx["outstanding_balance"]] = rng.uniform(85000, 140000)
        else:
            data[i, idx["nsf_count_4w"]] = rng.randint(3, 6)
            data[i, idx["payment_reversal_count_4w"]] = rng.randint(2, 4)

    # Type B: korelasyon kirilmasi (multivariate)
    for i in b_idx:
        sc = rng.choice(["high_util_low_txn", "hidden_stress", "payment_break", "income_erosion"])
        if sc == "high_util_low_txn":
            data[i, idx["utilization_ratio"]] = rng.uniform(0.88, 0.98)
            data[i, idx["txn_count_weekly"]] = rng.randint(0, 2)
            data[i, idx["txn_amount_weekly"]] = rng.uniform(50, 500)
            data[i, idx["util_slope_4w"]] = rng.uniform(0.04, 0.10)
            data[i, idx["util_acceleration"]] = rng.uniform(0.01, 0.03)
        elif sc == "hidden_stress":
            data[i, idx["dpd_current"]] = 0
            data[i, idx["dpd_max_4w"]] = rng.randint(0, 3)
            data[i, idx["min_payment_only_count_4w"]] = rng.randint(3, 5)
            data[i, idx["payment_to_min_ratio_4w"]] = rng.uniform(1.0, 1.1)
            data[i, idx["cash_advance_ratio_4w"]] = rng.uniform(0.25, 0.50)
            data[i, idx["checking_balance"]] = rng.uniform(100, 500)
            data[i, idx["checking_balance_min_4w"]] = rng.uniform(10, 80)
            data[i, idx["deposit_change_pct"]] = rng.uniform(-50, -25)
        elif sc == "payment_break":
            data[i, idx["avg_days_to_payment_4w"]] = rng.uniform(28, 38)
            data[i, idx["payment_reversal_count_4w"]] = rng.randint(2, 4)
            data[i, idx["txn_count_change_pct"]] = rng.uniform(-60, -35)
            data[i, idx["channel_count_4w"]] = rng.uniform(1.0, 1.5)
            data[i, idx["payment_ratio_slope_4w"]] = rng.uniform(-0.08, -0.04)
        else:  # income_erosion
            data[i, idx["deposit_change_pct"]] = rng.uniform(-55, -30)
            data[i, idx["deposit_amount_avg_4w"]] = rng.uniform(500, 1500)
            data[i, idx["balance_slope_4w"]] = rng.uniform(2000, 5000)
            data[i, idx["checking_slope_4w"]] = rng.uniform(-2000, -800)
            data[i, idx["checking_balance"]] = rng.uniform(300, 1000)

    # Type C: subtle drift (8-12 feature birden kayiyor)
    stress_decrease = {
        "payment_to_min_ratio_4w", "checking_balance", "checking_balance_min_4w",
        "txn_count_weekly", "txn_count_change_pct", "channel_count_4w",
        "deposit_change_pct", "deposit_amount_avg_4w", "checking_slope_4w",
        "payment_ratio_slope_4w",
    }
    for i in c_idx:
        seg_params = _SEGMENTS.get(segments[i], _SEGMENTS["STANDARD"])["params"]
        n_drift = rng.randint(8, 13)
        feats = rng.choice(N_BASE, n_drift, replace=False)
        for j in feats:
            fn = BASE_FEATURES[j]
            std = seg_params[fn][1]
            d = -1 if fn in stress_decrease else 1
            data[i, j] += d * std * rng.uniform(1.8, 3.0)

    data = _constrain(data)
    df = pd.DataFrame(data, columns=BASE_FEATURES)
    df.insert(0, "customer_id", [f"CUST_{i:05d}" for i in range(n)])
    df["segment"] = segments
    df = _compute_interactions(df)
    df["snapshot_date"] = datetime(2026, 4, 8)

    labels = pd.DataFrame({"customer_id": df["customer_id"], "is_anomaly": False, "anomaly_type": "NORMAL"})
    labels.loc[a_idx, ["is_anomaly", "anomaly_type"]] = [True, "A_UNIVARIATE"]
    labels.loc[b_idx, ["is_anomaly", "anomaly_type"]] = [True, "B_MULTIVARIATE"]
    labels.loc[c_idx, ["is_anomaly", "anomaly_type"]] = [True, "C_SUBTLE_DRIFT"]
    return df, labels


def generate_inference_data(n=N_CUSTOMERS):
    return generate_scoring_data(n)


if __name__ == "__main__":
    print("Training data...")
    train = generate_training_data()
    print(f"  {len(train)} rows, {len(ALL_FEATURES)} features")
    print(f"  Segments: {train['segment'].value_counts().to_dict()}")
    print(f"  Split: {train['split_flag'].value_counts().to_dict()}")
    print(f"\nScoring data...")
    score, labels = generate_scoring_data()
    print(f"  {len(score)} rows, {labels['is_anomaly'].sum()} anomalies")
    print(f"  {labels['anomaly_type'].value_counts().to_dict()}")
