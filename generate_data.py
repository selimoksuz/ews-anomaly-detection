"""
Synthetic data generator for EWS Anomaly Detection.

4 musteri segmenti ile gercekci veri uretir:
  - PREMIUM:  Yuksek gelir, dusuk risk, yuksek islem hacmi
  - STANDARD: Orta gelir, normal davranis
  - RISKY:    Dusuk gelir, yuksek kullanim, gecikme egilimi
  - NEW:      Yeni musteriler, dusuk islem gecmisi

Training data:  5000 musteri (TRAIN %80 / TEST %20 split)
Scoring data:   5000 musteri (bugunun verisi, anomaliler enjekte)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import ALL_FEATURES

N_CUSTOMERS = 5000
N_ANOMALY_A = 50   # univariate
N_ANOMALY_B = 80   # multivariate
N_ANOMALY_C = 70   # subtle drift
N_ANOMALIES = N_ANOMALY_A + N_ANOMALY_B + N_ANOMALY_C  # 200 = %4


# ═══════════════════════════════════════════════════════════════
# SEGMENT TABANLI PARAMETRE TANIMLARI
# ═══════════════════════════════════════════════════════════════

SEGMENT_PARAMS = {
    "PREMIUM": {
        # Yuksek gelirli, disiplinli, cok islem, dusuk kullanim
        "weight": 0.20,
        "params": {
            "days_past_due":                (0.0, 0.5),
            "worst_dpd_last_6m":            (0.0, 1.0),
            "min_payment_only_count_6m":    (0.0, 0.1),
            "payment_to_minimum_ratio":     (8.0, 3.0),     # minimumun 8 kati oder
            "avg_days_to_payment":          (5.0, 2.0),      # hemen oder
            "payment_amount_cv":            (0.15, 0.08),    # cok duzgun
            "payment_reversal_count":       (0.0, 0.05),
            "number_of_30dpd_last_12m":     (0.0, 0.05),
            "utilization_ratio":            (0.25, 0.12),    # dusuk kullanim
            "utilization_delta_3m":         (0.0, 0.03),
            "credit_limit_change_pct":      (0.02, 0.03),    # limit artisi alir
            "overlimit_frequency_6m":       (0.0, 0.05),
            "cash_advance_to_spending_ratio": (0.01, 0.02),  # nakit avans kullanmaz
            "draw_ratio":                   (0.20, 0.10),
            "txn_count_monthly":            (45.0, 15.0),    # cok islem
            "total_amount_monthly":         (35000.0, 15000.0),
            "avg_txn_amount":               (800.0, 400.0),
            "txn_count_ratio_vs_3m":        (1.0, 0.10),     # stabil
            "credit_turnover_trend":        (0.02, 0.03),    # gelir artiyor
            "debit_turnover_trend":         (0.01, 0.03),
            "channel_diversity_score":      (4.5, 0.8),      # cok kanal
            "new_merchant_category_count":  (2.0, 1.5),
            "outstanding_balance":          (15000.0, 10000.0),
            "balance_to_income_ratio":      (0.15, 0.08),    # dusuk kaldirac
            "avg_checking_balance_3m":      (25000.0, 12000.0),  # yuksek likidite
            "min_checking_balance_3m":      (12000.0, 8000.0),
            "deposit_volatility_6m":        (0.10, 0.05),    # duzgun gelir
            "nsf_event_count_12m":          (0.0, 0.02),
            "util_ratio_slope_3m":          (0.0, 0.01),
            "payment_ratio_slope_3m":       (0.01, 0.02),    # odeme orani artiyor
            "balance_change_pct_1m":        (0.0, 0.05),
            "dpd_trend_3m":                 (0.0, 0.1),
            "spending_to_limit_ratio_delta": (0.0, 0.02),
            "amount_volatility_change":     (0.0, 0.05),
            "avg_txn_amount_delta_3m":      (0.0, 0.05),
        },
    },
    "STANDARD": {
        # Orta gelir, normal davranis, ara sira kucuk gecikmeler
        "weight": 0.45,
        "params": {
            "days_past_due":                (2.0, 3.0),
            "worst_dpd_last_6m":            (5.0, 5.0),
            "min_payment_only_count_6m":    (0.5, 0.7),
            "payment_to_minimum_ratio":     (3.5, 1.5),
            "avg_days_to_payment":          (12.0, 4.0),
            "payment_amount_cv":            (0.30, 0.12),
            "payment_reversal_count":       (0.1, 0.3),
            "number_of_30dpd_last_12m":     (0.2, 0.4),
            "utilization_ratio":            (0.40, 0.15),
            "utilization_delta_3m":         (0.0, 0.04),
            "credit_limit_change_pct":      (0.0, 0.02),
            "overlimit_frequency_6m":       (0.1, 0.2),
            "cash_advance_to_spending_ratio": (0.04, 0.04),
            "draw_ratio":                   (0.35, 0.12),
            "txn_count_monthly":            (22.0, 10.0),
            "total_amount_monthly":         (12000.0, 6000.0),
            "avg_txn_amount":               (550.0, 250.0),
            "txn_count_ratio_vs_3m":        (1.0, 0.12),
            "credit_turnover_trend":        (0.0, 0.04),
            "debit_turnover_trend":         (0.0, 0.04),
            "channel_diversity_score":      (3.0, 1.0),
            "new_merchant_category_count":  (1.0, 1.2),
            "outstanding_balance":          (22000.0, 12000.0),
            "balance_to_income_ratio":      (0.35, 0.12),
            "avg_checking_balance_3m":      (6000.0, 3500.0),
            "min_checking_balance_3m":      (2000.0, 1800.0),
            "deposit_volatility_6m":        (0.22, 0.10),
            "nsf_event_count_12m":          (0.1, 0.2),
            "util_ratio_slope_3m":          (0.0, 0.02),
            "payment_ratio_slope_3m":       (0.0, 0.03),
            "balance_change_pct_1m":        (0.0, 0.06),
            "dpd_trend_3m":                 (0.0, 0.4),
            "spending_to_limit_ratio_delta": (0.0, 0.02),
            "amount_volatility_change":     (0.0, 0.08),
            "avg_txn_amount_delta_3m":      (0.0, 0.06),
        },
    },
    "RISKY": {
        # Dusuk gelir, yuksek kullanim, gecikme egilimi, nakit avans kullanan
        "weight": 0.25,
        "params": {
            "days_past_due":                (8.0, 7.0),
            "worst_dpd_last_6m":            (18.0, 12.0),
            "min_payment_only_count_6m":    (2.0, 1.5),      # sik minimum odeme
            "payment_to_minimum_ratio":     (1.5, 0.8),      # minimuma yakin
            "avg_days_to_payment":          (20.0, 6.0),      # gec oder
            "payment_amount_cv":            (0.50, 0.20),     # duzensiz
            "payment_reversal_count":       (0.5, 0.6),
            "number_of_30dpd_last_12m":     (1.0, 1.0),
            "utilization_ratio":            (0.70, 0.15),     # yuksek kullanim
            "utilization_delta_3m":         (0.03, 0.06),     # artiyor
            "credit_limit_change_pct":      (-0.02, 0.03),    # limit kesilebilir
            "overlimit_frequency_6m":       (0.5, 0.6),
            "cash_advance_to_spending_ratio": (0.15, 0.10),   # nakit avans kullaniyor
            "draw_ratio":                   (0.60, 0.15),
            "txn_count_monthly":            (15.0, 8.0),
            "total_amount_monthly":         (8000.0, 4000.0),
            "avg_txn_amount":               (550.0, 300.0),
            "txn_count_ratio_vs_3m":        (0.95, 0.18),     # hafif dusus
            "credit_turnover_trend":        (-0.02, 0.05),    # gelir azaliyor
            "debit_turnover_trend":         (0.01, 0.05),     # harcama artiyor
            "channel_diversity_score":      (2.0, 0.8),
            "new_merchant_category_count":  (0.5, 0.8),
            "outstanding_balance":          (40000.0, 18000.0),  # yuksek borc
            "balance_to_income_ratio":      (0.55, 0.15),     # yuksek kaldirac
            "avg_checking_balance_3m":      (2000.0, 1500.0), # dusuk likidite
            "min_checking_balance_3m":      (300.0, 400.0),   # cok dusuk dip
            "deposit_volatility_6m":        (0.40, 0.15),     # duzensiz gelir
            "nsf_event_count_12m":          (0.5, 0.6),
            "util_ratio_slope_3m":          (0.02, 0.03),     # artiyor
            "payment_ratio_slope_3m":       (-0.02, 0.03),    # odemeler azaliyor
            "balance_change_pct_1m":        (0.05, 0.10),     # borc artiyor
            "dpd_trend_3m":                 (0.5, 0.8),
            "spending_to_limit_ratio_delta": (0.02, 0.03),
            "amount_volatility_change":     (0.05, 0.12),
            "avg_txn_amount_delta_3m":      (0.0, 0.10),
        },
    },
    "NEW": {
        # Yeni musteri, dusuk islem gecmisi, dusuk bakiye
        "weight": 0.10,
        "params": {
            "days_past_due":                (0.5, 1.5),
            "worst_dpd_last_6m":            (1.0, 2.5),
            "min_payment_only_count_6m":    (0.2, 0.4),
            "payment_to_minimum_ratio":     (4.0, 2.0),
            "avg_days_to_payment":          (10.0, 5.0),
            "payment_amount_cv":            (0.35, 0.18),
            "payment_reversal_count":       (0.1, 0.2),
            "number_of_30dpd_last_12m":     (0.1, 0.2),
            "utilization_ratio":            (0.30, 0.18),
            "utilization_delta_3m":         (0.02, 0.05),     # yeni, kesfediyor
            "credit_limit_change_pct":      (0.0, 0.01),
            "overlimit_frequency_6m":       (0.0, 0.1),
            "cash_advance_to_spending_ratio": (0.02, 0.03),
            "draw_ratio":                   (0.25, 0.15),
            "txn_count_monthly":            (10.0, 6.0),      # az islem
            "total_amount_monthly":         (5000.0, 3000.0),
            "avg_txn_amount":               (500.0, 300.0),
            "txn_count_ratio_vs_3m":        (1.05, 0.20),     # artiyor
            "credit_turnover_trend":        (0.01, 0.04),
            "debit_turnover_trend":         (0.01, 0.04),
            "channel_diversity_score":      (2.0, 1.0),
            "new_merchant_category_count":  (2.0, 2.0),       # yeni yerler deniyor
            "outstanding_balance":          (8000.0, 5000.0),
            "balance_to_income_ratio":      (0.20, 0.10),
            "avg_checking_balance_3m":      (4000.0, 2500.0),
            "min_checking_balance_3m":      (1500.0, 1200.0),
            "deposit_volatility_6m":        (0.30, 0.15),
            "nsf_event_count_12m":          (0.0, 0.1),
            "util_ratio_slope_3m":          (0.01, 0.02),
            "payment_ratio_slope_3m":       (0.0, 0.02),
            "balance_change_pct_1m":        (0.02, 0.07),
            "dpd_trend_3m":                 (0.0, 0.3),
            "spending_to_limit_ratio_delta": (0.01, 0.03),
            "amount_volatility_change":     (0.0, 0.10),
            "avg_txn_amount_delta_3m":      (0.02, 0.08),
        },
    },
}


# ═══════════════════════════════════════════════════════════════
# KORELASYON YAPISI
# ═══════════════════════════════════════════════════════════════

CORRELATION_PAIRS = [
    # Odeme davranisi ic korelasyonlar
    ("days_past_due", "worst_dpd_last_6m", 0.80),
    ("days_past_due", "min_payment_only_count_6m", 0.50),
    ("days_past_due", "number_of_30dpd_last_12m", 0.60),
    ("days_past_due", "dpd_trend_3m", 0.40),
    ("payment_to_minimum_ratio", "days_past_due", -0.40),
    ("payment_to_minimum_ratio", "min_payment_only_count_6m", -0.55),
    ("payment_reversal_count", "nsf_event_count_12m", 0.45),
    ("avg_days_to_payment", "days_past_due", 0.35),

    # Kullanim / limit ic korelasyonlar
    ("utilization_ratio", "draw_ratio", 0.70),
    ("utilization_ratio", "outstanding_balance", 0.55),
    ("utilization_ratio", "overlimit_frequency_6m", 0.40),
    ("utilization_ratio", "cash_advance_to_spending_ratio", 0.30),
    ("utilization_delta_3m", "util_ratio_slope_3m", 0.80),

    # Islem ic korelasyonlar
    ("total_amount_monthly", "txn_count_monthly", 0.60),
    ("total_amount_monthly", "avg_txn_amount", 0.40),
    ("credit_turnover_trend", "debit_turnover_trend", 0.35),

    # Bakiye ic korelasyonlar
    ("avg_checking_balance_3m", "min_checking_balance_3m", 0.70),
    ("outstanding_balance", "balance_to_income_ratio", 0.50),
    ("nsf_event_count_12m", "min_checking_balance_3m", -0.35),

    # Capraz korelasyonlar (en onemli — multivariate anomali bunlardan cikar)
    ("utilization_ratio", "txn_count_monthly", 0.30),
    ("avg_checking_balance_3m", "balance_to_income_ratio", -0.30),
    ("cash_advance_to_spending_ratio", "min_checking_balance_3m", -0.25),
    ("min_payment_only_count_6m", "deposit_volatility_6m", 0.30),
    ("payment_to_minimum_ratio", "avg_checking_balance_3m", 0.25),
]


def _build_corr(n_features):
    rng = np.random.RandomState(42)
    A = rng.randn(n_features, n_features) * 0.03
    cov = A @ A.T + np.eye(n_features)

    idx = {f: i for i, f in enumerate(ALL_FEATURES)}
    for f1, f2, c in CORRELATION_PAIRS:
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
        "new_merchant_category_count", "channel_diversity_score",
    ]
    for f in non_neg:
        data[:, idx[f]] = np.clip(data[:, idx[f]], 0, None)

    # Ratio'lar 0-1 arasi
    for f in ["utilization_ratio", "draw_ratio", "cash_advance_to_spending_ratio"]:
        data[:, idx[f]] = np.clip(data[:, idx[f]], 0, 1)

    data[:, idx["payment_to_minimum_ratio"]] = np.clip(data[:, idx["payment_to_minimum_ratio"]], 0.5, 20)
    data[:, idx["channel_diversity_score"]] = np.clip(data[:, idx["channel_diversity_score"]], 1, 6)
    data[:, idx["avg_days_to_payment"]] = np.clip(data[:, idx["avg_days_to_payment"]], 1, 40)
    data[:, idx["payment_amount_cv"]] = np.clip(data[:, idx["payment_amount_cv"]], 0.02, 2.0)
    data[:, idx["deposit_volatility_6m"]] = np.clip(data[:, idx["deposit_volatility_6m"]], 0.02, 1.0)

    # worst_dpd >= current dpd
    data[:, idx["worst_dpd_last_6m"]] = np.maximum(
        data[:, idx["worst_dpd_last_6m"]], data[:, idx["days_past_due"]]
    )
    # min_balance <= avg_balance
    data[:, idx["min_checking_balance_3m"]] = np.minimum(
        data[:, idx["min_checking_balance_3m"]], data[:, idx["avg_checking_balance_3m"]]
    )
    # avg_txn_amount tutarlilik: total / count yakininda olmali
    safe_txn = np.clip(data[:, idx["txn_count_monthly"]], 1, None)
    implied_avg = data[:, idx["total_amount_monthly"]] / safe_txn
    data[:, idx["avg_txn_amount"]] = 0.6 * data[:, idx["avg_txn_amount"]] + 0.4 * implied_avg

    # Tamsayi degiskenler
    counts = [
        "days_past_due", "worst_dpd_last_6m", "min_payment_only_count_6m",
        "payment_reversal_count", "number_of_30dpd_last_12m",
        "overlimit_frequency_6m", "txn_count_monthly",
        "nsf_event_count_12m", "new_merchant_category_count",
    ]
    for f in counts:
        data[:, idx[f]] = np.round(data[:, idx[f]])

    return data


# ═══════════════════════════════════════════════════════════════
# SEGMENT BAZLI VERI URETIMI
# ═══════════════════════════════════════════════════════════════

def _generate_segment_data(n, rng):
    """Segment bazli musteri verisi uret."""
    corr = _build_corr(len(ALL_FEATURES))
    all_data = []
    all_segments = []

    for seg_name, seg_info in SEGMENT_PARAMS.items():
        seg_n = int(n * seg_info["weight"])
        params = seg_info["params"]

        means = np.array([params[f][0] for f in ALL_FEATURES])
        stds = np.array([params[f][1] for f in ALL_FEATURES])

        raw = rng.multivariate_normal(np.zeros(len(ALL_FEATURES)), corr, size=seg_n)
        data = raw * stds + means
        all_data.append(data)
        all_segments.extend([seg_name] * seg_n)

    # Eksik kalanlari STANDARD ile doldur
    remaining = n - len(all_segments)
    if remaining > 0:
        params = SEGMENT_PARAMS["STANDARD"]["params"]
        means = np.array([params[f][0] for f in ALL_FEATURES])
        stds = np.array([params[f][1] for f in ALL_FEATURES])
        raw = rng.multivariate_normal(np.zeros(len(ALL_FEATURES)), corr, size=remaining)
        all_data.append(raw * stds + means)
        all_segments.extend(["STANDARD"] * remaining)

    data = np.vstack(all_data)

    # Shuffle
    order = rng.permutation(len(data))
    data = data[order]
    segments = [all_segments[i] for i in order]

    data = _constrain(data)
    return data, segments


# ═══════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════

def generate_normal_data(n=N_CUSTOMERS, seed=42):
    """Eski API uyumlulugu icin — segmentsiz normal veri."""
    rng = np.random.RandomState(seed)
    data, segments = _generate_segment_data(n, rng)
    df = pd.DataFrame(data, columns=ALL_FEATURES)
    df.insert(0, "customer_id", [f"CUST_{i:05d}" for i in range(n)])
    return df


def generate_training_data(n=N_CUSTOMERS, seed=42):
    """
    Training data: %80 TRAIN, %20 TEST (holdout).
    snapshot_date: son 6 aydan rastgele tarihler.
    """
    rng = np.random.RandomState(seed)
    data, segments = _generate_segment_data(n, rng)

    df = pd.DataFrame(data, columns=ALL_FEATURES)
    df.insert(0, "customer_id", [f"CUST_{i:05d}" for i in range(n)])
    df["segment"] = segments

    # Train/test split
    test_idx = rng.choice(n, int(n * 0.2), replace=False)
    df["split_flag"] = "TRAIN"
    df.loc[test_idx, "split_flag"] = "TEST"

    # Snapshot dates (son 6 ay icinden)
    base = datetime(2026, 4, 7)
    dates = [base - timedelta(days=int(rng.randint(7, 180))) for _ in range(n)]
    df["snapshot_date"] = dates

    return df


def generate_scoring_data(n=N_CUSTOMERS, seed=99):
    """
    Bugunun verisi (skorlama icin). %4 anomali enjekte edilmis.
    """
    rng = np.random.RandomState(seed)
    data, segments = _generate_segment_data(n, rng)
    idx = {f: i for i, f in enumerate(ALL_FEATURES)}

    # Anomali enjeksiyonu
    anom_idx = rng.choice(n, N_ANOMALIES, replace=False)
    a_idx = anom_idx[:N_ANOMALY_A]
    b_idx = anom_idx[N_ANOMALY_A:N_ANOMALY_A + N_ANOMALY_B]
    c_idx = anom_idx[N_ANOMALY_A + N_ANOMALY_B:]

    # ── Type A: Tek degisken asiri sapma ──
    for i in a_idx:
        scenario = rng.choice([
            "extreme_dpd", "extreme_util", "extreme_balance",
            "extreme_nsf", "extreme_reversal"
        ])
        if scenario == "extreme_dpd":
            data[i, idx["days_past_due"]] = rng.randint(25, 60)
            data[i, idx["worst_dpd_last_6m"]] = data[i, idx["days_past_due"]] + rng.randint(0, 10)
        elif scenario == "extreme_util":
            data[i, idx["utilization_ratio"]] = rng.uniform(0.95, 1.0)
            data[i, idx["draw_ratio"]] = rng.uniform(0.90, 1.0)
        elif scenario == "extreme_balance":
            data[i, idx["outstanding_balance"]] = rng.uniform(80000, 150000)
            data[i, idx["balance_to_income_ratio"]] = rng.uniform(0.80, 1.20)
        elif scenario == "extreme_nsf":
            data[i, idx["nsf_event_count_12m"]] = rng.randint(3, 8)
            data[i, idx["payment_reversal_count"]] = rng.randint(2, 5)
        else:
            data[i, idx["payment_reversal_count"]] = rng.randint(3, 7)

    # ── Type B: Korelasyon yapisi bozulmus (multivariate) ──
    for i in b_idx:
        scenario = rng.choice([
            "high_util_low_txn",      # limit dolmus ama islem yok
            "hidden_stress",           # DPD=0 ama her sey stres
            "payment_pattern_break",   # odeme aliskanligi kirilmis
            "income_erosion",          # gelir eriyor ama borc artiyor
        ])

        if scenario == "high_util_low_txn":
            # Normalde util ve txn pozitif korelasyon gosterir
            # Burada util cok yuksek ama islem neredeyse yok
            data[i, idx["utilization_ratio"]] = rng.uniform(0.85, 0.98)
            data[i, idx["draw_ratio"]] = rng.uniform(0.80, 0.95)
            data[i, idx["txn_count_monthly"]] = rng.randint(1, 4)
            data[i, idx["total_amount_monthly"]] = rng.uniform(300, 1500)
            data[i, idx["util_ratio_slope_3m"]] = rng.uniform(0.04, 0.12)
            data[i, idx["utilization_delta_3m"]] = rng.uniform(0.10, 0.25)

        elif scenario == "hidden_stress":
            # DPD sifir — gecikme yok gibi gorunuyor
            # Ama: sadece minimum odeme, nakit avans yuksek, bakiye dusuk
            data[i, idx["days_past_due"]] = 0
            data[i, idx["worst_dpd_last_6m"]] = rng.randint(0, 3)
            data[i, idx["min_payment_only_count_6m"]] = rng.randint(3, 6)
            data[i, idx["payment_to_minimum_ratio"]] = rng.uniform(1.0, 1.15)
            data[i, idx["cash_advance_to_spending_ratio"]] = rng.uniform(0.25, 0.55)
            data[i, idx["avg_checking_balance_3m"]] = rng.uniform(150, 600)
            data[i, idx["min_checking_balance_3m"]] = rng.uniform(10, 100)
            data[i, idx["deposit_volatility_6m"]] = rng.uniform(0.55, 0.85)

        elif scenario == "payment_pattern_break":
            # Odeme suresi, tutari, kanali — hepsi degismis
            data[i, idx["avg_days_to_payment"]] = rng.uniform(26, 38)
            data[i, idx["payment_amount_cv"]] = rng.uniform(0.80, 1.40)
            data[i, idx["payment_reversal_count"]] = rng.randint(2, 4)
            data[i, idx["channel_diversity_score"]] = rng.uniform(1.0, 1.5)
            data[i, idx["new_merchant_category_count"]] = rng.randint(5, 9)
            data[i, idx["payment_ratio_slope_3m"]] = rng.uniform(-0.08, -0.04)

        elif scenario == "income_erosion":
            # Gelir azaliyor ama harcama/borc artiyor
            data[i, idx["credit_turnover_trend"]] = rng.uniform(-0.10, -0.05)
            data[i, idx["debit_turnover_trend"]] = rng.uniform(0.03, 0.08)
            data[i, idx["balance_change_pct_1m"]] = rng.uniform(0.15, 0.35)
            data[i, idx["avg_checking_balance_3m"]] = rng.uniform(500, 1500)
            data[i, idx["deposit_volatility_6m"]] = rng.uniform(0.50, 0.80)
            data[i, idx["balance_to_income_ratio"]] = rng.uniform(0.65, 0.95)

    # ── Type C: Yaygin subtle drift (8-12 feature birden kayiyor) ──
    stress_decrease = {
        "payment_to_minimum_ratio", "avg_checking_balance_3m",
        "min_checking_balance_3m", "txn_count_monthly",
        "txn_count_ratio_vs_3m", "channel_diversity_score",
        "credit_turnover_trend",
    }
    for i in c_idx:
        seg_params = SEGMENT_PARAMS.get(segments[i], SEGMENT_PARAMS["STANDARD"])["params"]
        n_drift = rng.randint(8, 13)
        feats = rng.choice(len(ALL_FEATURES), n_drift, replace=False)
        for j in feats:
            feat_name = ALL_FEATURES[j]
            std = seg_params[feat_name][1]
            direction = -1 if feat_name in stress_decrease else 1
            data[i, j] += direction * std * rng.uniform(1.8, 3.0)

    data = _constrain(data)

    df = pd.DataFrame(data, columns=ALL_FEATURES)
    df.insert(0, "customer_id", [f"CUST_{i:05d}" for i in range(n)])
    df["segment"] = segments
    df["snapshot_date"] = datetime(2026, 4, 8)

    # Anomali labels (sadece validation icin — Oracle'a yazilmaz)
    labels = pd.DataFrame({
        "customer_id": df["customer_id"],
        "is_anomaly": False,
        "anomaly_type": "NORMAL",
    })
    labels.loc[a_idx, ["is_anomaly", "anomaly_type"]] = [True, "A_UNIVARIATE"]
    labels.loc[b_idx, ["is_anomaly", "anomaly_type"]] = [True, "B_MULTIVARIATE"]
    labels.loc[c_idx, ["is_anomaly", "anomaly_type"]] = [True, "C_SUBTLE_DRIFT"]

    return df, labels


# Eski API uyumlulugu
def generate_inference_data(n=N_CUSTOMERS):
    df, labels = generate_scoring_data(n)
    return df, labels


if __name__ == "__main__":
    print("Training data...")
    train = generate_training_data()
    print(f"  {len(train)} rows, segments: {train['segment'].value_counts().to_dict()}")
    print(f"  split: {train['split_flag'].value_counts().to_dict()}")
    print(f"\nScoring data...")
    score, labels = generate_scoring_data()
    print(f"  {len(score)} rows, {labels['is_anomaly'].sum()} anomalies")
    print(f"  {labels['anomaly_type'].value_counts().to_dict()}")
