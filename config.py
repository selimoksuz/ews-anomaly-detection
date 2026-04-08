"""EWS Anomaly Detection - Configuration."""

# --- Feature Definitions (5 category, 35 features) ---

PAYMENT_FEATURES = [
    "days_past_due",
    "worst_dpd_last_6m",
    "min_payment_only_count_6m",
    "payment_to_minimum_ratio",
    "avg_days_to_payment",
    "payment_amount_cv",
    "payment_reversal_count",
    "number_of_30dpd_last_12m",
]

UTILIZATION_FEATURES = [
    "utilization_ratio",
    "utilization_delta_3m",
    "credit_limit_change_pct",
    "overlimit_frequency_6m",
    "cash_advance_to_spending_ratio",
    "draw_ratio",
]

TRANSACTION_FEATURES = [
    "txn_count_monthly",
    "total_amount_monthly",
    "avg_txn_amount",
    "txn_count_ratio_vs_3m",
    "credit_turnover_trend",
    "debit_turnover_trend",
    "channel_diversity_score",
    "new_merchant_category_count",
]

BALANCE_FEATURES = [
    "outstanding_balance",
    "balance_to_income_ratio",
    "avg_checking_balance_3m",
    "min_checking_balance_3m",
    "deposit_volatility_6m",
    "nsf_event_count_12m",
]

TREND_FEATURES = [
    "util_ratio_slope_3m",
    "payment_ratio_slope_3m",
    "balance_change_pct_1m",
    "dpd_trend_3m",
    "spending_to_limit_ratio_delta",
    "amount_volatility_change",
    "avg_txn_amount_delta_3m",
]

ALL_FEATURES = (
    PAYMENT_FEATURES
    + UTILIZATION_FEATURES
    + TRANSACTION_FEATURES
    + BALANCE_FEATURES
    + TREND_FEATURES
)

assert len(ALL_FEATURES) == 35, f"Expected 35, got {len(ALL_FEATURES)}"

FEATURE_LABELS = {
    "days_past_due": "Gecikme Gunu",
    "worst_dpd_last_6m": "En Kotu Gecikme (6 ay)",
    "min_payment_only_count_6m": "Sadece Min. Odeme Sayisi (6 ay)",
    "payment_to_minimum_ratio": "Odeme / Minimum Odeme Orani",
    "avg_days_to_payment": "Ort. Odeme Suresi (gun)",
    "payment_amount_cv": "Odeme Tutari Degiskenlik Katsayisi",
    "payment_reversal_count": "Odeme Iade/Iptal Sayisi",
    "number_of_30dpd_last_12m": "30+ Gun Gecikme Sayisi (12 ay)",
    "utilization_ratio": "Limit Kullanim Orani",
    "utilization_delta_3m": "Limit Kullanim Degisimi (3 ay)",
    "credit_limit_change_pct": "Kredi Limiti Degisimi %",
    "overlimit_frequency_6m": "Limit Asim Sikligi (6 ay)",
    "cash_advance_to_spending_ratio": "Nakit Avans / Harcama Orani",
    "draw_ratio": "Kullanilmis / Tahsis Orani",
    "txn_count_monthly": "Aylik Islem Sayisi",
    "total_amount_monthly": "Aylik Islem Tutari",
    "avg_txn_amount": "Ort. Islem Tutari",
    "txn_count_ratio_vs_3m": "Islem Sayisi Degisimi (vs 3 ay)",
    "credit_turnover_trend": "Alacak Cirosu Trendi",
    "debit_turnover_trend": "Borc Cirosu Trendi",
    "channel_diversity_score": "Kanal Cesitlilik Skoru",
    "new_merchant_category_count": "Yeni Isyeri Kategori Sayisi",
    "outstanding_balance": "Mevcut Borc Bakiyesi",
    "balance_to_income_ratio": "Borc / Gelir Orani",
    "avg_checking_balance_3m": "Ort. Vadesiz Hesap Bakiyesi (3 ay)",
    "min_checking_balance_3m": "Min. Vadesiz Hesap Bakiyesi (3 ay)",
    "deposit_volatility_6m": "Mevduat Duzensizligi (6 ay)",
    "nsf_event_count_12m": "Karsilaksiz Islem Sayisi (12 ay)",
    "util_ratio_slope_3m": "Kullanim Orani Egimi (3 ay)",
    "payment_ratio_slope_3m": "Odeme Orani Egimi (3 ay)",
    "balance_change_pct_1m": "Bakiye Degisimi % (1 ay)",
    "dpd_trend_3m": "Gecikme Trendi (3 ay)",
    "spending_to_limit_ratio_delta": "Harcama/Limit Orani Degisimi",
    "amount_volatility_change": "Tutar Volatilite Degisimi",
    "avg_txn_amount_delta_3m": "Ort. Islem Tutari Degisimi (3 ay)",
}

# --- Model Parameters ---
AE_LATENT_DIM = 6
AE_HIDDEN_LAYERS = [24, 12]
AE_EPOCHS = 150
AE_LEARNING_RATE = 1e-3

ISO_N_ESTIMATORS = 200
ISO_CONTAMINATION = 0.05

# --- Ensemble Weights ---
WEIGHT_AE = 0.5
WEIGHT_IF = 0.3
WEIGHT_MD = 0.2

# --- Alert Bands ---
BAND_THRESHOLDS = {
    "NORMAL": (0, 60),
    "SARI": (60, 75),
    "TURUNCU": (75, 90),
    "KIRMIZI": (90, 100),
}
