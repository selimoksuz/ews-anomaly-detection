"""
EWS Anomaly Detection - Configuration.

Degisken katmanlari:
  Katman 1 — Anlik/Haftalik: Her hafta farkli deger uretir
  Katman 2 — Rolling 4 Hafta: Kisa pencere, haftalik kayar
  Katman 3 — Trend/Ivme: Degisimin yonu ve hizi
  Katman 4 — Risk Grubu Interaction: Is mantigi gruplari
"""

# ═══════════════════════════════════════════════════════════════
# KATMAN 1 — ANLIK / HAFTALIK (her hafta degisen degerler)
# ═══════════════════════════════════════════════════════════════

INSTANT_FEATURES = [
    "dpd_current",                  # Bu haftaki gecikme gun sayisi
    "utilization_ratio",            # Bu haftaki limit kullanim orani
    "outstanding_balance",          # Bu haftaki toplam borc bakiyesi
    "checking_balance",             # Bu haftaki vadesiz hesap bakiyesi
    "txn_count_weekly",             # Bu haftaki islem sayisi
    "txn_amount_weekly",            # Bu haftaki islem tutari
    "avg_txn_amount_weekly",        # Bu haftaki ort islem tutari
    "payment_amount_this_week",     # Bu hafta yapilan odeme tutari
]

# ═══════════════════════════════════════════════════════════════
# KATMAN 2 — ROLLING 4 HAFTA (kisa pencere, haftalik kayar)
# ═══════════════════════════════════════════════════════════════

ROLLING_4W_FEATURES = [
    "dpd_max_4w",                   # Son 4 haftadaki max gecikme
    "min_payment_only_count_4w",    # Son 4 haftada sadece min odeme yapilan hafta sayisi
    "payment_to_min_ratio_4w",      # Son 4 hafta ort odeme/min orani
    "avg_days_to_payment_4w",       # Son 4 hafta ort fatura-odeme arasi gun
    "payment_reversal_count_4w",    # Son 4 haftada odeme iade/iptal sayisi
    "nsf_count_4w",                 # Son 4 haftada karsilaksiz islem sayisi
    "overlimit_count_4w",           # Son 4 haftada limit asim sayisi
    "cash_advance_ratio_4w",        # Son 4 hafta nakit avans / toplam harcama
    "checking_balance_min_4w",      # Son 4 haftanin min vadesiz bakiyesi
    "deposit_amount_avg_4w",        # Son 4 hafta ort mevduat girisi
    "channel_count_4w",             # Son 4 haftada kullanilan farkli kanal sayisi
]

# ═══════════════════════════════════════════════════════════════
# KATMAN 3 — TREND / IVME (degisimin yonu ve hizi)
# ═══════════════════════════════════════════════════════════════

TREND_FEATURES = [
    "util_slope_4w",                # Kullanim orani 4 haftalik egim
    "balance_slope_4w",             # Borc bakiyesi 4 haftalik egim
    "checking_slope_4w",            # Vadesiz bakiye 4 haftalik egim
    "payment_ratio_slope_4w",       # Odeme/min orani 4 haftalik egim
    "txn_count_change_pct",         # Islem sayisi: bu hafta vs 4 hafta ort (%)
    "txn_amount_change_pct",        # Islem tutari: bu hafta vs 4 hafta ort (%)
    "deposit_change_pct",           # Mevduat: bu hafta vs 4 hafta ort (%)
    "util_acceleration",            # Kullanim ivmesi (slope degisimi)
    "dpd_direction_4w",             # Gecikme yonu: kac haftadir artiyor
]

# ═══════════════════════════════════════════════════════════════
# KATMAN 4 — RISK GRUBU INTERACTION
# ═══════════════════════════════════════════════════════════════

INTERACTION_FEATURES = [
    "liquidity_squeeze_score",      # util_high * checking_low * cash_adv_high
    "hidden_stress_score",          # dpd_low * min_pay_high * deposit_volatile
    "income_erosion_score",         # deposit_falling * balance_rising
    "payment_breakdown_score",      # days_to_pay_high * reversal_high * cv_high
]

# ═══════════════════════════════════════════════════════════════

ALL_FEATURES = (
    INSTANT_FEATURES
    + ROLLING_4W_FEATURES
    + TREND_FEATURES
    + INTERACTION_FEATURES
)

assert len(ALL_FEATURES) == 32, f"Expected 32, got {len(ALL_FEATURES)}"

# ═══════════════════════════════════════════════════════════════
# TURKCE ETIKETLER (human-readable ciktilar icin)
# ═══════════════════════════════════════════════════════════════

FEATURE_LABELS = {
    # Katman 1 — Anlik
    "dpd_current":                  "Gecikme Gunu (bu hafta)",
    "utilization_ratio":            "Limit Kullanim Orani",
    "outstanding_balance":          "Toplam Borc Bakiyesi",
    "checking_balance":             "Vadesiz Hesap Bakiyesi",
    "txn_count_weekly":             "Haftalik Islem Sayisi",
    "txn_amount_weekly":            "Haftalik Islem Tutari",
    "avg_txn_amount_weekly":        "Ort. Islem Tutari (haftalik)",
    "payment_amount_this_week":     "Bu Hafta Odeme Tutari",

    # Katman 2 — Rolling 4W
    "dpd_max_4w":                   "Max Gecikme (son 4 hafta)",
    "min_payment_only_count_4w":    "Sadece Min. Odeme Sayisi (4 hafta)",
    "payment_to_min_ratio_4w":      "Odeme / Min. Odeme Orani (4 hafta)",
    "avg_days_to_payment_4w":       "Ort. Odeme Suresi (4 hafta, gun)",
    "payment_reversal_count_4w":    "Odeme Iade/Iptal (4 hafta)",
    "nsf_count_4w":                 "Karsilaksiz Islem (4 hafta)",
    "overlimit_count_4w":           "Limit Asim Sayisi (4 hafta)",
    "cash_advance_ratio_4w":        "Nakit Avans Orani (4 hafta)",
    "checking_balance_min_4w":      "Min. Vadesiz Bakiye (4 hafta)",
    "deposit_amount_avg_4w":        "Ort. Mevduat Girisi (4 hafta)",
    "channel_count_4w":             "Kullanilan Kanal Sayisi (4 hafta)",

    # Katman 3 — Trend
    "util_slope_4w":                "Kullanim Orani Egimi (4 hafta)",
    "balance_slope_4w":             "Borc Bakiye Egimi (4 hafta)",
    "checking_slope_4w":            "Vadesiz Bakiye Egimi (4 hafta)",
    "payment_ratio_slope_4w":       "Odeme Orani Egimi (4 hafta)",
    "txn_count_change_pct":         "Islem Sayisi Degisimi %",
    "txn_amount_change_pct":        "Islem Tutari Degisimi %",
    "deposit_change_pct":           "Mevduat Degisimi %",
    "util_acceleration":            "Kullanim Ivmesi",
    "dpd_direction_4w":             "Gecikme Yonu (artan hafta sayisi)",

    # Katman 4 — Interaction
    "liquidity_squeeze_score":      "Likidite Sikismasi Skoru",
    "hidden_stress_score":          "Gizli Stres Skoru",
    "income_erosion_score":         "Gelir Erozyonu Skoru",
    "payment_breakdown_score":      "Odeme Kirilma Skoru",
}

# ═══════════════════════════════════════════════════════════════
# RISK GRUPLARI (human-readable grup anomali aciklamalari)
# ═══════════════════════════════════════════════════════════════

RISK_GROUPS = {
    "likidite_sikismasi": {
        "label": "Likidite Sikismasi",
        "interaction_feature": "liquidity_squeeze_score",
        "component_features": [
            "utilization_ratio", "checking_balance",
            "checking_balance_min_4w", "cash_advance_ratio_4w",
        ],
        "description": "Kredi kullanimi artarken nakit varliklari eriyor",
        "stress_direction": {
            "utilization_ratio": "high",
            "checking_balance": "low",
            "checking_balance_min_4w": "low",
            "cash_advance_ratio_4w": "high",
        },
    },
    "gizli_stres": {
        "label": "Gizli Stres",
        "interaction_feature": "hidden_stress_score",
        "component_features": [
            "dpd_current", "min_payment_only_count_4w",
            "payment_to_min_ratio_4w", "deposit_change_pct",
        ],
        "description": "Gecikme yok ama odeme davranisi bozulmus",
        "stress_direction": {
            "dpd_current": "low",
            "min_payment_only_count_4w": "high",
            "payment_to_min_ratio_4w": "low",
            "deposit_change_pct": "low",
        },
    },
    "gelir_erozyonu": {
        "label": "Gelir Erozyonu",
        "interaction_feature": "income_erosion_score",
        "component_features": [
            "deposit_change_pct", "deposit_amount_avg_4w",
            "balance_slope_4w", "checking_slope_4w",
        ],
        "description": "Gelen para azaliyor, borc buyuyor, bakiye eriyor",
        "stress_direction": {
            "deposit_change_pct": "low",
            "deposit_amount_avg_4w": "low",
            "balance_slope_4w": "high",
            "checking_slope_4w": "low",
        },
    },
    "odeme_kirilmasi": {
        "label": "Odeme Davranisi Kirilmasi",
        "interaction_feature": "payment_breakdown_score",
        "component_features": [
            "avg_days_to_payment_4w", "payment_reversal_count_4w",
            "txn_count_change_pct", "channel_count_4w",
        ],
        "description": "Odeme zamanlama, tutar ve kanali degismis",
        "stress_direction": {
            "avg_days_to_payment_4w": "high",
            "payment_reversal_count_4w": "high",
            "txn_count_change_pct": "low",
            "channel_count_4w": "low",
        },
    },
}

# ═══════════════════════════════════════════════════════════════
# MODEL PARAMETRELERI
# ═══════════════════════════════════════════════════════════════

AE_LATENT_DIM = 6
AE_HIDDEN_LAYERS = [24, 12]
AE_EPOCHS = 150
AE_LEARNING_RATE = 1e-3

ISO_N_ESTIMATORS = 200
ISO_CONTAMINATION = 0.05

WEIGHT_AE = 0.5
WEIGHT_IF = 0.3
WEIGHT_MD = 0.2

BAND_THRESHOLDS = {
    "NORMAL": (0, 60),
    "SARI": (60, 75),
    "TURUNCU": (75, 90),
    "KIRMIZI": (90, 100),
}
