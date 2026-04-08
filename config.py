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

ALL_FEATURES = (
    INSTANT_FEATURES
    + ROLLING_4W_FEATURES
    + TREND_FEATURES
)

assert len(ALL_FEATURES) == 28, f"Expected 28, got {len(ALL_FEATURES)}"

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

}

# Not: Eski config.py'deki model parametreleri ve risk gruplari artik
# config/pipeline_config.yaml'a tasindi. Bu dosya sadece geriye uyumluluk
# icin feature listesi ve label'lari tutuyor.
