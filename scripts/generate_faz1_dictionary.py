from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "dictionary" / "faz1_variable_dictionary.md"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.config_loader import get_feature_list, load_config

NATIVE_META = {
    "customer_id": ("identifier", "Musteri anahtari", "zorunlu"),
    "snapshot_date": ("date", "Ay sonu snapshot tarihi", "zorunlu"),
    "segment": ("text", "Segment etiketi", "zorunlu"),
    "is_balance_sheet_customer": ("flag", "Bilancolu musteri isareti", "cohort filtresinde kullanilir"),
    "has_pos": ("flag", "Herhangi bir bankada POS varligi", "cohort filtresinde kullanilir"),
    "bank_total_risk": ("numeric", "Tum bankalardaki toplam risk", "cohort filtresinde `>= 1m`"),
    "nace_section": ("text", "Sektor ust siniflamasi", "destekleyici alan"),
    "nace_main": ("text", "Sektor ana etiketi", "destekleyici alan"),
    "fs_period_code": ("text", "Mali veri donem kodu", "annualization icin kullanilir"),
    "fs_last_update_date": ("date", "Mali verinin son guncelleme tarihi", "freshness kontrolunde kullanilir"),
    "memzuc_total_cash_risk_0_24m": ("numeric", "0-24 ay nakdi risk toplami", "missing/outlier bulunabilir"),
    "memzuc_business_loan_risk_0_24m": ("numeric", "0-24 ay isletme kredisi riski", "missing/outlier bulunabilir"),
    "tlref_factor": ("numeric", "TLREF normalizasyon carpani", "zorunlu"),
    "inflation_yoy_rate": ("numeric", "Yillik enflasyon orani", "dis referans seri"),
    "fs_net_sales_cumulative": ("numeric", "Donemsel kumulatif net satis", "annualize edilir"),
    "fs_ebitda_cumulative": ("numeric", "Donemsel kumulatif EBITDA", "missing/outlier bulunabilir"),
    "fs_trade_receivables": ("numeric", "Ticari alacak tutari", "missing/outlier bulunabilir"),
    "fs_notes_receivable": ("numeric", "Alacak senetleri tutari", "zorunlu"),
    "fs_net_profit_cumulative": ("numeric", "Donemsel kumulatif net kar", "annualize edilir"),
    "fs_equity": ("numeric", "Ozkaynak seviyesi", "missing olabilir"),
    "pos_monthly_volume": ("numeric", "Tum bankalar POS aylik hacmi", "missing/outlier bulunabilir"),
    "ifrs9_behavioral_pd": ("numeric", "Davranissal temerrut olasiligi", "missing/outlier bulunabilir"),
    "kkb_commercial_score": ("numeric", "KKB ticari kredi notu", "missing/outlier bulunabilir"),
    "kkb_indebtedness_index": ("numeric", "KKB ticari borcluluk endeksi", "zorunlu"),
    "memzuc_total_limit": ("numeric", "Memzuc toplam limit", "missing olabilir"),
    "memzuc_total_risk": ("numeric", "Memzuc toplam risk", "zorunlu"),
    "bank_asset_average_balance": ("numeric", "Bankamizdaki ortalama mevduat/varlik seviyesi", "missing/outlier bulunabilir"),
}

BASE_META = {
    "bank_debt_to_turnover": {
        "label": "Banka Borclulugu / Ciro",
        "business_heading": "Banka Borclulugu ve Ciro Iliskisi",
        "calculation": "memzuc_total_cash_risk_0_24m / annualized(fs_net_sales_cumulative)",
        "inputs": "memzuc_total_cash_risk_0_24m, fs_net_sales_cumulative, fs_period_code",
        "direction": "artmasi kotu, azalmasi iyi",
    },
    "pos_volume_change": {
        "label": "Tum Bankalar POS Hacmi Degisimi",
        "business_heading": "Musterinin Tum Bankalardaki Aylik POS Hacmi",
        "calculation": "(pos_monthly_volume - lag_12(pos_monthly_volume)) / abs(lag_12(pos_monthly_volume))",
        "inputs": "pos_monthly_volume",
        "direction": "azalmasi kotu, artmasi iyi",
    },
    "bank_debt_to_ebitda": {
        "label": "Banka Borclulugu / EBITDA",
        "business_heading": "Banka Borclulugu ve EBITDA Iliskisi",
        "calculation": "(memzuc_total_cash_risk_0_24m * tlref_factor) / annualized(fs_ebitda_cumulative)",
        "inputs": "memzuc_total_cash_risk_0_24m, tlref_factor, fs_ebitda_cumulative, fs_period_code",
        "direction": "artmasi kotu, azalmasi iyi",
    },
    "trade_receivables_to_turnover": {
        "label": "Ticari Alacak / Ciro",
        "business_heading": "Bilanco Ticari Alacak ve Ciro",
        "calculation": "(fs_trade_receivables + fs_notes_receivable) / annualized(fs_net_sales_cumulative)",
        "inputs": "fs_trade_receivables, fs_notes_receivable, fs_net_sales_cumulative, fs_period_code",
        "direction": "artmasi kotu, azalmasi iyi",
    },
    "profitability_to_turnover": {
        "label": "Karlilik / Ciro",
        "business_heading": "Bilanco Karlilik ve Ciro",
        "calculation": "annualized(fs_net_profit_cumulative) / annualized(fs_net_sales_cumulative)",
        "inputs": "fs_net_profit_cumulative, fs_net_sales_cumulative, fs_period_code",
        "direction": "azalmasi kotu, artmasi iyi",
    },
    "business_loan_vs_inflation": {
        "label": "Isletme Borcu / Enflasyon Farki",
        "business_heading": "Bankalardaki Isletme Borclari ve Enflasyon Iliskisi",
        "calculation": "yoy_pct_change(memzuc_business_loan_risk_0_24m) - inflation_yoy_rate",
        "inputs": "memzuc_business_loan_risk_0_24m, inflation_yoy_rate",
        "direction": "artmasi kotu, azalmasi iyi",
    },
    "equity_change": {
        "label": "Ozkaynak Degisimi",
        "business_heading": "Bilanco Ozkaynak Bilgisi",
        "calculation": "(fs_equity - lag_12(fs_equity)) / abs(lag_12(fs_equity))",
        "inputs": "fs_equity",
        "direction": "azalmasi kotu, artmasi iyi",
    },
    "ifrs9_behavioral_pd": {
        "label": "TFRS Davranis Temerrut Olasiligi",
        "business_heading": "TFRS Davranis Temerrut Olasiligi",
        "calculation": "native ifrs9_behavioral_pd",
        "inputs": "ifrs9_behavioral_pd",
        "direction": "artmasi kotu, azalmasi iyi",
    },
    "kkb_commercial_score": {
        "label": "KKB Ticari Kredi Notu",
        "business_heading": "KKB Ticari Kredi Notu",
        "calculation": "native kkb_commercial_score",
        "inputs": "kkb_commercial_score",
        "direction": "azalmasi kotu, artmasi iyi",
    },
    "kkb_indebtedness_index": {
        "label": "KKB Ticari Borcluluk Endeksi",
        "business_heading": "KKB Ticari Borcluluk Endeksi",
        "calculation": "native kkb_indebtedness_index",
        "inputs": "kkb_indebtedness_index",
        "direction": "artmasi kotu, azalmasi iyi",
    },
    "net_sales_change": {
        "label": "Net Satis Degisimi",
        "business_heading": "Net Satis (Ciro)",
        "calculation": "(annualized(fs_net_sales_cumulative) - lag_12(annualized(fs_net_sales_cumulative))) / abs(lag_12(annualized(fs_net_sales_cumulative)))",
        "inputs": "fs_net_sales_cumulative, fs_period_code",
        "direction": "azalmasi kotu, artmasi iyi",
    },
    "memzuc_limit_utilization_increase": {
        "label": "Memzuc Limit Doluluk Orani",
        "business_heading": "Memzuc Limit Doluluk Artisi",
        "calculation": "memzuc_total_risk / memzuc_total_limit",
        "inputs": "memzuc_total_risk, memzuc_total_limit",
        "direction": "artmasi kotu, azalmasi iyi",
    },
    "bank_asset_average_change": {
        "label": "Banka Varlik Ortalamasi Degisimi",
        "business_heading": "Bankamizda Bulunan Mevduat / Varlik Ortalamalari",
        "calculation": "(bank_asset_average_balance - lag_12(bank_asset_average_balance)) / abs(lag_12(bank_asset_average_balance))",
        "inputs": "bank_asset_average_balance",
        "direction": "azalmasi kotu, artmasi iyi",
    },
}


def derive_meta(feature_name: str) -> tuple[str, str, str, str, str, str]:
    if feature_name in {"customer_id", "snapshot_date", "segment"}:
        return (
            "technical_pass_through",
            "Teknik tasiyici kolon",
            f"native `{feature_name}` aynen tasinir",
            feature_name,
            "teknik metadata",
            "teknik metadata",
        )

    base_name = feature_name.split("__", 1)[0]
    base_meta = BASE_META[base_name]
    if feature_name == base_name:
        return (
            "base_feature",
            base_meta["label"],
            base_meta["calculation"],
            base_meta["inputs"],
            base_meta["business_heading"],
            base_meta["direction"],
        )
    if feature_name.endswith("__delta_1"):
        return (
            "self_history_delta",
            f"{base_meta['label']} Son Aya Gore Fark",
            f"{base_name} - lag_1({base_name})",
            base_name,
            base_meta["business_heading"],
            base_meta["direction"],
        )
    if feature_name.endswith("__self_zscore_6"):
        return (
            "self_history_zscore",
            f"{base_meta['label']} Self-Z(6)",
            f"({base_name} - shift(1).rolling_mean_6({base_name})) / shift(1).rolling_std_6({base_name})",
            base_name,
            base_meta["business_heading"],
            base_meta["direction"],
        )
    if feature_name.endswith("__trend_slope_6"):
        return (
            "trend_slope",
            f"{base_meta['label']} Trend Egimi(6)",
            f"rolling_slope_6({base_name})",
            base_name,
            base_meta["business_heading"],
            base_meta["direction"],
        )
    if feature_name.endswith("__population_percentile"):
        return (
            "population_reference",
            f"{base_meta['label']} Populasyon Percentile",
            f"pct_rank({base_name}) within snapshot",
            base_name,
            base_meta["business_heading"],
            base_meta["direction"],
        )
    if feature_name.endswith("__vs_population_median_delta"):
        return (
            "population_reference",
            f"{base_meta['label']} Snapshot Median Farki",
            f"{base_name} - snapshot_median({base_name})",
            base_name,
            base_meta["business_heading"],
            base_meta["direction"],
        )
    raise KeyError(feature_name)


def build_markdown() -> str:
    config = load_config()
    native_columns = list(NATIVE_META.keys())
    feature_names = get_feature_list(config)
    final_columns = ["customer_id", "snapshot_date", "segment", *feature_names]

    lines = [
        "# Faz 1 Variable Dictionary",
        "",
        "Bu sozluk yalnizca Ticari Orta Faz 1 kapsamini mapler.",
        "",
        "## Faz 1 Cohort",
        "",
        "- `segment = TICARI_ORTA`",
        "- `bank_total_risk >= 1_000_000`",
        "- `is_balance_sheet_customer = 1`",
        "- `has_pos = 1`",
        "",
        "## Donusum Kurallari",
        "",
        "- `Q1 -> *4`, `Q2 -> *2`, `Q3 -> *4/3`, `Q4/YE -> *1`",
        "- Yillik karsilastirmali degiskenler `lag_12` kullanir.",
        "- Self history turevleri: `__delta_1`, `__self_zscore_6`",
        "- Trend turevi: `__trend_slope_6`",
        "- Populasyon turevleri: `__population_percentile`, `__vs_population_median_delta`",
        "- Warm-up nedeniyle ilk `18` snapshot final derived tabloda tutulmaz.",
        "",
        "## 1. Native Source Dictionary",
        "",
        "- Oracle anahtari: `native_features`",
        "- Fiziksel tablo: `EWS_TO_FAZ1_NATIVE`",
        "- Grain: `customer_id + snapshot_date`",
        "",
        "| Native Column | Tip | Aciklama | Missing / Outlier Notu |",
        "|---|---|---|---|",
    ]
    for column in native_columns:
        meta = NATIVE_META[column]
        lines.append(f"| `{column}` | {meta[0]} | {meta[1]} | {meta[2]} |")

    lines.extend(
        [
            "",
            "## 2. Final Long List Dictionary",
            "",
            "- Oracle anahtari: `input_features`",
            "- Fiziksel tablo: `EWS_TO_FAZ1_INPUT`",
            "- Grain: `customer_id + snapshot_date`",
            "",
            "| Final Column | Column Type | Aciklama | Hesaplama / Donusum | Uretildigi Native / Base Alanlar | Business Basligi | Yon Kurali |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for column in final_columns:
        col_type, description, calculation, inputs, heading, direction = derive_meta(column)
        lines.append(
            f"| `{column}` | {col_type} | {description} | {calculation} | {inputs} | {heading} | {direction} |"
        )

    lines.extend(
        [
            "",
            "## Ozet",
            "",
            f"- Native source dictionary kolon sayisi: `{len(native_columns)}`",
            f"- Final long list toplam kolon sayisi: `{len(final_columns)}`",
            f"- Model feature sayisi: `{len(feature_names)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_markdown(), encoding="utf-8")


if __name__ == "__main__":
    main()
