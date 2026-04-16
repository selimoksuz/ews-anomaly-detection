# Faz 1 Variable Dictionary

Bu sozluk, `bilancolu + POS'u olan` Ticari Orta Faz 1 cohort'u icin sadece iki tablo uzerinden okunur:

1. `Native source dictionary`
   Oracle `native_features` tablosuna basilan ham/atomik kaynak alanlar
2. `Derived final long list`
   Oracle `input_features` tablosuna yazilan, modelin gordugu tum final kolonlar

Yani bu dosyada:
- tek ham tablo vardir
- tek final veri sozlugu tablosu vardir
- derived tarafta native'den aynen tasinan alanlar da, turetilen tum alanlar da ayni tabloda birlikte durur

Faz 1 aylik cadence ile calisir; snapshot'lar ay sonu birikir.

## Faz 1 Warm-Up Kurali

- Native tablo tum ay sonu snapshot'larini tutar.
- Derived/input tablo ise history feature'lar anlamli olustugu noktadan baslar.
- Faz 1'de `lag_12` kullanan yillik karsilastirma degiskenleri ve bunlarin history turevleri nedeniyle her musteri icin ilk `18` snapshot warm-up olarak disarida birakilir.
- Bu nedenle `derived row count`, `native row count` ile ayni olmak zorunda degildir; genellikle daha dusuktur.

## Ortak Donusum Notlari

- `annualization_factor(fs_period_code)`
  - `Q1 -> 4.0`
  - `Q2 -> 2.0`
  - `Q3 -> 4/3`
  - `Q4 -> 1.0`
  - `YE -> 1.0`
- `annualized_net_sales = fs_net_sales_cumulative * annualization_factor`
- `annualized_ebitda = fs_ebitda_cumulative * annualization_factor`
- `annualized_net_profit = fs_net_profit_cumulative * annualization_factor`
- `total_trade_receivables = fs_trade_receivables + fs_notes_receivable`
- `lag_12_pos_monthly_volume = lag_12(pos_monthly_volume)`
- `lag_12_annualized_net_sales = lag_12(annualized_net_sales)`
- `lag_12_equity = lag_12(fs_equity)`
- `__delta_1 = current_value - lag_1(current_value)`
- `__self_zscore_6 = (current_value - shift(1).rolling_mean_6) / shift(1).rolling_std_6`
- `__trend_slope_6 = trailing 6 snapshot lineer egim`
- `__population_percentile = ayni snapshot icinde percent rank`
- `__vs_population_median_delta = current_value - snapshot median`

## 1. Native Source Dictionary

- Oracle tablo anahtari: `native_features`
- Oracle fiziksel tablo: `EWS_TO_FAZ1_NATIVE`
- Grain: `customer_id + snapshot_date`

| Native Column | Tip | Aciklama | Missing / Outlier Notu |
|---|---|---|---|
| `customer_id` | identifier | Musteri anahtari | zorunlu |
| `snapshot_date` | date | Ay sonu snapshot tarihi | zorunlu |
| `segment` | text | Faz 1 cohort segment etiketi (`TICARI_ORTA_FAZ1`) | zorunlu |
| `is_balance_sheet_customer` | flag | Cohort filtreleme icin bilancolu musteri isareti | bu fazda `1` |
| `has_pos` | flag | Cohort filtreleme icin POS sahipligi | bu fazda `1` |
| `nace_section` | text | Sektor ust siniflamasi | nadir null yok |
| `nace_main` | text | Is kolu ana etiketi | nadir null yok |
| `fs_period_code` | text | Mali veri donemi (`Q1`, `Q2`, `Q3`, `Q4`, `YE`) | zorunlu |
| `fs_last_update_date` | date | Mali verinin son guncelleme tarihi | zorunlu |
| `memzuc_total_cash_risk_0_24m` | numeric | Memzuc 0-24 ay nakdi risk toplami | outlier enjekte edilir |
| `tlref_factor` | numeric | TLREF ile risk/EBITDA normalizasyon carpanidir | zorunlu |
| `fs_net_sales_cumulative` | numeric | Donemsel kumulatif net satis | zorunlu |
| `fs_ebitda_cumulative` | numeric | Donemsel kumulatif EBITDA | operasyonel missing enjekte edilir |
| `fs_trade_receivables` | numeric | Donemsel ticari alacak tutari | operasyonel missing ve outlier enjekte edilir |
| `fs_notes_receivable` | numeric | Donemsel alacak senetleri | zorunlu |
| `fs_net_profit_cumulative` | numeric | Donemsel kumulatif net kar | zorunlu |
| `fs_equity` | numeric | Ozkaynak seviyesi | zorunlu |
| `pos_monthly_volume` | numeric | Tum bankalar POS aylik hacmi | operasyonel missing ve outlier enjekte edilir |
| `ifrs9_behavioral_pd` | numeric | Davranissal temerrut olasiligi | operasyonel missing ve outlier enjekte edilir |
| `kkb_commercial_score` | numeric | KKB ticari kredi notu | operasyonel missing ve low-score outlier enjekte edilir |
| `kkb_indebtedness_index` | numeric | KKB ticari borcluluk endeksi | zorunlu |
| `memzuc_total_limit` | numeric | Memzuc toplam limit | operasyonel missing enjekte edilir |
| `memzuc_total_risk` | numeric | Memzuc toplam risk | zorunlu |

## 2. Derived Final Long List

- Oracle input tablo anahtari: `input_features`
- Oracle fiziksel tablo: `EWS_TO_FAZ1_INPUT`
- Grain: `customer_id + snapshot_date`

| Final Column | Column Type | Aciklama | Hesaplama / Donusum | Uretildigi Native / Base Alanlar | Business Basligi |
|---|---|---|---|---|---|
| `customer_id` | technical_pass_through | Input tablosu musteri anahtari | native `customer_id` aynen tasinir | `customer_id` | teknik metadata |
| `snapshot_date` | technical_pass_through | Input tablosu snapshot tarihi | native `snapshot_date` aynen tasinir | `snapshot_date` | teknik metadata |
| `segment` | technical_pass_through | Cohort etiketi | native `segment` aynen tasinir | `segment` | teknik metadata |
| `bank_debt_to_turnover` | derived_base | Nakit riskin yilliklastirilmis ciroya oranı | `memzuc_total_cash_risk_0_24m / annualized_net_sales` | `memzuc_total_cash_risk_0_24m`, `fs_net_sales_cumulative`, `fs_period_code` | `Banka Borclulugu ve Ciro Iliskisi Degiskeni` |
| `pos_volume_change` | derived_base | POS hacminde gecen yilin ayni ayina gore degisim | `(pos_monthly_volume - lag_12_pos_monthly_volume) / abs(lag_12_pos_monthly_volume)` | `pos_monthly_volume` | `Musterinin Tum Bankalardaki Aylik POS Hacmi Degiskeni` |
| `bank_debt_to_ebitda` | derived_base | TLREF ile normalize edilmis risk / EBITDA orani | `(memzuc_total_cash_risk_0_24m * tlref_factor) / annualized_ebitda` | `memzuc_total_cash_risk_0_24m`, `tlref_factor`, `fs_ebitda_cumulative`, `fs_period_code` | `Banka Borclulugu ve EBITDA Iliskisi Degiskeni` |
| `trade_receivables_to_turnover` | derived_base | Ticari alacak + alacak senetleri / ciro orani | `(fs_trade_receivables + fs_notes_receivable) / annualized_net_sales` | `fs_trade_receivables`, `fs_notes_receivable`, `fs_net_sales_cumulative`, `fs_period_code` | `Bilanco Ticari Alacak Bilgisi ve Ciro Bilgisi Degiskeni` |
| `profitability_to_turnover` | derived_base | Yilliklastirilmis net karin ciroya orani | `annualized_net_profit / annualized_net_sales` | `fs_net_profit_cumulative`, `fs_net_sales_cumulative`, `fs_period_code` | `Bilanco Karlilik Bilgisi ve Ciro Bilgisi Degiskeni` |
| `equity_change` | derived_base | Ozkaynagin gecen yilin ayni ayina gore degisimi | `(fs_equity - lag_12_equity) / abs(lag_12_equity)` | `fs_equity` | `Bilanco Ozkaynak Bilgisi Degiskeni` |
| `ifrs9_behavioral_pd` | passthrough_base | Davranissal PD authoritative skor | native `ifrs9_behavioral_pd` aynen tasinir | `ifrs9_behavioral_pd` | `TFRS Davranis Temerrut Olasiligi Degiskeni` |
| `kkb_commercial_score` | passthrough_base | KKB ticari kredi notu authoritative skor | native `kkb_commercial_score` aynen tasinir | `kkb_commercial_score` | `KKB Ticari Kredi Notu Degiskeni` |
| `kkb_indebtedness_index` | passthrough_base | KKB ticari borcluluk endeksi authoritative skor | native `kkb_indebtedness_index` aynen tasinir | `kkb_indebtedness_index` | `KKB Ticari Borcluluk Endeksi Degiskeni` |
| `net_sales_change` | derived_base | Yilliklastirilmis net satisin gecen yilin ayni ayina gore degisimi | `(annualized_net_sales - lag_12_annualized_net_sales) / abs(lag_12_annualized_net_sales)` | `fs_net_sales_cumulative`, `fs_period_code` | `Net Satis (Ciro) Degiskeni` |
| `memzuc_limit_utilization_increase` | derived_base | Toplam risk / toplam limit oranı | `memzuc_total_risk / memzuc_total_limit` | `memzuc_total_risk`, `memzuc_total_limit` | `Memzuc Limit Doluluk Artisi Degiskeni` |
| `bank_debt_to_turnover__delta_1` | self_history | Son aya gore mutlak fark | `bank_debt_to_turnover - lag_1(bank_debt_to_turnover)` | `bank_debt_to_turnover` | `Banka Borclulugu ve Ciro Iliskisi Degiskeni` |
| `bank_debt_to_turnover__self_zscore_6` | self_history | Son 6 gozleme gore self-z score | `zscore_6(bank_debt_to_turnover)` | `bank_debt_to_turnover` | `Banka Borclulugu ve Ciro Iliskisi Degiskeni` |
| `pos_volume_change__delta_1` | self_history | Son aya gore mutlak fark | `pos_volume_change - lag_1(pos_volume_change)` | `pos_volume_change` | `Musterinin Tum Bankalardaki Aylik POS Hacmi Degiskeni` |
| `pos_volume_change__self_zscore_6` | self_history | Son 6 gozleme gore self-z score | `zscore_6(pos_volume_change)` | `pos_volume_change` | `Musterinin Tum Bankalardaki Aylik POS Hacmi Degiskeni` |
| `bank_debt_to_ebitda__delta_1` | self_history | Son aya gore mutlak fark | `bank_debt_to_ebitda - lag_1(bank_debt_to_ebitda)` | `bank_debt_to_ebitda` | `Banka Borclulugu ve EBITDA Iliskisi Degiskeni` |
| `bank_debt_to_ebitda__self_zscore_6` | self_history | Son 6 gozleme gore self-z score | `zscore_6(bank_debt_to_ebitda)` | `bank_debt_to_ebitda` | `Banka Borclulugu ve EBITDA Iliskisi Degiskeni` |
| `trade_receivables_to_turnover__delta_1` | self_history | Son aya gore mutlak fark | `trade_receivables_to_turnover - lag_1(trade_receivables_to_turnover)` | `trade_receivables_to_turnover` | `Bilanco Ticari Alacak Bilgisi ve Ciro Bilgisi Degiskeni` |
| `trade_receivables_to_turnover__self_zscore_6` | self_history | Son 6 gozleme gore self-z score | `zscore_6(trade_receivables_to_turnover)` | `trade_receivables_to_turnover` | `Bilanco Ticari Alacak Bilgisi ve Ciro Bilgisi Degiskeni` |
| `profitability_to_turnover__delta_1` | self_history | Son aya gore mutlak fark | `profitability_to_turnover - lag_1(profitability_to_turnover)` | `profitability_to_turnover` | `Bilanco Karlilik Bilgisi ve Ciro Bilgisi Degiskeni` |
| `profitability_to_turnover__self_zscore_6` | self_history | Son 6 gozleme gore self-z score | `zscore_6(profitability_to_turnover)` | `profitability_to_turnover` | `Bilanco Karlilik Bilgisi ve Ciro Bilgisi Degiskeni` |
| `equity_change__delta_1` | self_history | Son aya gore mutlak fark | `equity_change - lag_1(equity_change)` | `equity_change` | `Bilanco Ozkaynak Bilgisi Degiskeni` |
| `equity_change__self_zscore_6` | self_history | Son 6 gozleme gore self-z score | `zscore_6(equity_change)` | `equity_change` | `Bilanco Ozkaynak Bilgisi Degiskeni` |
| `ifrs9_behavioral_pd__delta_1` | self_history | Son aya gore mutlak fark | `ifrs9_behavioral_pd - lag_1(ifrs9_behavioral_pd)` | `ifrs9_behavioral_pd` | `TFRS Davranis Temerrut Olasiligi Degiskeni` |
| `ifrs9_behavioral_pd__self_zscore_6` | self_history | Son 6 gozleme gore self-z score | `zscore_6(ifrs9_behavioral_pd)` | `ifrs9_behavioral_pd` | `TFRS Davranis Temerrut Olasiligi Degiskeni` |
| `kkb_commercial_score__delta_1` | self_history | Son aya gore mutlak fark | `kkb_commercial_score - lag_1(kkb_commercial_score)` | `kkb_commercial_score` | `KKB Ticari Kredi Notu Degiskeni` |
| `kkb_commercial_score__self_zscore_6` | self_history | Son 6 gozleme gore self-z score | `zscore_6(kkb_commercial_score)` | `kkb_commercial_score` | `KKB Ticari Kredi Notu Degiskeni` |
| `kkb_indebtedness_index__delta_1` | self_history | Son aya gore mutlak fark | `kkb_indebtedness_index - lag_1(kkb_indebtedness_index)` | `kkb_indebtedness_index` | `KKB Ticari Borcluluk Endeksi Degiskeni` |
| `kkb_indebtedness_index__self_zscore_6` | self_history | Son 6 gozleme gore self-z score | `zscore_6(kkb_indebtedness_index)` | `kkb_indebtedness_index` | `KKB Ticari Borcluluk Endeksi Degiskeni` |
| `net_sales_change__delta_1` | self_history | Son aya gore mutlak fark | `net_sales_change - lag_1(net_sales_change)` | `net_sales_change` | `Net Satis (Ciro) Degiskeni` |
| `net_sales_change__self_zscore_6` | self_history | Son 6 gozleme gore self-z score | `zscore_6(net_sales_change)` | `net_sales_change` | `Net Satis (Ciro) Degiskeni` |
| `memzuc_limit_utilization_increase__delta_1` | self_history | Son aya gore mutlak fark | `memzuc_limit_utilization_increase - lag_1(memzuc_limit_utilization_increase)` | `memzuc_limit_utilization_increase` | `Memzuc Limit Doluluk Artisi Degiskeni` |
| `memzuc_limit_utilization_increase__self_zscore_6` | self_history | Son 6 gozleme gore self-z score | `zscore_6(memzuc_limit_utilization_increase)` | `memzuc_limit_utilization_increase` | `Memzuc Limit Doluluk Artisi Degiskeni` |
| `pos_volume_change__trend_slope_6` | trend | Son 6 snapshot'taki lineer egim | `rolling_slope_6(pos_volume_change)` | `pos_volume_change` | `Musterinin Tum Bankalardaki Aylik POS Hacmi Degiskeni` |
| `net_sales_change__trend_slope_6` | trend | Son 6 snapshot'taki lineer egim | `rolling_slope_6(net_sales_change)` | `net_sales_change` | `Net Satis (Ciro) Degiskeni` |
| `pos_volume_change__population_percentile` | population_reference | Ayni snapshot icinde percent rank | `pct_rank(pos_volume_change)` | `pos_volume_change` | `Musterinin Tum Bankalardaki Aylik POS Hacmi Degiskeni` |
| `pos_volume_change__vs_population_median_delta` | population_reference | Snapshot medyanindan fark | `pos_volume_change - median(pos_volume_change)` | `pos_volume_change` | `Musterinin Tum Bankalardaki Aylik POS Hacmi Degiskeni` |
| `net_sales_change__population_percentile` | population_reference | Ayni snapshot icinde percent rank | `pct_rank(net_sales_change)` | `net_sales_change` | `Net Satis (Ciro) Degiskeni` |
| `net_sales_change__vs_population_median_delta` | population_reference | Snapshot medyanindan fark | `net_sales_change - median(net_sales_change)` | `net_sales_change` | `Net Satis (Ciro) Degiskeni` |

## Faz 1 Final Long List Ozeti

- Final input tablosunda toplam `42` kolon vardir.
- Bunlarin `39` tanesi modelin gordugu numeric feature'dir.
- Model gormeyen ama input tablosunda tasinan teknik kolonlar:
  - `customer_id`
  - `snapshot_date`
  - `segment`

Bu dosyada artik alt alta birden fazla derived alt-tablosu yoktur. `Derived Final Long List` tek tablodur ve finalde Oracle `input_features` tablosunda duran tum kolonlarin teknik kontratini birlikte verir.
