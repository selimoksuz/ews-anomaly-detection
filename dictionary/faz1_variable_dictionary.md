# Faz 1 Variable Dictionary

Bu sozluk, `bilancolu + POS'u olan` Ticari Orta Faz 1 cohort'u icin hem Oracle native tabloya basilan atomik alanlari hem de bu native tablodan turetilip Oracle `input_features` tablosuna yazilan generated/model input alanlarini birlikte tanimlar. Faz 1 arti̧k haftalik degil, ay sonu snapshot'lariyla aylik cadence uzerinden calisir.

## Faz 1 Warm-Up Kurali

- Native tablo tum ay sonu snapshot'larini tutar.
- Derived/input tablo ise history feature'lar anlamli olustugu noktadan baslar.
- Faz 1'de yillik karsilastirma yapan degiskenler icin `lag_12`, bunlarin `self_zscore_6` turevleri icin de 6 ek gecmis gerektigi icin her musteri icin ilk `18` snapshot warm-up olarak disarida birakilir.
- Bu nedenle `derived row count`, `native row count` ile ayni olmak zorunda degildir; genellikle daha dusuktur.

## 1. Native Katman

### Tablo

- Oracle tablo anahtari: `native_features`
- Oracle fiziksel tablo: `EWS_TO_FAZ1_NATIVE`
- Grain: `customer_id + snapshot_date`

### Native Alanlar

| Native Column | Tip | Aciklama | Missing / Outlier Notu |
|---|---|---|---|
| `customer_id` | identifier | Musteri anahtari | zorunlu |
| `snapshot_date` | date | Ay sonu snapshot tarihi | zorunlu |
| `segment` | text | Faz 1 cohort segment etiketi (`TICARI_ORTA_FAZ1`) | zorunlu |
| `is_balance_sheet_customer` | flag | Cohort filtreleme icin bilancolu musteri isareti | bu fazda `1` |
| `has_pos` | flag | Cohort filtreleme icin POS sahipligi | bu fazda `1` |
| `nace_section` | text | Sektor ust siniflamasi | nadir null yok |
| `nace_main` | text | Is kolu ana etiketi | nadir null yok |
| `fs_period_code` | text | Mali veri donemi (`Q1`, `Q2`, `Q3`, `YE`) | zorunlu |
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

### Native Veri Uretim Notu

- Faz 1 native veri sentetiktir ama business semantigine uygun atomik alan yapisini taklit eder.
- Missing degerler native seviyede enjekte edilir; generated feature tablosunda bu missing'ler korunur ve pipeline preprocessing katmaninda ele alinir.
- Outlier'lar native seviyede enjekte edilir; winsorization ve robust scaler davranisini test etmek icin kullanilir.

## 2. Derived Katman

### Tablo

- Oracle input tablo anahtari: `input_features`
- Oracle fiziksel tablo: `EWS_TO_FAZ1_INPUT`
- Grain: `customer_id + snapshot_date`

### 2.1 Base Business Features

| Derived Column | Kategori | Hesap Mantigi |
|---|---|---|
| `bank_debt_to_turnover` | ratio | `memzuc_total_cash_risk_0_24m / annualized_net_sales` |
| `pos_volume_change` | change | `((current_pos_monthly_volume - lag_12_pos_monthly_volume) / abs(lag_12_pos_monthly_volume))` |
| `bank_debt_to_ebitda` | ratio | `(memzuc_total_cash_risk_0_24m * tlref_factor) / annualized_ebitda` |
| `trade_receivables_to_turnover` | ratio | `(fs_trade_receivables + fs_notes_receivable) / annualized_net_sales` |
| `profitability_to_turnover` | ratio | `annualized_net_profit / annualized_net_sales` |
| `equity_change` | change | `((current_equity - lag_12_equity) / abs(lag_12_equity))` |
| `ifrs9_behavioral_pd` | score | native authoritative score |
| `kkb_commercial_score` | score | native authoritative score |
| `kkb_indebtedness_index` | score | native authoritative score |
| `net_sales_change` | change | `((annualized_net_sales - lag_12_annualized_net_sales) / abs(lag_12_annualized_net_sales))` |
| `memzuc_limit_utilization_increase` | ratio | `memzuc_total_risk / memzuc_total_limit` |

### 2.2 Self-History Features

Her base feature icin iki ortak history turevi uretilir:

- `__delta_1`
  - bir onceki snapshot'a gore mutlak fark
- `__self_zscore_6`
  - onceki 6 gozleme gore self-z score

Ornek:

- `bank_debt_to_turnover__delta_1`
- `bank_debt_to_turnover__self_zscore_6`
- `kkb_commercial_score__delta_1`
- `kkb_commercial_score__self_zscore_6`

### 2.3 Trend Features

Yalniz trend-anlamli iki aile icin ek trend turevi uretilir:

| Derived Column | Hesap Mantigi |
|---|---|
| `pos_volume_change__trend_slope_6` | trailing 6 snapshot lineer egim |
| `net_sales_change__trend_slope_6` | trailing 6 snapshot lineer egim |

### 2.4 Population-Reference Features

Yalniz business beklentisi geregi `POS` ve `net sales` aileleri icin snapshot-ici populasyon referansi uretilir.

| Derived Column | Hesap Mantigi |
|---|---|
| `pos_volume_change__population_percentile` | ayni snapshot icinde percent rank |
| `pos_volume_change__vs_population_median_delta` | `pos_volume_change - snapshot_median(pos_volume_change)` |
| `net_sales_change__population_percentile` | ayni snapshot icinde percent rank |
| `net_sales_change__vs_population_median_delta` | `net_sales_change - snapshot_median(net_sales_change)` |

### 2.5 Model Kullanim Kapsami

Bu Faz 1 demo cohort'unda modele yazilan tum generated feature'lar numeric'tir ve asagidaki sira ile uretilir:

1. native atomik alanlar
2. base business features
3. self-history features
4. trend features
5. population-reference features

Warm-up sonrasi kalan derived satirlar Oracle `input_features` tablosuna yazilir. Sonraki asamada preprocessing, feature selection, AE/IF/MD fit, calibration ve live scoring mevcut proje lifecycle'i tarafindan yapilir.
