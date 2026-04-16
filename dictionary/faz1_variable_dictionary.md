# Faz 1 Variable Dictionary

Bu sozluk, `bilancolu + POS'u olan` Ticari Orta Faz 1 cohort'u icin iki farkli katmani birlikte tanimlar:

1. `Native source dictionary`
   Oracle `native_features` tablosuna basilan atomik kaynak alanlar
2. `Derived final long list`
   Oracle `input_features` tablosuna yazilan, modelin gordugu tum final alanlar

Bu ikinci liste `final long list` olarak okunmalidir. Yani:
- native'den aynen tasinan ama derived/input tablosunda yer alan alanlar da burada vardir
- is kurali / history / trend / population-reference ile uretilen tum alanlar da burada vardir
- `customer_id`, `snapshot_date`, `segment` gibi metadata alanlari input tablosunda bulundugu icin bu listede teknik pass-through olarak yer alir

Faz 1 artiÌ§k haftalik degil, ay sonu snapshot'lariyla aylik cadence uzerinden calisir.

## 1. Faz 1 Warm-Up Kurali

- Native tablo tum ay sonu snapshot'larini tutar.
- Derived/input tablo ise history feature'lar anlamli olustugu noktadan baslar.
- Faz 1'de `lag_12` kullanan yillik karsilastirma degiskenleri ve bunlarin history turevleri nedeniyle her musteri icin ilk `18` snapshot warm-up olarak disarida birakilir.
- Bu nedenle `derived row count`, `native row count` ile ayni olmak zorunda degildir; genellikle daha dusuktur.

## 2. Native Source Dictionary

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

### Native Veri Uretim Notu

- Faz 1 native veri sentetiktir ama business semantigine uygun atomik alan yapisini taklit eder.
- Missing degerler native seviyede enjekte edilir; generated feature tablosunda bu missing'ler korunur ve pipeline preprocessing katmaninda ele alinir.
- Outlier'lar native seviyede enjekte edilir; winsorization ve robust scaler davranisini test etmek icin kullanilir.

## 3. Native -> Derived Donusum Mantigi

Derived long list'teki bircok alanin arkasinda ortak ara hesaplar vardir:

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

History katmaninda ortak pattern'ler:

- `__delta_1 = current_value - lag_1(current_value)`
- `__self_zscore_6 = (current_value - shift(1).rolling_mean_6) / shift(1).rolling_std_6`

Trend ve population katmaninda ortak pattern'ler:

- `__trend_slope_6`
  - trailing 6 snapshot lineer egim
- `__population_percentile`
  - ayni snapshot icinde percent rank
- `__vs_population_median_delta`
  - current_value - snapshot median

## 4. Derived Final Long List

### Tablo

- Oracle input tablo anahtari: `input_features`
- Oracle fiziksel tablo: `EWS_TO_FAZ1_INPUT`
- Grain: `customer_id + snapshot_date`

### 4.1 Technical Pass-Through Columns

Bu alanlar model feature'i degildir ama input tablosunda bulunduğu icin final long list'e dahildir.

| Derived Column | Derived Type | Aciklama | Hesaplama / Donusum | Business Basligi |
|---|---|---|---|---|
| `customer_id` | technical_pass_through | Input tablosu musteri anahtari | native `customer_id` aynen tasinir | teknik metadata |
| `snapshot_date` | technical_pass_through | Input tablosu snapshot tarihi | native `snapshot_date` aynen tasinir | teknik metadata |
| `segment` | technical_pass_through | Cohort etiketi | native `segment` aynen tasinir | teknik metadata |

### 4.2 Base Business Features

Bu liste derived/input tablosunda yer alan 11 temel business feature'i gosterir. Bunlarin 3 tanesi native authoritative skorun aynen tasindigi passthrough-base yapidadir.

| Derived Column | Derived Type | Aciklama | Hesaplama / Donusum | Uretildigi Native Alanlar | Business Basligi |
|---|---|---|---|---|---|
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

### 4.3 Self-History Features

Bu liste final derived/input tablosunda explicit duran tum `__delta_1` ve `__self_zscore_6` kolonlarini tek tek gosterir.

| Derived Column | Derived Type | Aciklama | Hesaplama / Donusum | Uretildigi Alan | Business Basligi |
|---|---|---|---|---|---|
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

### 4.4 Trend Features

| Derived Column | Derived Type | Aciklama | Hesaplama / Donusum | Uretildigi Alan | Business Basligi |
|---|---|---|---|---|---|
| `pos_volume_change__trend_slope_6` | trend | Son 6 snapshot'taki lineer egim | `rolling_slope_6(pos_volume_change)` | `pos_volume_change` | `Musterinin Tum Bankalardaki Aylik POS Hacmi Degiskeni` |
| `net_sales_change__trend_slope_6` | trend | Son 6 snapshot'taki lineer egim | `rolling_slope_6(net_sales_change)` | `net_sales_change` | `Net Satis (Ciro) Degiskeni` |

### 4.5 Population-Reference Features

Bu katman business beklentisine gore yalniz `POS` ve `Net Satis` aileleri icin uretilir.

| Derived Column | Derived Type | Aciklama | Hesaplama / Donusum | Uretildigi Alan | Business Basligi |
|---|---|---|---|---|---|
| `pos_volume_change__population_percentile` | population_reference | Ayni snapshot icinde percent rank | `pct_rank(pos_volume_change)` | `pos_volume_change` | `Musterinin Tum Bankalardaki Aylik POS Hacmi Degiskeni` |
| `pos_volume_change__vs_population_median_delta` | population_reference | Snapshot medyanindan fark | `pos_volume_change - median(pos_volume_change)` | `pos_volume_change` | `Musterinin Tum Bankalardaki Aylik POS Hacmi Degiskeni` |
| `net_sales_change__population_percentile` | population_reference | Ayni snapshot icinde percent rank | `pct_rank(net_sales_change)` | `net_sales_change` | `Net Satis (Ciro) Degiskeni` |
| `net_sales_change__vs_population_median_delta` | population_reference | Snapshot medyanindan fark | `net_sales_change - median(net_sales_change)` | `net_sales_change` | `Net Satis (Ciro) Degiskeni` |

## 5. Faz 1 Derived Long List Ozeti

Derived/input tablosundaki toplam final kolon yapisi:

- `3` teknik pass-through kolon
- `11` base business feature
- `22` self-history feature
- `2` trend feature
- `4` population-reference feature

Toplam:
- `42` kolonluk final input long list

Modelin gordugu numeric feature set:
- `11 + 22 + 2 + 4 = 39` numeric feature

Model gormeyen ama input tablosunda tasinan teknik kolonlar:
- `customer_id`
- `snapshot_date`
- `segment`

## 6. Faz 1 Okuma Notu

Bu sozlukte:

- `Native source dictionary` ham atomik kaynak katmanidir
- `Derived final long list` ise model-ready input tablosunun explicit ve tam kolon listesidir

Yani derived taraf, native'in kisa bir ozetini degil; finalde Oracle `input_features` tablosunda duran tum alanlarin teknik kontratini verir.
