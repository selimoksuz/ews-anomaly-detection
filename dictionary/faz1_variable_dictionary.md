# Faz 1 Variable Dictionary

Bu sozluk yalnizca Ticari Orta Faz 1 kapsamini mapler.

## Faz 1 Cohort

- `segment = TICARI_ORTA`
- `bank_total_risk >= 1_000_000`
- `is_balance_sheet_customer = 1`
- `has_pos = 1`

## Donusum Kurallari

- `Q1 -> *4`, `Q2 -> *2`, `Q3 -> *4/3`, `Q4/YE -> *1`
- Yillik karsilastirmali degiskenler `lag_12` kullanir.
- Self history turevleri: `__delta_1`, `__self_zscore_6`
- Trend turevi: `__trend_slope_6`
- Populasyon turevleri: `__population_percentile`, `__vs_population_median_delta`
- Warm-up nedeniyle ilk `18` snapshot final derived tabloda tutulmaz.

## 1. Native Source Dictionary

- Oracle anahtari: `native_features`
- Fiziksel tablo: `EWS_TO_FAZ1_NATIVE`
- Grain: `customer_id + snapshot_date`

| Native Column | Tip | Aciklama | Missing / Outlier Notu |
|---|---|---|---|
| `customer_id` | identifier | Musteri anahtari | zorunlu |
| `snapshot_date` | date | Ay sonu snapshot tarihi | zorunlu |
| `segment` | text | Segment etiketi | zorunlu |
| `is_balance_sheet_customer` | flag | Bilancolu musteri isareti | cohort filtresinde kullanilir |
| `has_pos` | flag | Herhangi bir bankada POS varligi | cohort filtresinde kullanilir |
| `bank_total_risk` | numeric | Tum bankalardaki toplam risk | cohort filtresinde `>= 1m` |
| `nace_section` | text | Sektor ust siniflamasi | destekleyici alan |
| `nace_main` | text | Sektor ana etiketi | destekleyici alan |
| `fs_period_code` | text | Mali veri donem kodu | annualization icin kullanilir |
| `fs_last_update_date` | date | Mali verinin son guncelleme tarihi | freshness kontrolunde kullanilir |
| `memzuc_total_cash_risk_0_24m` | numeric | 0-24 ay nakdi risk toplami | missing/outlier bulunabilir |
| `memzuc_business_loan_risk_0_24m` | numeric | 0-24 ay isletme kredisi riski | missing/outlier bulunabilir |
| `tlref_factor` | numeric | TLREF normalizasyon carpani | zorunlu |
| `inflation_yoy_rate` | numeric | Yillik enflasyon orani | dis referans seri |
| `fs_net_sales_cumulative` | numeric | Donemsel kumulatif net satis | annualize edilir |
| `fs_ebitda_cumulative` | numeric | Donemsel kumulatif EBITDA | missing/outlier bulunabilir |
| `fs_trade_receivables` | numeric | Ticari alacak tutari | missing/outlier bulunabilir |
| `fs_notes_receivable` | numeric | Alacak senetleri tutari | zorunlu |
| `fs_net_profit_cumulative` | numeric | Donemsel kumulatif net kar | annualize edilir |
| `fs_equity` | numeric | Ozkaynak seviyesi | missing olabilir |
| `pos_monthly_volume` | numeric | Tum bankalar POS aylik hacmi | missing/outlier bulunabilir |
| `ifrs9_behavioral_pd` | numeric | Davranissal temerrut olasiligi | missing/outlier bulunabilir |
| `kkb_commercial_score` | numeric | KKB ticari kredi notu | missing/outlier bulunabilir |
| `kkb_indebtedness_index` | numeric | KKB ticari borcluluk endeksi | zorunlu |
| `memzuc_total_limit` | numeric | Memzuc toplam limit | missing olabilir |
| `memzuc_total_risk` | numeric | Memzuc toplam risk | zorunlu |
| `bank_asset_average_balance` | numeric | Bankamizdaki ortalama mevduat/varlik seviyesi | missing/outlier bulunabilir |

## 2. Final Long List Dictionary

- Oracle anahtari: `input_features`
- Fiziksel tablo: `EWS_TO_FAZ1_INPUT`
- Grain: `customer_id + snapshot_date`

| Final Column | Column Type | Aciklama | Hesaplama / Donusum | Uretildigi Native / Base Alanlar | Business Basligi | Yon Kurali |
|---|---|---|---|---|---|---|
| `customer_id` | technical_pass_through | Teknik tasiyici kolon | native `customer_id` aynen tasinir | customer_id | teknik metadata | teknik metadata |
| `snapshot_date` | technical_pass_through | Teknik tasiyici kolon | native `snapshot_date` aynen tasinir | snapshot_date | teknik metadata | teknik metadata |
| `segment` | technical_pass_through | Teknik tasiyici kolon | native `segment` aynen tasinir | segment | teknik metadata | teknik metadata |
| `bank_debt_to_turnover` | base_feature | Banka Borclulugu / Ciro | memzuc_total_cash_risk_0_24m / annualized(fs_net_sales_cumulative) | memzuc_total_cash_risk_0_24m, fs_net_sales_cumulative, fs_period_code | Banka Borclulugu ve Ciro Iliskisi | artmasi kotu, azalmasi iyi |
| `pos_volume_change` | base_feature | Tum Bankalar POS Hacmi Degisimi | (pos_monthly_volume - lag_12(pos_monthly_volume)) / abs(lag_12(pos_monthly_volume)) | pos_monthly_volume | Musterinin Tum Bankalardaki Aylik POS Hacmi | azalmasi kotu, artmasi iyi |
| `bank_debt_to_ebitda` | base_feature | Banka Borclulugu / EBITDA | (memzuc_total_cash_risk_0_24m * tlref_factor) / annualized(fs_ebitda_cumulative) | memzuc_total_cash_risk_0_24m, tlref_factor, fs_ebitda_cumulative, fs_period_code | Banka Borclulugu ve EBITDA Iliskisi | artmasi kotu, azalmasi iyi |
| `trade_receivables_to_turnover` | base_feature | Ticari Alacak / Ciro | (fs_trade_receivables + fs_notes_receivable) / annualized(fs_net_sales_cumulative) | fs_trade_receivables, fs_notes_receivable, fs_net_sales_cumulative, fs_period_code | Bilanco Ticari Alacak ve Ciro | artmasi kotu, azalmasi iyi |
| `profitability_to_turnover` | base_feature | Karlilik / Ciro | annualized(fs_net_profit_cumulative) / annualized(fs_net_sales_cumulative) | fs_net_profit_cumulative, fs_net_sales_cumulative, fs_period_code | Bilanco Karlilik ve Ciro | azalmasi kotu, artmasi iyi |
| `business_loan_vs_inflation` | base_feature | Isletme Borcu / Enflasyon Farki | yoy_pct_change(memzuc_business_loan_risk_0_24m) - inflation_yoy_rate | memzuc_business_loan_risk_0_24m, inflation_yoy_rate | Bankalardaki Isletme Borclari ve Enflasyon Iliskisi | artmasi kotu, azalmasi iyi |
| `equity_change` | base_feature | Ozkaynak Degisimi | (fs_equity - lag_12(fs_equity)) / abs(lag_12(fs_equity)) | fs_equity | Bilanco Ozkaynak Bilgisi | azalmasi kotu, artmasi iyi |
| `ifrs9_behavioral_pd` | base_feature | TFRS Davranis Temerrut Olasiligi | native ifrs9_behavioral_pd | ifrs9_behavioral_pd | TFRS Davranis Temerrut Olasiligi | artmasi kotu, azalmasi iyi |
| `kkb_commercial_score` | base_feature | KKB Ticari Kredi Notu | native kkb_commercial_score | kkb_commercial_score | KKB Ticari Kredi Notu | azalmasi kotu, artmasi iyi |
| `kkb_indebtedness_index` | base_feature | KKB Ticari Borcluluk Endeksi | native kkb_indebtedness_index | kkb_indebtedness_index | KKB Ticari Borcluluk Endeksi | artmasi kotu, azalmasi iyi |
| `net_sales_change` | base_feature | Net Satis Degisimi | (annualized(fs_net_sales_cumulative) - lag_12(annualized(fs_net_sales_cumulative))) / abs(lag_12(annualized(fs_net_sales_cumulative))) | fs_net_sales_cumulative, fs_period_code | Net Satis (Ciro) | azalmasi kotu, artmasi iyi |
| `memzuc_limit_utilization_increase` | base_feature | Memzuc Limit Doluluk Orani | memzuc_total_risk / memzuc_total_limit | memzuc_total_risk, memzuc_total_limit | Memzuc Limit Doluluk Artisi | artmasi kotu, azalmasi iyi |
| `bank_asset_average_change` | base_feature | Banka Varlik Ortalamasi Degisimi | (bank_asset_average_balance - lag_12(bank_asset_average_balance)) / abs(lag_12(bank_asset_average_balance)) | bank_asset_average_balance | Bankamizda Bulunan Mevduat / Varlik Ortalamalari | azalmasi kotu, artmasi iyi |
| `bank_debt_to_turnover__delta_1` | self_history_delta | Banka Borclulugu / Ciro Son Aya Gore Fark | bank_debt_to_turnover - lag_1(bank_debt_to_turnover) | bank_debt_to_turnover | Banka Borclulugu ve Ciro Iliskisi | artmasi kotu, azalmasi iyi |
| `pos_volume_change__delta_1` | self_history_delta | Tum Bankalar POS Hacmi Degisimi Son Aya Gore Fark | pos_volume_change - lag_1(pos_volume_change) | pos_volume_change | Musterinin Tum Bankalardaki Aylik POS Hacmi | azalmasi kotu, artmasi iyi |
| `bank_debt_to_ebitda__delta_1` | self_history_delta | Banka Borclulugu / EBITDA Son Aya Gore Fark | bank_debt_to_ebitda - lag_1(bank_debt_to_ebitda) | bank_debt_to_ebitda | Banka Borclulugu ve EBITDA Iliskisi | artmasi kotu, azalmasi iyi |
| `trade_receivables_to_turnover__delta_1` | self_history_delta | Ticari Alacak / Ciro Son Aya Gore Fark | trade_receivables_to_turnover - lag_1(trade_receivables_to_turnover) | trade_receivables_to_turnover | Bilanco Ticari Alacak ve Ciro | artmasi kotu, azalmasi iyi |
| `profitability_to_turnover__delta_1` | self_history_delta | Karlilik / Ciro Son Aya Gore Fark | profitability_to_turnover - lag_1(profitability_to_turnover) | profitability_to_turnover | Bilanco Karlilik ve Ciro | azalmasi kotu, artmasi iyi |
| `business_loan_vs_inflation__delta_1` | self_history_delta | Isletme Borcu / Enflasyon Farki Son Aya Gore Fark | business_loan_vs_inflation - lag_1(business_loan_vs_inflation) | business_loan_vs_inflation | Bankalardaki Isletme Borclari ve Enflasyon Iliskisi | artmasi kotu, azalmasi iyi |
| `equity_change__delta_1` | self_history_delta | Ozkaynak Degisimi Son Aya Gore Fark | equity_change - lag_1(equity_change) | equity_change | Bilanco Ozkaynak Bilgisi | azalmasi kotu, artmasi iyi |
| `ifrs9_behavioral_pd__delta_1` | self_history_delta | TFRS Davranis Temerrut Olasiligi Son Aya Gore Fark | ifrs9_behavioral_pd - lag_1(ifrs9_behavioral_pd) | ifrs9_behavioral_pd | TFRS Davranis Temerrut Olasiligi | artmasi kotu, azalmasi iyi |
| `kkb_commercial_score__delta_1` | self_history_delta | KKB Ticari Kredi Notu Son Aya Gore Fark | kkb_commercial_score - lag_1(kkb_commercial_score) | kkb_commercial_score | KKB Ticari Kredi Notu | azalmasi kotu, artmasi iyi |
| `kkb_indebtedness_index__delta_1` | self_history_delta | KKB Ticari Borcluluk Endeksi Son Aya Gore Fark | kkb_indebtedness_index - lag_1(kkb_indebtedness_index) | kkb_indebtedness_index | KKB Ticari Borcluluk Endeksi | artmasi kotu, azalmasi iyi |
| `net_sales_change__delta_1` | self_history_delta | Net Satis Degisimi Son Aya Gore Fark | net_sales_change - lag_1(net_sales_change) | net_sales_change | Net Satis (Ciro) | azalmasi kotu, artmasi iyi |
| `memzuc_limit_utilization_increase__delta_1` | self_history_delta | Memzuc Limit Doluluk Orani Son Aya Gore Fark | memzuc_limit_utilization_increase - lag_1(memzuc_limit_utilization_increase) | memzuc_limit_utilization_increase | Memzuc Limit Doluluk Artisi | artmasi kotu, azalmasi iyi |
| `bank_asset_average_change__delta_1` | self_history_delta | Banka Varlik Ortalamasi Degisimi Son Aya Gore Fark | bank_asset_average_change - lag_1(bank_asset_average_change) | bank_asset_average_change | Bankamizda Bulunan Mevduat / Varlik Ortalamalari | azalmasi kotu, artmasi iyi |
| `bank_debt_to_turnover__self_zscore_6` | self_history_zscore | Banka Borclulugu / Ciro Self-Z(6) | (bank_debt_to_turnover - shift(1).rolling_mean_6(bank_debt_to_turnover)) / shift(1).rolling_std_6(bank_debt_to_turnover) | bank_debt_to_turnover | Banka Borclulugu ve Ciro Iliskisi | artmasi kotu, azalmasi iyi |
| `pos_volume_change__self_zscore_6` | self_history_zscore | Tum Bankalar POS Hacmi Degisimi Self-Z(6) | (pos_volume_change - shift(1).rolling_mean_6(pos_volume_change)) / shift(1).rolling_std_6(pos_volume_change) | pos_volume_change | Musterinin Tum Bankalardaki Aylik POS Hacmi | azalmasi kotu, artmasi iyi |
| `bank_debt_to_ebitda__self_zscore_6` | self_history_zscore | Banka Borclulugu / EBITDA Self-Z(6) | (bank_debt_to_ebitda - shift(1).rolling_mean_6(bank_debt_to_ebitda)) / shift(1).rolling_std_6(bank_debt_to_ebitda) | bank_debt_to_ebitda | Banka Borclulugu ve EBITDA Iliskisi | artmasi kotu, azalmasi iyi |
| `trade_receivables_to_turnover__self_zscore_6` | self_history_zscore | Ticari Alacak / Ciro Self-Z(6) | (trade_receivables_to_turnover - shift(1).rolling_mean_6(trade_receivables_to_turnover)) / shift(1).rolling_std_6(trade_receivables_to_turnover) | trade_receivables_to_turnover | Bilanco Ticari Alacak ve Ciro | artmasi kotu, azalmasi iyi |
| `profitability_to_turnover__self_zscore_6` | self_history_zscore | Karlilik / Ciro Self-Z(6) | (profitability_to_turnover - shift(1).rolling_mean_6(profitability_to_turnover)) / shift(1).rolling_std_6(profitability_to_turnover) | profitability_to_turnover | Bilanco Karlilik ve Ciro | azalmasi kotu, artmasi iyi |
| `business_loan_vs_inflation__self_zscore_6` | self_history_zscore | Isletme Borcu / Enflasyon Farki Self-Z(6) | (business_loan_vs_inflation - shift(1).rolling_mean_6(business_loan_vs_inflation)) / shift(1).rolling_std_6(business_loan_vs_inflation) | business_loan_vs_inflation | Bankalardaki Isletme Borclari ve Enflasyon Iliskisi | artmasi kotu, azalmasi iyi |
| `equity_change__self_zscore_6` | self_history_zscore | Ozkaynak Degisimi Self-Z(6) | (equity_change - shift(1).rolling_mean_6(equity_change)) / shift(1).rolling_std_6(equity_change) | equity_change | Bilanco Ozkaynak Bilgisi | azalmasi kotu, artmasi iyi |
| `ifrs9_behavioral_pd__self_zscore_6` | self_history_zscore | TFRS Davranis Temerrut Olasiligi Self-Z(6) | (ifrs9_behavioral_pd - shift(1).rolling_mean_6(ifrs9_behavioral_pd)) / shift(1).rolling_std_6(ifrs9_behavioral_pd) | ifrs9_behavioral_pd | TFRS Davranis Temerrut Olasiligi | artmasi kotu, azalmasi iyi |
| `kkb_commercial_score__self_zscore_6` | self_history_zscore | KKB Ticari Kredi Notu Self-Z(6) | (kkb_commercial_score - shift(1).rolling_mean_6(kkb_commercial_score)) / shift(1).rolling_std_6(kkb_commercial_score) | kkb_commercial_score | KKB Ticari Kredi Notu | azalmasi kotu, artmasi iyi |
| `kkb_indebtedness_index__self_zscore_6` | self_history_zscore | KKB Ticari Borcluluk Endeksi Self-Z(6) | (kkb_indebtedness_index - shift(1).rolling_mean_6(kkb_indebtedness_index)) / shift(1).rolling_std_6(kkb_indebtedness_index) | kkb_indebtedness_index | KKB Ticari Borcluluk Endeksi | artmasi kotu, azalmasi iyi |
| `net_sales_change__self_zscore_6` | self_history_zscore | Net Satis Degisimi Self-Z(6) | (net_sales_change - shift(1).rolling_mean_6(net_sales_change)) / shift(1).rolling_std_6(net_sales_change) | net_sales_change | Net Satis (Ciro) | azalmasi kotu, artmasi iyi |
| `memzuc_limit_utilization_increase__self_zscore_6` | self_history_zscore | Memzuc Limit Doluluk Orani Self-Z(6) | (memzuc_limit_utilization_increase - shift(1).rolling_mean_6(memzuc_limit_utilization_increase)) / shift(1).rolling_std_6(memzuc_limit_utilization_increase) | memzuc_limit_utilization_increase | Memzuc Limit Doluluk Artisi | artmasi kotu, azalmasi iyi |
| `bank_asset_average_change__self_zscore_6` | self_history_zscore | Banka Varlik Ortalamasi Degisimi Self-Z(6) | (bank_asset_average_change - shift(1).rolling_mean_6(bank_asset_average_change)) / shift(1).rolling_std_6(bank_asset_average_change) | bank_asset_average_change | Bankamizda Bulunan Mevduat / Varlik Ortalamalari | azalmasi kotu, artmasi iyi |
| `bank_debt_to_turnover__trend_slope_6` | trend_slope | Banka Borclulugu / Ciro Trend Egimi(6) | rolling_slope_6(bank_debt_to_turnover) | bank_debt_to_turnover | Banka Borclulugu ve Ciro Iliskisi | artmasi kotu, azalmasi iyi |
| `pos_volume_change__trend_slope_6` | trend_slope | Tum Bankalar POS Hacmi Degisimi Trend Egimi(6) | rolling_slope_6(pos_volume_change) | pos_volume_change | Musterinin Tum Bankalardaki Aylik POS Hacmi | azalmasi kotu, artmasi iyi |
| `bank_debt_to_ebitda__trend_slope_6` | trend_slope | Banka Borclulugu / EBITDA Trend Egimi(6) | rolling_slope_6(bank_debt_to_ebitda) | bank_debt_to_ebitda | Banka Borclulugu ve EBITDA Iliskisi | artmasi kotu, azalmasi iyi |
| `trade_receivables_to_turnover__trend_slope_6` | trend_slope | Ticari Alacak / Ciro Trend Egimi(6) | rolling_slope_6(trade_receivables_to_turnover) | trade_receivables_to_turnover | Bilanco Ticari Alacak ve Ciro | artmasi kotu, azalmasi iyi |
| `profitability_to_turnover__trend_slope_6` | trend_slope | Karlilik / Ciro Trend Egimi(6) | rolling_slope_6(profitability_to_turnover) | profitability_to_turnover | Bilanco Karlilik ve Ciro | azalmasi kotu, artmasi iyi |
| `business_loan_vs_inflation__trend_slope_6` | trend_slope | Isletme Borcu / Enflasyon Farki Trend Egimi(6) | rolling_slope_6(business_loan_vs_inflation) | business_loan_vs_inflation | Bankalardaki Isletme Borclari ve Enflasyon Iliskisi | artmasi kotu, azalmasi iyi |
| `equity_change__trend_slope_6` | trend_slope | Ozkaynak Degisimi Trend Egimi(6) | rolling_slope_6(equity_change) | equity_change | Bilanco Ozkaynak Bilgisi | azalmasi kotu, artmasi iyi |
| `ifrs9_behavioral_pd__trend_slope_6` | trend_slope | TFRS Davranis Temerrut Olasiligi Trend Egimi(6) | rolling_slope_6(ifrs9_behavioral_pd) | ifrs9_behavioral_pd | TFRS Davranis Temerrut Olasiligi | artmasi kotu, azalmasi iyi |
| `kkb_commercial_score__trend_slope_6` | trend_slope | KKB Ticari Kredi Notu Trend Egimi(6) | rolling_slope_6(kkb_commercial_score) | kkb_commercial_score | KKB Ticari Kredi Notu | azalmasi kotu, artmasi iyi |
| `kkb_indebtedness_index__trend_slope_6` | trend_slope | KKB Ticari Borcluluk Endeksi Trend Egimi(6) | rolling_slope_6(kkb_indebtedness_index) | kkb_indebtedness_index | KKB Ticari Borcluluk Endeksi | artmasi kotu, azalmasi iyi |
| `net_sales_change__trend_slope_6` | trend_slope | Net Satis Degisimi Trend Egimi(6) | rolling_slope_6(net_sales_change) | net_sales_change | Net Satis (Ciro) | azalmasi kotu, artmasi iyi |
| `memzuc_limit_utilization_increase__trend_slope_6` | trend_slope | Memzuc Limit Doluluk Orani Trend Egimi(6) | rolling_slope_6(memzuc_limit_utilization_increase) | memzuc_limit_utilization_increase | Memzuc Limit Doluluk Artisi | artmasi kotu, azalmasi iyi |
| `bank_asset_average_change__trend_slope_6` | trend_slope | Banka Varlik Ortalamasi Degisimi Trend Egimi(6) | rolling_slope_6(bank_asset_average_change) | bank_asset_average_change | Bankamizda Bulunan Mevduat / Varlik Ortalamalari | azalmasi kotu, artmasi iyi |
| `bank_debt_to_turnover__population_percentile` | population_reference | Banka Borclulugu / Ciro Populasyon Percentile | pct_rank(bank_debt_to_turnover) within snapshot | bank_debt_to_turnover | Banka Borclulugu ve Ciro Iliskisi | artmasi kotu, azalmasi iyi |
| `bank_debt_to_turnover__vs_population_median_delta` | population_reference | Banka Borclulugu / Ciro Snapshot Median Farki | bank_debt_to_turnover - snapshot_median(bank_debt_to_turnover) | bank_debt_to_turnover | Banka Borclulugu ve Ciro Iliskisi | artmasi kotu, azalmasi iyi |
| `pos_volume_change__population_percentile` | population_reference | Tum Bankalar POS Hacmi Degisimi Populasyon Percentile | pct_rank(pos_volume_change) within snapshot | pos_volume_change | Musterinin Tum Bankalardaki Aylik POS Hacmi | azalmasi kotu, artmasi iyi |
| `pos_volume_change__vs_population_median_delta` | population_reference | Tum Bankalar POS Hacmi Degisimi Snapshot Median Farki | pos_volume_change - snapshot_median(pos_volume_change) | pos_volume_change | Musterinin Tum Bankalardaki Aylik POS Hacmi | azalmasi kotu, artmasi iyi |
| `bank_debt_to_ebitda__population_percentile` | population_reference | Banka Borclulugu / EBITDA Populasyon Percentile | pct_rank(bank_debt_to_ebitda) within snapshot | bank_debt_to_ebitda | Banka Borclulugu ve EBITDA Iliskisi | artmasi kotu, azalmasi iyi |
| `bank_debt_to_ebitda__vs_population_median_delta` | population_reference | Banka Borclulugu / EBITDA Snapshot Median Farki | bank_debt_to_ebitda - snapshot_median(bank_debt_to_ebitda) | bank_debt_to_ebitda | Banka Borclulugu ve EBITDA Iliskisi | artmasi kotu, azalmasi iyi |
| `trade_receivables_to_turnover__population_percentile` | population_reference | Ticari Alacak / Ciro Populasyon Percentile | pct_rank(trade_receivables_to_turnover) within snapshot | trade_receivables_to_turnover | Bilanco Ticari Alacak ve Ciro | artmasi kotu, azalmasi iyi |
| `trade_receivables_to_turnover__vs_population_median_delta` | population_reference | Ticari Alacak / Ciro Snapshot Median Farki | trade_receivables_to_turnover - snapshot_median(trade_receivables_to_turnover) | trade_receivables_to_turnover | Bilanco Ticari Alacak ve Ciro | artmasi kotu, azalmasi iyi |
| `profitability_to_turnover__population_percentile` | population_reference | Karlilik / Ciro Populasyon Percentile | pct_rank(profitability_to_turnover) within snapshot | profitability_to_turnover | Bilanco Karlilik ve Ciro | azalmasi kotu, artmasi iyi |
| `profitability_to_turnover__vs_population_median_delta` | population_reference | Karlilik / Ciro Snapshot Median Farki | profitability_to_turnover - snapshot_median(profitability_to_turnover) | profitability_to_turnover | Bilanco Karlilik ve Ciro | azalmasi kotu, artmasi iyi |
| `business_loan_vs_inflation__population_percentile` | population_reference | Isletme Borcu / Enflasyon Farki Populasyon Percentile | pct_rank(business_loan_vs_inflation) within snapshot | business_loan_vs_inflation | Bankalardaki Isletme Borclari ve Enflasyon Iliskisi | artmasi kotu, azalmasi iyi |
| `business_loan_vs_inflation__vs_population_median_delta` | population_reference | Isletme Borcu / Enflasyon Farki Snapshot Median Farki | business_loan_vs_inflation - snapshot_median(business_loan_vs_inflation) | business_loan_vs_inflation | Bankalardaki Isletme Borclari ve Enflasyon Iliskisi | artmasi kotu, azalmasi iyi |
| `equity_change__population_percentile` | population_reference | Ozkaynak Degisimi Populasyon Percentile | pct_rank(equity_change) within snapshot | equity_change | Bilanco Ozkaynak Bilgisi | azalmasi kotu, artmasi iyi |
| `equity_change__vs_population_median_delta` | population_reference | Ozkaynak Degisimi Snapshot Median Farki | equity_change - snapshot_median(equity_change) | equity_change | Bilanco Ozkaynak Bilgisi | azalmasi kotu, artmasi iyi |
| `ifrs9_behavioral_pd__population_percentile` | population_reference | TFRS Davranis Temerrut Olasiligi Populasyon Percentile | pct_rank(ifrs9_behavioral_pd) within snapshot | ifrs9_behavioral_pd | TFRS Davranis Temerrut Olasiligi | artmasi kotu, azalmasi iyi |
| `ifrs9_behavioral_pd__vs_population_median_delta` | population_reference | TFRS Davranis Temerrut Olasiligi Snapshot Median Farki | ifrs9_behavioral_pd - snapshot_median(ifrs9_behavioral_pd) | ifrs9_behavioral_pd | TFRS Davranis Temerrut Olasiligi | artmasi kotu, azalmasi iyi |
| `kkb_commercial_score__population_percentile` | population_reference | KKB Ticari Kredi Notu Populasyon Percentile | pct_rank(kkb_commercial_score) within snapshot | kkb_commercial_score | KKB Ticari Kredi Notu | azalmasi kotu, artmasi iyi |
| `kkb_commercial_score__vs_population_median_delta` | population_reference | KKB Ticari Kredi Notu Snapshot Median Farki | kkb_commercial_score - snapshot_median(kkb_commercial_score) | kkb_commercial_score | KKB Ticari Kredi Notu | azalmasi kotu, artmasi iyi |
| `kkb_indebtedness_index__population_percentile` | population_reference | KKB Ticari Borcluluk Endeksi Populasyon Percentile | pct_rank(kkb_indebtedness_index) within snapshot | kkb_indebtedness_index | KKB Ticari Borcluluk Endeksi | artmasi kotu, azalmasi iyi |
| `kkb_indebtedness_index__vs_population_median_delta` | population_reference | KKB Ticari Borcluluk Endeksi Snapshot Median Farki | kkb_indebtedness_index - snapshot_median(kkb_indebtedness_index) | kkb_indebtedness_index | KKB Ticari Borcluluk Endeksi | artmasi kotu, azalmasi iyi |
| `net_sales_change__population_percentile` | population_reference | Net Satis Degisimi Populasyon Percentile | pct_rank(net_sales_change) within snapshot | net_sales_change | Net Satis (Ciro) | azalmasi kotu, artmasi iyi |
| `net_sales_change__vs_population_median_delta` | population_reference | Net Satis Degisimi Snapshot Median Farki | net_sales_change - snapshot_median(net_sales_change) | net_sales_change | Net Satis (Ciro) | azalmasi kotu, artmasi iyi |
| `memzuc_limit_utilization_increase__population_percentile` | population_reference | Memzuc Limit Doluluk Orani Populasyon Percentile | pct_rank(memzuc_limit_utilization_increase) within snapshot | memzuc_limit_utilization_increase | Memzuc Limit Doluluk Artisi | artmasi kotu, azalmasi iyi |
| `memzuc_limit_utilization_increase__vs_population_median_delta` | population_reference | Memzuc Limit Doluluk Orani Snapshot Median Farki | memzuc_limit_utilization_increase - snapshot_median(memzuc_limit_utilization_increase) | memzuc_limit_utilization_increase | Memzuc Limit Doluluk Artisi | artmasi kotu, azalmasi iyi |
| `bank_asset_average_change__population_percentile` | population_reference | Banka Varlik Ortalamasi Degisimi Populasyon Percentile | pct_rank(bank_asset_average_change) within snapshot | bank_asset_average_change | Bankamizda Bulunan Mevduat / Varlik Ortalamalari | azalmasi kotu, artmasi iyi |
| `bank_asset_average_change__vs_population_median_delta` | population_reference | Banka Varlik Ortalamasi Degisimi Snapshot Median Farki | bank_asset_average_change - snapshot_median(bank_asset_average_change) | bank_asset_average_change | Bankamizda Bulunan Mevduat / Varlik Ortalamalari | azalmasi kotu, artmasi iyi |

## Ozet

- Native source dictionary kolon sayisi: `27`
- Final long list toplam kolon sayisi: `81`
- Model feature sayisi: `78`
