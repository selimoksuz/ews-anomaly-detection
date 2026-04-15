# Business Variable Dictionary

Bu sozluk, is biriminin talep ettigi business basliklarini birebir mapleyen degisken kontratidir.

## 1. Ortak Kontrat Tanimlari

### Portfolio Scope

- `general`: Ticari Orta icinde genis kapsama sahip aile
- `conditional_general`: genel kullanima aday ama applicability veya coverage guard gerekir
- `segment_specific`: yalniz belirli alt portfoylerde anlamli
- `not_model_scope`: ham feature degil, rule/filter artifact

### Model Placement

- `broad_core`: genel modelin omurgasina dogrudan girer
- `conditional_core`: ayni genel modele guardrail ile girer
- `segment_specific`: yalniz ilgili alt portfoy/submodel icinde modele girer
- `enrichment_only`: modele girmez, sadece baglam ve aciklama icindir
- `rule_only`: deterministic rule/filter katmanina gider

### Missing Semantics

- `Event absence`: olay yoktur, uygun ise `0` verilebilir
- `Structural missing`: feature bu musteri icin uygulanamaz, `NA / not_applicable`
- `Operational missing`: veri gelmeliydi ama gelmedi, imputasyon yapilsa bile quality flag ile izlenir

### Ortak Is Kurallari

- Annualization: `Q1 * 4`, `Q2 * 2`, `Q3 * 4 / 3`, `YE * 1`
- FS freshness: son 12 ayda mali veri yoksa kayit uretilmez
- Population-reference: yalniz `pos_volume_change` ve `net_sales_change` icin acik peer/population turevi onerilir

## 2. Business Basliklari ve Dictionary Karsiliklari

## Sayfa 1

### Banka Borclulugu ve Ciro Iliskisi

- Canonical Name: `bank_debt_to_turnover`
- Priority: `high`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: Memzuc nakdi risk 0-24 ay; FS net sales
- Period / Annualization: nearest FS period; `Q1*4`, `Q2*2`, `Q3*4/3`, `YE*1`
- Calculation Logic: `total_cash_credit_risk_0_24m / annualized_net_sales`
- Direction: `increase_is_risk`
- Refresh Frequency: `monthly_or_quarterly`
- Freshness Rule: FS > 12 ay ise kayit yok
- Applicable Segments: bilancolu musteri
- Model Treatment: operational missing halinde quality flag ile kullanilir
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`, `_self_zscore_6`, `_vs_6m_median_ratio`
- Population Reference: `no`

### Musterinin Tum Bankalardaki Aylik POS Hacmi

- Canonical Name: `pos_volume_change`
- Priority: `high`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: FDSTCIRO-BKM
- Period / Annualization: aylik kumulatif POS hacmi; son 2 yil
- Calculation Logic: aylik POS hacmi current vs prior/peer
- Direction: `decrease_is_risk_but_ambiguous`
- Refresh Frequency: `monthly`
- Freshness Rule: `>45 gun caution; >90 gun enrichment_only`
- Applicable Segments: POS applicability olan musteri
- Model Treatment: applicability yoksa `NA`
- Companion Rules / Filters: mevsimsellik ve faaliyet grubu kiyasi zorunlu
- Recommended History Features: `_current`, `_delta_1`, `_delta_pct_1`, `_self_zscore_6`, `_trend_slope_6`
- Population Reference: `yes` (`_population_percentile`, `_vs_population_median_ratio`)

## Sayfa 2

### Haciz Tutari ve Ciro Iliskisi

- Canonical Name: `seizure_amount_to_turnover`
- Priority: `medium`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: ongoing seizure amounts; FS net sales
- Period / Annualization: son 5 yil seizure kayitlari; FS annualization
- Calculation Logic: `ongoing_seizure_amount / annualized_net_sales`
- Direction: `increase_is_risk`
- Refresh Frequency: `event_based`
- Freshness Rule: FS > 12 ay ise kayit yok
- Applicable Segments: hukuki olay coverage olan musteri
- Model Treatment: event absence halinde `0` verilebilir
- Companion Rules / Filters: `seizure_first_time_24m`
- Recommended History Features: `_current`, `_delta_1`, `_self_zscore_6`
- Population Reference: `no`

## Sayfa 3

### Cek Odemesi Zamani

- Canonical Name: `check_payment_time_shift`
- Priority: `high`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: bank internal cek odemeleri ve saat bucket'i
- Period / Annualization: son 1 yil
- Calculation Logic: ogleden sonra/gec saat bucket'ina kayma
- Direction: `increase_is_risk`
- Refresh Frequency: `weekly_or_monthly`
- Freshness Rule: `>30 gun caution`
- Applicable Segments: cek kullanan musteri
- Model Treatment: event absence halinde `0` verilebilir
- Companion Rules / Filters: `check_payment_time_shift_first_time_6m`, `check_payment_time_shift_consecutive_3m_distinct_days`
- Recommended History Features: `_current`, `_delta_1`, `_rolling_mean_3`
- Population Reference: `no`

### Faktoring

- Canonical Name: `factoring_risk_presence`
- Priority: `medium`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: TACRCSIE
- Period / Annualization: son 2 yil lookback; aylik/event
- Calculation Logic: factoring risk event count / indicator
- Direction: `increase_is_risk`
- Refresh Frequency: `monthly_or_event_based`
- Freshness Rule: `>60 gun stale ise enrichment_only`
- Applicable Segments: factoring coverage olan musteri
- Model Treatment: event absence halinde `0` verilebilir
- Companion Rules / Filters: `factoring_risk_first_time_24m`, `factoring_risk_consecutive_2m`, `factoring_risk_frequency_3_of_12m`
- Recommended History Features: `_current`, `_delta_1`, `_rolling_mean_6`
- Population Reference: `no`

### Banka Borclulugu ve EBITDA Iliskisi

- Canonical Name: `bank_debt_to_ebitda`
- Priority: `high`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: Memzuc nakdi risk 0-24 ay; TLREF; FS EBITDA
- Period / Annualization: nearest FS period; `Q1*4`, `Q2*2`, `Q3*4/3`, `YE*1`
- Calculation Logic: `(total_cash_credit_risk_0_24m * tlref_factor) / annualized_ebitda`
- Direction: `increase_is_risk`
- Refresh Frequency: `quarterly_or_annual`
- Freshness Rule: FS > 12 ay ise kayit yok
- Applicable Segments: bilancolu musteri
- Model Treatment: EBITDA yoksa `NA`
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`, `_self_zscore_6`, `_vs_6m_median_ratio`
- Population Reference: `no`

## Sayfa 4

### Bilanco Ticari Alacak ve Ciro

- Canonical Name: `trade_receivables_to_turnover`
- Priority: `high`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: FS `120-220 + 121-221`; net sales
- Period / Annualization: nearest FS period; `Q1*4`, `Q2*2`, `Q3*4/3`, `YE*1`
- Calculation Logic: `(trade_receivables + notes_receivable) / annualized_net_sales`
- Direction: `increase_is_risk`
- Refresh Frequency: `quarterly_or_annual`
- Freshness Rule: FS > 12 ay ise kayit yok
- Applicable Segments: bilancolu musteri
- Model Treatment: bilancosuz ise `NA`
- Companion Rules / Filters: `trade_receivables_over_1_flag`
- Recommended History Features: `_current`, `_delta_1`, `_trend_slope_6`
- Population Reference: `no`

## Sayfa 5

### Bilanco Karlilik ve Ciro

- Canonical Name: `profitability_to_turnover`
- Priority: `medium`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: FS net profit; net sales
- Period / Annualization: nearest FS period; annualized
- Calculation Logic: `net_profit / annualized_net_sales`
- Direction: `decrease_is_risk`
- Refresh Frequency: `quarterly_or_annual`
- Freshness Rule: FS > 12 ay ise kayit yok
- Applicable Segments: bilancolu musteri
- Model Treatment: bilancosuz ise `NA`
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`, `_self_zscore_6`
- Population Reference: `no`

### Musterinin Aylik Elektrik Fatura Tutarlari (NACE imalat)

- Canonical Name: `electricity_bill_amount_change`
- Priority: `low`
- Portfolio Scope: `segment_specific`
- Model Placement: `segment_specific`
- Source / Accounts: utility invoice source / abone no
- Period / Annualization: ceyreklik toplu sorgu, abone bazli tutar; aylik/cari degisim
- Calculation Logic: elektrik fatura toplami current vs prior
- Direction: `decrease_is_risk_but_ambiguous`
- Refresh Frequency: `monthly`
- Freshness Rule: `>60 gun stale ise enrichment_only`
- Applicable Segments: `nace_section = C`
- Model Treatment: applicability yoksa `NA`
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`, `_self_zscore_6`
- Population Reference: `no`

### Musterinin Bankalardaki Isletme Borclari ve Enflasyon Iliskisi

- Canonical Name: `business_loan_vs_inflation`
- Priority: `medium`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: Memzuc 0-24 ay isletme kredileri; inflation series
- Period / Annualization: aylik
- Calculation Logic: kredi buyume hizi vs enflasyon
- Direction: `increase_is_risk`
- Refresh Frequency: `monthly`
- Freshness Rule: `>60 gun stale ise enrichment_only`
- Applicable Segments: isletme kredisi olan musteri
- Model Treatment: urun yoksa `NA`
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`, `_trend_slope_6`
- Population Reference: `no`

### Bilanco Ozkaynak Bilgisi

- Canonical Name: `equity_change`
- Priority: `medium`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: FS equity
- Period / Annualization: nearest FS period
- Calculation Logic: current equity vs prior period
- Direction: `decrease_is_risk`
- Refresh Frequency: `quarterly_or_annual`
- Freshness Rule: FS > 12 ay ise kayit yok
- Applicable Segments: bilancolu musteri
- Model Treatment: bilancosuz ise `NA`
- Companion Rules / Filters: `equity_negative_flag`
- Recommended History Features: `_current`, `_delta_1`, `_trend_slope_6`
- Population Reference: `no`

## Sayfa 6

### Musterinin Bankamizda veya Diger Bankalarda Gecikmeye Girme Degiskeni

- Canonical Name: `delinquency_entry_or_frequency`
- Priority: `high`
- Portfolio Scope: `general`
- Model Placement: `broad_core`
- Source / Accounts: bank internal + external delinquency signals
- Period / Annualization: son 1-2 yil lookback
- Calculation Logic: delinquency event count / indicator
- Direction: `increase_is_risk`
- Refresh Frequency: `weekly_or_monthly`
- Freshness Rule: `>30 gun quality flag`
- Applicable Segments: tum Ticari Orta
- Model Treatment: event absence halinde `0` verilebilir
- Companion Rules / Filters: `delinquency_first_time_24m`, `delinquency_consecutive_2m`, `delinquency_frequency_3_of_12m`
- Recommended History Features: `_current`, `_delta_1`, `_rolling_mean_6`
- Population Reference: `no`

### Musterinin Bankamizdaki Kredi Karti Borcunun Tamamini Odememesi

- Canonical Name: `credit_card_full_payment_break`
- Priority: `high`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: bank internal card statement/payment
- Period / Annualization: aylik
- Calculation Logic: full payment break count / ratio
- Direction: `increase_is_risk`
- Refresh Frequency: `monthly`
- Freshness Rule: `>45 gun caution`
- Applicable Segments: kredi karti olan musteri
- Model Treatment: event absence halinde `0` verilebilir
- Companion Rules / Filters: `card_full_payment_break_first_time_24m`, `card_full_payment_break_consecutive_2m`
- Recommended History Features: `_current`, `_delta_1`, `_rolling_mean_6`
- Population Reference: `no`

### TFRS Davranis Temerrut Olasiligi

- Canonical Name: `ifrs9_behavioral_pd`
- Priority: `high`
- Portfolio Scope: `general`
- Model Placement: `broad_core`
- Source / Accounts: IFRS9 / TFRS PD
- Period / Annualization: aylik
- Calculation Logic: monthly behavioral PD
- Direction: `increase_is_risk`
- Refresh Frequency: `monthly`
- Freshness Rule: `>45 gun caution; >90 gun enrichment_only`
- Applicable Segments: coverage olan tum Ticari Orta
- Model Treatment: operational missing halinde quality flag ile kullanilir
- Companion Rules / Filters: `pd_below_2pct_exclusion_rule`
- Recommended History Features: `_current`, `_delta_1`, `_self_zscore_6`
- Population Reference: `no`

### KKB Ticari Kredi Notu

- Canonical Name: `kkb_commercial_score`
- Priority: `high`
- Portfolio Scope: `general`
- Model Placement: `broad_core`
- Source / Accounts: KKB
- Period / Annualization: sorgu bazli / aylik
- Calculation Logic: current score and change
- Direction: `decrease_is_risk`
- Refresh Frequency: `monthly_or_query_based`
- Freshness Rule: `>60 gun caution`
- Applicable Segments: KKB coverage olan tum Ticari Orta
- Model Treatment: operational missing halinde quality flag ile kullanilir
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`, `_self_zscore_6`
- Population Reference: `no`

### KKB Ticari Borcluluk Endeksi

- Canonical Name: `kkb_indebtedness_index`
- Priority: `high`
- Portfolio Scope: `general`
- Model Placement: `broad_core`
- Source / Accounts: KKB
- Period / Annualization: sorgu bazli / aylik
- Calculation Logic: current index and change
- Direction: `increase_is_risk`
- Refresh Frequency: `monthly_or_query_based`
- Freshness Rule: `>60 gun caution`
- Applicable Segments: KKB coverage olan tum Ticari Orta
- Model Treatment: operational missing halinde quality flag ile kullanilir
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`, `_self_zscore_6`
- Population Reference: `no`

## Sayfa 6-7

### Net Satis (Ciro)

- Canonical Name: `net_sales_change`
- Priority: `medium`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: FS net sales
- Period / Annualization: nearest FS period; `Q1*4`, `Q2*2`, `Q3*4/3`, `YE*1`; sektor normalize
- Calculation Logic: annualized net sales current vs comparable prior
- Direction: `decrease_is_risk_but_ambiguous`
- Refresh Frequency: `quarterly_or_annual`
- Freshness Rule: FS > 12 ay ise kayit yok
- Applicable Segments: bilancolu musteri
- Model Treatment: bilancosuz ise `NA`
- Companion Rules / Filters: Yap-Sat / Taahhut sector normalization
- Recommended History Features: `_current`, `_delta_1`, `_trend_slope_6`
- Population Reference: `yes` (`_population_percentile`, `_vs_population_median_ratio`)

## Sayfa 7

### Memzuc Limit Azalisi

- Canonical Name: `memzuc_limit_change`
- Priority: `low`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: Memzuc toplam limit
- Period / Annualization: aylik, 3 yil lookback
- Calculation Logic: current total limit vs prior
- Direction: `bidirectional`
- Refresh Frequency: `monthly`
- Freshness Rule: `>60 gun stale ise enrichment_only`
- Applicable Segments: Memzuc coverage olan musteri
- Model Treatment: partial `NA` olabilir
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`
- Population Reference: `no`

### Memzuc Banka Sayisi Azalisi

- Canonical Name: `memzuc_bank_count_change`
- Priority: `low`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: Memzuc banka sayisi
- Period / Annualization: aylik, 3 yil lookback
- Calculation Logic: current bank count vs prior
- Direction: `bidirectional`
- Refresh Frequency: `monthly`
- Freshness Rule: `>60 gun stale ise enrichment_only`
- Applicable Segments: Memzuc coverage olan musteri
- Model Treatment: partial `NA` olabilir
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`
- Population Reference: `no`

### Elektrik Borcunu Odeyemeyenler

- Canonical Name: `electricity_payment_failure`
- Priority: `low`
- Portfolio Scope: `segment_specific`
- Model Placement: `segment_specific`
- Source / Accounts: utility payment events
- Period / Annualization: aylik/event
- Calculation Logic: unpaid/late electricity bill event count
- Direction: `increase_is_risk`
- Refresh Frequency: `monthly_or_event_based`
- Freshness Rule: `>60 gun stale ise enrichment_only`
- Applicable Segments: elektrik abonesi / tuketimi izlenebilir alt portfoy
- Model Treatment: event absence halinde `0` verilebilir
- Companion Rules / Filters: `electricity_payment_failure_first_time_24m`, `electricity_payment_failure_consecutive_2m`
- Recommended History Features: `_current`, `_delta_1`
- Population Reference: `no`

### KKB Cek Portfoy Kalitesinde Bozulma

- Canonical Name: `kkb_check_portfolio_quality_deterioration`
- Priority: `low`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: KKB cek raporu (hamili/ciranta, 1/3/12 ay)
- Period / Annualization: sorgu bazli
- Calculation Logic: ibrazinda odeme orani / kalite metrigi bozulmasi
- Direction: `decrease_is_risk`
- Refresh Frequency: `monthly_or_query_based`
- Freshness Rule: `>60 gun stale ise enrichment_only`
- Applicable Segments: cek raporu coverage olan musteri
- Model Treatment: partial `NA` olabilir
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`
- Population Reference: `no`

## Sayfa 8

### Memzuc Limit Doluluk Artisi

- Canonical Name: `memzuc_limit_utilization_increase`
- Priority: `medium`
- Portfolio Scope: `conditional_general`
- Model Placement: `broad_core`
- Source / Accounts: Memzuc toplam risk, toplam limit
- Period / Annualization: aylik, 3 yil lookback
- Calculation Logic: `total_risk / total_limit` ve artis
- Direction: `increase_is_risk`
- Refresh Frequency: `monthly`
- Freshness Rule: `>45 gun caution; >90 gun enrichment_only`
- Applicable Segments: `segment in (Ticari_Orta, Buyuk)` ve `nace_main != Tarim`
- Model Treatment: applicability yoksa `NA`
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`, `_self_zscore_6`
- Population Reference: `no`

### Hayvan Sayisi

- Canonical Name: `livestock_count_change`
- Priority: `low`
- Portfolio Scope: `segment_specific`
- Model Placement: `segment_specific`
- Source / Accounts: Turkvet
- Period / Annualization: rating raporu sonrasi ilk Turkvet sorgusu baseline; sonraki sorgularla karsilastirma
- Calculation Logic: baseline veya sonraki maksimum hayvan sayisina gore azalis
- Direction: `decrease_is_risk_but_ambiguous`
- Refresh Frequency: `monthly_or_query_based`
- Freshness Rule: `>90 gun stale ise enrichment_only`
- Applicable Segments: `nace_main = Tarim` ve `nace_sub = Hayvancilik`
- Model Treatment: applicability yoksa `NA`
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`, `_trend_slope_6`
- Population Reference: `no`

## Sayfa 8-9

### Kesideci Olumsuzlugu ve Bilanco Ticari Alacak Iliskisi

- Canonical Name: `issuer_adverse_to_receivables`
- Priority: `low`
- Portfolio Scope: `segment_specific`
- Model Placement: `segment_specific`
- Source / Accounts: problematic issuer amount; FS `120-220 + 121-221`
- Period / Annualization: latest valid FS within 12m
- Calculation Logic: `problematic_issuer_amount / total_trade_receivables`
- Direction: `increase_is_risk`
- Refresh Frequency: `event_based`
- Freshness Rule: son 12 ayda mali veri yoksa kayit yok
- Applicable Segments: `segment in (Ticari_Orta, KOBI, Mikro)` ve `nace_main != Tarim`
- Model Treatment: event absence halinde `0` verilebilir
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`
- Population Reference: `no`

## Sayfa 9

### Supheli Ticari Alacaklar ve Ticari Alacak Iliskisi

- Canonical Name: `suspicious_receivables_to_receivables`
- Priority: `medium`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: FS `128-228`; `120-220 + 121-221`
- Period / Annualization: nearest FS period
- Calculation Logic: `suspicious_trade_receivables / total_trade_receivables`
- Direction: `increase_is_risk`
- Refresh Frequency: `quarterly_or_annual`
- Freshness Rule: FS > 12 ay ise kayit yok
- Applicable Segments: bilancolu musteri
- Model Treatment: bilancosuz ise `NA`
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`, `_trend_slope_6`
- Population Reference: `no`

### KKB Ileri Vadeli Cek ve Bilanco Senetli Borc / Verilen Cek Iliskisi

- Canonical Name: `forward_check_to_notes_payable_ratio`
- Priority: `medium`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: KKB 3 aylik ileri vadeli cek; FS `321-421 + 103`
- Period / Annualization: nearest valid FS within 12m
- Calculation Logic: `forward_dated_check_amount / (notes_payable + issued_checks)`
- Direction: `increase_is_risk`
- Refresh Frequency: `quarterly_or_query_based`
- Freshness Rule: son 12 ayda mali veri yoksa kayit yok
- Applicable Segments: cek/senet yogun musteri
- Model Treatment: applicability yoksa `NA`
- Companion Rules / Filters: `forward_check_above_1_flag`
- Recommended History Features: `_current`, `_delta_1`, `_self_zscore_6`
- Population Reference: `no`

## Sayfa 10

### Bankamizdaki Cek/Senet Iade Orani

- Canonical Name: `returned_check_note_ratio`
- Priority: `medium`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: TACRCSIE
- Period / Annualization: son 1 yil olaylari
- Calculation Logic: `returned_instrument_amount / total_processed_instrument_amount`
- Direction: `increase_is_risk`
- Refresh Frequency: `weekly_or_monthly`
- Freshness Rule: `>30 gun caution`
- Applicable Segments: cek/senet isleyen musteri
- Model Treatment: event absence halinde `0` verilebilir
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`, `_rolling_mean_6`
- Population Reference: `no`

### Bankamizda Bulunan Mevduat / Varlik Ortalamalari

- Canonical Name: `bank_asset_average_change`
- Priority: `medium`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: bank internal mevduat/varlik bakiyeleri
- Period / Annualization: periyodik ortalamalar, 2 yil lookback
- Calculation Logic: current average asset/deposit balance vs prior
- Direction: `decrease_is_risk_but_ambiguous`
- Refresh Frequency: `weekly_or_monthly`
- Freshness Rule: `>30 gun caution; >60 gun enrichment_only`
- Applicable Segments: bankada mevduat/varlik urunu olan musteri
- Model Treatment: urun yoksa `NA`
- Companion Rules / Filters: yok
- Recommended History Features: `_current`, `_delta_1`, `_trend_slope_6`
- Population Reference: `no`

## Sayfa 10-11

### Alacak Sigortasi Tazmin Verisi ve Bilanco Ticari Alacak Iliskisi

- Canonical Name: `insurance_claim_to_receivables`
- Priority: `medium`
- Portfolio Scope: `conditional_general`
- Model Placement: `conditional_core`
- Source / Accounts: insurance paid claim amount; FS `120-220 + 121-221`
- Period / Annualization: latest valid FS within 12m
- Calculation Logic: `insurance_paid_claim_amount / total_trade_receivables`
- Direction: `increase_is_risk`
- Refresh Frequency: `quarterly_or_event_based`
- Freshness Rule: son 12 ayda mali veri yoksa kayit yok
- Applicable Segments: alacak sigortasi kullanan, bilancolu, tarim disi, Ticari Orta ve alti musteri
- Model Treatment: event absence halinde `0` verilebilir
- Companion Rules / Filters: `insurance_claim_below_15pct_filter`
- Recommended History Features: `_current`, `_delta_1`, `_trend_slope_6`
- Population Reference: `no`
