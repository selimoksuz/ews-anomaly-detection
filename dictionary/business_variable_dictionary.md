# Business Variable Dictionary

Bu sozluk, business tarafindan onerilen degiskenleri Ticari Orta segmenti icin implementation-grade teknik kontrata donusturmek amaciyla hazirlanmistir.

## Scope

- segment modele feature olarak girmez
- segment, run scope / partition mantigiyla kullanilir
- `priority`, `portfolio scope` ve `model placement` ayni sey degildir
  - `priority`: sinyalin is/risk onemi
  - `portfolio scope`: Ticari Orta'nin ne kadar genis kismina uygulanabildigi
  - `model placement`: bugunku model kurgusunda nereye yerlestirildigi

## Missing Semantics

### 1. Event Absence

- olay hic yasanmamistir
- ornek: gecikme yok, iade cek yok, factoring riski yok
- bu durumda `0` verilebilir

### 2. Structural Missing

- degisken bu musteri icin uygulanamaz
- ornek: bilancosuz musteri, POS applicability olmayan musteri, tarim disi musteri icin hayvan sayisi
- bu durumda `0` verilmez
- `NA / not_applicable` olarak ele alinir

### 3. Operational Missing

- veri olmaliydi ama gelmedi
- ETL, sorgu, kaynak problemi
- imputasyon yapilsa bile quality issue olarak izlenir

## Model Placement Definitions

- `broad_core`
  - genel modelin omurgasina dogrudan girer
- `conditional_core`
  - ayni genel modele girer ama freshness/applicability guard ister
- `segment_specific`
  - yalniz ilgili alt portfoy/submodel icinde modele girer
- `enrichment_only`
  - skora/model input'una girmez; sadece baglam/aciklama icindir
- `rule_only`
  - deterministic business rule/filter katmanina gider

## Model Treatment Definitions

Missing semantics ve model placement'tan ayri olarak, feature degerinin model icinde nasil ele alinacagini tanimlar:

- `core_with_zero_absence`
  - event absence durumunda `0` verilir ve model input'una dogrudan girer (gecikme, factoring, iade cek gibi event-based feature'lar)
- `core_with_quality_impute`
  - operasyonel missing durumunda imputasyon + `<feature>_quality_flag = 1` ile model'e girer (KKB, IFRS9 PD gibi veri coverage'i genis olanlar)
- `enrichment_only_when_structural_missing`
  - structural missing oranı yuksek oldugunda model input'u disinda kalir, yalniz baglam icin gosterilir (FS-bagimli oranlar)
- `rule_only`
  - feature olarak modele girmez; yalniz companion rule veya value filter olarak kullanilir

## Global Calculation Rules

### Annualization

FS kaynakli kumulatif kalemler icin:

- `Q1 * 4`
- `Q2 * 2`
- `Q3 * 4 / 3`
- `YE * 1`

Bu kural su ailelerde kullanilir:

- `bank_debt_to_turnover`
- `bank_debt_to_ebitda`
- `trade_receivables_to_turnover`
- `profitability_to_turnover`
- `net_sales_change`
- `suspicious_receivables_to_receivables`
- `insurance_claim_to_receivables`
- `seizure_amount_to_turnover`
- `issuer_adverse_to_receivables`
- `forward_check_to_notes_payable_ratio`

### FS Freshness

- business sert kural: son 12 ayda mali veri yoksa feature kaydi uretilmez / yok sayilir
- bu nedenle FS-based variable'larda `freshness_rule` temel referans `exclude_if_fs_older_than_12m` olarak alinmistir

### Sectoral Normalization

- `pos_volume_change`
  - faaliyet grubu / mevsimsellik kiyasi gerekir
- `net_sales_change`
  - Yap-Sat ve Taahhut icin sektor-normalizasyon gerekir
  - Taahhut sektorunde **duzeltilmis mali tablolar** (rated/audited FS) kullanilir
- `electricity_bill_amount_change`
  - yalniz `nace_section = C` (imalat)
- `livestock_count_change`
  - yalniz `nace_main = Tarim` ve `nace_sub = Hayvancilik`

### Peer / Faaliyet Grubu Tanimi

`pos_volume_change` ve `net_sales_change` icin kullanilan "faaliyet grubu / peer group" asagidaki sekilde tanimlanir:

- `peer_group_key = (nace_section, segment)` default tanim
- minimum peer grup buyuklugu `min_peer_size = 30`
- grup buyuklugu esigin altindaysa bir ust seviye (`nace_section` tek basina, sonra `segment` tek basina) fallback kullanilir
- peer aggregation donemi POS icin `aylik`, net_sales icin `en yakin FS donemi` (Q1/Q2/Q3/YE)

### TLREF Tanimi

`bank_debt_to_ebitda` hesaplamasinda kullanilan `tlref_factor`:

- TLREF = Turk Lirasi Gecelik Referans Faiz orani (Turkish Lira Overnight Reference Rate)
- `tlref_factor`, Memzuc nakdi risk tutarinin donem icinde yarattigi finansman maliyetinin yaklasik TLREF-turevli carpani olarak business tarafindan saglanir
- formul uygulamasi: `(total_cash_credit_risk_0_24m * tlref_factor) / annualized_ebitda`
- business tarafi bu carpani donemsel olarak gunceller; sabit bir katsayi degildir

### Population-Reference Policy

Acik population-reference feature uretimi tum degiskenlere yayilmaz.

- `yes`
  - `pos_volume_change`
  - `net_sales_change`
- diger tum aileler
  - `no`
  - cunku mevcut anomaly motoru population pattern'i zaten ogrenmektedir

## Feature Families

### A. Broad Core

Bu aileler coverage ve refresh acisindan Ticari Orta genel modelinin omurgasina en yakin ailelerdir:

- `bank_limit_utilization`
- `delinquency_entry_or_frequency`
- `ifrs9_behavioral_pd`
- `kkb_commercial_score`
- `kkb_indebtedness_index`
- `memzuc_limit_utilization_increase`

### B. Conditional Core

Bu aileler modele girebilir ama guard ister:

- `bank_debt_to_turnover`
- `bank_debt_to_ebitda`
- `pos_volume_change`
- `credit_card_full_payment_break`
- `trade_receivables_to_turnover`
- `profitability_to_turnover`
- `business_loan_vs_inflation`
- `equity_change`
- `net_sales_change`
- `suspicious_receivables_to_receivables`
- `insurance_claim_to_receivables`
- `seizure_amount_to_turnover`
- `forward_check_to_notes_payable_ratio`
- `kkb_check_portfolio_quality_deterioration`
- `check_payment_time_shift`
- `returned_check_note_ratio`
- `memzuc_limit_change`
- `memzuc_bank_count_change`
- `bank_asset_average_change`
- `factoring_risk_presence`

### C. Segment-Specific

Bu aileler yalniz ilgili alt portfoylerde anlamlidir:

- `electricity_bill_amount_change`
- `electricity_payment_failure`
- `livestock_count_change`
- `issuer_adverse_to_receivables`

### D. Rule-Only

Bu maddeler modele feature olarak girmez; companion rule veya value filter olarak kalir:

- `factoring_risk_first_time_24m`
- `factoring_risk_consecutive_2m`
- `factoring_risk_frequency_3_of_12m`
- `delinquency_first_time_24m`
- `delinquency_consecutive_2m`
- `delinquency_frequency_3_of_12m`
- `card_full_payment_break_first_time_24m`
- `card_full_payment_break_consecutive_2m`
- `check_payment_time_shift_first_time_6m`
- `check_payment_time_shift_consecutive_3m_distinct_days`
- `electricity_payment_failure_first_time_24m`
- `electricity_payment_failure_consecutive_2m`
- `seizure_first_time_24m`
- `pd_below_2pct_exclusion_rule`
- `insurance_claim_below_15pct_filter`
- `forward_check_above_1_flag`
- `trade_receivables_over_1_flag`
- `equity_negative_flag`

## Exact Business-Specific Applicability Rules

### `memzuc_limit_utilization_increase`

- business metnine gore:
  - `segment IN (Ticari_Orta, Buyuk)`
  - `nace_main != Tarim`
- Ticari Orta run'inda bu variable:
  - tarim disi musterilerle sinirlanir

### `livestock_count_change`

- `nace_main = Tarim`
- `nace_sub = Hayvancilik`
- veri kaynagi `Turkvet`
- baseline:
  - rating raporu sonrasi ilk Turkvet sorgusu
  - sonraki sorgularda artis varsa en yuksek hayvan sayisi esas alinabilir
  - baseline'a gore azalis risk sinyali sayilir

### `issuer_adverse_to_receivables`

- business metnine gore:
  - `segment IN (Ticari_Orta, KOBI, Mikro)`
  - `nace_main != Tarim`
  - son 12 ayda gecerli mali veri yoksa kayit uretilmez

### `electricity_bill_amount_change`

- `nace_section = C`
- elektrik tuketimi/faturasi anlamli olan imalat alt evreninde calisir

### `electricity_payment_failure`

- yalniz elektrik abonesi / tuketimi izlenebilir alt portfoyde anlamlidir

### `insurance_claim_to_receivables`

- alacak sigortasi kullanan
- bilancolu
- tarim disi
- Ticari Orta ve alti segmentler
- son 12 ayda gecerli mali veri yoksa kayit uretilmez

## Exact Threshold / Value Filters

- `ifrs9_behavioral_pd`
  - `pd < 0.02` ise kullanilmaz
- `insurance_claim_to_receivables`
  - `ratio < 0.15` ise dikkate alinmaz
- `forward_check_to_notes_payable_ratio`
  - `ratio > 1` ise risk flag yukseltilir
- `trade_receivables_to_turnover`
  - `ratio > 1` ise ek risk flag olusur
- `equity_change`
  - `equity < 0` ise negatif ozkaynak flag olusur

## Exact Rule Coverage

### `factoring_risk_presence`

- `factoring_risk_first_time_24m`
- `factoring_risk_consecutive_2m`
- `factoring_risk_frequency_3_of_12m`

### `delinquency_entry_or_frequency`

- `delinquency_first_time_24m`
- `delinquency_consecutive_2m`
- `delinquency_frequency_3_of_12m`

### `credit_card_full_payment_break`

- `card_full_payment_break_first_time_24m`
- `card_full_payment_break_consecutive_2m`

### `check_payment_time_shift`

- `check_payment_time_shift_first_time_6m`
- `check_payment_time_shift_consecutive_3m_distinct_days`

### `electricity_payment_failure`

- `electricity_payment_failure_first_time_24m`
- `electricity_payment_failure_consecutive_2m`

### `seizure_amount_to_turnover`

- `seizure_first_time_24m`

## Standard Self-History Features

| Derived Feature | Formula | Purpose |
|---|---|---|
| `_current` | cari deger | mutlak seviye |
| `_lag_1` | onceki snapshot | en yakin gecmis |
| `_delta_1` | `current - lag_1` | kisa donem fark |
| `_delta_pct_1` | `(current - lag_1) / abs(lag_1)` | oransal degisim |
| `_rolling_mean_3` | son 3 snapshot ortalamasi | kisa donem baz |
| `_rolling_mean_6` | son 6 snapshot ortalamasi | orta donem baz |
| `_rolling_median_6` | son 6 snapshot mediani | dayanikli baz |
| `_rolling_std_6` | son 6 snapshot std | musteri oynakligi |
| `_self_zscore_6` | `(current - rolling_mean_6) / rolling_std_6` | kendi normuna gore sapma |
| `_vs_6m_median_ratio` | `current / rolling_median_6` | kendi medianina gore oran |
| `_trend_slope_6` | son 6 snapshot lineer egim | bozulma/iyilesme hizi |

## Recommended Population-Reference Features

Yalnizca iki aile icin acik population-reference feature onerilir:

| Canonical Name | Population Features |
|---|---|
| `pos_volume_change` | `_population_percentile`, `_vs_population_median_ratio` |
| `net_sales_change` | `_population_percentile`, `_vs_population_median_ratio` |

## Phase Roadmap

Business analysis dokumaniyla bire bir hizalanmistir.

### Faz 1 (Birincil, 11 aile)

Davranissal veya risk-state temelli, yuksek oncelikli:

- `bank_debt_to_turnover`
- `bank_debt_to_ebitda`
- `pos_volume_change`
- `delinquency_entry_or_frequency`
- `credit_card_full_payment_break`
- `ifrs9_behavioral_pd`
- `kkb_commercial_score`
- `kkb_indebtedness_index`
- `returned_check_note_ratio`
- `check_payment_time_shift`
- `memzuc_limit_utilization_increase` (segment filter sonrasinda)

### Faz 2 (Ikinci seviye, 14 aile)

Coverage / freshness / sektor normalizasyonu gerektiren:

- `trade_receivables_to_turnover`
- `suspicious_receivables_to_receivables`
- `profitability_to_turnover`
- `equity_change`
- `net_sales_change` (sektor normalization zorunlu)
- `insurance_claim_to_receivables`
- `seizure_amount_to_turnover`
- `forward_check_to_notes_payable_ratio`
- `business_loan_vs_inflation`
- `memzuc_limit_change`
- `memzuc_bank_count_change`
- `kkb_check_portfolio_quality_deterioration`
- `bank_asset_average_change` (urun sahipligi kosullu)
- `factoring_risk_presence` (companion rule'lari ile)

### Faz 3 (Segment-ozel, 4 aile)

Alt-portfoye ozel:

- `electricity_bill_amount_change` (NACE imalat)
- `electricity_payment_failure` (elektrik abonesi)
- `livestock_count_change` (tarim/hayvancilik + Turkvet)
- `issuer_adverse_to_receivables` (Ticari Orta ve alti, tarim disi)

### Ek (Business Onay Bekleyen)

Bu madde 11 sayfalik business ekran goruntusu envanterinde yok; teknik genisleme olarak oneriliyor:

- `bank_limit_utilization` (banka-ici limit doluluk; Memzuc versiyonundan ayri tutulan internal metric)

## Pipeline Integration Order

1. Oracle input'tan ham degiskenleri al
2. FS-based ailelerde donem secimi + annualization uygula
3. applicability filtrelerini uygula
4. companion rules ve value filters'i hesapla
5. self-history turevlerini uret
6. yalniz `pos_volume_change` ve `net_sales_change` icin population-reference turevleri uret
7. sonra mevcut pipeline'a gir:
   - missing handling
   - hard bounds
   - categorical handling
   - feature selection
   - AE / IF / MD
   - calibration
   - scoring

## Open Questions (Business Tarafina)

Bu maddeler dokumantasyon disi, operasyonel karar bekler:

- `tlref_factor` donemsel guncelleme sikligi ve kaynak dosyasi (hangi internal rate table)
- `peer_group_key` icin `nace_section + segment` tanimi onaylanmis mi, yoksa `nace_main + segment` mi tercih ediliyor
- `min_peer_size = 30` esik degeri business tarafindan teyit edilecek
- `kkb_check_portfolio_quality_deterioration` metriginin hangi alt bucket'inin (hamili/ciranta, 1/3/12 ay) bozulma referansi olarak alinacagi
- `memzuc_bank_count_change` icin yon belirsizligi (azalis = iliski sadelestirme mi, sikintili kapanis mi)
- `bank_asset_average_change` icin "periyodik ortalama" pencere boyu (haftalik mi aylik mi)
- Taahhut sektoru icin "duzeltilmis FS" source table'inin FS standardindan farki
- `returned_check_note_ratio` kaynagi: business notunda TACRCSIE geciyor, ancak "Bankamizdaki" ifadesi bank internal'e isaret ediyor — hangisi otoriter
- Annualization ceyrek carpanlarinin (Q1×4, Q2×2, Q3×4/3) sektor bazinda degistirilmesi gerekiyor mu
