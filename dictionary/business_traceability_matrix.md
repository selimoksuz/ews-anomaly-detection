# Business Traceability Matrix

Bu dokumanin amaci, is biriminin 11 sayfalik ekran goruntusu talebindeki maddeleri mevcut dictionary yapisiyla birebir eslestirmek ve hangi maddelerin:

- `full`
- `partial`
- `missing`

seviyesinde karsilandigini gostermektir.

Guncel durumda hedef, business isterlerini implementation-grade dictionary kontratina tam map etmektir.

## 1. Raw / Model Feature Coverage

| ID | Sayfa | Business Maddesi | Beklenen Canonical | Dictionary Karsiligi | Coverage | Not |
|---|---|---|---|---|---|---|
| 1 | 1 | Banka Borclulugu ve Ciro Iliskisi | `bank_debt_to_turnover` | Var | `full` | exact source, annualization ve freshness kuralina baglandi |
| 2 | 1 | Tum Bankalardaki Aylik POS Hacmi | `pos_volume_change` | Var | `full` | faaliyet grubu / seasonality ve peer kullanimi ayrica tanimli |
| 3 | 2 | Haciz Tutari ve Ciro Iliskisi | `seizure_amount_to_turnover` | Var | `full` | raw feature + companion rule tanimli |
| 4 | 3 | Cek Odemesi Zamani | `check_payment_time_shift` | Var | `full` | raw feature + exact companion rules tanimli |
| 5 | 3 | Faktoring | `factoring_risk_presence` | Var | `full` | raw feature + 3 companion rule tanimli |
| 6 | 3-4 | Banka Borclulugu ve EBITDA Iliskisi | `bank_debt_to_ebitda` | Var | `full` | TLREF, annualization, freshness encode edildi |
| 7 | 4 | Bilanco Ticari Alacak ve Ciro | `trade_receivables_to_turnover` | Var | `full` | account codes ve `>1` threshold flag tanimli |
| 8 | 5 | Bilanco Karlilik ve Ciro | `profitability_to_turnover` | Var | `full` | annualization ve stale kuralina baglandi |
| 9 | 5 | Aylik Elektrik Fatura Tutarlari | `electricity_bill_amount_change` | Var | `full` | exact `nace_section = C` applicability yazildi |
| 10 | 5 | Bankalardaki Isletme Borclari ve Enflasyon | `business_loan_vs_inflation` | Var | `full` | source ve hesap mantigi yazildi |
| 11 | 5 | Bilanco Ozkaynak | `equity_change` | Var | `full` | `equity_negative_flag` eklendi |
| 12 | 6 | Bankamizda veya Diger Bankalarda Gecikmeye Girme | `delinquency_entry_or_frequency` | Var | `full` | raw feature + 3 companion rule var |
| 13 | 6 | Bankamizdaki Kredi Karti Borcunun Tamamini Odememesi | `credit_card_full_payment_break` | Var | `full` | raw feature + 2 companion rule var |
| 14 | 6 | TFRS Davranis Temerrut Olasiligi | `ifrs9_behavioral_pd` | Var | `full` | `%2 alti dislama` rule'u var |
| 15 | 6 | KKB Ticari Kredi Notu | `kkb_commercial_score` | Var | `full` | source ve treatment net |
| 16 | 6 | KKB Ticari Borcluluk Endeksi | `kkb_indebtedness_index` | Var | `full` | source ve treatment net |
| 17 | 6-7 | Net Satis (Ciro) | `net_sales_change` | Var | `full` | annualization + Yap-Sat/Taahhut normalization yazildi |
| 18 | 7 | Memzuc Limit Azalisi | `memzuc_limit_change` | Var | `full` | business trace icin label'da azalisi korundu, canonical change olarak aciklandi |
| 19 | 7 | Memzuc Banka Sayisi Azalisi | `memzuc_bank_count_change` | Var | `full` | business trace icin label'da azalisi korundu, canonical change olarak aciklandi |
| 20 | 7 | Elektrik Borcunu Odeyemeyenler | `electricity_payment_failure` | Var | `full` | raw feature + companion rules + applicability yazildi |
| 21 | 7 | KKB Cek Portfoy Kalitesinde Bozulma | `kkb_check_portfolio_quality_deterioration` | Var | `full` | source ve sorgu mantigi tanimli |
| 22 | 8 | Memzuc Limit Doluluk Artisi | `memzuc_limit_utilization_increase` | Var | `full` | `Ticari Orta/Buyuk`, `tarim disi` applicability yazildi |
| 23 | 8 | Hayvan Sayisi | `livestock_count_change` | Var | `full` | Turkvet baseline mantigi yazildi |
| 24 | 8-9 | Kesideci Olumsuzlugu ve Bilanco Ticari Alacak | `issuer_adverse_to_receivables` | Var | `full` | `Ticari Orta ve alti`, `tarim disi`, `12 ay FS` kuralina baglandi |
| 25 | 9 | Supheli Ticari Alacaklar ve Ticari Alacak | `suspicious_receivables_to_receivables` | Var | `full` | account codes ve formula yazildi |
| 26 | 9-10 | KKB Ileri Vadeli Cek ve Bilanco Senetli Borc / Verilen Cek | `forward_check_to_notes_payable_ratio` | Var | `full` | nearest FS, 3 aylik KKB ve `>1` flag tanimli |
| 27 | 10 | Bankamizdaki Cek/Senet Iade Orani | `returned_check_note_ratio` | Var | `full` | business label ve canonical eslesmesi net |
| 28 | 10 | Bankamizda Bulunan Mevduat/Varlik Ortalamalari | `bank_asset_average_change` | Var | `full` | urun sahipligi, refresh ve treatment net |
| 29 | 10-11 | Alacak Sigortasi Tazmin ve Bilanco Ticari Alacak | `insurance_claim_to_receivables` | Var | `full` | `%15` filter, 12 ay FS, applicability yazildi |

## 2. Companion Rule Coverage

| Parent Feature | Business Rule | Canonical | Coverage | Not |
|---|---|---|---|---|
| `factoring_risk_presence` | Son 24 ayda ilk kez factoring riski | `factoring_risk_first_time_24m` | `full` | mevcut |
| `factoring_risk_presence` | Iki ay pes pese factoring riski | `factoring_risk_consecutive_2m` | `full` | mevcut |
| `factoring_risk_presence` | Son 12 ayda 3 farkli ay factoring riski | `factoring_risk_frequency_3_of_12m` | `full` | eklendi |
| `delinquency_entry_or_frequency` | Son 24 ayda ilk kez gecikme | `delinquency_first_time_24m` | `full` | mevcut |
| `delinquency_entry_or_frequency` | Iki donem pes pese gecikme | `delinquency_consecutive_2m` | `full` | mevcut |
| `delinquency_entry_or_frequency` | Son 12 ayda 3 ayrik donem gecikme | `delinquency_frequency_3_of_12m` | `full` | mevcut |
| `credit_card_full_payment_break` | Son 24 ayda ilk kez tam odememe | `card_full_payment_break_first_time_24m` | `full` | mevcut |
| `credit_card_full_payment_break` | Iki donem pes pese tam odememe | `card_full_payment_break_consecutive_2m` | `full` | mevcut |
| `check_payment_time_shift` | Son 6 ayda ilk kez gec saate kayma | `check_payment_time_shift_first_time_6m` | `full` | mevcut |
| `check_payment_time_shift` | Son 3 ayda farkli gunlerde pes pese gec saate kayma | `check_payment_time_shift_consecutive_3m_distinct_days` | `full` | canonical business wording'e yaklastirildi |
| `electricity_payment_failure` | Son 24 ayda ilk kez odememe | `electricity_payment_failure_first_time_24m` | `full` | mevcut |
| `electricity_payment_failure` | Iki donem pes pese odememe | `electricity_payment_failure_consecutive_2m` | `full` | mevcut |
| `seizure_amount_to_turnover` | Son 24 ayda ilk kez devam eden haciz | `seizure_first_time_24m` | `full` | eklendi |

## 3. Value Filter / Applicability / Formula Coverage

| Rule Family | Coverage | Not |
|---|---|---|
| Annualization `Q1*4`, `Q2*2`, `Q3*4/3`, `YE*1` | `full` | global rule ve variable-level period alanlarina islendi |
| FS stale rule (12 ay) | `full` | variable bazli freshness rule'lara islendi |
| Sector normalization | `full` | POS, net sales, imalat, tarim/hayvancilik kurallari yazildi |
| PD threshold `%2` | `full` | `pd_below_2pct_exclusion_rule` var |
| Insurance claim `%15` | `full` | `insurance_claim_below_15pct_filter` var |
| Forward-check `>1` | `full` | `forward_check_above_1_flag` var |
| Trade receivables `>1` | `full` | `trade_receivables_over_1_flag` var |
| Equity negative sign-flip | `full` | `equity_negative_flag` var |
| POS peer comparison | `full` | sadece `pos_volume_change` icin population-reference acildi |
| Net sales sector normalization | `full` | `net_sales_change` icin yazildi |
| Memzuc limit utilization applicability | `full` | `Ticari Orta/Buyuk`, `tarim disi` yazildi |
| Livestock applicability | `full` | `Tarim + Hayvancilik + Turkvet baseline` yazildi |
| Issuer adverse applicability | `full` | `Ticari Orta ve alti`, `tarim disi`, `12 ay FS` yazildi |
| Electricity bill applicability | `full` | `nace_section = C` yazildi |
| Insurance claim FS freshness/applicability | `full` | `12 ay FS`, bilancolu, tarim disi, alacak sigortali musteri yazildi |

## 4. Overall Assessment

Guncel durumda:

- 29 business feature basliginin tamami dictionary'de exact canonical, formula, source, freshness ve applicability bilgisiyle karsilanmistir.
- Companion rule seti business notundaki maddelerle hizalanmistir.
- Generic threshold/filter mantigi kaldirilmis, feature-spesifik kurallara donusturulmustur.
- Population-reference politikasi tek karara indirilmistir:
  - `yes`: `pos_volume_change`, `net_sales_change`
  - digerleri: `no`

Bu nedenle guncel dictionary icin net karar:

- `screening / classification` icin yeterlidir
- `implementation-grade technical contract` olarak kullanilabilir

## 5. Remaining Caveats

### 5.1 Implementation Backlog (Runtime Koda Tasinma)

- Bu sozlukte tanimlanan exact source-period-rule mantigi henuz runtime koda tasinmamis olabilir.
- Yani dictionary tamamlansa da uygulama ayri backlog ister.
- Gerekli yeni moduller: `engine/annualization.py`, `engine/applicability.py`, `engine/history_features.py`, `engine/peer_features.py`, `engine/rules.py`
- Gerekli config: `config/pipeline_config_ticari_orta.yaml`, `business/feature_family_map.yaml`

### 5.2 Business Tarafi Onay Bekleyen Noktalar

Dictionary'nin `Open Questions` bolumunde listelenen maddeler:

- `tlref_factor` kaynak ve guncelleme sikligi
- `peer_group_key` tanimi (`nace_section + segment` vs `nace_main + segment`)
- `min_peer_size` esik degeri
- `kkb_check_portfolio_quality_deterioration` icin alt bucket secimi
- `memzuc_bank_count_change` yon semantigi
- `bank_asset_average_change` periyot tanimi
- Taahhut "duzeltilmis FS" kaynak tablosu
- `returned_check_note_ratio` otoriter kaynak (TACRCSIE vs bank internal)
- Annualization carpanlarinin sektor bazinda override durumu

### 5.3 Coverage Tutarliligi

- Business analysis 11 sayfa envanter: 29 canonical.
- Business analysis Faz 1=11, Faz 2=14 (13 + `factoring_risk_presence`), Faz 3=4; toplam 29.
- Dictionary Section A=6, B=20, C=4; toplam 30.
- Aradaki 1 fark: `bank_limit_utilization` (banka-ici limit doluluk), business ekranlarinda bagimsiz madde olarak yok. Business sayfa 8 madde 22 `memzuc_limit_utilization_increase` (tum bankalar / Memzuc) olarak listeliyor. Dictionary bu iki kavrami ayri tutuyor ve bank-internal versiyonu genisletme olarak ekliyor.
- Oneri: `bank_limit_utilization` business'a "business dahili isterimiz disinda teknik ek" olarak bildirilsin; onaylanirsa Section A'da kalir, onaylanmazsa Section B'ye dusurulur veya cikarilir.
- Canonical name alignment tamamlandi (`memzuc_limit_change` her iki dokumanda ayni).
