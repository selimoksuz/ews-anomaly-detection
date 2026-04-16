# Business Traceability Matrix

Bu dokuman, is biriminin 11 sayfalik talep dokumanindaki business basliklarini dictionary kontratiyla birebir eslestirir.

## 1. Business Heading -> Dictionary Mapping

| ID | Sayfa | Business Basligi | Canonical Name | Dictionary Heading | Table Coverage |
|---|---|---|---|---|---|
| 1 | 1 | Banka Borclulugu ve Ciro Iliskisi | `bank_debt_to_turnover` | `Sayfa 1 > Banka Borclulugu ve Ciro Iliskisi` | `full` |
| 2 | 1 | Musterinin Tum Bankalardaki Aylik POS Hacmi | `pos_volume_change` | `Sayfa 1 > Musterinin Tum Bankalardaki Aylik POS Hacmi` | `full` |
| 3 | 2 | Haciz Tutari ve Ciro Iliskisi | `seizure_amount_to_turnover` | `Sayfa 2 > Haciz Tutari ve Ciro Iliskisi` | `full` |
| 4 | 3 | Cek Odemesi Zamani | `check_payment_time_shift` | `Sayfa 3 > Cek Odemesi Zamani` | `full` |
| 5 | 3 | Faktoring | `factoring_risk_presence` | `Sayfa 3 > Faktoring` | `full` |
| 6 | 3 | Banka Borclulugu ve EBITDA Iliskisi | `bank_debt_to_ebitda` | `Sayfa 3 > Banka Borclulugu ve EBITDA Iliskisi` | `full` |
| 7 | 4 | Bilanco Ticari Alacak ve Ciro | `trade_receivables_to_turnover` | `Sayfa 4 > Bilanco Ticari Alacak ve Ciro` | `full` |
| 8 | 5 | Bilanco Karlilik ve Ciro | `profitability_to_turnover` | `Sayfa 5 > Bilanco Karlilik ve Ciro` | `full` |
| 9 | 5 | Musterinin Aylik Elektrik Fatura Tutarlari (NACE imalat) | `electricity_bill_amount_change` | `Sayfa 5 > Musterinin Aylik Elektrik Fatura Tutarlari (NACE imalat)` | `full` |
| 10 | 5 | Musterinin Bankalardaki Isletme Borclari ve Enflasyon Iliskisi | `business_loan_vs_inflation` | `Sayfa 5 > Musterinin Bankalardaki Isletme Borclari ve Enflasyon Iliskisi` | `full` |
| 11 | 5 | Bilanco Ozkaynak Bilgisi | `equity_change` | `Sayfa 5 > Bilanco Ozkaynak Bilgisi` | `full` |
| 12 | 6 | Musterinin Bankamizda veya Diger Bankalarda Gecikmeye Girme Degiskeni | `delinquency_entry_or_frequency` | `Sayfa 6 > Musterinin Bankamizda veya Diger Bankalarda Gecikmeye Girme Degiskeni` | `full` |
| 13 | 6 | Musterinin Bankamizdaki Kredi Karti Borcunun Tamamini Odememesi | `credit_card_full_payment_break` | `Sayfa 6 > Musterinin Bankamizdaki Kredi Karti Borcunun Tamamini Odememesi` | `full` |
| 14 | 6 | TFRS Davranis Temerrut Olasiligi | `ifrs9_behavioral_pd` | `Sayfa 6 > TFRS Davranis Temerrut Olasiligi` | `full` |
| 15 | 6 | KKB Ticari Kredi Notu | `kkb_commercial_score` | `Sayfa 6 > KKB Ticari Kredi Notu` | `full` |
| 16 | 6 | KKB Ticari Borcluluk Endeksi | `kkb_indebtedness_index` | `Sayfa 6 > KKB Ticari Borcluluk Endeksi` | `full` |
| 17 | 6-7 | Net Satis (Ciro) | `net_sales_change` | `Sayfa 6-7 > Net Satis (Ciro)` | `full` |
| 18 | 7 | Memzuc Limit Azalisi | `memzuc_limit_change` | `Sayfa 7 > Memzuc Limit Azalisi` | `full` |
| 19 | 7 | Memzuc Banka Sayisi Azalisi | `memzuc_bank_count_change` | `Sayfa 7 > Memzuc Banka Sayisi Azalisi` | `full` |
| 20 | 7 | Elektrik Borcunu Odeyemeyenler | `electricity_payment_failure` | `Sayfa 7 > Elektrik Borcunu Odeyemeyenler` | `full` |
| 21 | 7 | KKB Cek Portfoy Kalitesinde Bozulma | `kkb_check_portfolio_quality_deterioration` | `Sayfa 7 > KKB Cek Portfoy Kalitesinde Bozulma` | `full` |
| 22 | 8 | Memzuc Limit Doluluk Artisi | `memzuc_limit_utilization_increase` | `Sayfa 8 > Memzuc Limit Doluluk Artisi` | `full` |
| 23 | 8 | Hayvan Sayisi | `livestock_count_change` | `Sayfa 8 > Hayvan Sayisi` | `full` |
| 24 | 8-9 | Kesideci Olumsuzlugu ve Bilanco Ticari Alacak Iliskisi | `issuer_adverse_to_receivables` | `Sayfa 8-9 > Kesideci Olumsuzlugu ve Bilanco Ticari Alacak Iliskisi` | `full` |
| 25 | 9 | Supheli Ticari Alacaklar ve Ticari Alacak Iliskisi | `suspicious_receivables_to_receivables` | `Sayfa 9 > Supheli Ticari Alacaklar ve Ticari Alacak Iliskisi` | `full` |
| 26 | 9 | KKB Ileri Vadeli Cek ve Bilanco Senetli Borc / Verilen Cek Iliskisi | `forward_check_to_notes_payable_ratio` | `Sayfa 9 > KKB Ileri Vadeli Cek ve Bilanco Senetli Borc / Verilen Cek Iliskisi` | `full` |
| 27 | 10 | Bankamizdaki Cek/Senet Iade Orani | `returned_check_note_ratio` | `Sayfa 10 > Bankamizdaki Cek/Senet Iade Orani` | `full` |
| 28 | 10 | Bankamizda Bulunan Mevduat / Varlik Ortalamalari | `bank_asset_average_change` | `Sayfa 10 > Bankamizda Bulunan Mevduat / Varlik Ortalamalari` | `full` |
| 29 | 10-11 | Alacak Sigortasi Tazmin Verisi ve Bilanco Ticari Alacak Iliskisi | `insurance_claim_to_receivables` | `Sayfa 10-11 > Alacak Sigortasi Tazmin Verisi ve Bilanco Ticari Alacak Iliskisi` | `full` |

## 2. Companion Rule / Filter Mapping

| Parent Feature | Business Rule / Filter | Canonical Artifact | Coverage |
|---|---|---|---|
| `factoring_risk_presence` | son 24 ayda ilk kez factoring riski | `factoring_risk_first_time_24m` | `full` |
| `factoring_risk_presence` | iki ay pes pese factoring riski | `factoring_risk_consecutive_2m` | `full` |
| `factoring_risk_presence` | son 12 ayda uc farkli ay factoring riski | `factoring_risk_frequency_3_of_12m` | `full` |
| `delinquency_entry_or_frequency` | son 24 ayda ilk kez gecikme | `delinquency_first_time_24m` | `full` |
| `delinquency_entry_or_frequency` | iki donem pes pese gecikme | `delinquency_consecutive_2m` | `full` |
| `delinquency_entry_or_frequency` | son 12 ayda uc ayrik donemde gecikme | `delinquency_frequency_3_of_12m` | `full` |
| `credit_card_full_payment_break` | son 24 ayda ilk kez tam odememe | `card_full_payment_break_first_time_24m` | `full` |
| `credit_card_full_payment_break` | iki donem pes pese tam odememe | `card_full_payment_break_consecutive_2m` | `full` |
| `check_payment_time_shift` | son 6 ayda ilk kez gec saate kayma | `check_payment_time_shift_first_time_6m` | `full` |
| `check_payment_time_shift` | son 3 ayda farkli gunlerde pes pese gec saate kayma | `check_payment_time_shift_consecutive_3m_distinct_days` | `full` |
| `electricity_payment_failure` | son 24 ayda ilk kez elektrik odememe | `electricity_payment_failure_first_time_24m` | `full` |
| `electricity_payment_failure` | iki donem pes pese elektrik odememe | `electricity_payment_failure_consecutive_2m` | `full` |
| `seizure_amount_to_turnover` | son 24 ayda ilk kez devam eden haciz kaydi | `seizure_first_time_24m` | `full` |
| `ifrs9_behavioral_pd` | PD %2 alti dislama | `pd_below_2pct_exclusion_rule` | `full` |
| `insurance_claim_to_receivables` | %15 alti dikkate alma | `insurance_claim_below_15pct_filter` | `full` |
| `forward_check_to_notes_payable_ratio` | oran 1 uzeri flag | `forward_check_above_1_flag` | `full` |
| `trade_receivables_to_turnover` | oran 1 uzeri flag | `trade_receivables_over_1_flag` | `full` |
| `equity_change` | negatif ozkaynak flag | `equity_negative_flag` | `full` |
