# Business Scope

Kaynak: Is biriminin 2026-04-14 tarihinde paylastigi 11 sayfalik degisken talep dokumani.

Bu dokuman yalnizca business kapsamini toplar:
- hangi degisken ailelerinin istendigi
- bu degiskenler icin ortak is kurallari
- hangi basliklarin hangi sayfada ve hangi oncelikte geldigi

Bu dokuman teknik backlog, implementasyon notu veya acik soru listesi icermez.

## 1. Business Oncelik Prensibi

- Is biriminin gonderdigi dokumanda sayfa sirasi ayni zamanda oncelik sirasidir.
- Birinci sayfadan on birinci sayfaya ilerledikce business onceligi azalir.
- Asagidaki envanter business dokumaniyla ayni sirayi korur.

## 2. Is Biriminin Talep Ettigi Degisken Envanteri

| # | Sayfa | Business Basligi | Canonical Name |
|---|---|---|---|
| 1 | 1 | Banka Borclulugu ve Ciro Iliskisi | `bank_debt_to_turnover` |
| 2 | 1 | Musterinin Tum Bankalardaki Aylik POS Hacmi | `pos_volume_change` |
| 3 | 2 | Haciz Tutari ve Ciro Iliskisi | `seizure_amount_to_turnover` |
| 4 | 3 | Cek Odemesi Zamani | `check_payment_time_shift` |
| 5 | 3 | Faktoring | `factoring_risk_presence` |
| 6 | 3 | Banka Borclulugu ve EBITDA Iliskisi | `bank_debt_to_ebitda` |
| 7 | 4 | Bilanco Ticari Alacak ve Ciro | `trade_receivables_to_turnover` |
| 8 | 5 | Bilanco Karlilik ve Ciro | `profitability_to_turnover` |
| 9 | 5 | Musterinin Aylik Elektrik Fatura Tutarlari (NACE imalat) | `electricity_bill_amount_change` |
| 10 | 5 | Musterinin Bankalardaki Isletme Borclari ve Enflasyon Iliskisi | `business_loan_vs_inflation` |
| 11 | 5 | Bilanco Ozkaynak Bilgisi | `equity_change` |
| 12 | 6 | Musterinin Bankamizda veya Diger Bankalarda Gecikmeye Girme Degiskeni | `delinquency_entry_or_frequency` |
| 13 | 6 | Musterinin Bankamizdaki Kredi Karti Borcunun Tamamini Odememesi | `credit_card_full_payment_break` |
| 14 | 6 | TFRS Davranis Temerrut Olasiligi | `ifrs9_behavioral_pd` |
| 15 | 6 | KKB Ticari Kredi Notu | `kkb_commercial_score` |
| 16 | 6 | KKB Ticari Borcluluk Endeksi | `kkb_indebtedness_index` |
| 17 | 6-7 | Net Satis (Ciro) | `net_sales_change` |
| 18 | 7 | Memzuc Limit Azalisi | `memzuc_limit_change` |
| 19 | 7 | Memzuc Banka Sayisi Azalisi | `memzuc_bank_count_change` |
| 20 | 7 | Elektrik Borcunu Odeyemeyenler | `electricity_payment_failure` |
| 21 | 7 | KKB Cek Portfoy Kalitesinde Bozulma | `kkb_check_portfolio_quality_deterioration` |
| 22 | 8 | Memzuc Limit Doluluk Artisi | `memzuc_limit_utilization_increase` |
| 23 | 8 | Hayvan Sayisi | `livestock_count_change` |
| 24 | 8-9 | Kesideci Olumsuzlugu ve Bilanco Ticari Alacak Iliskisi | `issuer_adverse_to_receivables` |
| 25 | 9 | Supheli Ticari Alacaklar ve Ticari Alacak Iliskisi | `suspicious_receivables_to_receivables` |
| 26 | 9 | KKB Ileri Vadeli Cek ve Bilanco Senetli Borc / Verilen Cek Iliskisi | `forward_check_to_notes_payable_ratio` |
| 27 | 10 | Bankamizdaki Cek/Senet Iade Orani | `returned_check_note_ratio` |
| 28 | 10 | Bankamizda Bulunan Mevduat / Varlik Ortalamalari | `bank_asset_average_change` |
| 29 | 10-11 | Alacak Sigortasi Tazmin Verisi ve Bilanco Ticari Alacak Iliskisi | `insurance_claim_to_receivables` |

## 3. Ortak Business Kurallari

### 3.1 Donem Secimi ve Annualization

Mali veri kullanan oran ailesinde yil sonu disindaki donemler annualize edilir:

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

### 3.2 Mali Veri Tazeligi

- Bilginin kaydedildigi tarihte son 12 aya ait mali veri yoksa kayit uretilmez.
- Bu kural mali tablo kullanan tum oran ve trend aileleri icin gecerlidir.

### 3.3 Sektorel ve Applicability Kurallari

- `pos_volume_change`: faaliyet grubu ve mevsimsellik dikkate alinmalidir.
- `net_sales_change`: Yap-Sat ve Taahhut sektorlerinde sektor-normalizasyon gerekir.
- `electricity_bill_amount_change`: yalniz `NACE imalat`.
- `livestock_count_change`: yalniz `Tarim + Hayvancilik`.
- `memzuc_limit_utilization_increase`: `Ticari Orta/Buyuk`, tarim disi.
- `issuer_adverse_to_receivables`: `Ticari Orta ve alti`, tarim disi.
- `insurance_claim_to_receivables`: alacak sigortasi kullanan, bilancolu, tarim disi alt evren.

### 3.4 Feature-Ici Companion Rule Mantigi

Business talebi rule'lari ayri bir jenerik havuz olarak degil, ilgili degiskenin kendi icinde tanimlar.

#### Delinquency
- son 24 ayda ilk kez gecikmeye girme
- iki donem pes pese gecikme
- son 12 ayda uc ayrik donemde gecikme

#### Kredi karti tam odememe
- son 24 ayda ilk kez tam odememe
- iki donem pes pese tam odememe

#### Cek odemesi zamani
- son 6 ayda ilk kez ogleden sonra/gec saate kayma
- son 3 ayda farkli gunlerde pes pese gec saate kayma

#### Faktoring
- son 24 ayda ilk kez factoring riski
- iki ay pes pese factoring riski
- son 12 ayda uc farkli ay factoring riski

#### Elektrik odememe
- son 24 ayda ilk kez elektrik odememe
- iki donem pes pese elektrik odememe

#### Haciz
- son 24 ayda ilk kez devam eden haciz kaydi

### 3.5 Deger Esikleri ve Filtreler

- `ifrs9_behavioral_pd`: `PD >= %2`
- `insurance_claim_to_receivables`: `%15` alti dikkate alinmaz
- `forward_check_to_notes_payable_ratio`: `> 1` oldugunda ilave risk sinyali
- `trade_receivables_to_turnover`: `> 1` oldugunda ilave risk sinyali
- `equity_change`: ozkaynak negatif oldugunda ilave risk sinyali

## 4. Veri Kaynaklari

| Canonical Name | Ana Kaynak |
|---|---|
| `pos_volume_change` | FDSTCIRO-BKM Uye Is Yeri Ciro Sorgulama |
| `bank_debt_to_turnover` | Memzuc + FS |
| `bank_debt_to_ebitda` | Memzuc + TLREF + FS |
| `business_loan_vs_inflation` | Memzuc + enflasyon verisi |
| `returned_check_note_ratio` | TACRCSIE |
| `kkb_commercial_score` | KKB |
| `kkb_indebtedness_index` | KKB |
| `kkb_check_portfolio_quality_deterioration` | KKB cek raporu |
| `forward_check_to_notes_payable_ratio` | KKB cek raporu + FS |
| `livestock_count_change` | Turkvet |
| `trade_receivables_to_turnover` | FS 120-220 + 121-221 / ciro |
| `suspicious_receivables_to_receivables` | FS 128-228 / (120-220 + 121-221) |
| `issuer_adverse_to_receivables` | kesideci olumsuzluk tutari / FS ticari alacak |
| `insurance_claim_to_receivables` | alacak sigortasi tazmin / FS ticari alacak |

## 5. Sayfa Bazli Is Odagi

### Sayfa 1
- banka borclulugu / ciro
- tum bankalardaki aylik POS hacmi

### Sayfa 2
- haciz tutari / ciro

### Sayfa 3
- cek odemesi zamani
- factoring
- banka borclulugu / EBITDA

### Sayfa 4
- bilanco ticari alacak / ciro

### Sayfa 5
- bilanco karlilik / ciro
- aylik elektrik fatura tutarlari
- isletme borcu / enflasyon
- bilanco ozkaynak

### Sayfa 6
- gecikmeye girme
- kredi karti tam odememe
- TFRS davranis temerrut olasiligi
- KKB ticari kredi notu
- KKB ticari borcluluk endeksi

### Sayfa 6-7
- net satis (ciro)

### Sayfa 7
- memzuc limit azalisi
- memzuc banka sayisi azalisi
- elektrik borcunu odeyemeyenler
- KKB cek portfoy kalitesinde bozulma

### Sayfa 8
- memzuc limit doluluk artisi
- hayvan sayisi

### Sayfa 8-9
- kesideci olumsuzlugu / ticari alacak

### Sayfa 9
- supheli ticari alacaklar / ticari alacak
- ileri vadeli cek / senetli borc-verilen cek

### Sayfa 10
- cek/senet iade orani
- bankamizdaki mevduat/varlik ortalamalari

### Sayfa 10-11
- alacak sigortasi tazmin / ticari alacak
