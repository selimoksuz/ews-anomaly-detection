# Ticari Orta Faz 1 Business Scope

Bu dokuman yalnizca Faz 1 business kapsamını tanimlar. Teknik implementasyon, pipeline notu veya acik aksiyon listesi icermez.

## Portfolio Tanimi

Faz 1 cohort'u asagidaki musteri evrenidir:

- `segment = TICARI_ORTA`
- `bank_total_risk >= 1_000_000`
- `is_balance_sheet_customer = 1`
- `has_pos = 1`

## Faz 1 Nihai Degisken Seti

Is biriminin Faz 1 icin nihai onay verdigi ve kapsama alinacak tek degiskenler bunlardir:

| # | Business Basligi | Canonical Name |
|---|---|---|
| 1 | Banka Borclulugu ve Ciro Iliskisi | `bank_debt_to_turnover` |
| 2 | Musterinin Tum Bankalardaki Aylik POS Hacmi | `pos_volume_change` |
| 3 | Banka Borclulugu ve EBITDA Iliskisi | `bank_debt_to_ebitda` |
| 4 | Bilanco Ticari Alacak ve Ciro | `trade_receivables_to_turnover` |
| 5 | Bilanco Karlilik ve Ciro | `profitability_to_turnover` |
| 6 | Musterinin Bankalardaki Isletme Borclari ve Enflasyon Iliskisi | `business_loan_vs_inflation` |
| 7 | Bilanco Ozkaynak Bilgisi | `equity_change` |
| 8 | TFRS Davranis Temerrut Olasiligi | `ifrs9_behavioral_pd` |
| 9 | KKB Ticari Kredi Notu | `kkb_commercial_score` |
| 10 | KKB Ticari Borcluluk Endeksi | `kkb_indebtedness_index` |
| 11 | Net Satis (Ciro) | `net_sales_change` |
| 12 | Memzuc Limit Doluluk Artisi | `memzuc_limit_utilization_increase` |
| 13 | Bankamizda Bulunan Mevduat / Varlik Ortalamalari | `bank_asset_average_change` |

Faz 1'de bunun disindaki hicbir business degiskeni kapsama alinmaz.

## Ortak Is Kurallari

### 1. Mali Veri Annualization

Mali veri donem kodu yil sonu degilse annualization uygulanir:

- `Q1 -> *4`
- `Q2 -> *2`
- `Q3 -> *4/3`
- `Q4 / YE -> *1`

Bu kural asagidaki ailelerde gecerlidir:

- `bank_debt_to_turnover`
- `bank_debt_to_ebitda`
- `trade_receivables_to_turnover`
- `profitability_to_turnover`
- `net_sales_change`

### 2. Zaman Serisi Bakisi

- Faz 1 aylik cadence ile calisir.
- Snapshot'lar ay sonu goruntusu olarak tutulur.
- Business yorumunda degisim tek snapshot degil, musteri trendi uzerinden okunur.

### 3. Yorumlama Mantigi

- Oran veya seviye degiskenleri tek basina esik degil, musteri gecmisi ve populasyon ile birlikte okunur.
- Faz 1'de reason yaziminda su referanslar beklenir:
  - `gerceklesen`
  - `musteri_gecmis_referansi`
  - `populasyon_referansi`
  - `ae_referansi`
  - `ensemble_katki`

### 4. Veri Tazeligi

- Mali veri kullanan degiskenlerde stale veri riski dikkate alinmalidir.
- Taze olmayan mali veri dominant alarm kaynagi olmamalidir.

### 5. Aylik Isletim Beklentisi

- Skorlama aylik calisir.
- Her ayin `1.` gunu bir onceki ay sonu snapshot'i skora girer.
- Sonuclarin aylik izlenebilir ve onceki ayla kiyaslanabilir olmasi beklenir.

## Faz 1 Business Beklentisi

Business tarafinin Faz 1 icin bekledigi ana ciktilar:

- anomali skoru
- alarm bandi
- top 3 reason
- tum reason effect'leri
- musteri gecmisi / populasyon / AE referansi ile okunabilir aciklama
- aylik izleme ve alarm esikleri

Bu dokumanin dictionary karsiligi [dictionary/faz1_variable_dictionary.md](/C:/Users/Acer/ews-anomaly-detection/dictionary/faz1_variable_dictionary.md) dosyasidir.
