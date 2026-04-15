# Business Analysis

Kaynak: Is biriminin paylastigi 11 sayfalik degisken talep dokumani (WhatsApp gorselleri, 2026-04-14).

Bu dokuman:
- Ekran goruntulerinden cikarilmistir
- Birebir OCR transkripti degil, teknik yorumlama ve spec'e donusturme amaclidir
- Soz konusu 11 sayfa icindeki her basligi ve ilgili hesap/rule/eslik detaylarini yakalar
- Once tamamen ekrana sadakatle degisken kayitlarini listeler, sonra mimari yorumlari ekler

## 0. Hizli Envanter (11 Sayfadaki 29 Degisken Basligi)

| # | Sayfa | Ekrandaki Baslik | Canonical Name |
|---|---|---|---|
| 1 | 1 | Banka Borcluluğu ve Ciro Iliskisi | `bank_debt_to_turnover` |
| 2 | 1 | Musterinin Tum Bankalardaki Aylik Pos Hacmi | `pos_volume_change` |
| 3 | 2 | Haciz Tutari ve Ciro Iliskisi | `seizure_amount_to_turnover` |
| 4 | 3 | Cek Odemesi Zamani | `check_payment_time_shift` |
| 5 | 3 | Faktoring | `factoring_risk_presence` |
| 6 | 3 | Banka Borcluluğu ve Ebitda Iliskisi | `bank_debt_to_ebitda` |
| 7 | 4 | Bilanco Ticari Alacak ve Ciro | `trade_receivables_to_turnover` |
| 8 | 5 | Bilanco Karlilik ve Ciro | `profitability_to_turnover` |
| 9 | 5 | Aylik Elektrik Fatura Tutarlari (NACE imalat) | `electricity_bill_amount_change` |
| 10 | 5 | Bankalardaki Isletme Borclari ve Enflasyon | `business_loan_vs_inflation` |
| 11 | 5 | Bilanco Ozkaynak | `equity_change` |
| 12 | 6 | Bankamizda veya Diger Bankalarda Gecikmeye Girme | `delinquency_entry_or_frequency` |
| 13 | 6 | Bankamizdaki Kredi Karti Borcunun Tamamini Odememe | `credit_card_full_payment_break` |
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
| 24 | 8-9 | Kesideci Olumsuzlugu ve Bilanco Ticari Alacak | `issuer_adverse_to_receivables` |
| 25 | 9 | Supheli Ticari Alacaklar ve Ticari Alacak | `suspicious_receivables_to_receivables` |
| 26 | 9 | KKB Ileri Vadeli Cek ve Bilanco Senetli Borc/Verilen Cek | `forward_check_to_notes_payable_ratio` |
| 27 | 10 | Bankamizdaki Cek/Senet Iade Orani | `returned_check_note_ratio` |
| 28 | 10 | Bankamizda Bulunan Mevduat/Varlik Ortalamalari | `bank_asset_average_change` |
| 29 | 10-11 | Alacak Sigortasi Tazmin ve Bilanco Ticari Alacak | `insurance_claim_to_receivables` |

## 1. Ana Is Ihtiyaci

Is birimi temelde su problemi cozmek istiyor:

- kurumsal/ticari musteride odeme davranisi ve iliskili finansal davranis bozulmasini erken yakalamak
- bunu sadece tek bir skor olarak degil, is diliyle okunabilir bir aciklamaya donusturmek
- aciklamayi birden fazla sinyal ailesiyle desteklemek

Bu talep cok-katmanli bir yapi gerektiriyor:

1. core anomaly scoring (davranissal + risk-state degiskenleri)
2. context enrichment (mali tablo ve alacak tabanli oranlar)
3. narrative / explanation (insan dostu yorum)
4. **feature-companion rule layer** (jenerik rule havuzu degil — her feature'in kendi rule'u ile birlikte)

## 2. Ticari Orta Segmentinin Dogal Heterojenligi

Ticari Orta segment tek bir homogen populasyon degil. Icinde:

- sahis firma ve tuzel musteri
- bilancosu olan ve olmayan
- tarim ve tarim disi sektorler
- bankayla belirli urunlerde calisan ve calismayan musteri

bir arada bulunur. Bu durum is biriminin listeledigi degiskenlerin bir kisminin:

- bazi alt-portfoylerde **yapisal olarak tanimsiz** olmasi (bilancosuz musteri icin mali oranlar)
- bazi alt-portfoylerde **applicability dogurmasi** (POS'u olmayan musteri icin POS hacmi)
- bazi alt-portfoylerde **anlamsiz olmasi** (tarim disi musteri icin hayvan sayisi)

anlamina gelir. Bu, modellemede missing'i tek tip ele almanin yanlis oldugunu gosterir.

## 3. Null Semantigi — Uc Tur

Missing tek bir kavram degil, ucune ayrilir:

### A. Event absence
- olay hic yasanmamistir
- ornek: iade cek yok, gecikme yok, factoring riski yok
- model muamelesi: `0` mantikli, `core_with_zero_absence`

### B. Structural missing
- degisken bu musteri icin uygulanamaz
- ornek: bilancosuz musteri icin `trade_receivables_to_turnover`, POS urunu olmayan musteri icin POS hacmi, tarim disi musteri icin hayvan sayisi
- model muamelesi: `0` verilmez; `NA / not_applicable` + companion `<feature>_is_applicable = 0` flag emit edilir

### C. Operational missing
- veri aslinda olmaliydi ama gelmedi (ETL eksikligi, sorgu donmedi)
- model muamelesi: impute yapilabilir ama `<feature>_quality_flag = 1` ile izlenir

## 4. Genel Hesaplama Kurallari

### 4.1 Annualization (Donem Carpanlari) — Zorunlu Standart

Mali veri tabanli tum oran degiskenlerinde (bank_debt_*, seizure_amount_to_turnover, trade_receivables_to_turnover, suspicious_receivables_to_receivables, profitability_to_turnover, insurance_claim_to_receivables vb.):

- 1. donem (Q1): deger × 4
- 2. donem (Q2): deger × 2
- 3. donem (Q3): deger × 4/3
- Yil sonu (YE): dogrudan

Ornek (ekrandan): "Memzuc 2026 Haziran 24 ay vade ≤ nakdi risk toplami 200 lira, 2026 ikinci donem net satislar tutari 95 lira → 200 / (95 × 2) = 200/190 ≈ 1.05"

### 4.2 Stale Mali Veri Kurali

- Bilginin kaydedildigi tarihte son 1 yila ait mali verisi sistemde bulunmuyorsa bilgi kaydedilmeyecek ve yok sayilacaktir
- FS-based oran degiskenlerinde `freshness_rule: exclude_if_fs_older_than_12m` uygulanir
- Taahhut sektorundeki firmalarda **duzeltilmis mali tablolar** kullanilir

### 4.3 Sektorel Normalizasyon

- **Yap-Sat (insaat)**: satislari proje tamamlanma surecinde ciroya yansimaz; net satis dusus sinyali yanilticidir. Model bu sektoru ayri normalize etmelidir.
- **Taahhut**: duzeltilmis FS zorunlu.
- **NACE imalat**: elektrik fatura degisimi anlamli.
- **Tarim / hayvancilik**: hayvan sayisi anlamli; diger degiskenlerden buyuk kismi anlamsiz.

### 4.4 Peer/Segment Referans — Zorunlu vs Opsiyonel

Ekranlar peer/segment-reference kullanimini **sinirli** sekilde isiyor:

- **ZORUNLU**: POS Hacmi (`pos_volume_change`) — "onceki ay ile degil, faaliyet grubundaki gecmis degerlerle mutlaka mukayese"
- **ZORUNLU**: Net Satis (`net_sales_change`) — sektor-ozel normalization (Yap-Sat)
- **OPSIYONEL**: Diger core degiskenler — mevcut populasyon-anomaly motoruna ek deger katiyorsa

Yani peer features'i "tum degiskenlere" yaymak over-engineering'dir; ekranlar bunu istemiyor.

## 5. Companion Rule'lar (Feature-Spesifik, Jenerik Degil)

Ekranlar "ilk kez", "2 kez pes pese", "3 farkli ay" tipi rule'lari **her degiskenin kendi icinde** tanimliyor. Bu rule'lar bagimsiz kategori degil, feature'a baglanir.

### `delinquency_entry_or_frequency` (sayfa 6)
- R1: son 2 yilda hic gecikme yokken ilk kez gecikmeye girme
- R2: son 1 yilda ilk kez 2 ay pes pese gecikmeye girme
- R3: son 1 yilda toplamda 3 farkli ayda gecikme bilgisi olmasi

### `credit_card_full_payment_break` (sayfa 6)
- R1: son 2 yilda hic asgari/eksik odeme yokken ilk kez asgari/eksik odeme
- R2: son 1 yilda ilk kez 2 ay pes pese asgari/eksik odeme

### `check_payment_time_shift` (sayfa 3)
- R1: son 6 ayda ilk kez cek odemesini ogleden sonra yapmasi
- R2: son 3 ayda ilk kez farkli gunlerde pes pese cek odemesini ogleden sonra yapmasi

### `factoring_risk_presence` (sayfa 3)
- R1: son 2 yilda hic factoring riski yokken ilk kez gorulmesi
- R2: son 1 yilda ilk kez 2 ay pes pese factoring riski
- R3: son 1 yilda toplam 3 farkli ayda factoring riski

### `electricity_payment_failure` (sayfa 7)
- R1: son 2 yilda hic geciktirmemisken ilk kez geciktirmesi
- R2: son 1 yilda ilk kez 2 kez pes pese geciktirmesi

### `seizure_amount_to_turnover` (sayfa 3)
- R1: son 2 yilda hic haciz kaydi yokken ilk kez devam eden haciz kaydinin olusmasi (feature ile birlikte rule)

## 6. Value Filter'lar (Feature-Spesifik Esikler)

### `ifrs9_behavioral_pd`
- `value >= 0.02` (PD %2 esigi). PD %2 altinda olan aylik bilgi yok sayilir.

### `insurance_claim_to_receivables`
- `value >= 0.15` (%15 esigi). %15 altindaki degerler dikkate alinmayabilir.

### `forward_check_to_notes_payable_ratio`
- `value > 1` kosuluyla arttikca kotu (unit threshold). 1 altinda normal kabul edilebilir.

### `trade_receivables_to_turnover`
- Deger 1 uzerinde olmasi kosuluyla artmasi cok daha kotu bir gelisme (companion threshold indicator).

### `equity_change`
- Ozkaynak **negatif olmasi** cok daha kotu (sign_flip indicator). Sadece trend degil, sifir-gecis ayri sinyal.

## 7. Segment Filtreleri (Ekrandan Aynen)

### `memzuc_limit_utilization_increase` (sayfa 8)
- Yalnizca Buyuk segment musterilerine ek olarak Ticari Orta segment ve uzeri musteriler icin
- Ana faaliyet bilgisi **Tarimsal olan musteriler icin calismayacaktir**
- Applicability: `segment IN (Ticari_Orta, Buyuk) AND nace_main != 'Tarim'`

### `livestock_count_change` (sayfa 8)
- Yalnizca ana faaliyet Tarim olan ve faaliyeti hayvancilik olan musteriler
- Veri kaynagi: Turkvet sorgusu
- Applicability: `nace_main = 'Tarim' AND nace_sub = 'Hayvancilik'`

### `issuer_adverse_to_receivables` (sayfa 8-9)
- Yalnizca Ticari Orta segment **ve alti** segmentler
- Ana faaliyet **Tarimsal olmayan** musteriler
- Applicability: `segment IN (Ticari_Orta, KOBI, Mikro) AND nace_main != 'Tarim'`

### `electricity_bill_amount_change` (sayfa 5)
- NACE kodu imalat olan musteriler
- Applicability: `nace_section = 'C'` (imalat)

### `electricity_payment_failure` (sayfa 7)
- Yalnizca elektrik abonesi olan / tuketim izlenebilir musteriler

## 8. Veri Kaynaklari (Ekrandan)

| Degisken | Veri Kaynagi |
|---|---|
| `pos_volume_change` | FDSTCIRO-BKM Uye Is Yeri Ciro Sorgulama |
| `bank_debt_to_turnover` | Memzuc (0-24 ay nakdi krediler) + FS |
| `bank_debt_to_ebitda` | Memzuc nakdi risk * TLREF / EBITDA |
| `business_loan_vs_inflation` | Memzuc 0-24 ay + Enflasyon verisi |
| `returned_check_note_ratio` | TACRCSIE (Tahsil/Senet) |
| `kkb_commercial_score` | KKB |
| `kkb_indebtedness_index` | KKB |
| `kkb_check_portfolio_quality_deterioration` | KKB Cek raporu (1/3/12 ay, hamili/ciranta) |
| `forward_check_to_notes_payable_ratio` | KKB Cek raporu (3 aylik sorgu) + Bilanco 321-421 + 103 |
| `livestock_count_change` | Turkvet |
| `trade_receivables_to_turnover` | Bilanco 120-220 Alicilar + 121-221 Alacak Senetleri / Ciro |
| `suspicious_receivables_to_receivables` | Bilanco 128-228 / (120-220 + 121-221) |
| `issuer_adverse_to_receivables` | Kesideci olumsuzluk tutari / Bilanco 120-220 + 121-221 |
| `insurance_claim_to_receivables` | Alacak sigortasi tazmin / Bilanco 120-220 + 121-221 |

## 9. Rule Layer — Dogru Yaklasim

Onceki business notunda "jenerik rule havuzu" (`first_time_event_rules`, `threshold_exclusion_rules`) diye ayri kategori yazilmisti. Bu yanlisti.

Dogru yaklasim:
- **Companion rule'lar**: her feature'in kendi icinde "ilk kez / pes pese / N farkli ay" pattern'leri (Bolum 5).
- **Value filter'lar**: her feature'in kendi value esigi (Bolum 6).
- **Applicability filter'lar**: her feature'in kendi segment/NACE kosulu (Bolum 7).

Pipeline'da bunlar:
- anomaly model input'u degil
- feature calculation adiminin parcasi
- family registry (feature_family_map.yaml) icinde bu alanlarla kayitli

Yani rule layer ayri bir motor **degil**; feature calculation'in bir katmanidir.

## 10. Ticari Orta icin Modele Girebilecek Degiskenler

### 10.1 Birincil (Faz 1)

Davranissal veya risk-state temelli, ekranda yuksek onceliği olan 11 aile:

1. `bank_debt_to_turnover`
2. `bank_debt_to_ebitda`
3. `pos_volume_change`
4. `delinquency_entry_or_frequency`
5. `credit_card_full_payment_break`
6. `ifrs9_behavioral_pd`
7. `kkb_commercial_score`
8. `kkb_indebtedness_index`
9. `returned_check_note_ratio`
10. `check_payment_time_shift`
11. `memzuc_limit_utilization_increase` (segment filter sonrasinda)

### 10.2 Ikinci seviye (Faz 2)

Coverage / freshness / sektor normalizasyonu gerektiren 13 aile:

1. `trade_receivables_to_turnover`
2. `suspicious_receivables_to_receivables`
3. `profitability_to_turnover`
4. `equity_change`
5. `net_sales_change` (sektor normalization zorunlu)
6. `insurance_claim_to_receivables`
7. `seizure_amount_to_turnover`
8. `forward_check_to_notes_payable_ratio`
9. `business_loan_vs_inflation`
10. `memzuc_limit_change`
11. `memzuc_bank_count_change`
12. `kkb_check_portfolio_quality_deterioration`
13. `bank_asset_average_change` (urun sahipligi kosullu)

### 10.3 Segment-ozel (Faz 3)

Alt-portfoye ozel 4 aile:

1. `electricity_bill_amount_change` (NACE imalat)
2. `electricity_payment_failure` (elektrik abonesi)
3. `livestock_count_change` (tarim/hayvancilik + Turkvet)
4. `issuer_adverse_to_receivables` (Ticari Orta ve alti, tarim disi)

### 10.4 Rule-Only

Artik **yok**. Onceki notta `factoring_risk_presence` rule_only diye gecirilmisti; aslinda feature + 3 companion rule'dur. `factoring_risk_presence` Faz 2'ye alinir, rule'lari feature'in yaninda cikari.

Not: Faz 2 shortlist'e `factoring_risk_presence` da dahil edilir (toplam 14 aile). Envantere dahil ama Faz 2 listesinde tekrarlanmadi; yerini koruyabilmek icin operasyonda `factoring_risk_presence`'i asagidaki sekilde ekle:

14. `factoring_risk_presence` (companion rule'lari ile birlikte)

## 11. Self-History Turev Politikasi

Her core/second-level degiskene standart self-history turevleri uretilir. Ekrandaki "N yil geriye gidilebilir" ifadeleri pencere uzunluğunu belirler:

| Degisken | History Window | Turevler |
|---|---|---|
| Oran-based (bank_debt_*, trade_receivables_*, profitability_*, suspicious_*, forward_check_*) | 3 yil | `_current`, `_delta_1`, `_self_zscore_6`, `_vs_6m_median_ratio` |
| PD / KKB skorlari | 3 yil | `_current`, `_delta_1`, `_self_zscore_6` |
| POS hacmi, net satis | 2 yil / 3 yil | `_current`, `_delta_1`, `_delta_pct_1`, `_self_zscore_6`, `_trend_slope_6` |
| Gecikme / kart / cek event | 1-2 yil | `_current`, `_delta_1`, `_rolling_mean_6` |
| Memzuc limit/sayi | 3 yil | `_current`, `_delta_1` |
| Haciz | 5 yil | `_current`, `_delta_1`, `_self_zscore_6` |
| Bankamizda mevduat/varlik | 2 yil | `_current`, `_delta_1`, `_trend_slope_6` |

Rule-based feature'lar icin (delinquency, factoring, check_payment_time, electricity_payment_failure) companion rule'lar history'nin fonksiyonudur; ayrica `_rolling_mean_6` gibi count turevleri eklenir.

## 12. Mevcut Projeye Eklenmesi Gerekenler (Teknik)

Mevcut `engine/` motoru retail DPD-izleme icin olgun, ancak Ticari Orta icin su moduller eksik:

### 12.1 Yeni moduller

- `business/feature_family_map.yaml` — sozlugun makine-okunur versiyonu (role, null_semantics, applicable_segments, directionality, refresh_frequency, freshness_rule, companion_rules, value_filter, data_source, history_window, peer_required)
- `engine/applicability.py` — structural missing icin `<feature>_is_applicable` companion kolon emitter
- `engine/history_features.py` — `_delta_1`, `_self_zscore_6`, `_rolling_mean_6`, `_trend_slope_6` turetici
- `engine/peer_features.py` — yalniz POS ve Net Satis icin peer/sektor turetici
- `engine/rules.py` — companion rule hesaplayici (first_time, consecutive, distinct_months)
- `engine/annualization.py` — Q1×4, Q2×2, Q3×4/3, YE×1 ceyreklik carpanlarini uygular

### 12.2 Mevcut modullerde degisiklikler

- `engine/preprocessing.py` — null_semantics farkindaligi; structural semantic ile companion flag uretimi
- `engine/feature_selection.py` — `mahalanobis_eligible: false` feature'lari MD branch'inden cikar
- `engine/calibration.py` — applicability subset uzerinden fit
- `engine/scorer.py` — per-model (AE vs IF vs MD) contribution kirilimini explainability'ye ekle
- `engine/lifecycle.py` — `run_partitions` (bilancolu/bilancosuz, POS var/yok) sub-run destegi
- `engine/monitoring.py` — structural missing orani trend + applicability drift metrikleri
- `config/pipeline_config_ticari_orta.yaml` — ayri config (retail config korunarak)

### 12.3 Pipeline akis sirasi (Ticari Orta)

1. Oracle input'tan ham degiskenleri oku
2. `annualization.py` — mali veri tabanli oranlarda donem carpanlarini uygula
3. `applicability.py` — structural missing feature'larina companion flag emit et
4. `rules.py` — companion rule feature'larini hesapla (first_time, consecutive, distinct_months)
5. `history_features.py` — self-history turevlerini uret
6. `peer_features.py` — POS ve Net Satis icin peer/sektor turevleri
7. Mevcut pipeline: missing + hard bounds + categorical + feature_selection + AE/IF/MD + calibration + scoring

## 13. Mantikli ve Riskli Taraflar

### Mantikli taraflar (ekrana sadik)

- davranis bozulmasi odakli dusunme
- kendi gecmisi + (sinirli) peer kiyasi ayrimi
- oran ve iliski degiskeni kurma
- sektor farkindaligi (Yap-Sat, Taahhut, imalat, tarim)
- stale FS kurali
- segment filtreleri (memzuc doluluk, hayvan, kesideci)
- rule'larin feature'lara bagli olmasi

### Riskli taraflar (ekrandaki belirsizlikler)

- memzuc_bank_count_change yonu ekranda "azalma kotu" diyor ama bankayla iliski sadelestirilmesi de bu sinyali verebilir — yon belirsiz
- bank_asset_average_change urun sahipligine kosullu; coverage dusuk olabilir
- net_sales_change sektor normalization'i model tarafinda net tanimlanmamis
- Yap-Sat dahil bazi sektorlerde "faturasizlik" durumu FS-based oranlari bozabilir
- annualization ceyrek carpanlarini sektor bazinda degistiren is tarafi bekleniyor mu, sabit mi — ekran sabit kabul etmis

## 14. Net Yol Haritasi

1. **Sozlugu duzelt**: 29 feature icin tam field setini (role, applicable_segments, companion_rules, value_filter, data_source, annualization, directionality, freshness_rule, peer_required, history_window, account_codes) tabloya ve detay kartlarina yaz.
2. **`business/feature_family_map.yaml`** dosyasini sozlukten uret (makine-okunur registry).
3. **`engine/annualization.py`, `engine/applicability.py`, `engine/rules.py`, `engine/history_features.py`** modullerini yaz.
4. **`config/pipeline_config_ticari_orta.yaml`** olustur; run_partitions (bilancolu/bilancosuz) destegini lifecycle'a ekle.
5. Faz 1 shortlist (11 degisken) ile PoC; calibration sub-segment subset'te fit edilsin.
6. Monitoring'e structural missing + applicability drift metrikleri eklensin.
7. Faz 2: mali tablo tabanli oranlar + sektor normalization (Yap-Sat/Taahhut).
8. Faz 3: segment-ozel feature'lar (elektrik, hayvan, kesideci).
9. Per-model contribution explainability + human-readable narrative layer.
