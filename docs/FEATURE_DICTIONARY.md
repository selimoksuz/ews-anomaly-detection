# EWS Feature Dictionary — Degisken Sozlugu

Bu dokuman, EWS Multivariate Anomaly Detection sisteminde kullanilan 32 degiskenin detayli tanimini icerir.

**Tasarim prensibi:** Haftalik anomali tespiti yapildigindan, tum degiskenlerin haftalik bazda degisen deger uretmesi gerekir. 6 ay / 12 ay gibi uzun pencereli degiskenler kullanilmaz — bunlar haftalik tespitte oldu agirlik tasir.

---

## Katman Yapisi

| Katman | Amac | Degisken Sayisi | Degisim Frekansi |
|--------|------|-----------------|------------------|
| **Katman 1 — Anlik** | Su anki durum | 8 | Her hafta kesin degisir |
| **Katman 2 — Rolling 4W** | Kisa gecmis ozeti | 11 | Her hafta kayar pencereyle degisir |
| **Katman 3 — Trend** | Degisimin yonu ve hizi | 9 | Her hafta guncel slope/ivme |
| **Katman 4 — Interaction** | Risk grubu bilesik sinyalleri | 4 | Bilesenleri degistikce degisir |

---

## Katman 1 — Anlik / Haftalik (8 degisken)

Bu degiskenler musterinin **su anki durumunu** gosterir. Her hafta farkli deger uretir.

### `dpd_current`
- **Tanim:** Bu haftaki gecikme gun sayisi
- **Hesaplama:** `MAX(0, bugun - vade_tarihi)`. Odeme yapildiysa 0.
- **Birim:** Gun (tamsayi)
- **Normal aralik:** 0-5 (cogu musteri)
- **Anomali sinyali:** Kendi gecmisinde hic gecikmemis musteride 3+ gun bile anormal
- **IFRS 9 baglantisi:** 30 gunu astiginda Stage 1→2 gecisi tetiklenir
- **Korelasyon:** `dpd_max_4w` (r≈0.7), `dpd_direction_4w` (r≈0.5)

### `utilization_ratio`
- **Tanim:** Kullanilmis kredi / toplam kredi limiti
- **Hesaplama:** `toplam_kullanim / toplam_limit`
- **Birim:** Oran (0-1)
- **Normal aralik:** Segmente gore degisir. PREMIUM: 0.20-0.35, STANDARD: 0.30-0.55, RISKY: 0.55-0.85
- **Anomali sinyali:** Kendi ortalamasindan sapma onemli. %40→%85 cikis, sabit %85'ten daha guclu sinyal
- **Neden onemli:** BIS'in en guclu mikro EWI'larindan. Tek basina yuksek olabilir ama `txn_count_weekly` dusukken yuksek olmasi multivariate anomali (limit dolmus ama islem yok → odeme yapamadigi icin limit dolmus)
- **Korelasyon:** `outstanding_balance` (r≈0.55), `checking_balance` (r≈-0.25)

### `outstanding_balance`
- **Tanim:** Toplam borc bakiyesi (TL)
- **Hesaplama:** Tum kredi urunlerindeki kullanim bakiyesi toplami
- **Birim:** TL
- **Normal aralik:** Segmente gore. PREMIUM: 5K-25K, STANDARD: 10K-35K, RISKY: 20K-60K
- **Anomali sinyali:** Ani artis → hizli borclanma. `checking_balance` dusuyorken `outstanding_balance` artiyorsa → cift yonlu sikisma
- **Korelasyon:** `utilization_ratio` (r≈0.55), `balance_slope_4w` (r≈0.4)

### `checking_balance`
- **Tanim:** Bu haftaki vadesiz hesap bakiyesi
- **Hesaplama:** Haftalik snapshot tarihindeki vadesiz hesap bakiyesi
- **Birim:** TL
- **Normal aralik:** PREMIUM: 15K-35K, STANDARD: 3K-10K, RISKY: 500-4K
- **Anomali sinyali:** Dusus trendi → likidite erimesi. Khandani et al. (2010) en guclu predictor'lardan biri olarak tanimlamistir. Sifira yaklasma veya eksiye dusme acil sinyal.
- **Korelasyon:** `checking_balance_min_4w` (r≈0.8), `checking_slope_4w` ile trend iliskisi

### `txn_count_weekly`
- **Tanim:** Bu haftaki toplam islem sayisi
- **Hesaplama:** Hafta icindeki tum borc/alacak islem sayisi
- **Birim:** Adet (tamsayi)
- **Normal aralik:** PREMIUM: 8-15, STANDARD: 4-8, RISKY: 2-6, NEW: 1-4
- **Anomali sinyali:** Ani dusus → aktivite azalmasi. `utilization_ratio` yuksek + `txn_count_weekly` dusuk → en klasik multivariate anomali pattern'i (limit dolmus ama islem yok)
- **Korelasyon:** `txn_amount_weekly` (r≈0.6)

### `txn_amount_weekly`
- **Tanim:** Bu haftaki toplam islem tutari
- **Hesaplama:** Hafta icindeki tum islem tutarlari toplami
- **Birim:** TL
- **Normal aralik:** PREMIUM: 5K-12K, STANDARD: 2K-5K, RISKY: 1K-3K
- **Anomali sinyali:** `txn_count_weekly` ile birlikte okunmali. Ayni islem sayisi ama cok farkli tutar → ortalama islem buyuklugu degismis → davranis kirilmasi
- **Korelasyon:** `txn_count_weekly` (r≈0.6), `avg_txn_amount_weekly` (r≈0.4)

### `avg_txn_amount_weekly`
- **Tanim:** Bu haftaki ortalama islem tutari
- **Hesaplama:** `txn_amount_weekly / txn_count_weekly`
- **Birim:** TL
- **Normal aralik:** 200-800 (segmente gore)
- **Anomali sinyali:** Normalde 300 TL'lik islemler yapan birinin 5.000 TL'lik islemler yapmasi → aliskanlik kirilmasi. Her iki yonde sapma anlamli.
- **Korelasyon:** Dolaylidir — `txn_amount_weekly` ve `txn_count_weekly`'den turetilir

### `payment_amount_this_week`
- **Tanim:** Bu hafta yapilan odeme tutari
- **Hesaplama:** Hafta icinde yapilan tum odeme islemleri toplami
- **Birim:** TL
- **Normal aralik:** Musteriye gore degisir (kendi gecmisine kiyaslanir)
- **Anomali sinyali:** Sifir olmasi (bu hafta hic odeme yok) veya cok dusuk olmasi sinyal. `payment_to_min_ratio_4w` ile birlikte degerlendirilir.

---

## Katman 2 — Rolling 4 Hafta (11 degisken)

Bu degiskenler son 4 haftanin ozetini verir. Her hafta pencere 1 hafta kayar → her hafta farkli deger uretir. "6 aylik" veya "12 aylik" pencere yerine 4 hafta kullanilir cunku haftalik anomali tespitinde yeterince hassas ve yeterince stabil.

### `dpd_max_4w`
- **Tanim:** Son 4 haftadaki en yuksek gecikme gun sayisi
- **Hesaplama:** `MAX(dpd_current)` over last 4 weeks
- **Birim:** Gun (tamsayi)
- **Neden onemli:** `dpd_current` su an 0 olabilir ama 2 hafta once 15 gune cikti ise bu bilgiyi tutar. Yakin gecmisteki stres izi.
- **Anomali sinyali:** 0'dan farkli olmasi dikkat cekici. 15+ ciddi.

### `min_payment_only_count_4w`
- **Tanim:** Son 4 haftada sadece minimum odeme yapilan hafta sayisi
- **Hesaplama:** Son 4 haftada `odeme_tutari ≈ minimum_odeme (±%5)` olan hafta sayisi
- **Birim:** Adet (0-4)
- **Neden onemli:** Butaru et al. (2016) en guclu leading indicator. DPD sifirken bile stres gosterir. DPD'den 2-3 ay once sinyal verir.
- **Anomali sinyali:** 0→2 gecisi ciddi. 3+ kesin stres.

### `payment_to_min_ratio_4w`
- **Tanim:** Son 4 haftada yapilan odemelerin minimum odemeye orani (ortalama)
- **Hesaplama:** `AVG(yapilan_odeme / minimum_odeme)` over last 4 weeks
- **Birim:** Oran (>= 1.0)
- **Normal aralik:** PREMIUM: 5-10, STANDARD: 2-5, RISKY: 1.0-2.0
- **Anomali sinyali:** 1.0'a yaklasma → sadece minimum oduyor. Dusus trendi → bozulma baslangici.

### `avg_days_to_payment_4w`
- **Tanim:** Son 4 haftada faturadan odemeye gecen ortalama gun sayisi
- **Hesaplama:** `AVG(odeme_tarihi - fatura_tarihi)` over last 4 weeks
- **Birim:** Gun
- **Normal aralik:** 3-15
- **Anomali sinyali:** Uzama → odeme disiplini bozuluyor. 25+ gun → vade sonuna yakin oduyor.

### `payment_reversal_count_4w`
- **Tanim:** Son 4 haftada geri donen / iptal edilen odeme sayisi
- **Hesaplama:** Karsilaksiz cek, geri donen EFT, iptal otomatik odeme sayisi
- **Birim:** Adet (tamsayi)
- **Anomali sinyali:** 0'dan farkli olmasi bile guclu sinyal. 2+ ciddi alarm.

### `nsf_count_4w`
- **Tanim:** Son 4 haftada karsilaksiz islem sayisi (Non-Sufficient Funds)
- **Hesaplama:** Bakiye yetersizliginden reddedilen islem sayisi
- **Birim:** Adet (tamsayi)
- **Anomali sinyali:** Direkt likidite sorununa isaret eder. 0'dan farkli = alarm.
- **Korelasyon:** `checking_balance_min_4w` ile negatif (r≈-0.35)

### `overlimit_count_4w`
- **Tanim:** Son 4 haftada kredi limitinin asildigi hafta sayisi
- **Hesaplama:** `balance > limit` olan hafta sayisi
- **Birim:** Adet (0-4)
- **Anomali sinyali:** 2+ → musteri limitini yonetemiypr.

### `cash_advance_ratio_4w`
- **Tanim:** Son 4 haftada nakit avans cekiminin toplam harcamaya orani
- **Hesaplama:** `SUM(nakit_avans) / SUM(toplam_harcama)` over last 4 weeks
- **Birim:** Oran (0-1)
- **Normal aralik:** %0-5
- **Anomali sinyali:** %15+ → nakde sikismis. %30+ → acil. Butaru et al. en onemli behavioral indicator.

### `checking_balance_min_4w`
- **Tanim:** Son 4 haftanin en dusuk vadesiz hesap bakiyesi
- **Hesaplama:** `MIN(gunluk_vadesiz_bakiye)` over last 4 weeks
- **Birim:** TL
- **Neden onemli:** Ortalama iyi gorunebilir ama bir gun sifira dustuyse → o an likidite krizi yasanmis. Khandani et al. (2010) avg_balance'tan bile daha prediktif bulmultur.
- **Anomali sinyali:** Sifira yaklasma veya negatife dusme.

### `deposit_amount_avg_4w`
- **Tanim:** Son 4 haftada hesaba giren ortalama mevduat tutari
- **Hesaplama:** `AVG(haftalik_mevduat_girisi)` over last 4 weeks
- **Birim:** TL
- **Neden onemli:** Gelir proxy'si. Dusmesi → gelir azaliyor. Tiger Analytics SME EWS'de en guclu sinyal.
- **Anomali sinyali:** Kendi gecmisine gore %30+ dusus dikkat cekici.

### `channel_count_4w`
- **Tanim:** Son 4 haftada kullanilan farkli kanal sayisi
- **Hesaplama:** `COUNT(DISTINCT channel)` — ATM, POS, online, mobil, sube
- **Birim:** Adet (tamsayi)
- **Normal aralik:** 2-5
- **Anomali sinyali:** 4→1 dusus → sadece nakit cekime donmus. Figini et al. (2017) leading indicator.

---

## Katman 3 — Trend / Ivme (9 degisken)

Bu degiskenler **degisimin yonunu ve hizini** olcer. McKinsey EWS prensibi: "Seviyeden cok trend onemlidir."

### `util_slope_4w`
- **Tanim:** Kullanim oraninin son 4 haftalik lineer egimi
- **Hesaplama:** 4 haftalik `utilization_ratio` serisine linear regression → slope
- **Birim:** Hafta basina degisim
- **Anomali sinyali:** Pozitif slope = artan baski. 0.03+/hafta → agresif borclanma.

### `balance_slope_4w`
- **Tanim:** Borc bakiyesinin son 4 haftalik egimi
- **Hesaplama:** 4 haftalik `outstanding_balance` → linear regression slope
- **Birim:** TL/hafta
- **Anomali sinyali:** Pozitif = borc artiyor. `checking_slope_4w` ile ters yonde hareket ediyorsa → kapasite sikismasi.

### `checking_slope_4w`
- **Tanim:** Vadesiz bakiyenin son 4 haftalik egimi
- **Hesaplama:** 4 haftalik `checking_balance` → linear regression slope
- **Birim:** TL/hafta
- **Anomali sinyali:** Negatif = bakiye eriyor. `balance_slope_4w` pozitifken bu negatifse → cift yonlu sikisma.

### `payment_ratio_slope_4w`
- **Tanim:** Odeme/minimum oraninin 4 haftalik egimi
- **Hesaplama:** 4 haftalik `payment_to_min_ratio` → slope
- **Birim:** Hafta basina degisim
- **Anomali sinyali:** Negatif = odeme kapasitesi daraliyor. 1.0'a dogru inis trendi → yakin gelecekte sadece minimum odeme.

### `txn_count_change_pct`
- **Tanim:** Bu haftaki islem sayisinin son 4 hafta ortalamasina gore degisimi
- **Hesaplama:** `(txn_count_weekly - avg_4w) / avg_4w * 100`
- **Birim:** Yuzde (%)
- **Anomali sinyali:** -%50'nin altina dusus ciddi. `utilization_ratio` yuksekken islem dususu → en guclu multivariate sinyal.

### `txn_amount_change_pct`
- **Tanim:** Bu haftaki islem tutarinin son 4 hafta ortalamasina gore degisimi
- **Hesaplama:** `(txn_amount_weekly - avg_4w) / avg_4w * 100`
- **Birim:** Yuzde (%)
- **Anomali sinyali:** Her iki yonde sapma anlamli. `txn_count_change_pct` ile farkli yonde hareket etmesi → islem buyuklugu degismis.

### `deposit_change_pct`
- **Tanim:** Bu haftaki mevduat girisinin son 4 hafta ortalamasina gore degisimi
- **Hesaplama:** `(deposit_this_week - avg_4w) / avg_4w * 100`
- **Birim:** Yuzde (%)
- **Anomali sinyali:** -%40'in altina dusus → gelir kaybi olabilir. `balance_slope_4w` pozitifken bu negatifse → gelir erozyonu.

### `util_acceleration`
- **Tanim:** Kullanim orani ivmesi — egimin degisim hizi
- **Hesaplama:** `util_slope_son_2_hafta - util_slope_onceki_2_hafta`
- **Birim:** Ivme (slope degisimi)
- **Anomali sinyali:** Pozitif = giderek daha hizli borclanma. Sabit %80 util'den farkli olarak ivmelenen %80 cok daha tehlikeli.

### `dpd_direction_4w`
- **Tanim:** Son 4 haftada gecikmenin kac haftadir arttigini gosterir
- **Hesaplama:** Son 4 haftanin DPD serisinde art arda artan hafta sayisi
- **Birim:** Hafta (0-4)
- **Anomali sinyali:** 3-4 → sistematik bozulma. Tek seferlik gecikme (1) ile kronik artis (4) arasindaki farki yakalar.

---

## Katman 4 — Risk Grubu Interaction (4 degisken)

Bu degiskenler birden fazla degiskenin BIRLIKTE anormal olmasini tek bir sayida ozetler. Ikili korelasyonda gorunmeyen ama grup olarak bozuldiginda ortaya cikan anomalileri yakalar.

### `liquidity_squeeze_score`
- **Tanim:** Likidite sikismasi bilesik skoru
- **Bilesenleri:** `utilization_ratio` (↑) + `checking_balance` (↓) + `checking_balance_min_4w` (↓) + `cash_advance_ratio_4w` (↑)
- **Hesaplama:** Her bilesenin stres yonundeki z-score'u → pozitif olanlarin ortalamasi
- **Is birimi aciklamasi:** "Kredi kullanimi artarken nakit varliklari eriyor ve nakit avansa basvuruyor"
- **Neden onemli:** Bu 4 degisken normalde birlikte bu yonde hareket etmez. Kullanim artarken bakiyenin de belirli seviyede kalmasi beklenir. Hepsinin birlikte stres yonunde sapmasi → likidite krizi.

### `hidden_stress_score`
- **Tanim:** Gizli stres bilesik skoru
- **Bilesenleri:** `dpd_current` (↓ VEYA 0) + `min_payment_only_count_4w` (↑) + `payment_to_min_ratio_4w` (↓) + `deposit_change_pct` (↓)
- **Hesaplama:** DPD dusuk/sifir IKen diger 3 bilesenin stres yonunde olmasi
- **Is birimi aciklamasi:** "Gecikme yok ama sadece minimum oduyor, mevduati dusuyor — 2-3 ay icinde gecikmeye dusme riski yuksek"
- **Neden onemli:** Klasik DPD-bazli EWS bunu KACIR. Bu skor DPD sifirken bile stres tespit eder.

### `income_erosion_score`
- **Tanim:** Gelir erozyonu bilesik skoru
- **Bilesenleri:** `deposit_change_pct` (↓) + `deposit_amount_avg_4w` (↓) + `balance_slope_4w` (↑) + `checking_slope_4w` (↓)
- **Hesaplama:** Gelen para azalirken borc artiyorsa ve bakiye eriyorsa → yuksek skor
- **Is birimi aciklamasi:** "Hesaba gelen para azaliyor, borc artiyor, vadesiz bakiye eriyor — gelir kaybi yasaniyor olabilir"
- **Neden onemli:** Tiger Analytics SME EWS'de en guclu sinyal. Gelir dususu temmerrutten 4-6 ay once baslar.

### `payment_breakdown_score`
- **Tanim:** Odeme davranisi kirilma skoru
- **Bilesenleri:** `avg_days_to_payment_4w` (↑) + `payment_reversal_count_4w` (↑) + `txn_count_change_pct` (↓) + `channel_count_4w` (↓)
- **Hesaplama:** Odeme suresi uzamis, iadeler artmis, islem sayisi dusmus, kanal cesitliligi azalmis → bilesik skor
- **Is birimi aciklamasi:** "Odeme aliskanligi tamamen degismis — gec oduyor, odemeler geri donuyor, islem yapmıyor, tek kanala donmus"
- **Neden onemli:** Figini et al. (2017) behavioral scoring calismasi. Davranis kirilmasi temmerrutten once olusur.

---

## Degiskenler Arasi Kritik Iliskiler

Autoencoder bu iliskileri ogrenir. Bozulduklarinda multivariate anomali:

| Iliski | Normal | Anomali |
|--------|--------|---------|
| `utilization_ratio` ↔ `txn_count_weekly` | Kullanim artinca islem de artar | Kullanim yuksek ama islem yok → limit dolmus, odeyemiyor |
| `dpd_current` ↔ `min_payment_only_count_4w` | Gecikme yoksa min odeme de yok | DPD=0 ama 4 haftadir min oduyor → gizli stres |
| `checking_balance` ↔ `outstanding_balance` | Bakiye yuksekken borc kontrol altinda | Bakiye erirken borc artiyor → cift sikisma |
| `deposit_amount_avg_4w` ↔ `balance_slope_4w` | Mevduat geliyorsa borc stabil | Mevduat azaliyor ama borc artiyor → gelir erozyonu |
| `txn_count_weekly` ↔ `txn_amount_weekly` | Islem sayisi ve tutar birlikte hareket eder | Az islem ama yuksek tutar → aliskanlik kirilmasi |

---

## Referanslar

- Khandani, Kim & Lo (2010) — Consumer Credit-Risk Models via ML: 71 degisken, top-10 predictor listesi
- Butaru et al. (2016) — Risk and Risk Management in Credit Card Industry: min_payment, cash_advance
- Figini et al. (2017) — Behavioral Scoring for Retail Credit: channel diversity, payment pattern
- Tiger Analytics — ML-driven EWS for SME Credit: credit/debit turnover, fund transfer
- BIS/BCBS — Credit-to-GDP gap, Debt Service Ratio
- McKinsey — Credit Monitoring for Competitive Advantage: "trend > seviye"
- IFRS 9 — SICR backstop: 30 DPD, PD artis, watchlist
