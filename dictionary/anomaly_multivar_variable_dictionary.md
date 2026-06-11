# anomaly_multivar Variable Dictionary

Bu sozluk `anomaly_multivar` aylik musteri anomali akisi icindir.

## Veri Tane Yapisi

| Alan | Deger |
| --- | --- |
| Entity | `mono_id` |
| Zaman | `cohort_dt` |
| Frekans | aylik |
| Skorlanan ay | varsayilan en guncel `cohort_dt` |
| Train kapsamı | skorlanan ay oncesindeki tum prior kayitlar; cap yalniz CLI/config ile sayi verilirse uygulanir |
| Skor kalibrasyonu | train cok buyukse deterministik kalibrasyon orneklemi kullanilir; model fit ve peer istatistikleri tum train kapsamindan gelir |
| Kaynak tablo | `ZT_VAR2.EWS_ANOMALY_MULTIVAR_INPUT` |
| Skor tablo | `ZT_VAR2.EWS_ANOMALY_MULTIVAR_RESULTS` |
| Reason tablo | `ZT_VAR2.EWS_ANOMALY_MULTIVAR_DETAILS` |

## Model Disi Metadata

Bu kolonlar model feature'i degildir; veri anlamlandirma, donem, yukleme veya filtre bilgisi icin tutulur.

| Kolon | Kullanim |
| --- | --- |
| `financial_term_l1y` | L1Y finansalin ait oldugu mali donem |
| `bilanco_flg` | Bu veri setinde tum kayitlarda 1; model disi |
| `financial_term_q` | Ara donem finansalin ait oldugu donem |
| `annualization_q` | Ara donem yilliklandirma bilgisi |
| `ref_donem_id` | Referans donem id |
| `kkbguncelsorgu_no` | Sorgu teknik anahtari |
| `yukleme_zmn` | Yukleme zamani |

## Feature Politikasi

Bu versiyonda finansal module kendi icinde rasyo uretilmez. Finansal kolonlar ayni anda doldugu icin bu tip oranlar musteriyi izleyen ekip tarafindan dogrudan fark edilebilir ve reason kalitesini dusurebilir.

Tum oranlarda paydanin pozitif ve anlamli buyuklukte olmasi beklenir. Payda sifir, negatif veya ilgili paydanin tipik seviyesine gore cok kucukse oran uretilmez; bu durum finansal bozulma reason'i yerine missing/veri kalitesi sinyali olarak ele alinir. Mutlak orani `1000` uzerine tasiyan kayitlar da is kararinda yorumlanabilir olmadigi icin model feature'ina alinmaz.

Finansal module kabul edilen kolonlar:

| Grup | Kolon ornekleri |
| --- | --- |
| L1Y finansal | `fs_net_sales_cumulative_l1y`, `fs_trade_receivables_l1y`, `fs_notes_receivable_l1y`, `fs_net_profit_cumulative_l1y`, `equity_l1y`, `supheli_ticari_alacaklar_l1y` |
| Ara donem finansal | `fs_net_sales_cumulative_q`, `fs_ebitda_cumulative_q`, `fs_net_profit_cumulative_q`, `fs_trade_receivables_q`, `fs_notes_receivable_q`, `fs_equity_q`, `supheli_alacaklar_q` |

Kullanilmayan turev tipleri:

| Tip | Ornek | Durum |
| --- | --- | --- |
| Finansal / finansal | `fs_net_profit_cumulative_l1y / fs_net_sales_cumulative_l1y` | kullanilmaz |
| Q finansal / L1Y finansal | `fs_net_sales_cumulative_q / fs_net_sales_cumulative_l1y` | kullanilmaz |
| Finansal alacak / finansal satis | `fs_trade_receivables_l1y / fs_net_sales_cumulative_l1y` | kullanilmaz |
| Finansal supheli alacak / finansal alacak | `supheli_ticari_alacaklar_l1y / fs_trade_receivables_l1y` | kullanilmaz |
| `A * B` carpim | `pd * debt` | kullanilmaz |
| `A - B` cikarma | `pd_rating - pd_model` | kullanilmaz |
| `A + B` toplam | `bank_risk + memzuc_risk` | kullanilmaz |
| cross/stress/weighted feature | `cross_pd_debt_stress` | kullanilmaz |

Izin verilen tipler:

| Tip | Ornek |
| --- | --- |
| Kredi risk / varlik | `bank_total_risk / toplam_varlik_ttr` |
| Memzuc / banka kredi riski | `memzuc_total_risk / bank_total_risk` |
| Kredi risk / finansal olcek | `bank_total_risk / fs_net_sales_cumulative_l1y` |
| Finansal kalem / dis varlik normalizer | `fs_trade_receivables_l1y / toplam_varlik_ttr` |
| Internal / finansal olcek | `gunceltkn_dgr / fs_net_sales_cumulative_l1y` |
| PD orani | `irb_rating_pd / irb_model_pd` |
| Peer-relative z-score | secilen oranin peer grubuna gore robust z-score'u |

## Uretilen Feature'lar

| Feature | Tanim | Formul |
| --- | --- | --- |
| `memzuc_limit_utilization` | Memzuc limit kullanim orani | `memzuc_total_risk / memzuc_total_limit` |
| `memzuc_st_mt_cash_share` | Memzuc KV/OV nakdi risk payi | `memzuc_st_mt_cash_risk / memzuc_total_risk` |
| `bank_risk_to_assets` | Banka risk / varlik | `bank_total_risk / toplam_varlik_ttr` |
| `memzuc_risk_to_assets` | Memzuc risk / varlik | `memzuc_total_risk / toplam_varlik_ttr` |
| `l1y_equity_to_assets` | L1Y ozkaynak / varlik | `equity_l1y / toplam_varlik_ttr` |
| `q_equity_to_assets` | Ara donem ozkaynak / varlik | `fs_equity_q / toplam_varlik_ttr` |
| `l1y_debt_to_sales` | Banka risk / L1Y satis | `bank_total_risk / fs_net_sales_cumulative_l1y` |
| `q_debt_to_sales` | Banka risk / ara donem satis | `bank_total_risk / fs_net_sales_cumulative_q` |
| `memzuc_debt_to_l1y_sales` | Memzuc risk / L1Y satis | `memzuc_total_risk / fs_net_sales_cumulative_l1y` |
| `memzuc_debt_to_q_sales` | Memzuc risk / ara donem satis | `memzuc_total_risk / fs_net_sales_cumulative_q` |
| `memzuc_to_bank_risk_ratio` | Memzuc risk / banka risk | `memzuc_total_risk / bank_total_risk` |
| `bank_to_memzuc_risk_ratio` | Banka risk / memzuc risk | `bank_total_risk / memzuc_total_risk` |
| `l1y_trade_receivables_to_assets` | L1Y ticari alacak / varlik | `fs_trade_receivables_l1y / toplam_varlik_ttr` |
| `l1y_notes_receivable_to_assets` | L1Y senetli alacak / varlik | `fs_notes_receivable_l1y / toplam_varlik_ttr` |
| `q_trade_receivables_to_assets` | Ara donem ticari alacak / varlik | `fs_trade_receivables_q / toplam_varlik_ttr` |
| `q_notes_receivable_to_assets` | Ara donem senetli alacak / varlik | `fs_notes_receivable_q / toplam_varlik_ttr` |
| `pd_ratio` | IRB rating PD / model PD | `irb_rating_pd / irb_model_pd` |
| `internal_tkn_to_assets` | TKN / varlik | `gunceltkn_dgr / toplam_varlik_ttr` |
| `internal_tbe_to_assets` | TBE / varlik | `gunceltbe_dgr / toplam_varlik_ttr` |
| `internal_tkn_to_sales` | TKN / L1Y satis | `gunceltkn_dgr / fs_net_sales_cumulative_l1y` |
| `internal_tbe_to_sales` | TBE / L1Y satis | `gunceltbe_dgr / fs_net_sales_cumulative_l1y` |
| `internal_tkn_tbe_ratio` | TKN / TBE | `gunceltkn_dgr / gunceltbe_dgr` |

Not: `pd_to_rating_group` uretilmez. PD ve rating grup ayni risk bilgisinin farkli sayisal gosterimleri oldugu icin model feature'i veya reason kaynagi olarak birbirine oranlanmaz.

## Modelde Kullanilan Nihai Deger

Model ham oranlari dogrudan degil, peer grubuna gore olusturulan robust z-score hallerini kullanir:

`<feature>__peer_z`

Reason secimi model katkisini baz alir, ancak erken uyari incelemesi icin risk artisi veya missing/veri kalitesi sinyalleri ilk siraya alinir. Risk azalisi olan anomaliler ancak yeterli risk-artisi reason yoksa ilk uc reason'a girer.

Peer hiyerarsisi:

1. `cohort_dt + musteri_segment + rating_group + sektor + size_bucket`
2. `cohort_dt + musteri_segment + rating_group + size_bucket`
3. `cohort_dt + musteri_segment + sektor`
4. `cohort_dt + musteri_segment + size_bucket`
5. `cohort_dt + musteri_segment`
6. `cohort_dt`

PD ile baslayan feature'larda rating_group peer kirilimi kullanilmaz. Bu durumda hiyerarsi `cohort_dt + musteri_segment + sektor + size_bucket` seviyesinden baslar.

## Peer Temsil Kabiliyeti

Her reason icin peer kalitesi ayrica uretilir:

| Alan | Tanim |
| --- | --- |
| `peer_level` | Kullanilan peer fallback seviyesi |
| `peer_support` | O feature icin peer medyanini olusturan musteri sayisi |
| `peer_representativeness_score` | Peer seviyesinin darligi ve support sayisindan 0-100 arasi temsil skoru |
| `peer_quality` | `GUCLU`, `KABUL_EDILEBILIR`, `ZAYIF` |

Kurumsal yorum:

| Sinif | Yorum |
| --- | --- |
| `GUCLU` | Dar segment/rating/sektor/size seviyesinde yeterli support var; reason karar icin guclu kabul edilir |
| `KABUL_EDILEBILIR` | Peer anlamli ancak izleme ekibi feature ve support ile birlikte okumali |
| `ZAYIF` | Peer fazla genis veya support zayif; reason tek basina karar nedeni olmamali |

Run summary icinde hem tum skor-feature peer karsilastirmalari hem de final reason detaylari icin peer kalite dagilimi, support quantile'lari ve kurumsal peer assessment yazilir.

Peer anlamlilik testi run summary icinde `meaningfulness_test` olarak PASS/WARN/FAIL uretir. Test kriterleri:

| Kriter | Esik |
| --- | --- |
| Kabul edilebilir veya guclu peer orani | >= %95 |
| Zayif peer orani | <= %5 |
| P10 peer support | >= 50 |
| Medyan peer support | >= 100 |
| Dar peer seviyesi orani | >= %75 |
