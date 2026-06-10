# anomaly_multivar Variable Dictionary

Bu sozluk `anomaly_multivar` aylik musteri anomali akisi icindir.

## Veri Tane Yapisi

| Alan | Deger |
| --- | --- |
| Entity | `mono_id` |
| Zaman | `cohort_dt` |
| Frekans | aylik |
| Skorlanan ay | varsayilan en guncel `cohort_dt` |
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
| `pd_to_rating_group` | PD / rating grup | `irb_rating_pd / rating_group` |
| `internal_tkn_to_assets` | TKN / varlik | `gunceltkn_dgr / toplam_varlik_ttr` |
| `internal_tbe_to_assets` | TBE / varlik | `gunceltbe_dgr / toplam_varlik_ttr` |
| `internal_tkn_to_sales` | TKN / L1Y satis | `gunceltkn_dgr / fs_net_sales_cumulative_l1y` |
| `internal_tbe_to_sales` | TBE / L1Y satis | `gunceltbe_dgr / fs_net_sales_cumulative_l1y` |
| `internal_tkn_tbe_ratio` | TKN / TBE | `gunceltkn_dgr / gunceltbe_dgr` |

## Modelde Kullanilan Nihai Deger

Model ham oranlari dogrudan degil, peer grubuna gore olusturulan robust z-score hallerini kullanir:

`<feature>__peer_z`

Peer hiyerarsisi:

1. `cohort_dt + musteri_segment + rating_group + sektor + size_bucket`
2. `cohort_dt + musteri_segment + rating_group + size_bucket`
3. `cohort_dt + musteri_segment + sektor`
4. `cohort_dt + musteri_segment + size_bucket`
5. `cohort_dt + musteri_segment`
6. `cohort_dt`
