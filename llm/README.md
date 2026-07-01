# LLM Anomaly Prototype

Bu klasor, ekran goruntulerindeki LLM scriptinin revize edilmis ve proje icinde izole edilmis halidir.

Ana farklar:

- API key kodda tutulmaz; `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` env degiskenleri veya `secret/secrets.yaml` kullanilir.
- LLM'e ham tablo degil, denetlenebilir `evidence JSON` verilir.
- Evidence icinde veri sozlugu, risk yonu, musteri gecmisi, rolling medyanlar, trend, sezon, cari peer ve tarihsel musteri/peer snapshot serileri birlikte bulunur.
- Gelecek donemler LLM'e verilmez; skorlanan ay sadece onceki aylarla kiyaslanir.
- LLM skor motoru degil, dogrudan `is_anomaly` karari veren uzman karar katmani olarak calisir.

## Variable Dictionary

Degisken sozlugu ve feature deneme noktasi `config/dictionaries.yaml` dosyasidir.

- `raw_variables.groups`: ham kolonlari kaynak basliklarina gore tutar (`time_var`, `id_var`, `demographic_var`, `internal_risk_var`, `financial_l1_var`, `financial_q_var`, `memzuc_var`, `internal_rate_var`, `varlik_var`, `kkb_var`, `technical_var`). Her kolon icin kaynak, tanim, rol ve okunabilir aciklama vardir.
- `generated_variables.variables`: ham kolonlardan uretilen oran/transform feature'larini tutar. `formula` alaninda `+`, `-`, `*`, `/` ve parantez kullanilabilir. `/` islemi pipeline'in `safe_divide` kuralini kullanir.
- `final_llm_features.include`: LLM'e gitmesini istedigin final feature listesidir.
- `final_llm_features.exclude`: LLM'e gitmeyecek degiskenlerdir. Numeric PD degerleri ve PD/rating oranlari burada kapali tutulur.
- `risk_direction`: `HIGHER_IS_RISKY` ise artan deger risk bozulmasi, azalan deger iyilesme; `LOWER_IS_RISKY` ise azalan deger risk bozulmasi, artan deger iyilesme olarak yorumlanir.

Yeni bir deneme icin once `generated_variables.variables` altina feature ekle, sonra LLM'e gitsin istiyorsan ayni feature adini `final_llm_features.include` listesine ekle. `enabled` veya `source_module` kullanilmaz; kapatmak istedigin feature'i `final_llm_features.exclude` listesine al.

## Evidence Uretme

Ham input CSV/Excel varsa:

```powershell
python -m llm.llm_anomaly build-evidence anomaly_multivar.csv runtime/llm/evidence.jsonl --max-customers 10
```

Sadece mevcut runtime skor ciktisi varsa, score alanlarini LLM'e vermeden `reason_details` uzerinden evidence uret:

```powershell
python -m llm.llm_anomaly build-evidence runtime/multivar_anomaly/20251231_20260611_165408/anomaly_multivar_scores_20251231.csv runtime/llm/evidence_from_results.jsonl --from-results --max-customers 10
```

Oracle input tablosundan tam history/season/peer evidence uretmek icin:

```powershell
python -m llm.llm_anomaly build-oracle runtime/llm/evidence_oracle.jsonl --max-customers 10 --max-train-rows 300000 --series-periods 6
```

Bu komut sadece evidence dosyasi uretir. LLM'e gitmez ve Oracle output tablolarina insert yapmaz.

Bu komut `ZT_VAR2.EWS_ANOMALY_MULTIVAR_INPUT` ham input tablosundan okur, gerekli turetilmis feature'lari uretir ve LLM'e gidecek evidence JSONL dosyasini yazar.

Oracle baglanti ayari secret YAML dosyasindan okunur. Kod once env ile verilen path'i, sonra repo ve notebook workspace koklerini dener:

1. `EWS_ANOMALY_SECRETS_PATH` veya `RISK_PIPELINE_SECRETS_PATH`
2. Calisilan dizin/repo ve parent dizinlerde `secret/secrets.yaml`
3. Calisilan dizin/repo ve parent dizinlerde `secrets.yaml`
4. Notebook workspace'te yanlislikla olusan `secrets .yaml`

Standart repo ici kullanim icin dosyayi `secret/secrets.yaml` olarak tut. Kurumsal notebook workspace'inde secret parent dizinde duruyorsa ekstra path vermeden bulunur.

Direkt kullanici ile baglanacaksan:

```yaml
oracle:
  sections:
    ORA_PRD_ZTUSER:
      user: "<oracle-user>"
      password: "<oracle-user-password>"
      host: "<oracle-host>"
      port: 1521
      service_name: "<oracle-service>"
```

Proxy auth kullanacaksan:

```yaml
oracle:
  sections:
    ORA_PRD_ZTUSER:
      user: "<proxy-user>[<target-schema-or-user>]"
      password: "<proxy-user-password>"
      host: "<oracle-host>"
      port: 1521
      service_name: "<oracle-service>"
```

Input/output tablo owner bilgileri credential tarafinda degil `config/pipeline_config.yaml` icindedir. Her tablo ayri owner/schema ile yazilabilir:

Config dosyasi secilirken `pipeline` ve `oracle` root alanlari zorunlu kabul edilir. Parent workspace'te baska projeye ait `pipeline_config.yaml` varsa ve bu alanlari icermiyorsa otomatik atlanir; repo icindeki `config/pipeline_config.yaml` kullanilir.

```yaml
oracle:
  section: ORA_PRD_ZTUSER
  tables:
    multivar_input:
      owner: X1
      table: EWS_ANOMALY_MULTIVAR_INPUT
    llm_results:
      owner: X2
      table: EWS_ANOMALY_LLM_RESULTS
    llm_reason_details:
      owner: X2
      table: EWS_ANOMALY_LLM_REASONS
    llm_feature_details:
      owner: X2
      table: EWS_ANOMALY_LLM_FEATURES

llm:
  outputs:
    oracle:
      # replace: ayni scoring month icin eski LLM output satirlarini silip yeni run'i yazar.
      # append: eski satirlari silmeden yeni run'i insert eder.
      write_mode: replace
```

`append` ayni musteri/ay icin birden fazla run saklamak icin kullanilacaksa LLM output tablolarinin primary key yapisi `RUN_ID` icermelidir. Yeni olusturulan tablolar bu yapiyla kurulur; eski tabloda eski PK varsa kontrollu migration gerekir.

## LLM Key ve Endpoint Ayarlari

LLM ayarlari su sirayla okunur:

1. Terminal env degiskenleri: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
2. Lokal dosyalar: repo kokundeki `.env` ve `llm/.env.local`
3. Secret dosyasi: env ile verilen path, repo/workspace `secret/secrets.yaml`, repo/workspace `secrets.yaml`
4. Model cagrisinda `ChatOpenAI` icine timeout parametresi verilmez; referans notebooktaki davranis korunur.
5. Internal endpoint icin `httpx.Client(trust_env=False, timeout=None)` kullanilir; yani `HTTP_PROXY/HTTPS_PROXY` env degerleri varsayilan olarak LLM request'ine uygulanmaz.

`base_url`, `api_key` ve `model` zorunludur. Bu proje on-prem/internal endpoint kullanir; dis `https://api.openai.com/v1` adresine otomatik gecis yapmaz.

Env degiskeni vermek istemiyorsan `secret/secrets.yaml` icine sunu koy:

```yaml
llm:
  section: OPENSHIFT_LLM
  sections:
    OPENSHIFT_LLM:
      base_url: "https://manavgat.yzyonetim.zb/v1"
      api_key: "<valid-key>"
      model: "gpt-oss-20b"
      # Sadece LLM endpoint'e proxy uzerinden gitmek zorundaysan true yap.
      # http_trust_env: false
      # Kurum/internal CA bundle path'i. Bos birakilirsa env ve yaygin Linux path'leri denenir.
      # ca_bundle: "/etc/pki/tls/certs/ca-bundle.crt"
      ssl_verify: false
```

Baska bir LLM section secmek icin:

```bash
export LLM_SECTION=OPENSHIFT_LLM
```

veya PowerShell:

```powershell
$env:LLM_SECTION="OPENSHIFT_LLM"
```

Logda key yazilmaz; sadece `key_source=env:LLM_API_KEY` veya `key_source=secret/secrets.yaml ...` gibi kaynak bilgisi gorulur.

Structured response ilk prototipteki gibi LangChain/Pydantic uzerinden zorlanir: `ChatOpenAI`, `ChatPromptTemplate`, Pydantic `BaseModel/Field`, `with_structured_output(...)`, `chain.invoke(...)`.

Varsayilan LLM cagri sekli ilk calisan kaynak kodun operasyonel kalibiyla aynidir: `llm.with_structured_output(AnomalyBatchResult)`, `prompt | structured_llm`, `chain.invoke({"input_records": ...})`. `method=...` override verilmez; internal endpoint LangChain default structured davranisiyla cagrilir. Her musteri snapshot'i tek prompt olarak gider ve tek structured karar objesi beklenir.

Cikti tipi tek satirlik musteri-snapshot karari olacak sekilde tutulur. Structured obje `period_position`, `is_anomaly`, `anomaly_type`, `anomaly_score`, `reason_summary`, `reason_1/2/3`, `reason_1_weight/2_weight/3_weight`, `risk_level` doner. `mono_id`, `cohort_dt` ve Oracle output icin gerekli uyumluluk alanlari kod tarafinda evidence kaydindan doldurulur.

Model feature veya neden bazinda birden fazla `results` item'i dondurmemelidir. Musteri datasinin history/series bilgisi yeterliyse peer tek basina anomali nedeni olamaz; peer sadece musteri history bozulmasini destekleyen kanit olarak kullanilir.

Output format parse edilebilir olmak zorundadir: tek satir JSON object, markdown/code fence yok, JSON string wrapper yok, Python repr yok. Internal endpoint bazen structured parser'a tool-call objesi vermek yerine JSON'u `AIMessage.content` icinde duz metin olarak dondurur; kod bu durumda raw content icindeki JSON objeyi okuyup ayni tek satir karar kontratina sokar.

`reason_summary`, `reason_1`, `reason_2`, `reason_3` ve LLM kaynakli aciklama alanlari Turkce olmalidir. Alan adlari Oracle/JSON kontrati geregi teknik isim olarak kalir.

Evidence, ham nested JSON dump olarak degil ayni bilgileri tasiyan kompakt text olarak gonderilir; bu feature veya veri azaltma degildir, gereksiz token sismesini azaltmak icindir.

`timeout_seconds` eski config/secret dosyalarinda kalsa bile LLM scoring cagrisina aktarilmaz. Bu bilerek yapildi; onceki calisan notebook `ChatOpenAI` icine timeout vermedigi icin uzun structured cevaplarda client tarafinda erken kesme olmuyordu.

`httpcore/_sync/http_proxy.py` ve `[SSL: UNEXPECTED_EOF_WHILE_READING]` gorulurse request env proxy uzerinden gitmis olabilir. Varsayilan `http_trust_env=False` bunu kapatir. Gercekten proxy gerekiyorsa `LLM_HTTP_TRUST_ENV=true` veya secret altinda `http_trust_env: true` kullan.

Internal endpoint icin `ssl_verify` varsayilan olarak `false` gelir; sertifika zinciri dogrulamasi yapilmaz. Daha sonra kurum CA'si ile dogrulama acilmak istenirse `ssl_verify: true` ve gerekirse `ca_bundle: "/path/to/kurum-ca.pem"` verilebilir. Kod sirasiyla `LLM_CA_BUNDLE`, `LLM_SSL_CERT_FILE`, secret `ca_bundle`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` ve yaygin Linux CA path'lerini dener.

Logda su satirlar gorulmelidir:

```text
LLM settings resolved: ... timeout_seconds=None max_retries=0 max_tokens=None http_trust_env=False proxy_env_present=True ssl_verify=False ca_bundle=... structured_call=with_structured_output_schema_only client=langchain_structured
LangChain structured LLM chain initialized: model=gpt-oss-20b structured_call=with_structured_output_schema_only include_raw=True max_retries=0 max_tokens=None http_trust_env=False ssl_verify=False ca_bundle=... raw_response_file=runtime/llm/raw_model_responses.jsonl
LLM request payload prepared: mono_id=... decision_items=... formatter=compact_text
========== LLM PAYLOAD PREVIEW 1/3 START | mono_id=... chars=... ==========
period_position=0 | mono_id=... | cohort_dt=... | context=... | decision_contract=... | peer_definition=... | data_quality=...
feature name=... | current=... | history=... | trend=... | seasonality=... | peer=...
========== LLM PAYLOAD PREVIEW 1/3 END | mono_id=... ==========
```

Ilk 3 musteri icin bu preview bloklari loga basilir. Daha sonra `ConnectionError`, route kopmasi veya endpoint hatasi olursa hata satirinda da `mono_id`, `decision_items`, payload `chars` ve kisaltilmis `payload_preview` gorulur.

Model HTTP 200 donup structured parse basarisiz olursa ham model cevabi su dosyaya append edilir. `parsed=None` ama `raw_response.content` icinde gecerli tek JSON object varsa kod bunu parse edip devam eder:

```text
runtime/llm/raw_model_responses.jsonl
```

Farkli dosya icin:

```bash
export LLM_RAW_RESPONSE_FILE="runtime/llm/raw_model_responses_debug.jsonl"
```

Endpoint ve key saglik kontrolu icin notebook:

```text
llm/llm_endpoint_healthcheck.ipynb
```

Bu notebook sirasiyla repo root bulma, `secret/secrets.yaml` path/section/key varligi kontrolu, config okuma, TCP/TLS handshake, `/models`, repo icindeki LangChain/Pydantic structured call ve opsiyonel ham LangChain call testlerini yapar. Ana saglik kontrolu 4. hucredeki structured call'dir; opsiyonel ham LangChain testi varsayilan olarak skip edilir.

## Promptu Dry Run Gormek

```powershell
python -m llm.llm_anomaly run runtime/llm/evidence.jsonl runtime/llm/prompts.jsonl --from-evidence --dry-run
```

## LLM Karari Almak

Yerel env dosyasi kullanmak icin `llm/.env.local.example` dosyasini `llm/.env.local` olarak kopyala ve key'i oraya yaz. `llm/.env.local` git'e alinmaz.

Kurum ici OpenAI-compatible endpoint icin:

```powershell
$env:LLM_BASE_URL="https://manavgat.yzyonetim.zb/v1"
$env:LLM_API_KEY="<rotated-or-valid-key>"
$env:LLM_MODEL="gpt-oss-20b"
python -m llm.llm_anomaly run runtime/llm/evidence_oracle_sample.jsonl runtime/llm/decisions.jsonl --from-evidence
```

Linux/OpenShift terminalinde ayni ayarlar:

```bash
export LLM_BASE_URL="https://manavgat.yzyonetim.zb/v1"
export LLM_API_KEY="<valid-key>"
export LLM_MODEL="gpt-oss-20b"
```

Oracle ham inputtan oku, LLM kararini al ve ayri Oracle tablolarina yaz:

```powershell
python -m llm.llm_anomaly run-oracle runtime/llm/decisions_oracle.jsonl --max-customers 10 --max-train-rows 300000
```

Linux/OpenShift terminalinde:

```bash
python -m llm.llm_anomaly run-oracle runtime/llm/decisions_oracle_10.jsonl \
  --max-customers 10 \
  --max-train-rows 300000 \
  --top-features 12 \
  --series-periods 6 \
  --customer-selection-mode ml-balanced
```

Belirli bir cohort ayini skorlamak icin `--scoring-month` ver:

```bash
python -m llm.llm_anomaly run-oracle runtime/llm/decisions_oracle_20260531.jsonl \
  --scoring-month 2026-05-31 \
  --max-customers 10 \
  --max-train-rows 300000
```

`--scoring-month` verilmezse kaynak Oracle input tablosundaki en buyuk `cohort_dt` otomatik secilir. Logda su satirla gorulur:

```text
SCORING COHORT SELECTED | requested=latest selected=2026-05-31 selection_mode=auto latest cohort_dt ...
```

Manuel secimde:

```text
SCORING COHORT SELECTED | requested=2026-04-30 selected=2026-04-30 selection_mode=manual --scoring-month ...
```

Onemli:

- `build-oracle`: sadece evidence JSONL uretir, LLM'e gitmez, Oracle output insert yapmaz.
- `run-oracle`: Oracle'dan okur, evidence uretir, LLM'e gider, structured output'u Oracle tablolarina yazar.
- `run-oracle` varsayilan olarak once tum scoring cohort'u ML ile skorlar. Sonra LLM'e gidecek 10 musteriyi bu ML sonucundan secer: `--ml-balanced-anomaly-count 5` kadar en yuksek ML anomaly bucket'i ve kalan kadar `NORMAL` bucket referans musteri.
- Eski Oracle sirasi ile secim istenirse `--customer-selection-mode first` kullan.
- LLM sonucuna ML karsilastirma kolonlari eklenir: `ML_ENSEMBLE_SCORE`, `ML_ANOMALY_SCORE` (geriye uyumlu ensemble alias), `ML_IS_ANOMALY`, `ML_ALERT_BAND`, `ML_IF_SCORE`, `ML_RESIDUAL_SCORE`, `ML_AUTOENCODER_SCORE`. Bu skorlar LLM promptuna verilmez; sadece karsilastirma icin output satirina yazilir.
- Result tablosunda `LLM_CONFIDENCE` geriye uyumlu alan olarak `ANOMALY_SCORE` ile doldurulur; mevcut kontratta ayrica confidence kavrami yoktur.
- `SEASONALITY_ASSESSMENT`, `TREND_ASSESSMENT` ve `PEER_ASSESSMENT` LLM decision satirina bagli evidence feature detaylarindan uretilen kisa sayisal ozetlerdir.
- `CAVEAT`, karar yorumunun hangi veri kisitlariyla okunmasi gerektigini anlatan uyari/not alanidir; ornegin dusuk coverage, eksik history, stale veri veya zayif peer gibi durumlari belirtir.
- ML companion skoru istenmezse `--skip-ml-companion` kullan.
- `--dry-run`: LLM'e gitmez ve Oracle output insert yapmaz; sadece prompt/evidence kontrolu icindir.
- `--max-customers 10`: LLM'e gidecek scoring snapshot/musteri sayisidir. Her musteri scoring ayinda 1 karar satiri olarak gider; history satirlari insert edilmez, evidence icinde baglam olarak kullanilir.
- `--customer-selection-mode ml-balanced`: ML skorlamayi full cohort uzerinde yapar, LLM ornegini ML sonucuna gore dengeli secer. Secilen musteriler `runtime/llm/ml_balanced_selected_customers.csv` dosyasina yazilir.
- `--max-train-rows 300000`: LLM'e 300k satir gondermez. History, peer, trend ve seasonality referanslarini hesaplamak icin kullanilan gecmis/reference ust limitidir.
- `--series-periods 6`: Her feature icin LLM promptuna girecek musteri ve peer snapshot serisi uzunlugudur. Musterinin toplam 5 snapshot'i varsa 5'i de gider; daha uzun pencere icin bu degeri artir.
- Secilen musterilerin tam gecmisi ayrica cekilir; bu sayede `max_train_rows` sampling'i secilen musterinin history'sini dusurmez.
- Cari peer referansi `max-customers` ile secilen 10 musteri uzerinden degil, skorlanan ayin tum scoring cohort'u uzerinden hesaplanir. `snapshot_series.peer` ise her tarihsel snapshot ayi icin ayni peer hiyerarsisiyle tekrar hesaplanan peer median/support/quality bilgisini tasir.
- Evidence hazirligi Oracle path'te full train/score pencerelerini korur ama artik dev bir `combined` dataframe olusturup tekrar split etmez. Secilen musteri history'si onceden gruplanir, seasonal peer medyanlari ve robust scale degerleri scoring ayina gore cache'lenir. Bu veri veya feature azaltma degildir; ayni referans veriyi tekrar tekrar taramayi azaltir.

Output tablolarini run oncesi olusturmak/kontrol etmek icin:

```bash
python -m llm.llm_anomaly ensure-output-tables --scoring-month 2026-05-31
```

LLM karar tablolari:

- `ZT_VAR2.EWS_ANOMALY_LLM_RESULTS`: musteri-donem seviyesinde `IS_ANOMALY`, `ANOMALY_TYPE`, `RISK_LEVEL`, LLM `ANOMALY_SCORE`, ML comparison kolonlari (`ML_ENSEMBLE_SCORE`, `ML_ANOMALY_SCORE`, `ML_IS_ANOMALY`, `ML_ALERT_BAND`, `ML_IF_SCORE`, `ML_RESIDUAL_SCORE`, `ML_AUTOENCODER_SCORE`), `REASON_SUMMARY`, `REASON_1/2/3`, `REASON_1_WEIGHT/2_WEIGHT/3_WEIGHT` ve raw JSON response.
- `ZT_VAR2.EWS_ANOMALY_LLM_REASONS`: tek karar icindeki top reason alanlarinin detay satirlari.
- `ZT_VAR2.EWS_ANOMALY_LLM_FEATURES`: her karar satiri icin hesaplanan tum feature evidence satirlari. Current/previous/change, history median-p25-p75-robust scale-history z, rolling medianlar, trend, sezon, peer median/z/support/quality, snapshot series JSON ve full feature JSON burada gorulur.

## Terminalde Beklenen Akis

`run-oracle` calistiginda terminalde ve `runtime/logs/cli/llm_anomaly.log` dosyasinda su basliklar gorulmelidir:

```text
STEP 00 START | LLM Oracle anomaly run basladi
STEP 00M START/DONE | LLM oncesi tum scoring cohort icin ML anomaly skorlamasi yapiliyor
STEP 01 START/DONE | Oracle kaynak tablo ve ay profili okunuyor
STEP 02 START/DONE | Ham tablo kolonlari ve veri sozlugu denetleniyor
STEP 03 START/DONE | Musteri bazli history ve aylik peer gruplariyla LLM evidence uretiliyor
STEP 04 START/DONE | LLM modelinden anomali karari aliniyor
STEP 04M START/DONE | Full cohort ML skorlarindan LLM karar satirlarina karsilastirma kolonlari ekleniyor
STEP 05 START/DONE | LLM kararlari Oracle output tablolarina yaziliyor
```

STEP 03 icinde secilen her musteri icin su ayrim gorulur:

```text
LLM scoring payload prepared: mono_id=... scoring_cohort_dt=2026-05-31 customer_history_periods=5 history_first_cohort_dt=... history_last_cohort_dt=... output_rows_for_customer=1
```

STEP 04 icinde `decision_items=1` scoring snapshot karar sayisidir; `customer_history_periods` ise bu tek karar icin prompta giren musteri gecmisini gosterir.

Bir adim yapilamazsa logda `FAILED` veya `SKIPPED` ve nedeni yazilir. Ornek:

```text
STEP 04 FAILED | LLM API key is required. Set LLM_API_KEY/OPENAI_API_KEY env variable or secret/secrets.yaml llm.api_key / llm.sections.<section>.api_key.
STEP 05 SKIPPED | LLM karar uretilemedi; Oracle output tablolari doldurulmadi
```

## Degisken Aileleri

Loglarda ham ve turetilmis degiskenler kategoriyle yazilir:

- `memzuc`: memzuc toplam risk/limit/nakdi risk ve bunlardan uretilen oranlar.
- `bank_risk`: banka toplam risk ve banka risk oranlari.
- `financial`: mali tablo, L1Y ve ara donem finansal alanlar.
- `internal_kkb`: TKN/TBE/KKB tabanli internal sinyaller.
- `pd_rating`: rating grubu ve PD alanlari. LLM feature setinde rating kullanilir; PD numeric degerleri kullanilmaz.
- `context`: segment, sektor, NACE, referans donem gibi gruplama/aciklama alanlari.
- `technical`: `data_time`, `created_at`, teknik yukleme alanlari.

PD/rating notu:

- `rating_group` ve varsa ham `irb_rating` sinyali kullanilabilir.
- `irb_rating_pd`, `irb_model_pd`, `pd_ratio`, `pd_to_rating_group` gibi PD degeri veya PD karsilastirmasi iceren feature'lar LLM inputundan cikarilir.
- Peer grubu rating ile daraltilmaz; ana hiyerarsi ay + segment + sektor + aylik buyukluk sirasidir.

## Output Insert Kontrolu

Run sonunda logda su satirlar gorulmelidir:

```text
AUDIT OUTPUT TABLE | table_key=llm_results ... inserted=10 ... run_rows_after=10
AUDIT OUTPUT TABLE | table_key=llm_reason_details ... inserted=<reason_count> ... run_rows_after=<reason_count>
AUDIT OUTPUT TABLE | table_key=llm_feature_details ... inserted=<feature_count> ... run_rows_after=<feature_count>
```

Manuel Oracle kontrolu icin:

```sql
SELECT COUNT(*) FROM ZT_VAR2.EWS_ANOMALY_LLM_RESULTS
WHERE TRUNC(COHORT_DT) = DATE '2026-05-31';

SELECT COUNT(*) FROM ZT_VAR2.EWS_ANOMALY_LLM_REASONS
WHERE TRUNC(COHORT_DT) = DATE '2026-05-31';

SELECT COUNT(*) FROM ZT_VAR2.EWS_ANOMALY_LLM_FEATURES
WHERE TRUNC(COHORT_DT) = DATE '2026-05-31';
```

Model cagrisi ilk prototipteki operasyonel kalipla yapilir: `ChatOpenAI`, `ChatPromptTemplate`, Pydantic `BaseModel/Field`, `llm.with_structured_output(...)` ve `chain.invoke(...)`. Basarili cevap tek structured karar objesi olarak okunur. Ek parser veya dis endpoint gecisi yoktur; sadece hata analizinde kullanmak icin LangChain'in raw modeli `runtime/llm/raw_model_responses.jsonl` dosyasina yazilir.

Eger healthcheck'te `TypeError('issubclass() arg 1 must be a class')` gorursen once repo kodunun guncel oldugunu ve kernelin yeniden baslatildigini kontrol et. Guncel kod schema'yi `with_structured_output` oncesi class olarak dogrular; hata devam ederse notebook 4. hucrede `STRUCTURED SCHEMA OK AnomalyBatchResult` satiri gorunmez.

Eger logda HTTP 200 OK sonrasi `LLM structured response returned None` gorulurse endpoint cevap vermis ama ne LangChain parser ne de raw content JSON parser tek karar objesi uretmemis demektir. Eger `LLM returned a results list` gorulurse model feature nedenlerini ayri karar gibi dondurmustur; bu durumda `runtime/llm/raw_model_responses.jsonl` dosyasindaki son satiri paylas, prompt/kontrat tarafinda hangi alanin yanlis yorumlandigini gorebiliriz.
