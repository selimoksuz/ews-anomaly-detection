# LLM Anomaly Prototype

Bu klasor, ekran goruntulerindeki LLM scriptinin revize edilmis ve proje icinde izole edilmis halidir.

Ana farklar:

- API key kodda tutulmaz; `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` env degiskenleri veya `secret/secrets.yaml` kullanilir.
- LLM'e ham tablo degil, denetlenebilir `evidence JSON` verilir.
- Evidence icinde veri sozlugu, risk yonu, musteri gecmisi, rolling medyanlar, trend, sezon, cari peer ve tarihsel musteri/peer snapshot serileri birlikte bulunur.
- Gelecek donemler LLM'e verilmez; skorlanan ay sadece onceki aylarla kiyaslanir.
- LLM skor motoru degil, dogrudan `is_anomaly` karari veren uzman karar katmani olarak calisir.

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

Oracle baglanti ayari `secret/secrets.yaml` icindedir. Bu dosya `secret/secrets.yaml.example` dosyasindan kopyalanir ve git'e alinmaz.

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
```

## LLM Key ve Endpoint Ayarlari

LLM ayarlari su sirayla okunur:

1. Terminal env degiskenleri: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TIMEOUT_SECONDS`
2. Lokal dosyalar: repo kokundeki `.env` ve `llm/.env.local`
3. Secret dosyasi: `secret/secrets.yaml`
4. `timeout_seconds` verilmezse default 120 kullanilir.

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
      timeout_seconds: 120
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

Varsayilan LLM cagri sekli ilk calisan kaynak kodun operasyonel kalibiyla aynidir: `llm.with_structured_output(AnomalyBatchResult)`, `prompt | structured_llm`, `chain.invoke({"input_records": ...})`, sonra dogrudan `response.results` okunur. `method=...` override verilmez; internal endpoint LangChain default structured davranisiyla cagrilir. Her musteri snapshot'i tek prompt olarak gider; donen `results` listesi `period_position` ile tekrar ilgili kayda baglanir.

Cikti tipi sade degildir. `AnomalyBatchResult.results` altinda genis `AnomalyRecord` doner: `period_position`, `mono_id`, `cohort_dt`, `is_anomaly`, `anomaly_type`, `risk_level`, `confidence`, `seasonality_assessment`, `trend_assessment`, `peer_assessment`, `main_reasons`, `caveat`, `recommended_action`. Oracle output tablolari bu genis yapiyi kullanir.

Evidence, ham nested JSON dump olarak degil ayni bilgileri tasiyan kompakt text olarak gonderilir; bu feature veya veri azaltma degildir, token sismesini ve response timeout riskini azaltmak icindir.

Timeout/retry ayarlari env veya `secret/secrets.yaml` altindan verilebilir:

```yaml
llm:
  sections:
    OPENSHIFT_LLM:
      timeout_seconds: 300
      max_retries: 0
```

Logda su satirlar gorulmelidir:

```text
LLM settings resolved: ... timeout_seconds=300 max_retries=0 max_tokens=None structured_call=with_structured_output_schema_only client=langchain_structured
LangChain structured LLM chain initialized: model=gpt-oss-20b structured_call=with_structured_output_schema_only max_retries=0 max_tokens=None
LLM request payload prepared: mono_id=... decision_items=... formatter=compact_text
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
  --series-periods 6
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
- `--dry-run`: LLM'e gitmez ve Oracle output insert yapmaz; sadece prompt/evidence kontrolu icindir.
- `--max-customers 10`: LLM'e gidecek scoring snapshot/musteri sayisidir. Her musteri scoring ayinda 1 karar satiri olarak gider; history satirlari insert edilmez, evidence icinde baglam olarak kullanilir.
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

- `ZT_VAR2.EWS_ANOMALY_LLM_RESULTS`: musteri-donem seviyesinde `IS_ANOMALY`, `ANOMALY_TYPE`, `RISK_LEVEL`, `LLM_CONFIDENCE`, ozet yorumlar ve raw JSON response.
- `ZT_VAR2.EWS_ANOMALY_LLM_REASONS`: LLM'in `main_reasons` listesindeki feature bazli reason detaylari.

## Terminalde Beklenen Akis

`run-oracle` calistiginda terminalde ve `runtime/logs/cli/llm_anomaly.log` dosyasinda su basliklar gorulmelidir:

```text
STEP 00 START | LLM Oracle anomaly run basladi
STEP 01 START/DONE | Oracle kaynak tablo ve ay profili okunuyor
STEP 02 START/DONE | Ham tablo kolonlari ve veri sozlugu denetleniyor
STEP 03 START/DONE | Musteri bazli history ve aylik peer gruplariyla LLM evidence uretiliyor
STEP 04 START/DONE | LLM modelinden anomali karari aliniyor
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
- `pd_rating`: `irb_rating_pd`, `irb_model_pd`, `rating_group` gibi direkt PD/rating sinyalleri.
- `context`: segment, sektor, NACE, referans donem gibi gruplama/aciklama alanlari.
- `technical`: `data_time`, `created_at`, teknik yukleme alanlari.

PD/rating notu:

- `irb_rating_pd`, `irb_model_pd`, `rating_group` exclude edilmez; direkt sinyal olarak kullanilabilir.
- `pd_ratio`, `pd_to_rating_group` gibi PD/rating alanlarini kendi arasinda oranlayan veya dogrudan karsilastiran turetilmis feature'lar kullanilmaz.
- Peer grubu rating ile daraltilmaz; ana hiyerarsi ay + segment + sektor + aylik buyukluk sirasidir.

## Output Insert Kontrolu

Run sonunda logda su satirlar gorulmelidir:

```text
AUDIT OUTPUT TABLE | table_key=llm_results ... inserted=10 ... run_rows_after=10
AUDIT OUTPUT TABLE | table_key=llm_reason_details ... inserted=<reason_count> ... run_rows_after=<reason_count>
```

Manuel Oracle kontrolu icin:

```sql
SELECT COUNT(*) FROM ZT_VAR2.EWS_ANOMALY_LLM_RESULTS
WHERE TRUNC(COHORT_DT) = DATE '2026-05-31';

SELECT COUNT(*) FROM ZT_VAR2.EWS_ANOMALY_LLM_REASONS
WHERE TRUNC(COHORT_DT) = DATE '2026-05-31';
```

Model cagrisi ilk prototipteki operasyonel kalipla yapilir: `ChatOpenAI`, `ChatPromptTemplate`, Pydantic `BaseModel/Field`, `llm.with_structured_output(...)` ve `chain.invoke(...)`. Basarili cevapta dogrudan `response.results` okunur. Ek parser, raw response parser veya dis endpoint gecisi yoktur.

Eger healthcheck'te `TypeError('issubclass() arg 1 must be a class')` gorursen once repo kodunun guncel oldugunu ve kernelin yeniden baslatildigini kontrol et. Guncel kod schema'yi `with_structured_output` oncesi class olarak dogrular; hata devam ederse notebook 4. hucrede `STRUCTURED SCHEMA OK AnomalyBatchResult` satiri gorunmez.

Eger logda HTTP 200 OK sonrasi `LLM structured response did not include results` gorulurse endpoint cevap vermis ama LangChain structured chain `AnomalyBatchResult.results` objesini uretmemis demektir. Bu durumda once notebook kernelinin guncel kodu import ettigini ve logda `structured_call=with_structured_output_schema_only` gorundugunu kontrol et.
