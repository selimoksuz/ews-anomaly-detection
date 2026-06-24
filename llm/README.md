# LLM Anomaly Prototype

Bu klasor, ekran goruntulerindeki LLM scriptinin revize edilmis ve proje icinde izole edilmis halidir.

Ana farklar:

- API key kodda tutulmaz; `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` env degiskenleri veya `secret/secrets.yaml` kullanilir.
- LLM'e ham tablo degil, denetlenebilir `evidence JSON` verilir.
- Evidence icinde veri sozlugu, risk yonu, musteri gecmisi, rolling medyanlar, trend, sezon, peer ve veri kalitesi birlikte bulunur.
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
python -m llm.llm_anomaly build-oracle runtime/llm/evidence_oracle.jsonl --max-customers 10 --max-train-rows 300000
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

1. Terminal env degiskenleri: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TIMEOUT_SECONDS`, `LLM_RESPONSE_FORMAT`
2. Lokal dosyalar: repo kokundeki `.env` ve `llm/.env.local`
3. Secret dosyasi: `secret/secrets.yaml`
4. Default: `base_url=https://api.openai.com/v1`, `model=gpt-4.1-mini`, `timeout_seconds=120`

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
      response_format: "json_object"
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

Kurum ici endpoint `response_format` desteklemiyorsa logda `Invalid parameter: response_format` / `unsupported value` gibi HTTP 400 hata gorulebilir. Kod otomatik fallback yapmaz; bu parametreyi kullanmak istemiyorsan bilincli olarak kapat:

```yaml
llm:
  sections:
    OPENSHIFT_LLM:
      response_format: "none"
```

veya:

```bash
export LLM_RESPONSE_FORMAT=none
```

Endpoint ve key saglik kontrolu icin notebook:

```text
llm/llm_endpoint_healthcheck.ipynb
```

Bu notebook sirasiyla repo root bulma, `secret/secrets.yaml` path/section/key varligi kontrolu, config okuma, TCP/TLS handshake, `/models`, repo icindeki chat call ve opsiyonel LangChain call testlerini yapar.

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
  --top-features 12
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
- `--max-customers 10`: LLM'e gidecek musteri-donem payload sayisidir. 10 verilirse LLM'e 10 evidence package gider.
- `--max-train-rows 300000`: LLM'e 300k satir gondermez. History, peer, trend ve seasonality referanslarini hesaplamak icin kullanilan gecmis/reference ust limitidir.
- Secilen musterilerin tam gecmisi ayrica cekilir; bu sayede `max_train_rows` sampling'i secilen musterinin history'sini dusurmez.
- Peer referansi `max-customers` ile secilen 10 musteri uzerinden degil, skorlanan ayin tum scoring cohort'u uzerinden hesaplanir.

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
- Peer grubu rating ile daraltilmaz; ana hiyerarsi ay + segment + sektor + aylik buyukluk fallback mantigidir.

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

OpenAI endpoint icin:

```powershell
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_API_KEY="..."
$env:LLM_MODEL="gpt-4.1-mini"
python -m llm.llm_anomaly run runtime/llm/evidence.jsonl runtime/llm/decisions.jsonl --from-evidence
```

OpenAI-compatible lokal veya kurum ici endpoint kullanilacaksa sadece `LLM_BASE_URL`, `LLM_MODEL` ve key degistirilir.
