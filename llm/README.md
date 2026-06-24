# LLM Anomaly Prototype

Bu klasor, ekran goruntulerindeki LLM scriptinin revize edilmis ve proje icinde izole edilmis halidir.

Ana farklar:

- API key kodda tutulmaz; `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` env degiskenleri kullanilir.
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

Oracle ham inputtan oku, LLM kararini al ve ayri Oracle tablolarina yaz:

```powershell
python -m llm.llm_anomaly run-oracle runtime/llm/decisions_oracle.jsonl --max-customers 10 --max-train-rows 300000
```

LLM karar tablolari:

- `ZT_VAR2.EWS_ANOMALY_LLM_RESULTS`: musteri-donem seviyesinde `IS_ANOMALY`, `ANOMALY_TYPE`, `RISK_LEVEL`, `LLM_CONFIDENCE`, ozet yorumlar ve raw JSON response.
- `ZT_VAR2.EWS_ANOMALY_LLM_REASONS`: LLM'in `main_reasons` listesindeki feature bazli reason detaylari.

OpenAI endpoint icin:

```powershell
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_API_KEY="..."
$env:LLM_MODEL="gpt-4.1-mini"
python -m llm.llm_anomaly run runtime/llm/evidence.jsonl runtime/llm/decisions.jsonl --from-evidence
```

OpenAI-compatible lokal veya kurum ici endpoint kullanilacaksa sadece `LLM_BASE_URL`, `LLM_MODEL` ve key degistirilir.
