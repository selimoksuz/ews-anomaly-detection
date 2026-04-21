# EWS Lifecycle Architecture

Bu proje Oracle-first, config-driven ve batch orchestrated anomaly lifecycle olarak calisir.

## Data Quality Layer

Pipeline yalnizca `native -> derived -> model` akisi degil, ayni zamanda zorunlu bir veri kalitesi katmani ile calisir.

- `engine/quality.py` native ve derived katmanda asagidaki kontrolleri uygular:
- satir sayisi ve benzersiz musteri sayisi esikleri
- duplicate `customer_id + snapshot_date` kontrolu
- feature coverage / missing ratio kontrolu
- robust outlier share kontrolu
- freshness kontrolu

Native katmanda freshness su an `fs_last_update_date` uzerinden olculur. Yani bilanço tarihi ile skor snapshot'i arasindaki yas farki izlenir. Derived katmanda ise artik model girdisi oldugu icin coverage, duplicate ve outlier kontrolleri onceliklidir.

Quality davranisi config ile yonetilir:

- `config/quality_rules.yaml`
- `development` run'larinda fail seviye quality sonucu run'i bloklar
- `live_scoring` run'larinda veri yoksa run fail olmaz, `skipped` olarak kapanir
- quality ozetleri run bazli monitoring bundle icine yazilir

## Simplified Folder-Mapped Flow

Bu diyagramda:

- `lane` = klasor / sorumluluk alani
- `node` = ilgili dosyanin yaptigi is
- `store` = Oracle tablo veya artifact deposu

```mermaid
flowchart LR
    START([Baslangic])
    END([Bitis])

    subgraph L0["config/ + root"]
        direction TB
        CLI["cli.py<br/>Komutu baslatir"]
        CFG["config/pipeline_config.yaml<br/>Tum davranis kurallari"]
        SEC["config/secrets.yaml<br/>Oracle baglanti bilgisi"]
        CLOAD["engine/config_loader.py<br/>Config'i yukler ve normalize eder"]
    end

    subgraph L1["engine/ - source + orchestration"]
        direction TB
        LIFE["engine/lifecycle.py<br/>Run turunu secip akisi yonetir"]
        SRC["engine/source_loader.py<br/>Oracle'dan frame okur"]
        DLOAD["engine/data_loader.py<br/>id/time zorunlu kontrol + frame validation"]
        WN["engine/windowing.py<br/>train/test/calibration/oot pencerelerini cozer"]
        SMP["engine/sampling.py<br/>Opsiyonelse sample alir ve validate eder"]
    end

    subgraph L2["engine/ - feature pipeline"]
        direction TB
        PRE["engine/preprocessing.py<br/>missing + hard bounds + categorical transform + scaler"]
        FS["engine/feature_selection.py<br/>exact duplicate drop + zero-variance continuous drop + branch routing"]
    end

    subgraph L3["engine/ - modeling"]
        direction TB
        MOD["engine/models.py<br/>AE + IF + MD fit/transform"]
        CAL["engine/calibration.py<br/>component raw score -> percentile map"]
        WT["engine/weight_tuning.py<br/>30+ varsa weight optimize eder"]
        SCR["engine/scorer.py<br/>Final score ve explainability uretir"]
    end

    subgraph L4["engine/ - runtime output"]
        direction TB
        OUT["engine/output_writer.py<br/>Delete + insert output yazar"]
        ORA["engine/oracle_io.py<br/>Oracle read/write ve DML"]
        REG["engine/registry.py<br/>candidate / champion / run metadata"]
        MON["engine/monitoring.py<br/>input ve score ozeti"]
        RET["engine/retention.py<br/>eski artifact/log temizligi"]
    end

    subgraph S1["Oracle stores"]
        direction TB
        INP[("EWS_INPUT_FEATURES")]
        OCM[("EWS_OUTCOME_LABELS")]
        RES[("EWS_ALERT_RESULTS")]
        DET[("EWS_ALERT_DETAILS")]
        EFF[("EWS_ALERT_FEATURE_EFFECTS")]
    end

    subgraph S2["runtime/"]
        direction TB
        ART[("run artifacts + run manifests + run logs + monitoring bundles")]
        META[("run_registry + model_registry + champions")]
    end

    START --> CLI --> CLOAD
    CFG --> CLOAD
    SEC --> CLOAD
    CLOAD --> LIFE

    LIFE --> RTYPE{"Run type?"}

    RTYPE -- "develop / retrain" --> DEV1["Oracle input'u oku"]
    INP --> SRC
    DEV1 --> SRC --> DLOAD --> WN
    WN --> DEV2["train/test/calibration/oot frame'leri"]
    DEV2 --> SAMPQ{"Sampling aktif mi?"}
    SAMPQ -- "Hayir" --> DEV3["Full train/test kullan"]
    SAMPQ -- "Evet" --> SMP --> SVAL{"Sample validation gecti mi?"}
    SVAL -- "Hayir + fallback=true" --> DEV3
    SVAL -- "Hayir + fallback=false" --> FAIL1([Fail])
    SVAL -- "Evet" --> DEV4["Sampled train/test kullan"]

    DEV3 --> PRE
    DEV4 --> PRE
    PRE --> CATQ{"Config'te include edilen kategorik var mi?"}
    CATQ -- "Yok" --> FEAT1["Numeric/raw feature space"]
    CATQ -- "Var" --> FEAT2["Generated feature space"]
    FEAT1 --> FS
    FEAT2 --> FS
    FS --> MOD
    MOD --> CALQ{"Calibration acik ve<br/>min_rows saglandi mi?"}
    CALQ -- "Hayir" --> MODART["Model artifact hazir"]
    CALQ -- "Evet" --> CAL --> MODART
    MODART --> REG

    REG --> WTQ{"Outcome ve tune-weights var mi?"}
    OCM --> WTQ
    WTQ -- "Hayir" --> PROMQ{"Promote edilsin mi?"}
    WTQ -- "Evet" --> WT --> PROMQ
    PROMQ -- "Hayir" --> REG
    PROMQ -- "Evet" --> REG
    REG --> ART
    REG --> META

    RTYPE -- "score-live" --> LIVE1["Champion'i yukle"]
    LIVE1 --> REG --> ART
    LIVE1 --> LIVESEL{"live_scoring.snapshot ne diyor?"}
    LIVESEL -- "today" --> LIVE2["SYSDATE gununu cek"]
    LIVESEL -- "explicit_date" --> LIVE3["O tarihi cek"]
    LIVESEL -- "range" --> LIVE4["start_date/end_date range cek"]
    LIVESEL -- "latest" --> LIVE5["En guncel snapshot'i cek"]

    LIVE2 --> SRC
    LIVE3 --> SRC
    LIVE4 --> SRC
    LIVE5 --> SRC
    SRC --> LIVEVAL{"Frame bos mu?"}
    LIVEVAL -- "Evet" --> FAIL2([Fail])
    LIVEVAL -- "Hayir" --> DLOAD
    DLOAD --> PRE
    PRE --> FS
    FS --> SCR
    ART --> SCR
    SCR --> CALUSE{"Calibration artifact var mi?"}
    CALUSE -- "Hayir" --> SCORE1["Raw component score ile devam"]
    CALUSE -- "Evet" --> SCORE2["Percentile-mapped component score kullan"]
    SCORE1 --> SCORE3["Active weights ile anomaly_score uret"]
    SCORE2 --> SCORE3
    SCORE3 --> SHADOWQ{"Shadow aktif mi?"}
    SHADOWQ -- "Hayir" --> OUTFRAME["Primary scored dataframe"]
    SHADOWQ -- "Evet" --> SHADOW["raw_shadow_score + score_delta ekle"] --> OUTFRAME

    OUTFRAME --> MON
    OUTFRAME --> OUT
    OUT --> ORA
    ORA --> RES
    ORA --> DET
    ORA --> EFF
    MON --> META
    OUT --> META

    RTYPE -- "compare / compare-preprocessing /\ncompare-feature-selection /\ncompare-sampling / evaluate-outcomes" --> EVAL["Ayni development artifacts ile\nanalitik comparison raporu uret"]
    EVAL --> REG
    EVAL --> META

    RET --> META
    RES --> END
    DET --> END
    EFF --> END
```

## Flow Notes

- `Missing handling` ve `hard bounds` akistan kopuk degil. Bunlar `shared data contract` icinde ard arda calisir ve sonra kategorik transform / preprocessing adimina gecilir.
- `Calibration enabled ve rows >= min_rows` kontrolu su anlama gelir:
  - calibration config'de acik degilse artifact uretilmez
  - calibration window bos ise artifact uretilmez
  - calibration window satir sayisi config'deki `calibration.min_rows` esiginden dusukse artifact uretilmez
  - bu durumda model yine calisir, sadece `raw -> percentile` mapping olmadan devam eder
- `Calibration fit: raw -> percentile mapping` demek:
  - calibration window uzerinde `ae_raw`, `if_raw`, `md_raw` score dagilimlari olculur
  - her bir raw score calibration dagilimindaki yuzdelik konumuna cevrilir
  - boylece farkli model skorlarini ortak 0-100 uzayinda karsilastirabiliriz
- `Shadow scoring` candidate yolunun devami degil, candidate artifact'e paralel opsiyonel bir diagnostic branch'tir. Champion secimini zorunlu olarak yonetmez; esas production score `primary/robust` branch'tir.
- `Live scoring snapshot` secimi `live_scoring.snapshot` config'i ile yonetilir:
  - `selector: today` ise sadece `TRUNC(SYSDATE)` gunu cekilir
  - `selector: latest` ise dogrudan en guncel snapshot cekilir
  - `explicit_date` verilirse selector override edilir
  - `explicit_date` ile basarili bir run tamamlanirsa alan tekrar `null` yapilir; ertesi run default `today` davranisina doner
  - `start_date/end_date` verilirse date range kullanilir
  - secilen kapsamda veri yoksa `score-live` run'i `failed` degil `skipped` status ile kapanir
- Oracle output tablolarinin olusumu:
  - `EWS_ALERT_RESULTS`: `PM24 scored dataframe` dogrudan bu tabloya yazilir
  - `EWS_ALERT_DETAILS`: `PM24` icindeki `top-N reason` alanlari `OB3` adiminda satirlastirilir
  - `EWS_ALERT_FEATURE_EFFECTS`: `PM24` icindeki tum feature katkilar `OB4` adiminda long-format'a acilir

## Batch Execution

`cli.py run-batch` config icindeki `batch_execution` bolumune gore su akisi yonetir:

1. Champion yoksa bootstrap `develop`
2. Gerekirse `tune-weights`
3. Gerekirse `evaluate-outcomes`
4. Gerekirse bootstrap `promote`
5. `score-live`

Champion varsa steady-state batch akisi:

1. `score-live`
2. `retrain` veya `develop` ile challenger uret
3. `tune-weights`
4. `evaluate-outcomes`
5. `compare`
6. Config isterse `promote`

## Oracle Tables

### Inputs

- `ZT_VAR2.EWS_INPUT_FEATURES`
  Tek append-only feature tablosu. Development ve live scoring ayni kaynaktan okunur.
- `ZT_VAR2.EWS_OUTCOME_LABELS`
  Outcome tablosu. Weight tuning ve validation icin `30+` primary, `default` monitoring olarak kullanilir.

### Outputs

- `ZT_VAR2.EWS_ALERT_RESULTS`
  Musteri-snapshot seviyesinde ozet skor, band ve metadata.
- `ZT_VAR2.EWS_ALERT_DETAILS`
  Alert alan musteriler icin top-N hizli explainability satirlari.
- `ZT_VAR2.EWS_ALERT_FEATURE_EFFECTS`
  Tum feature efektleri, human-readable uzun format explainability tablosu.

## Feature Inference And Categorical Config

Varsayilan davranis:

- `pipeline.id_column` ve `pipeline.time_column` feature degildir.
- Bunlar disindaki numeric kolonlar otomatik feature olarak infer edilir.
- Kategorik kolonlar varsayilan olarak modele girmez.

Bir kategorik kolonu modele dahil etmek istersen `features.categorical.per_feature` altinda acikca tanimlarsin. `transforms` alani her zaman YAML liste formunda yazilir:

```yaml
features:
  mode: infer
  categorical:
    default_include: false
    low_cardinality_threshold: 8
    per_feature:
      risk_band:
        include: true
        transforms:
          - ordinal
          - is_unseen
        order:
          - low
          - medium
          - high
      channel:
        include: true
        transforms:
          - one_hot
          - rarity
```

## Train-Only Sampling

Sampling varsayilan olarak sadece `development.train` penceresinde calisir. Istersen config ile `test`, `calibration` ve `oot` pencerelerine de acabilirsin. `live scoring` full data ile devam eder.

Varsayilan mantik:

- `development.sampling.enabled: false`
- `activate_if_rows_gt` esigi gecilmezse sample alinmaz
- sample alindiginda zaman, missing ve tail dagilimi korunmaya calisilir
- validation fail ederse sistem full train'e geri doner

Ornek config:

```yaml
development:
  sampling:
    enabled: true
    activate_if_rows_gt: 1000000
    max_rows: 500000
    compare_max_rows: 5000
    tail_z_threshold: 3.5
    random_seed: 42
    validation:
      max_snapshot_share_delta: 0.02
      max_tail_share_delta: 0.02
      max_missing_share_delta: 0.02
      max_feature_missing_delta: 0.01
      max_feature_ks: 0.10
      fallback_to_full_on_fail: true
```

Karsilastirma icin:

- `python cli.py compare-sampling [segment]`

Bu komut baseline vs sampled train kosusunu karsilastirir ve `sampling_comparison.json` ile `sampling_comparison.md` uretir.

Desteklenen kategorik transformlar:

- `one_hot`
- `freq`
- `rarity`
- `is_unseen`
- `changed_from_prev`
- `ordinal`

## Local Runtime State

```mermaid
flowchart LR
    A["runtime/registry/run_registry.json"] --> B["Batch / lifecycle run state"]
    C["runtime/registry/model_registry.json"] --> D["Model metadata + calibration + weights + evaluation"]
    E["runtime/registry/champions.json"] --> F["Active champion pointer"]
    G["runtime/runs/<run_id>/manifest.json"] --> H["Per-run manifest + run logs + monitoring bundle"]
    I["runtime/models/<segment>/<run_id>/"] --> J["model.pkl + calibration.json + weights.json + evaluation.json + stability.json"]
    K["runtime/logs/cli/"] --> L["CLI session logs"]
```

## Manual Reset

Local runtime state'i temizlemek icin:

```bash
python cli.py reset-runtime
```

Bu komut:

- `runtime/logs/`
- `runtime/models/`
- `runtime/runs/`
- legacy kalmis `runtime/monitoring/` veya config ile tanimlanmis eski monitoring klasorleri

temizler, `runtime/registry/` altindaki registry dosyalarini sifirdan olusturur ve Oracle tablolari silmez.

## Airflow Entry Point

`orchestration/airflow/ews_batch_dag.py` tek giris noktasi olarak `cli.py run-batch` cagirir. Boylece scheduling katmani ince kalir; is mantigi uygulama icinde kalir.

Airflow DAG icindeki `dag_id`, `schedule` ve `max_active_runs` degerleri root config'teki `orchestration.airflow` alanindan okunur. Boylece config ile DAG arasinda ayri bir cron drift'i olusmaz.
