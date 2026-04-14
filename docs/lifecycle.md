# EWS Lifecycle Architecture

Bu proje Oracle-first, config-driven ve batch orchestrated anomaly lifecycle olarak calisir.

## High-Level Flow

```mermaid
flowchart TD
    A["Oracle input table<br/>customer_id + snapshot_date + raw columns"] --> B{"id/time kolonlari var mi?"}
    B -- "Hayir" --> X1["Fail"]
    B -- "Evet" --> C["Feature inference<br/>id,time,non-feature kolonlar feature listesinden dislanir<br/>ama raw frame'de kalir"]

    C --> D["Snapshot listesini cek"]
    D --> E["Window resolver<br/>train / test / calibration / oot"]
    E --> F["OOT: latest N snapshot"]
    E --> G["Calibration: latest M snapshot"]
    E --> H["History: OOT baslangicindan once kalan tum snapshotlar"]
    H --> I["Train/Test split<br/>snapshot bazinda oransal bol"]

    I --> J{"Sampling aktif mi<br/>ve train/test icin gerekli mi?"}
    J -- "Hayir" --> K["Full train / full test"]
    J -- "Evet" --> L["Time + missing + tail stratified sample"]
    L --> M{"Validation gecti mi?"}
    M -- "Hayir ve fallback=true" --> K
    M -- "Hayir ve fallback=false" --> X2["Fail"]
    M -- "Evet" --> N["Sampled train / sampled test"]

    K --> O["Shared data contract"]
    N --> O
    G --> O
    F --> O

    O --> O1["Missing handling<br/>feature-level strategy uygula"]
    O1 --> O2["Hard bounds apply"]
    O2 --> O3{"Kategorik kolon include edildi mi?"}
    O3 -- "Hayir" --> O4["Sadece numeric/raw features ile devam"]
    O3 -- "Evet" --> O5["Secilen categorical transformlari uret"]
    O4 --> P["Generated feature space"]
    O5 --> P

    P --> Q["Robust preprocessing fit/apply<br/>winsor + scaler"]
    Q --> R["Feature selection + branch routing"]
    R --> S1["AE feature set"]
    R --> S2["IF feature set"]
    R --> S3["MD feature set"]
    S1 --> T["AE + IF + MD fit"]
    S2 --> T
    S3 --> T

    T --> U{"Calibration enabled mi<br/>ve calibration rows >= min_rows mi?"}
    U -- "Hayir" --> V["Calibration skip"]
    U -- "Evet" --> W["Calibration fit<br/>raw component score -> percentile mapping"]

    V --> Y["Candidate artifact + registry record"]
    W --> Y

    Y --> Z{"Shadow scoring aktif mi?"}
    Z -- "Hayir" --> AA["Candidate hazir"]
    Z -- "Evet" --> AB["Parallel shadow artifact fit<br/>opsiyonel diagnostic branch"]
    AB --> AA

    D2["Oracle outcome labels<br/>30+ primary, default monitoring"] --> AC{"Weight tuning yapilacak mi?"}
    AA --> AC
    AC -- "Hayir" --> AD["Manual weights"]
    AC -- "Evet" --> AE["Tune weights on test<br/>validate on oot"]
    AE --> AF["Weight version registry update"]
    AD --> AG{"Promote edilsin mi?"}
    AF --> AG
    AG -- "Hayir" --> AH["Candidate olarak kalir"]
    AG -- "Evet" --> AI["Champion pointer update"]

    AI --> AJ["Live scoring"]
    AH --> AJ
    AJ --> AK["Latest snapshot'i Oracle'dan oku"]
    AK --> AL["Ayni data contract'i apply et"]
    AL --> AM["Ayni preprocessing artifact'i apply et"]
    AM --> AN["AE / IF / MD raw score"]
    AN --> AO{"Calibration artifact var mi?"}
    AO -- "Hayir" --> AP["Raw score ile devam"]
    AO -- "Evet" --> AQ["Calibrated score uret"]
    AP --> AR["Final anomaly_score<br/>weight set ile"]
    AQ --> AR
    AR --> AS{"Shadow aktif mi?"}
    AS -- "Hayir" --> AT["Primary result set"]
    AS -- "Evet" --> AU["Raw shadow score + score_delta"]
    AU --> AT

    AT --> AV["Oracle output write"]
    AV --> AV1["EWS_ALERT_RESULTS"]
    AV --> AV2["EWS_ALERT_DETAILS"]
    AV --> AV3["EWS_ALERT_FEATURE_EFFECTS"]
    AT --> AW["Monitoring + metadata write"]
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
    A["meta/run_registry.json"] --> B["Batch / lifecycle run log"]
    C["meta/model_registry.json"] --> D["Model metadata + calibration + weights + evaluation"]
    E["meta/champions.json"] --> F["Active champion pointer"]
    G["meta/runs/<run_id>/manifest.json"] --> H["Per-run manifest + monitoring json"]
    I["artifacts/<segment>/<run_id>/"] --> J["model.pkl + calibration.json + weights.json + evaluation.json + stability.json"]
    K["logs/"] --> L["CLI and runtime logs"]
```

## Manual Reset

Local runtime state'i temizlemek icin:

```bash
python cli.py reset-runtime
```

Bu komut:

- `logs/`
- `artifacts/`
- `meta/runs/`
- `meta/monitoring/`

temizler ve registry dosyalarini sifirdan olusturur. Oracle tablolari silmez.

## Airflow Entry Point

`orchestration/airflow/ews_batch_dag.py` tek giris noktasi olarak `cli.py run-batch` cagirir. Boylece scheduling katmani ince kalir; is mantigi uygulama icinde kalir.
