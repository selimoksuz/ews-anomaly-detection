# EWS Lifecycle Architecture

Bu proje Oracle-first, config-driven ve batch orchestrated anomaly lifecycle olarak calisir.

## High-Level Flow

```mermaid
flowchart LR
    subgraph OI["Lane 1 - Oracle Inputs"]
        direction TB
        OI1["Input features table<br/>customer_id + snapshot_date + raw columns"]
        OI2["Outcome labels table<br/>30+ primary / default monitoring"]
    end

    subgraph LC["Lane 2 - Lifecycle / Orchestrator"]
        direction TB
        LC1{"id_column ve time_column var mi?"}
        LC2["Feature inference<br/>id,time,non-feature kolonlar feature listesinden dislanir<br/>ama raw frame'de kalir"]
        LC3["Snapshot listesini cek"]
        LC4["Window resolver"]
        LC5["OOT = latest N snapshot"]
        LC6["Calibration = latest M snapshot"]
        LC7["OOT baslangicindan once kalan history"]
        LC8["History'yi snapshot bazinda<br/>train / test oranla bol"]
        LC9{"Sampling aktif mi?"}
        LC10["Full train / full test"]
        LC11["Time + missing + tail stratified sample"]
        LC12{"Sample validation gecti mi?"}
        LC13["Fallback: full train / full test"]
        LC14["Prepared windows<br/>train / test / calibration / oot"]
        LC15{"Weight tuning yapilsin mi?"}
        LC16{"Promote edilsin mi?"}
        LC17{"Champion var mi?"}
        LC18["Config'e gore live scoring snapshot'ini Oracle'dan oku<br/>today / latest / explicit_date / range"]
    end

    subgraph PM["Lane 3 - Preprocessing / Modeling"]
        direction TB
        PM1["Shared data contract"]
        PM2["Missing handling<br/>feature-level strategy"]
        PM3["Hard bounds apply"]
        PM4{"Config'te include edilmis kategorik kolon var mi?"}
        PM5["Categorical transform uret<br/>one_hot / freq / rarity / is_unseen / changed_from_prev / ordinal"]
        PM6["Sadece numeric/raw feature ile devam"]
        PM7["Generated feature space"]
        PM8["Robust preprocessing<br/>winsor + scaler"]
        PM9["Feature selection + branch routing"]
        PM10["AE / IF / MD fit"]
        PM11{"Calibration enabled ve<br/>calibration rows >= min_rows mi?"}
        PM12["Calibration fit<br/>raw component score -> percentile mapping"]
        PM13["Calibration skip"]
        PM14{"Shadow scoring aktif mi?"}
        PM15["Parallel shadow branch fit"]
        PM16["Primary candidate hazir"]
        PM17["Latest snapshot'a scorer uygula"]
        PM18{"Calibration artifact var mi?"}
        PM19["Raw component score kullan"]
        PM20["Calibrated component score kullan"]
        PM21["Weight set ile final anomaly_score uret"]
        PM22{"Shadow aktif mi?"}
        PM23["raw_shadow_score + score_delta uret"]
        PM24["Scored dataframe<br/>1 row = 1 customer + 1 snapshot"]
    end

    subgraph RG["Lane 4 - Registry / Metadata"]
        direction TB
        RG1["Candidate model artifact yaz"]
        RG2["Calibration artifact yaz"]
        RG3["Shadow artifact yaz"]
        RG4["Weight version update"]
        RG5["Champion pointer update"]
        RG6["Run metadata / windows / sampling / monitoring yaz"]
        RG7["Champion + active artifacts yukle"]
    end

    subgraph OB["Lane 5 - Output Builder / Monitoring"]
        direction TB
        OB1["Input monitoring"]
        OB2["Score monitoring"]
        OB3["Top-N reason row'larini ac"]
        OB4["Tum feature effect row'larini ac"]
    end

    subgraph OO["Lane 6 - Oracle Outputs"]
        direction TB
        OO1["EWS_ALERT_RESULTS<br/>final score + band + metadata"]
        OO2["EWS_ALERT_DETAILS<br/>top-N explainability rows"]
        OO3["EWS_ALERT_FEATURE_EFFECTS<br/>all feature effect rows"]
    end

    OI1 --> LC1
    LC1 -- "Hayir" --> X1["Fail"]
    LC1 -- "Evet" --> LC2 --> LC3 --> LC4
    LC4 --> LC5
    LC4 --> LC6
    LC4 --> LC7 --> LC8 --> LC9

    LC9 -- "Hayir" --> LC10 --> LC14
    LC9 -- "Evet" --> LC11 --> LC12
    LC12 -- "Evet" --> LC14
    LC12 -- "Hayir ve fallback=true" --> LC13 --> LC14
    LC12 -- "Hayir ve fallback=false" --> X2["Fail"]

    LC14 --> PM1
    PM1 --> PM2 --> PM3 --> PM4
    PM4 -- "Evet" --> PM5 --> PM7
    PM4 -- "Hayir" --> PM6 --> PM7
    PM7 --> PM8 --> PM9 --> PM10 --> PM11

    PM11 -- "Evet" --> PM12 --> RG2
    PM11 -- "Hayir" --> PM13 --> RG1
    RG2 --> RG1

    RG1 --> PM14
    PM14 -- "Evet" --> PM15 --> RG3 --> PM16
    PM14 -- "Hayir" --> PM16

    OI2 --> LC15
    PM16 --> LC15
    LC15 -- "Hayir" --> LC16
    LC15 -- "Evet" --> RG4 --> LC16
    LC16 -- "Hayir" --> RG6
    LC16 -- "Evet" --> RG5 --> RG6

    RG6 --> LC17
    LC17 -- "Hayir" --> X3["Live scoring bekler"]
    LC17 -- "Evet" --> RG7 --> LC18
    OI1 --> LC18

    LC18 --> OB1 --> PM17 --> PM18
    PM18 -- "Hayir" --> PM19 --> PM21
    PM18 -- "Evet" --> PM20 --> PM21
    PM21 --> PM22
    PM22 -- "Evet" --> PM23 --> PM24
    PM22 -- "Hayir" --> PM24

    PM24 --> OB2
    PM24 --> OO1
    PM24 --> OB3 --> OO2
    PM24 --> OB4 --> OO3
    OB2 --> RG6
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
