# EWS Lifecycle Architecture

Bu proje artık Oracle-first, config-driven ve batch orchestrated anomaly lifecycle olarak çalışıyor.

## High-Level Flow

```mermaid
flowchart TD
    A["Oracle Input<br/>ZT_VAR2.EWS_INPUT_FEATURES"] --> B["Window Resolver<br/>train / dev / calibration / oot"]
    A --> C["Live Snapshot Selector<br/>latest snapshot"]
    D["Oracle Outcome Labels<br/>ZT_VAR2.EWS_OUTCOME_LABELS"] --> E["30+ Target Evaluation<br/>weight tuning + validation"]

    B --> F["Model Training<br/>AE + IF + MD"]
    F --> G["Calibration Fit<br/>empirical CDF"]
    G --> H["Candidate Artifact"]
    H --> E
    E --> I["Registry<br/>candidate / champion / runs"]

    I --> J["Live Scoring"]
    C --> J
    J --> K["ZT_VAR2.EWS_ALERT_RESULTS"]
    J --> L["ZT_VAR2.EWS_ALERT_DETAILS"]
    J --> M["ZT_VAR2.EWS_ALERT_FEATURE_EFFECTS"]
```

## Batch Execution

`cli.py run-batch` config içindeki `batch_execution` bölümüne göre şu akışı yönetir:

1. Champion yoksa bootstrap `develop`
2. Gerekirse `tune-weights`
3. Gerekirse `evaluate-outcomes`
4. Gerekirse bootstrap `promote`
5. `score-live`

Champion varsa steady-state batch akışı:

1. `score-live`
2. `retrain` veya `develop` ile challenger üret
3. `tune-weights`
4. `evaluate-outcomes`
5. `compare`
6. Config isterse `promote`

## Oracle Tables

### Inputs

- `ZT_VAR2.EWS_INPUT_FEATURES`
  Tek append-only feature tablosu. Development ve live scoring aynı kaynaktan okunur.
- `ZT_VAR2.EWS_OUTCOME_LABELS`
  Outcome tablosu. Weight tuning ve validation için `30+` primary, `default` monitoring olarak kullanılır.

### Outputs

- `ZT_VAR2.EWS_ALERT_RESULTS`
  Müşteri-snapshot seviyesinde özet skor, band ve metadata.
- `ZT_VAR2.EWS_ALERT_DETAILS`
  Alert alan müşteriler için top-N hızlı explainability satırları.
- `ZT_VAR2.EWS_ALERT_FEATURE_EFFECTS`
  Tüm feature efektleri, human-readable uzun format explainability tablosu.

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

## Notebook Simulation

`notebooks/ews_simulation.ipynb` notebook’u lifecycle’ı görsel olarak simüle etmek için üretilir.

Notebook içinde:

- aktif Oracle tablo isimleri
- local runtime klasörleri
- `reset-runtime`
- `prepare-demo-data`
- `run-batch`
- manuel `develop -> tune -> evaluate -> promote -> score-live`
- Oracle örnek çıktı okumaları
- registry dosyalarının görünümü

Notebook yeniden üretmek için:

```bash
python cli.py build-notebook
```

## Manual Reset

Local runtime state’i temizlemek için:

```bash
python cli.py reset-runtime
```

Bu komut:

- `logs/`
- `artifacts/`
- `meta/runs/`
- `meta/monitoring/`
- lokal CSV output klasörünü

temizler ve registry dosyalarını sıfırdan oluşturur. Oracle tablolarını silmez.

## Airflow Entry Point

`orchestration/airflow/ews_batch_dag.py` tek giriş noktası olarak `cli.py run-batch` çağırır. Böylece scheduling katmanı ince kalır; iş mantığı uygulama içinde kalır.
