# EWS Anomaly Detection Architecture

## Pipeline Flow

```text
python cli.py setup                -> legacy Oracle tablo kurulumu
python cli.py develop              -> aday model egit
python cli.py tune-weights         -> 30+ ile weight optimize et
python cli.py evaluate-outcomes    -> 30+ / default outcome metrikleri uret
python cli.py promote              -> champion belirle
python cli.py score-live           -> latest snapshot skorla
python cli.py run-batch            -> config-driven batch orchestration
python cli.py reset-runtime        -> local runtime state temizle
python cli.py cleanup              -> retention kurallarina gore local runtime temizligi
```

## Folder Structure

```text
ews-anomaly-detection/
|-- cli.py
|-- Dockerfile
|-- requirements.txt
|
|-- config/
|   |-- pipeline_config.yaml
|   `-- secrets.yaml
|
|-- docs/
|   |-- architecture.md
|   |-- FEATURE_DICTIONARY.md
|   `-- lifecycle.md
|
|-- engine/
|   |-- __init__.py
|   |-- calibration.py
|   |-- config_loader.py
|   |-- data_loader.py
|   |-- lifecycle.py
|   |-- models.py
|   |-- monitoring.py
|   |-- oracle_io.py
|   |-- output_writer.py
|   |-- pipeline.py
|   |-- preprocessing.py
|   |-- registry.py
|   |-- retention.py
|   |-- scorer.py
|   |-- source_loader.py
|   |-- weight_tuning.py
|   `-- windowing.py
|
|-- scripts/
|   |-- __init__.py
|   `-- oracle_config.py
|
|-- legacy/
|   |-- __init__.py
|   |-- config.py
|   |-- model.py
|   `-- run.py
|
|-- orchestration/
|   `-- airflow/
|       `-- ews_batch_dag.py
|
`-- tests/
    |-- helpers.py
    |-- test_calibration.py
    |-- test_full_effects.py
    |-- test_model_stability.py
    |-- test_preprocessing.py
    |-- test_registry.py
    |-- test_retention.py
    |-- test_scorer.py
    |-- test_weight_tuning.py
    `-- test_windowing.py
```

## Active Runtime Outputs

- `meta/`
  Run registry, model registry, champion pointer, run manifests
- `artifacts/`
  Model, calibration, weight and evaluation artifacts
- `logs/`
  CLI and lifecycle logs
- `output/`
  Opsiyonel lokal runtime ciktilari

## Oracle Tables

| Table | Direction | Purpose |
|---|---|---|
| `ZT_VAR2.EWS_INPUT_FEATURES` | Input | Tek append-only feature tablosu |
| `ZT_VAR2.EWS_OUTCOME_LABELS` | Input | `30+` primary, `default` monitoring labels |
| `ZT_VAR2.EWS_ALERT_RESULTS` | Output | Skor, band, metadata |
| `ZT_VAR2.EWS_ALERT_DETAILS` | Output | Top-N reason rows |
| `ZT_VAR2.EWS_ALERT_FEATURE_EFFECTS` | Output | Tum feature etkileri |
