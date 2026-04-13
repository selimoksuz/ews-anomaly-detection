# EWS Anomaly Detection Architecture

## Pipeline Flow

```text
python cli.py setup                -> legacy Oracle tablo kurulumu
python cli.py load                 -> sentetik veri uret ve yukle
python cli.py develop              -> aday model egit
python cli.py tune-weights         -> 30+ ile weight optimize et
python cli.py evaluate-outcomes    -> 30+ / default outcome metrikleri uret
python cli.py promote              -> champion belirle
python cli.py score-live           -> latest snapshot skorla
python cli.py run-batch            -> config-driven batch orchestration
python cli.py build-notebook       -> simulation notebook uret
python cli.py reset-runtime        -> local runtime state temizle
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
|   |-- notebook_builder.py
|   |-- oracle_io.py
|   |-- output_writer.py
|   |-- pipeline.py
|   |-- registry.py
|   |-- retention.py
|   |-- scorer.py
|   |-- source_loader.py
|   |-- weight_tuning.py
|   `-- windowing.py
|
|-- scripts/
|   |-- __init__.py
|   |-- generate_data.py
|   |-- oracle_config.py
|   `-- setup_oracle.py
|
|-- legacy/
|   |-- __init__.py
|   |-- config.py
|   |-- model.py
|   `-- run.py
|
|-- notebooks/
|   `-- ews_simulation.ipynb
|
|-- orchestration/
|   `-- airflow/
|       `-- ews_batch_dag.py
|
`-- tests/
    |-- test_calibration.py
    |-- test_full_effects.py
    |-- test_model_stability.py
    |-- test_notebook_builder.py
    |-- test_registry.py
    |-- test_retention.py
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
  Optional local CSV outputs

## Oracle Tables

| Table | Direction | Purpose |
|---|---|---|
| `ZT_VAR2.EWS_INPUT_FEATURES` | Input | Tek append-only feature tablosu |
| `ZT_VAR2.EWS_OUTCOME_LABELS` | Input | `30+` primary, `default` monitoring labels |
| `ZT_VAR2.EWS_ALERT_RESULTS` | Output | Skor, band, metadata |
| `ZT_VAR2.EWS_ALERT_DETAILS` | Output | Top-N reason rows |
| `ZT_VAR2.EWS_ALERT_FEATURE_EFFECTS` | Output | Tum feature etkileri |
