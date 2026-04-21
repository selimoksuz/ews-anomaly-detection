"""Airflow DAG scaffold for the config-driven EWS batch pipeline."""

from datetime import datetime
from pathlib import Path

import yaml

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline_config.yaml"
with PIPELINE_CONFIG.open("r", encoding="utf-8") as handle:
    root_config = yaml.safe_load(handle) or {}

airflow_cfg = ((root_config.get("orchestration", {}) or {}).get("airflow", {}) or {})
dag_id = airflow_cfg.get("dag_id", "ews_anomaly_detection")
schedule = airflow_cfg.get("schedule", "0 8 1 * *")
max_active_runs = int(airflow_cfg.get("max_active_runs", 1))


with DAG(
    dag_id=dag_id,
    start_date=datetime(2026, 1, 1),
    schedule=schedule,
    catchup=False,
    max_active_runs=max_active_runs,
) as dag:
    run_batch = BashOperator(
        task_id="run_batch",
        bash_command=f"cd {PROJECT_ROOT.as_posix()} && .venv/Scripts/python.exe cli.py run-batch",
    )

    run_batch
