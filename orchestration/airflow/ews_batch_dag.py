"""Airflow DAG scaffold for the config-driven EWS batch pipeline."""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="ews_anomaly_detection",
    start_date=datetime(2026, 1, 1),
    schedule="0 6 * * 1",
    catchup=False,
    max_active_runs=1,
) as dag:
    run_batch = BashOperator(
        task_id="run_batch",
        bash_command="cd C:/Users/Acer/ews-anomaly-detection && .venv/Scripts/python.exe cli.py run-batch",
    )

    run_batch
