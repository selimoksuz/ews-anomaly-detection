"""Airflow DAG scaffold for the EWS anomaly detection lifecycle."""

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
    score_live = BashOperator(
        task_id="score_live",
        bash_command="cd C:/Users/Acer/ews-anomaly-detection && .venv/Scripts/python.exe cli.py score-live",
    )

    retrain_candidate = BashOperator(
        task_id="retrain_candidate",
        bash_command="cd C:/Users/Acer/ews-anomaly-detection && .venv/Scripts/python.exe cli.py retrain",
    )

    compare_models = BashOperator(
        task_id="compare_models",
        bash_command="cd C:/Users/Acer/ews-anomaly-detection && .venv/Scripts/python.exe cli.py compare",
    )

    cleanup = BashOperator(
        task_id="cleanup_runtime",
        bash_command="cd C:/Users/Acer/ews-anomaly-detection && .venv/Scripts/python.exe cli.py cleanup",
    )

    score_live >> retrain_candidate >> compare_models >> cleanup
