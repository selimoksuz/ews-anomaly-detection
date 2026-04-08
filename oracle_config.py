"""Oracle connection configuration for EWS Anomaly Detection."""

import oracledb

ORACLE_HOST = "localhost"
ORACLE_PORT = 1521
ORACLE_SERVICE = "DEV.selimoksuz.com"
ORACLE_USER = "RISK_PIPELINE"
ORACLE_PASSWORD = "RiskPipe!2025"

SCHEMA = "RISK_PIPELINE"

# Table names
TABLE_TRAIN = f"{SCHEMA}.EWS_TRAINING_DATA"
TABLE_SCORING = f"{SCHEMA}.EWS_SCORING_DATA"
TABLE_RESULTS = f"{SCHEMA}.EWS_ALERT_RESULTS"
TABLE_DETAILS = f"{SCHEMA}.EWS_ALERT_DETAILS"


def get_connection():
    return oracledb.connect(
        user=ORACLE_USER,
        password=ORACLE_PASSWORD,
        dsn=f"{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE}",
    )
