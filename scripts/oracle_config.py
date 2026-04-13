"""Oracle connection configuration for EWS Anomaly Detection."""

import oracledb

ORACLE_HOST = "172.29.224.1"
ORACLE_PORT = 1521
ORACLE_SERVICE = "DEV.selimoksuz.com"

# Proxy auth: zt_var2 uzerinden ZT322168 schema'sina baglan
ORACLE_PROXY_USER = "ZT322168"
ORACLE_USER = "zt_var2"
ORACLE_PASSWORD = "ZtVar2Pass2025"

SCHEMA = "ZT_VAR2"

# Table names
TABLE_TRAIN = f"{SCHEMA}.EWS_TRAINING_DATA"
TABLE_SCORING = f"{SCHEMA}.EWS_SCORING_DATA"
TABLE_RESULTS = f"{SCHEMA}.EWS_ALERT_RESULTS"
TABLE_DETAILS = f"{SCHEMA}.EWS_ALERT_DETAILS"


def get_connection():
    return oracledb.connect(
        user=f"{ORACLE_PROXY_USER}[{ORACLE_USER}]",
        password=ORACLE_PASSWORD,
        dsn=f"{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE}",
    )
