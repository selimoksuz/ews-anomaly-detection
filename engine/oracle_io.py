"""Oracle I/O module — read/write operations."""

import logging
import oracledb
import pandas as pd
from engine.config_loader import get_feature_list

logger = logging.getLogger(__name__)


class OracleConnector:
    def __init__(self, config, secrets):
        self.config = config
        self.secrets = secrets
        self.schema = config["oracle"]["schema"]
        self.tables = config["oracle"]["tables"]
        self.features = get_feature_list(config)
        self._conn = None

    def connect(self):
        ora = self.secrets["oracle"]
        dsn = f"{ora['host']}:{ora['port']}/{ora['service_name']}"
        user = f"{ora['proxy_user']}[{ora['user']}]"
        self._conn = oracledb.connect(user=user, password=ora["password"], dsn=dsn)
        logger.info(f"Oracle connected: {self.schema}")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            self.connect()
        return self._conn

    def _table(self, key):
        return f"{self.schema}.{self.tables[key]}"

    def _feature_cols_sql(self):
        return ", ".join([f.upper() for f in self.features])

    # ── READ ──

    def read_training_data(self, split="TRAIN"):
        sql = f"SELECT CUSTOMER_ID, {self._feature_cols_sql()} FROM {self._table('training')} WHERE SPLIT_FLAG = :1"
        df = pd.read_sql(sql, self.conn, params=[split])
        df.columns = [self.config["pipeline"]["id_column"]] + self.features
        logger.info(f"Read {len(df)} rows from {self._table('training')} (split={split})")
        return df

    def read_scoring_data(self):
        sql = f"SELECT CUSTOMER_ID, {self._feature_cols_sql()} FROM {self._table('scoring')}"
        df = pd.read_sql(sql, self.conn)
        df.columns = [self.config["pipeline"]["id_column"]] + self.features
        logger.info(f"Read {len(df)} rows from {self._table('scoring')}")
        return df

    # ── WRITE ──

    def write_results(self, results_df, scoring_date):
        cursor = self.conn.cursor()
        cursor.execute(f"DELETE FROM {self._table('results')} WHERE SCORING_DATE = :1", [scoring_date])

        rows = []
        for _, row in results_df.iterrows():
            nedenler = []
            for feat, d in row["detay"].items():
                ico = "UP" if d["degisim_pct"] > 0 else "DN"
                nedenler.append(f"{d['label']}: {d['beklenen']}->{d['gerceklesen']} ({ico}%{abs(d['degisim_pct']):.0f})")
            while len(nedenler) < 3:
                nedenler.append(None)
            rows.append([
                row["customer_id"], scoring_date,
                float(row["anomaly_score"]), str(row["alert_band"]),
                float(row["ae_score"]), float(row["if_score"]), float(row["md_score"]),
                nedenler[0], nedenler[1], nedenler[2],
            ])

        sql = f"""INSERT INTO {self._table('results')}
            (CUSTOMER_ID, SCORING_DATE, ANOMALY_SCORE, ALERT_BAND,
             AE_SCORE, IF_SCORE, MD_SCORE, NEDEN_1, NEDEN_2, NEDEN_3)
            VALUES (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10)"""

        for i in range(0, len(rows), 500):
            cursor.executemany(sql, rows[i:i + 500])

        self.conn.commit()
        cursor.close()
        logger.info(f"Wrote {len(rows)} rows to {self._table('results')}")

    def write_details(self, results_df, scoring_date):
        cursor = self.conn.cursor()
        cursor.execute(f"DELETE FROM {self._table('details')} WHERE SCORING_DATE = :1", [scoring_date])

        alerts = results_df[results_df["alert_band"].isin(["KIRMIZI", "TURUNCU", "SARI"])]
        rows = []
        for _, row in alerts.iterrows():
            for sira, (feat, d) in enumerate(row["detay"].items(), 1):
                rows.append([
                    row["customer_id"], scoring_date, feat, d["label"],
                    float(d["beklenen"]), float(d["gerceklesen"]),
                    float(d["degisim_pct"]), float(d["katki_pct"]), sira,
                ])

        sql = f"""INSERT INTO {self._table('details')}
            (CUSTOMER_ID, SCORING_DATE, FEATURE_NAME, FEATURE_LABEL,
             BEKLENEN, GERCEKLESEN, DEGISIM_PCT, KATKI_PCT, SIRA)
            VALUES (:1,:2,:3,:4,:5,:6,:7,:8,:9)"""

        for i in range(0, len(rows), 500):
            cursor.executemany(sql, rows[i:i + 500])

        self.conn.commit()
        cursor.close()
        logger.info(f"Wrote {len(rows)} detail rows to {self._table('details')}")

    # ── SETUP ──

    def setup_tables(self):
        cursor = self.conn.cursor()
        feat_ddl = "\n".join([f"    {f.upper()} NUMBER," for f in self.features])

        ddls = {
            "training": f"""CREATE TABLE {self._table('training')} (
    CUSTOMER_ID VARCHAR2(20) NOT NULL,
    SNAPSHOT_DATE DATE NOT NULL,
    SPLIT_FLAG VARCHAR2(10) DEFAULT 'TRAIN',
{feat_ddl}
    CONSTRAINT PK_EWS_TRAIN PRIMARY KEY (CUSTOMER_ID, SNAPSHOT_DATE))""",

            "scoring": f"""CREATE TABLE {self._table('scoring')} (
    CUSTOMER_ID VARCHAR2(20) NOT NULL,
    SNAPSHOT_DATE DATE DEFAULT SYSDATE,
{feat_ddl}
    CONSTRAINT PK_EWS_SCORING PRIMARY KEY (CUSTOMER_ID))""",

            "results": f"""CREATE TABLE {self._table('results')} (
    CUSTOMER_ID VARCHAR2(20) NOT NULL,
    SCORING_DATE DATE DEFAULT SYSDATE,
    ANOMALY_SCORE NUMBER(5,1),
    ALERT_BAND VARCHAR2(10),
    AE_SCORE NUMBER(5,1),
    IF_SCORE NUMBER(5,1),
    MD_SCORE NUMBER(5,1),
    NEDEN_1 VARCHAR2(500),
    NEDEN_2 VARCHAR2(500),
    NEDEN_3 VARCHAR2(500),
    CONSTRAINT PK_EWS_RESULTS PRIMARY KEY (CUSTOMER_ID, SCORING_DATE))""",

            "details": f"""CREATE TABLE {self._table('details')} (
    CUSTOMER_ID VARCHAR2(20) NOT NULL,
    SCORING_DATE DATE DEFAULT SYSDATE,
    FEATURE_NAME VARCHAR2(50),
    FEATURE_LABEL VARCHAR2(100),
    BEKLENEN NUMBER(15,4),
    GERCEKLESEN NUMBER(15,4),
    DEGISIM_PCT NUMBER(10,2),
    KATKI_PCT NUMBER(5,1),
    SIRA NUMBER(2),
    CONSTRAINT PK_EWS_DETAILS PRIMARY KEY (CUSTOMER_ID, SCORING_DATE, FEATURE_NAME))""",
        }

        for key in ["details", "results", "scoring", "training"]:
            try:
                cursor.execute(f"DROP TABLE {self._table(key)} PURGE")
                logger.info(f"Dropped {self._table(key)}")
            except Exception:
                pass

        for key in ["training", "scoring", "results", "details"]:
            cursor.execute(ddls[key])
            logger.info(f"Created {self._table(key)}")

        self.conn.commit()
        cursor.close()

    def load_dataframe(self, df, table_key, extra_cols=None):
        """DataFrame'i Oracle tablosuna yukle."""
        cursor = self.conn.cursor()
        id_col = self.config["pipeline"]["id_column"]
        time_col = self.config["pipeline"]["time_column"]

        base_cols = [id_col, time_col]
        if extra_cols:
            base_cols += extra_cols
        all_cols = base_cols + self.features

        col_names = [c.upper() for c in all_cols]
        placeholders = ", ".join([f":{i + 1}" for i in range(len(col_names))])
        sql = f"INSERT INTO {self._table(table_key)} ({', '.join(col_names)}) VALUES ({placeholders})"

        rows = []
        for _, row in df.iterrows():
            vals = [row.get(c) for c in base_cols]
            vals += [float(row[f]) if pd.notna(row.get(f)) else None for f in self.features]
            rows.append(vals)

        for i in range(0, len(rows), 500):
            cursor.executemany(sql, rows[i:i + 500])

        self.conn.commit()
        cursor.close()
        logger.info(f"Loaded {len(rows)} rows into {self._table(table_key)}")
