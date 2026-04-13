"""
Oracle table setup and data loading for EWS Anomaly Detection.

Creates 4 tables:
  1. EWS_TRAINING_DATA  - historical normal data (train + holdout test)
  2. EWS_SCORING_DATA   - today's snapshot for scoring
  3. EWS_ALERT_RESULTS  - output: scores, bands, reasons
  4. EWS_ALERT_DETAILS  - output: per-feature contribution details

Usage:
    python scripts/setup_oracle.py
"""

import numpy as np
import pandas as pd
from legacy.config import ALL_FEATURES
from scripts.generate_data import generate_training_data, generate_scoring_data
from scripts.oracle_config import (
    SCHEMA,
    TABLE_DETAILS,
    TABLE_RESULTS,
    TABLE_SCORING,
    TABLE_TRAIN,
    get_connection,
)


# ── DDL ──────────────────────────────────────────────────────────

FEATURE_COLUMNS_DDL = "\n".join(
    [f"    {f.upper()} NUMBER," for f in ALL_FEATURES]
)

DDL_TRAINING = f"""
CREATE TABLE {TABLE_TRAIN} (
    CUSTOMER_ID VARCHAR2(20) NOT NULL,
    SNAPSHOT_DATE DATE NOT NULL,
    SPLIT_FLAG VARCHAR2(10) DEFAULT 'TRAIN',
{FEATURE_COLUMNS_DDL}
    CONSTRAINT PK_EWS_TRAIN PRIMARY KEY (CUSTOMER_ID, SNAPSHOT_DATE)
)
"""

DDL_SCORING = f"""
CREATE TABLE {TABLE_SCORING} (
    CUSTOMER_ID VARCHAR2(20) NOT NULL,
    SNAPSHOT_DATE DATE DEFAULT SYSDATE,
{FEATURE_COLUMNS_DDL}
    CONSTRAINT PK_EWS_SCORING PRIMARY KEY (CUSTOMER_ID)
)
"""

DDL_RESULTS = f"""
CREATE TABLE {TABLE_RESULTS} (
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
    CONSTRAINT PK_EWS_RESULTS PRIMARY KEY (CUSTOMER_ID, SCORING_DATE)
)
"""

DDL_DETAILS = f"""
CREATE TABLE {TABLE_DETAILS} (
    CUSTOMER_ID VARCHAR2(20) NOT NULL,
    SCORING_DATE DATE DEFAULT SYSDATE,
    FEATURE_NAME VARCHAR2(50),
    FEATURE_LABEL VARCHAR2(100),
    BEKLENEN NUMBER(15,4),
    GERCEKLESEN NUMBER(15,4),
    DEGISIM_PCT NUMBER(10,2),
    KATKI_PCT NUMBER(5,1),
    SIRA NUMBER(2),
    CONSTRAINT PK_EWS_DETAILS PRIMARY KEY (CUSTOMER_ID, SCORING_DATE, FEATURE_NAME)
)
"""


def create_tables(conn):
    """Create tables (drop if exist)."""
    cursor = conn.cursor()

    tables = [
        (TABLE_DETAILS, DDL_DETAILS),
        (TABLE_RESULTS, DDL_RESULTS),
        (TABLE_SCORING, DDL_SCORING),
        (TABLE_TRAIN, DDL_TRAINING),
    ]

    # Drop in reverse order (details depends on results conceptually)
    for table_name, _ in tables:
        try:
            cursor.execute(f"DROP TABLE {table_name} PURGE")
            print(f"  Dropped: {table_name}")
        except Exception:
            pass

    # Create
    for table_name, ddl in reversed(tables):
        cursor.execute(ddl)
        print(f"  Created: {table_name}")

    conn.commit()
    cursor.close()


def load_training_data(conn, df, batch_size=500):
    """Load training data into Oracle."""
    cursor = conn.cursor()

    cols = ["CUSTOMER_ID", "SNAPSHOT_DATE", "SPLIT_FLAG"] + [f.upper() for f in ALL_FEATURES]
    placeholders = ", ".join([f":{i+1}" for i in range(len(cols))])
    sql = f"INSERT INTO {TABLE_TRAIN} ({', '.join(cols)}) VALUES ({placeholders})"

    rows = []
    for _, row in df.iterrows():
        vals = [row["customer_id"], row["snapshot_date"], row["split_flag"]]
        vals += [float(row[f]) if pd.notna(row[f]) else None for f in ALL_FEATURES]
        rows.append(vals)

    for i in range(0, len(rows), batch_size):
        cursor.executemany(sql, rows[i:i+batch_size])

    conn.commit()
    cursor.close()
    print(f"  Loaded {len(rows)} rows into {TABLE_TRAIN}")


def load_scoring_data(conn, df, batch_size=500):
    """Load scoring data into Oracle."""
    cursor = conn.cursor()

    cols = ["CUSTOMER_ID", "SNAPSHOT_DATE"] + [f.upper() for f in ALL_FEATURES]
    placeholders = ", ".join([f":{i+1}" for i in range(len(cols))])
    sql = f"INSERT INTO {TABLE_SCORING} ({', '.join(cols)}) VALUES ({placeholders})"

    rows = []
    for _, row in df.iterrows():
        vals = [row["customer_id"], row["snapshot_date"]]
        vals += [float(row[f]) if pd.notna(row[f]) else None for f in ALL_FEATURES]
        rows.append(vals)

    for i in range(0, len(rows), batch_size):
        cursor.executemany(sql, rows[i:i+batch_size])

    conn.commit()
    cursor.close()
    print(f"  Loaded {len(rows)} rows into {TABLE_SCORING}")


def main():
    print("Oracle EWS Setup")
    print("=" * 50)

    print("\n1. Generating data...")
    train_df = generate_training_data()
    scoring_df, _labels = generate_scoring_data()

    print(f"\n2. Connecting to Oracle ({SCHEMA})...")
    conn = get_connection()

    print("\n3. Creating tables...")
    create_tables(conn)

    print("\n4. Loading training data...")
    load_training_data(conn, train_df)

    print("\n5. Loading scoring data...")
    load_scoring_data(conn, scoring_df)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
