"""Pipeline orchestrator — train, score, setup steps."""

import logging
import pickle
from pathlib import Path
from datetime import date, datetime

import numpy as np
import pandas as pd

from engine.config_loader import load_config, load_secrets, get_feature_list
from engine.oracle_io import OracleConnector
from engine.models import AnomalyModels
from engine.scorer import AnomalyScorer

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")


class EWSPipeline:
    """EWS Anomaly Detection Pipeline."""

    def __init__(self, config_path=None, secrets_path=None):
        self.config = load_config(config_path)
        self.secrets = load_secrets(secrets_path)
        self.features = get_feature_list(self.config)
        self.id_col = self.config["pipeline"]["id_column"]
        self.time_col = self.config["pipeline"]["time_column"]
        self.models = None
        self.scorer = None

    def _get_oracle(self):
        return OracleConnector(self.config, self.secrets)

    # ── SETUP ──

    def setup(self):
        logger.info("=== SETUP ===")
        with self._get_oracle() as ora:
            ora.setup_tables(drop_existing=True)
        logger.info("Setup complete")

    def load_data(self, train_df, scoring_df=None):
        """DataFrame'leri Oracle'a yukle."""
        logger.info("=== LOAD DATA ===")
        with self._get_oracle() as ora:
            ora.setup_tables(drop_existing=True)

            # Training data
            self._insert_dataframe(ora, train_df, "training",
                                   extra_cols=[self.config["pipeline"].get("split_column", "split_flag")])
            logger.info(f"Training: {len(train_df)} rows loaded")

            if scoring_df is not None:
                self._insert_dataframe(ora, scoring_df, "scoring")
                logger.info(f"Scoring: {len(scoring_df)} rows loaded")

    def _insert_dataframe(self, ora, df, table_key, extra_cols=None):
        """Generic DataFrame → Oracle insert."""
        conn = ora.connect()
        cursor = conn.cursor()

        cols = [self.id_col, self.time_col]
        if extra_cols:
            cols += extra_cols
        cols += self.features

        col_names = [c.upper() for c in cols]
        placeholders = ", ".join([f":{i+1}" for i in range(len(col_names))])
        table = ora._qualified_table_name(table_key)
        sql = f"INSERT INTO {table} ({', '.join(col_names)}) VALUES ({placeholders})"

        rows = []
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row.get(c)
                if isinstance(v, (pd.Timestamp, datetime)):
                    vals.append(v.to_pydatetime() if isinstance(v, pd.Timestamp) else v)
                elif pd.isna(v) if not isinstance(v, str) else False:
                    vals.append(None)
                else:
                    vals.append(v)
            rows.append(vals)

        for i in range(0, len(rows), 500):
            cursor.executemany(sql, rows[i:i+500])
        conn.commit()
        cursor.close()

    # ── TRAIN ──

    def train(self):
        logger.info("=== TRAIN ===")
        with self._get_oracle() as ora:
            train_df = ora.read_training_data(split="TRAIN")
            logger.info(f"Training data: {train_df.shape}")

            X_raw = train_df[self.features].fillna(0).values
            self.models = AnomalyModels(self.config)
            self.models.fit(X_raw)
            self._save_model()

            test_df = ora.read_training_data(split="TEST")
            if len(test_df) > 0:
                self._evaluate_stability(train_df, test_df)

        logger.info("Training complete")

    # ── SCORE ──

    def score(self):
        logger.info("=== SCORE ===")
        self._load_model()

        with self._get_oracle() as ora:
            scoring_df = ora.read_scoring_data()
            logger.info(f"Scoring data: {scoring_df.shape}")

            self.scorer = AnomalyScorer(self.config, self.models)
            results = self.scorer.score(scoring_df)

            scoring_date = date.today()
            self._write_to_oracle(ora, results, scoring_date)

        logger.info("Scoring complete")
        return results

    def _write_to_oracle(self, ora, results, scoring_date):
        """Scorer ciktisini Codex OracleConnector formatina cevir ve yaz."""
        # Results tablosu
        res = results.copy()
        res[self.time_col] = pd.Timestamp(scoring_date)

        # Reasons: detay dict'ten string listesine cevir
        reasons = []
        for _, row in res.iterrows():
            parts = []
            if isinstance(row.get("detay"), dict):
                for feat, d in row["detay"].items():
                    ico = "UP" if d["degisim_pct"] > 0 else "DN"
                    parts.append(f"{d['label']}: {d['beklenen']}->{d['gerceklesen']} ({ico}%{abs(d['degisim_pct']):.0f})")
            reasons.append(parts)
        res["reasons"] = reasons

        ora.write_results(res)
        logger.info(f"Wrote {len(res)} results")

        # Details tablosu
        alerts = results[results["alert_band"].isin(["KIRMIZI", "TURUNCU", "SARI"])]
        detail_rows = []
        for _, row in alerts.iterrows():
            if not isinstance(row.get("detay"), dict):
                continue
            for rank, (feat, d) in enumerate(row["detay"].items(), 1):
                detail_rows.append({
                    self.id_col: row[self.id_col],
                    self.time_col: pd.Timestamp(scoring_date),
                    "feature_name": feat,
                    "feature_label": d["label"],
                    "expected_value": d["beklenen"],
                    "actual_value": d["gerceklesen"],
                    "delta_pct": d["degisim_pct"],
                    "contribution_pct": d["katki_pct"],
                    "rank": rank,
                })

        if detail_rows:
            details_df = pd.DataFrame(detail_rows)
            ora.write_details(details_df)
            logger.info(f"Wrote {len(details_df)} detail rows")

    # ── TRAIN + SCORE ──

    def run(self):
        self.train()
        return self.score()

    # ── MODEL PERSISTENCE ──

    def _save_model(self):
        MODEL_DIR.mkdir(exist_ok=True)
        path = MODEL_DIR / "ews_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.models, f)
        logger.info(f"Model saved: {path}")

    def _load_model(self):
        path = MODEL_DIR / "ews_model.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model bulunamadi: {path}. Once 'train' calistirin.")
        with open(path, "rb") as f:
            self.models = pickle.load(f)
        logger.info(f"Model loaded: {path}")

    def _evaluate_stability(self, train_df, test_df):
        from scipy.stats import ks_2samp
        X_tr = self.models.transform(train_df[self.features].fillna(0).values)
        X_te = self.models.transform(test_df[self.features].fillna(0).values)
        tr_err = self.models._ae_total_error(X_tr)
        te_err = self.models._ae_total_error(X_te)
        ks_stat, ks_pval = ks_2samp(tr_err, te_err)
        ratio = te_err.mean() / tr_err.mean()
        logger.info(f"Stability: AE ratio={ratio:.3f}x, KS p={ks_pval:.4f}")
