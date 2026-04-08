"""Pipeline orchestrator — train, score, setup steps."""

import logging
import pickle
from pathlib import Path
from datetime import date

from engine.config_loader import load_config, load_secrets, get_feature_list
from engine.oracle_io import OracleConnector
from engine.models import AnomalyModels
from engine.scorer import AnomalyScorer

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")


class EWSPipeline:
    """EWS Anomaly Detection Pipeline."""

    def __init__(self, config_path="config/pipeline_config.yaml", secrets_path="config/secrets.yaml"):
        self.config = load_config(config_path)
        self.secrets = load_secrets(secrets_path)
        self.features = get_feature_list(self.config)
        self.oracle = OracleConnector(self.config, self.secrets)
        self.models = None
        self.scorer = None

    # ── SETUP ──

    def setup(self):
        """Oracle tablolarini olustur."""
        logger.info("=== SETUP ===")
        self.oracle.connect()
        self.oracle.setup_tables()
        self.oracle.close()
        logger.info("Setup complete")

    def load_data(self, train_df, scoring_df=None):
        """DataFrame'leri Oracle'a yukle."""
        logger.info("=== LOAD DATA ===")
        self.oracle.connect()
        self.oracle.setup_tables()

        self.oracle.load_dataframe(train_df, "training", extra_cols=["split_flag"])
        logger.info(f"Training data loaded: {len(train_df)} rows")

        if scoring_df is not None:
            self.oracle.load_dataframe(scoring_df, "scoring")
            logger.info(f"Scoring data loaded: {len(scoring_df)} rows")

        self.oracle.close()

    # ── TRAIN ──

    def train(self):
        """Oracle'dan training verisini oku, modeli egit, kaydet."""
        logger.info("=== TRAIN ===")
        self.oracle.connect()

        train_df = self.oracle.read_training_data(split="TRAIN")
        logger.info(f"Training data: {train_df.shape}")

        X_raw = train_df[self.features].fillna(0).values

        self.models = AnomalyModels(self.config)
        self.models.fit(X_raw)

        self._save_model()

        # Test seti ile stabilite kontrolu
        test_df = self.oracle.read_training_data(split="TEST")
        if len(test_df) > 0:
            self._evaluate_stability(train_df, test_df)

        self.oracle.close()
        logger.info("Training complete")

    # ── SCORE ──

    def score(self):
        """Oracle'dan scoring verisini oku, skorla, sonuclari yaz."""
        logger.info("=== SCORE ===")
        self._load_model()

        self.oracle.connect()
        scoring_df = self.oracle.read_scoring_data()
        logger.info(f"Scoring data: {scoring_df.shape}")

        self.scorer = AnomalyScorer(self.config, self.models)
        results = self.scorer.score(scoring_df)

        scoring_date = date.today()
        self.oracle.write_results(results, scoring_date)
        self.oracle.write_details(results, scoring_date)

        self.oracle.close()
        logger.info("Scoring complete")
        return results

    # ── TRAIN + SCORE (tek komut) ──

    def run(self):
        """Train + Score tek seferde."""
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
            raise FileNotFoundError(f"Model dosyasi bulunamadi: {path}. Once 'train' calistirin.")
        with open(path, "rb") as f:
            self.models = pickle.load(f)
        logger.info(f"Model loaded: {path}")

    # ── EVALUATION ──

    def _evaluate_stability(self, train_df, test_df):
        from scipy.stats import ks_2samp

        X_tr = self.models.transform(train_df[self.features].fillna(0).values)
        X_te = self.models.transform(test_df[self.features].fillna(0).values)

        tr_err = self.models._ae_total_error(X_tr)
        te_err = self.models._ae_total_error(X_te)
        ks_stat, ks_pval = ks_2samp(tr_err, te_err)

        ratio = te_err.mean() / tr_err.mean()
        logger.info(f"Stability: AE ratio={ratio:.3f}x, KS p={ks_pval:.4f}")

        if ks_pval < 0.05:
            logger.warning("AE reconstruction error distribution differs between train and test (possible overfit)")
        if ratio > 1.5:
            logger.warning(f"AE test/train loss ratio high: {ratio:.3f}x")
