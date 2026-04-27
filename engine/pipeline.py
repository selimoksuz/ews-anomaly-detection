"""Pipeline orchestrator - train, score, setup steps."""

import json
import logging
import pickle
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config_loader import (
    get_alert_bands,
    get_ensemble_weights,
    get_feature_list,
    load_config,
    load_secrets,
    resolve_project_path,
    resolve_feature_list,
)
from engine.models import AnomalyModels
from engine.oracle_io import OracleConnector
from engine.scorer import AnomalyScorer

logger = logging.getLogger(__name__)

MODEL_DIR = resolve_project_path("runtime/legacy_models")


class EWSPipeline:
    """EWS Anomaly Detection Pipeline."""

    def __init__(self, config_path=None, secrets_path=None):
        self.config = load_config(config_path)
        self.secrets = load_secrets(secrets_path)
        self.features = get_feature_list(self.config)
        self.id_col = self.config["pipeline"]["id_column"]
        self.time_col = self.config["pipeline"]["time_column"]
        self.stability_cfg = self.config.get("stability", {})
        self.models = None
        self.scorer = None

    def _get_oracle(self):
        return OracleConnector(self.config, self.secrets)

    # Setup

    def setup(self):
        logger.info("=== SETUP ===")
        with self._get_oracle() as ora:
            ora.setup_tables(drop_existing=True)
        logger.info("Setup complete")

    def load_data(self, train_df, scoring_df=None, outcomes_df=None):
        """DataFrame'leri Oracle'a yukle."""
        logger.info("=== LOAD DATA ===")
        with self._get_oracle() as ora:
            ora.setup_tables(drop_existing=True)
            ora.replace_rows("training", train_df)
            logger.info("Training: %s rows loaded", len(train_df))

            if scoring_df is not None:
                ora.replace_rows("scoring", scoring_df)
                logger.info("Scoring: %s rows loaded", len(scoring_df))
            if outcomes_df is not None and "outcomes" in self.config.get("oracle", {}).get("tables", {}):
                ora.replace_rows("outcomes", outcomes_df)
                logger.info("Outcomes: %s rows loaded", len(outcomes_df))

    # Train

    def train(self):
        logger.info("=== TRAIN ===")
        with self._get_oracle() as ora:
            train_df = ora.read_training_data(split="TRAIN")
            logger.info("Training data: %s", train_df.shape)

            self.features = resolve_feature_list(self.config, train_df)
            self.models = AnomalyModels(self.config, feature_names=self.features)
            self.models.fit(train_df, feature_names=self.features)
            self._save_model()

            test_df = ora.read_training_data(split="TEST")
            if len(test_df) > 0:
                self._evaluate_stability(train_df, test_df)

        logger.info("Training complete")

    # Score

    def score(self):
        logger.info("=== SCORE ===")
        self._load_model()
        self.features = list(self.models.feature_names)

        with self._get_oracle() as ora:
            scoring_df = ora.read_scoring_data()
            logger.info("Scoring data: %s", scoring_df.shape)

            self.scorer = AnomalyScorer(self.config, self.models)
            results = self.scorer.score(scoring_df)

            scoring_date = date.today()
            self._write_to_oracle(ora, results, scoring_date)

        logger.info("Scoring complete")
        return results

    def _write_to_oracle(self, ora, results, scoring_date):
        """Scorer ciktisini Codex OracleConnector formatina cevir ve yaz."""
        res = results.copy()
        res[self.time_col] = pd.Timestamp(scoring_date)

        reasons = []
        for _, row in res.iterrows():
            parts = []
            if isinstance(row.get("detay"), dict):
                for _, d in row["detay"].items():
                    parts.append(
                        "\n".join(
                            [
                                f"{d['label']}",
                                f"gerceklesen: {self._display_value(d.get('gerceklesen'))}",
                                f"musteri_gecmis_referansi: {self._display_value(d.get('musteri_gecmis_referansi'))}",
                                f"populasyon_referansi: {self._display_value(d.get('populasyon_referansi'))}",
                                f"ae_referansi: {self._display_value(d.get('ae_referansi', d.get('beklenen')))}",
                                *([f"yon: {d.get('yon')}"] if d.get("yon") else []),
                                *([f"yon_yorumu: {d.get('yon_yorumu')}"] if d.get("yon_yorumu") else []),
                                f"ensemble_katki: %{self._display_pct(d.get('ensemble_katki_pct', d.get('katki_pct')))} "
                                f"(AE %{self._display_pct(d.get('ae_katki_pct'))}, "
                                f"IF %{self._display_pct(d.get('if_katki_pct'))}, "
                                f"MD %{self._display_pct(d.get('md_katki_pct'))})",
                            ]
                        )
                    )
            reasons.append(parts)
        res["reasons"] = reasons

        ora.write_results(res)
        logger.info("Wrote %s results", len(res))

        alerts = results[results["alert_band"].isin(["KIRMIZI", "TURUNCU", "SARI"])]
        detail_rows = []
        for _, row in alerts.iterrows():
            if not isinstance(row.get("detay"), dict):
                continue
            for rank, (feat, d) in enumerate(row["detay"].items(), 1):
                detail_rows.append(
                    {
                        self.id_col: row[self.id_col],
                        self.time_col: pd.Timestamp(scoring_date),
                        "feature_name": feat,
                        "feature_label": d["label"],
                        "expected_value": d.get("expected_value", d.get("ae_referansi", d.get("beklenen"))),
                        "actual_value": d.get("actual_value", d.get("gerceklesen")),
                        "delta_pct": d.get("delta_pct", d.get("degisim_pct")),
                        "contribution_pct": d.get("contribution_pct", d.get("ensemble_katki_pct", d.get("katki_pct"))),
                        "customer_history_reference": d.get("musteri_gecmis_referansi"),
                        "population_reference": d.get("populasyon_referansi"),
                        "ae_reference": d.get("ae_referansi", d.get("beklenen")),
                        "ae_contribution_pct": d.get("ae_katki_pct"),
                        "if_contribution_pct": d.get("if_katki_pct"),
                        "md_contribution_pct": d.get("md_katki_pct"),
                        "rank": rank,
                    }
                )

        if detail_rows:
            details_df = pd.DataFrame(detail_rows)
            ora.write_details(details_df)
            logger.info("Wrote %s detail rows", len(details_df))

    # Train + Score

    def run(self):
        self.train()
        return self.score()

    # Model persistence

    def _save_model(self):
        MODEL_DIR.mkdir(exist_ok=True)
        path = MODEL_DIR / "ews_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.models, f)
        logger.info("Model saved: %s", path)

    def _load_model(self):
        path = MODEL_DIR / "ews_model.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model bulunamadi: {path}. Once 'train' calistirin.")
        with open(path, "rb") as f:
            self.models = pickle.load(f)
        logger.info("Model loaded: %s", path)

    def _score_to_band(self, scores):
        bands = get_alert_bands(self.config)
        result = []
        for score in scores:
            assigned = "NORMAL"
            for band_name, (lo, hi) in bands.items():
                if lo <= score < hi or (band_name == "KIRMIZI" and score >= lo):
                    assigned = band_name
            result.append(assigned)
        return result

    @staticmethod
    def _display_value(value):
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.2f}"

    @staticmethod
    def _display_pct(value):
        if value is None or pd.isna(value):
            return "0"
        return f"{float(value):.1f}".rstrip("0").rstrip(".")

    @staticmethod
    def _summarize_distribution(values):
        values = np.asarray(values, dtype=float)
        return {
            "mean": round(float(values.mean()), 4),
            "median": round(float(np.median(values)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "p99": round(float(np.percentile(values, 99)), 4),
        }

    def _band_share(self, scores):
        bands = pd.Series(self._score_to_band(scores))
        total = len(bands)
        return {
            band: round(float((bands == band).sum() / total), 4)
            for band in ("NORMAL", "SARI", "TURUNCU", "KIRMIZI")
        }

    def _save_stability_report(self, report):
        report_path = Path(
            resolve_project_path(
                self.stability_cfg.get("report_path", MODEL_DIR / "ews_model_stability.json")
            )
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        logger.info("Stability report saved: %s", report_path)

    def _evaluate_stability(self, train_df, test_df):
        from scipy.stats import ks_2samp

        weights = get_ensemble_weights(self.config)
        X_tr = self.models.transform(train_df)
        X_te = self.models.transform(test_df)

        train_metrics = {
            "ae_raw": self.models.raw_ae_scores(X_tr),
            "if_raw": self.models.raw_if_scores(X_tr),
            "md_raw": self.models.raw_md_scores(X_tr),
            "ae_score": self.models.ae_scores(X_tr),
            "if_score": self.models.if_scores(X_tr),
            "md_score": self.models.md_scores(X_tr),
        }
        test_metrics = {
            "ae_raw": self.models.raw_ae_scores(X_te),
            "if_raw": self.models.raw_if_scores(X_te),
            "md_raw": self.models.raw_md_scores(X_te),
            "ae_score": self.models.ae_scores(X_te),
            "if_score": self.models.if_scores(X_te),
            "md_score": self.models.md_scores(X_te),
        }

        train_metrics["ensemble_score"] = np.clip(
            weights["autoencoder"] * train_metrics["ae_score"]
            + weights["isolation_forest"] * train_metrics["if_score"]
            + weights["mahalanobis"] * train_metrics["md_score"],
            0,
            100,
        )
        test_metrics["ensemble_score"] = np.clip(
            weights["autoencoder"] * test_metrics["ae_score"]
            + weights["isolation_forest"] * test_metrics["if_score"]
            + weights["mahalanobis"] * test_metrics["md_score"],
            0,
            100,
        )

        report = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "metrics": {},
            "ensemble_alert_share": {
                "train": self._band_share(train_metrics["ensemble_score"]),
                "test": self._band_share(test_metrics["ensemble_score"]),
            },
        }

        for metric_name in (
            "ae_raw",
            "if_raw",
            "md_raw",
            "ae_score",
            "if_score",
            "md_score",
            "ensemble_score",
        ):
            train_values = train_metrics[metric_name]
            test_values = test_metrics[metric_name]
            ks_stat, ks_pval = ks_2samp(train_values, test_values)
            train_summary = self._summarize_distribution(train_values)
            test_summary = self._summarize_distribution(test_values)
            mean_ratio = None
            if abs(train_summary["mean"]) > 1e-12:
                mean_ratio = round(float(test_summary["mean"] / train_summary["mean"]), 4)

            report["metrics"][metric_name] = {
                "train": train_summary,
                "test": test_summary,
                "mean_ratio": mean_ratio,
                "ks_stat": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pval), 4),
            }
            logger.info(
                "Stability %s: mean_ratio=%s, train_p95=%.4f, test_p95=%.4f, KS p=%.4f",
                metric_name,
                mean_ratio,
                train_summary["p95"],
                test_summary["p95"],
                ks_pval,
            )

        logger.info(
            "Ensemble alert share train=%s test=%s",
            report["ensemble_alert_share"]["train"],
            report["ensemble_alert_share"]["test"],
        )
        self._save_stability_report(report)
