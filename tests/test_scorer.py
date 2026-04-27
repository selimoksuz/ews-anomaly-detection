import unittest

import numpy as np
import pandas as pd

from engine.config_loader import get_feature_list, load_config
from engine.scorer import AnomalyScorer


class _StubModels:
    def __init__(self, features):
        self.features = features
        self.feature_names = features
        self.raw_feature_names = features

    def transform(self, frame):
        return frame[self.raw_feature_names].fillna(0).to_numpy(dtype=float)

    def ae_reconstruct(self, X):
        return np.zeros_like(X)

    def actual_values(self, frame):
        return frame[self.raw_feature_names].fillna(0).to_numpy(dtype=float)

    def inverse_transform(self, X):
        return X

    def ae_contribution(self, X):
        return np.ones_like(X) / X.shape[1]

    def if_contribution(self, X):
        return np.ones_like(X) / X.shape[1]

    def md_contribution(self, X):
        return np.ones_like(X) / X.shape[1]

    def raw_ae_scores(self, X):
        return np.full(X.shape[0], 0.1)

    def raw_if_scores(self, X):
        return np.full(X.shape[0], 0.2)

    def raw_md_scores(self, X):
        return np.full(X.shape[0], 0.3)

    def ae_scores(self, X):
        return np.full(X.shape[0], 10.0)

    def if_scores(self, X):
        return np.full(X.shape[0], 20.0)

    def md_scores(self, X):
        return np.full(X.shape[0], 30.0)


class ScorerTests(unittest.TestCase):
    def test_metadata_does_not_override_row_level_segment(self):
        config = load_config()
        features = get_feature_list(config)
        row = {feature: 1.0 for feature in features}
        row["customer_id"] = "CUST_1"
        row["snapshot_date"] = "2026-01-01"
        row["segment"] = "ENTITY_123"
        frame = pd.DataFrame([row])

        scorer = AnomalyScorer(
            config,
            _StubModels(features),
            metadata={
                "run_id": "run-1",
                "segment": "ALL",
                "model_version": "model-1",
            },
        )

        scored = scorer.score(frame)
        self.assertEqual(scored.loc[0, "segment"], "ENTITY_123")
        self.assertEqual(scored.loc[0, "run_id"], "run-1")
        self.assertEqual(scored.loc[0, "model_version"], "model-1")

    def test_reason_contains_direction_metadata(self):
        config = load_config()
        features = get_feature_list(config)
        row = {feature: 1.0 for feature in features}
        row["bank_debt_to_turnover"] = 2.0
        row["bank_debt_to_turnover__delta_1"] = 0.5
        row["customer_id"] = "CUST_2"
        row["snapshot_date"] = "2026-01-31"
        row["segment"] = "TICARI_ORTA"
        frame = pd.DataFrame([row])

        scorer = AnomalyScorer(config, _StubModels(features))
        scored = scorer.score(frame)
        detail = scored.loc[0, "detay"]["bank_debt_to_turnover"]
        reason_text = scored.loc[0, "reason_1"]

        self.assertEqual(detail["yon"], "artmasi kotu, azalmasi iyi")
        self.assertIn("yon: artmasi kotu, azalmasi iyi", reason_text)
        self.assertIn("yon_yorumu: musteri gecmis referansina gore artmis ve kotulesme yonunde", reason_text)

    def test_full_detail_population_percentile_contains_all_references(self):
        config = load_config()
        features = get_feature_list(config)
        row = {feature: 0.0 for feature in features}
        row["bank_debt_to_turnover"] = 0.8
        row["bank_debt_to_turnover__delta_1"] = 0.1
        row["bank_debt_to_turnover__population_percentile"] = 0.9
        row["customer_id"] = "CUST_3"
        row["snapshot_date"] = "2026-01-31"
        row["segment"] = "TICARI_ORTA"
        frame = pd.DataFrame([row])

        scorer = AnomalyScorer(config, _StubModels(features))
        scored = scorer.score(frame)
        detail = scored.loc[0, "full_detay"]["bank_debt_to_turnover__population_percentile"]

        self.assertEqual(detail["musteri_gecmis_referansi"], 0.5)
        self.assertEqual(detail["populasyon_referansi"], 0.5)
        self.assertEqual(detail["yon"], "artmasi kotu, azalmasi iyi")
        self.assertEqual(
            detail["yon_yorumu"],
            "genel percentile referansina gore artmis ve kotulesme yonunde",
        )


if __name__ == "__main__":
    unittest.main()
