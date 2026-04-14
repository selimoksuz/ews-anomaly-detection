import unittest

import numpy as np
import pandas as pd

from engine.config_loader import get_feature_list, load_config
from engine.scorer import AnomalyScorer


class _StubModels:
    def __init__(self, features):
        self.features = features

    def transform(self, frame):
        return frame.fillna(0).to_numpy(dtype=float)

    def ae_reconstruct(self, X):
        return np.zeros_like(X)

    def actual_values(self, frame):
        return frame.fillna(0).to_numpy(dtype=float)

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


if __name__ == "__main__":
    unittest.main()
