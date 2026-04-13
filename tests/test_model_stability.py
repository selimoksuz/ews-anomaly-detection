import unittest

import numpy as np

from engine.config_loader import get_feature_list, load_config
from engine.models import AnomalyModels
from scripts.generate_data import generate_scoring_data, generate_training_data


class ModelStabilityTests(unittest.TestCase):
    def test_repeated_fit_is_deterministic(self):
        config = load_config()
        features = get_feature_list(config)

        train_df = generate_training_data(n=800, seed=42)
        train_df = train_df[train_df["split_flag"] == "TRAIN"].reset_index(drop=True)
        scoring_df, _ = generate_scoring_data(n=250, seed=99)

        X_train = train_df[features].fillna(0).values
        X_score = scoring_df[features].fillna(0).values

        model_a = AnomalyModels(config)
        model_a.fit(X_train)
        score_a = model_a.transform(X_score)

        model_b = AnomalyModels(config)
        model_b.fit(X_train)
        score_b = model_b.transform(X_score)

        np.testing.assert_allclose(model_a.raw_ae_scores(score_a), model_b.raw_ae_scores(score_b))
        np.testing.assert_allclose(model_a.raw_if_scores(score_a), model_b.raw_if_scores(score_b))
        np.testing.assert_allclose(model_a.raw_md_scores(score_a), model_b.raw_md_scores(score_b))
        np.testing.assert_allclose(model_a.ae_scores(score_a), model_b.ae_scores(score_b))
        np.testing.assert_allclose(model_a.if_scores(score_a), model_b.if_scores(score_b))
        np.testing.assert_allclose(model_a.md_scores(score_a), model_b.md_scores(score_b))


if __name__ == "__main__":
    unittest.main()
