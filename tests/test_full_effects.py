import unittest

from engine.config_loader import get_feature_list, load_config
from engine.models import AnomalyModels
from engine.scorer import AnomalyScorer
from scripts.generate_data import generate_scoring_data, generate_training_data


class FullEffectsTests(unittest.TestCase):
    def test_scorer_keeps_all_feature_effects(self):
        config = load_config()
        features = get_feature_list(config)

        train_df = generate_training_data(n=300, seed=42)
        train_df = train_df[train_df["split_flag"] == "TRAIN"].reset_index(drop=True)
        scoring_df, _ = generate_scoring_data(n=250, seed=99)

        model = AnomalyModels(config)
        model.fit(train_df[features].fillna(0).values)

        result = AnomalyScorer(config, model).score(scoring_df)
        self.assertIn("full_detay", result.columns)
        self.assertEqual(len(result.loc[0, "full_detay"]), len(features))
        self.assertEqual(len(result.loc[0, "detay"]), config["scoring"]["top_n_reasons"])


if __name__ == "__main__":
    unittest.main()
