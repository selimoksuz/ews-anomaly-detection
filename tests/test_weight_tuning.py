import unittest

import pandas as pd

from engine.weight_tuning import WeightOptimizer


class WeightOptimizerTests(unittest.TestCase):
    def test_optimizer_prefers_component_that_matches_target(self):
        config = {
            "weight_optimization": {
                "grid_step": 0.5,
                "objective": "precision_at_top_percent",
                "top_percent": 0.25,
            }
        }
        tuning_frame = pd.DataFrame(
            {
                "ae_cal": [95, 90, 85, 30, 20, 10, 5, 1],
                "if_cal": [10, 12, 15, 70, 65, 60, 55, 50],
                "md_cal": [20, 18, 16, 40, 38, 36, 34, 32],
                "label_30dpd_8w": [1, 1, 1, 0, 0, 0, 0, 0],
                "label_default_12m": [1, 0, 1, 0, 0, 0, 0, 0],
            }
        )
        validation_frame = tuning_frame.copy()

        optimizer = WeightOptimizer(config)
        artifact = optimizer.optimize(
            tuning_frame,
            validation_frame,
            target_column="label_30dpd_8w",
            monitoring_columns=["label_default_12m"],
            model_version="MODEL-1",
            segment="ALL",
        )

        self.assertGreaterEqual(artifact["weights"]["autoencoder"], artifact["weights"]["isolation_forest"])
        self.assertGreaterEqual(
            artifact["validation_metrics"]["primary"]["precision_at_top_percent"],
            0.5,
        )

    def test_optimizer_respects_min_component_weight_floor(self):
        config = {
            "weight_optimization": {
                "grid_step": 0.1,
                "min_component_weight": 0.1,
                "objective": "precision_at_top_percent",
                "top_percent": 0.25,
            }
        }
        frame = pd.DataFrame(
            {
                "ae_cal": [95, 90, 82, 75, 30, 20, 10, 5],
                "if_cal": [92, 86, 80, 72, 40, 25, 15, 5],
                "md_cal": [91, 87, 81, 70, 35, 22, 11, 4],
                "label_30dpd_8w": [1, 1, 1, 1, 0, 0, 0, 0],
            }
        )

        optimizer = WeightOptimizer(config)
        artifact = optimizer.optimize(
            frame,
            frame.copy(),
            target_column="label_30dpd_8w",
            monitoring_columns=[],
            model_version="MODEL-2",
            segment="ALL",
        )

        self.assertGreaterEqual(artifact["weights"]["autoencoder"], 0.1)
        self.assertGreaterEqual(artifact["weights"]["isolation_forest"], 0.1)
        self.assertGreaterEqual(artifact["weights"]["mahalanobis"], 0.1)


if __name__ == "__main__":
    unittest.main()
