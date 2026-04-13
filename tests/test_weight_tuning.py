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


if __name__ == "__main__":
    unittest.main()
