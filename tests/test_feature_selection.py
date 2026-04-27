import unittest

import pandas as pd

from engine.feature_selection import FeatureSelector


class FeatureSelectionTests(unittest.TestCase):
    def test_low_coverage_and_high_correlation_prune_longlist(self):
        config = {
            "feature_selection": {
                "enabled": True,
                "drop_zero_variance_continuous": False,
                "drop_exact_duplicates": False,
                "drop_low_coverage": {"enabled": True, "min_coverage": 0.60},
                "drop_high_correlation": {
                    "enabled": True,
                    "threshold": 0.97,
                    "priority_order": [
                        "base",
                        "delta_1",
                        "self_zscore_6",
                        "trend_slope_6",
                        "population_percentile",
                        "vs_population_median_delta",
                    ],
                },
                "branch_routing": {
                    "enabled": True,
                    "autoencoder": {"exclude_raw_types": [], "exclude_transforms": []},
                    "isolation_forest": {"exclude_raw_types": [], "exclude_transforms": []},
                    "mahalanobis": {"exclude_raw_types": ["binary"], "exclude_transforms": []},
                },
            }
        }
        selector = FeatureSelector(config)
        frame = pd.DataFrame(
            {
                "bank_debt_to_turnover": [1.0, 2.0, 3.0, 4.0, 5.0],
                "bank_debt_to_turnover__delta_1": [1.01, 2.01, 3.01, 4.01, 5.01],
                "pos_volume_change": [0.2, None, None, None, None],
                "risk_flag": [0, 1, 0, 1, 0],
            }
        )
        registry = {
            "bank_debt_to_turnover": {"raw_type": "continuous", "transform": "raw"},
            "bank_debt_to_turnover__delta_1": {"raw_type": "continuous", "transform": "delta_1"},
            "pos_volume_change": {"raw_type": "continuous", "transform": "raw"},
            "risk_flag": {"raw_type": "binary", "transform": "raw"},
        }

        summary = selector.select(frame, registry)

        self.assertIn("bank_debt_to_turnover", summary["kept_features"])
        self.assertNotIn("bank_debt_to_turnover__delta_1", summary["kept_features"])
        self.assertEqual(
            summary["dropped_features"]["bank_debt_to_turnover__delta_1"],
            "high_correlation_with:bank_debt_to_turnover",
        )
        self.assertNotIn("pos_volume_change", summary["kept_features"])
        self.assertEqual(summary["dropped_features"]["pos_volume_change"], "low_coverage<0.6")
        self.assertNotIn("risk_flag", summary["branch_features"]["mahalanobis"])


if __name__ == "__main__":
    unittest.main()
