import unittest

import pandas as pd

from engine.config_loader import get_feature_list, load_config
from engine.data_loader import DataLoader
from engine.preprocessing import FeaturePreprocessor


class PreprocessingTests(unittest.TestCase):
    def test_hard_bounds_are_applied_from_config(self):
        config = load_config()
        features = get_feature_list(config)
        preprocessor = FeaturePreprocessor(config, features)

        frame = pd.DataFrame([{feature: 1.0 for feature in features}])
        frame["cash_advance_ratio_4w"] = 1.8
        frame["dpd_current"] = -5
        transformed = preprocessor.fit_transform(frame)
        self.assertEqual(transformed.shape[1], len(features))

        actual = preprocessor.prepare_actual_values(frame)
        cash_idx = features.index("cash_advance_ratio_4w")
        dpd_idx = features.index("dpd_current")
        self.assertEqual(actual[0, cash_idx], 1.0)
        self.assertEqual(actual[0, dpd_idx], 0.0)

    def test_data_loader_allows_missing_feature_values(self):
        config = load_config()
        features = get_feature_list(config)
        loader = DataLoader(config)
        frame = pd.DataFrame([{feature: 1.0 for feature in features}])
        frame["customer_id"] = "CUST_1"
        frame["snapshot_date"] = "2026-01-01"
        frame["txn_amount_weekly"] = None

        validated = loader.validate_data(frame)
        self.assertTrue(validated["txn_amount_weekly"].isnull().any())

    def test_feature_level_missing_strategies_support_min_max_mean_constant(self):
        config = load_config()
        features = get_feature_list(config)
        config["preprocessing"]["missing"] = {
            "default_strategy": "median",
            "feature_strategies": {
                "txn_count_weekly": {"strategy": "constant", "value": 0},
                "channel_count_4w": {"strategy": "min"},
                "payment_reversal_count_4w": {"strategy": "max"},
                "checking_balance": {"strategy": "mean"},
            },
        }
        preprocessor = FeaturePreprocessor(config, features)

        rows = []
        for index, checking_balance in enumerate([100.0, 200.0, 300.0], start=1):
            row = {feature: float(index) for feature in features}
            row["txn_count_weekly"] = float(index)
            row["channel_count_4w"] = float(index + 2)
            row["payment_reversal_count_4w"] = float(index + 4)
            row["checking_balance"] = checking_balance
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.loc[1, "txn_count_weekly"] = None
        frame.loc[1, "channel_count_4w"] = None
        frame.loc[1, "payment_reversal_count_4w"] = None
        frame.loc[1, "checking_balance"] = None
        frame.loc[1, "outstanding_balance"] = None

        preprocessor.fit(frame)
        actual = pd.DataFrame(preprocessor.prepare_actual_values(frame), columns=features)

        self.assertEqual(actual.loc[1, "txn_count_weekly"], 0.0)
        self.assertEqual(actual.loc[1, "channel_count_4w"], 3.0)
        self.assertEqual(actual.loc[1, "payment_reversal_count_4w"], 7.0)
        self.assertEqual(actual.loc[1, "checking_balance"], 200.0)
        self.assertEqual(actual.loc[1, "outstanding_balance"], 2.0)


if __name__ == "__main__":
    unittest.main()
