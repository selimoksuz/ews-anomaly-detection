import unittest

import pandas as pd

from engine.config_loader import get_feature_list, load_config
from engine.data_loader import DataLoader
from engine.models import AnomalyModels
from engine.preprocessing import FeaturePreprocessor
from engine.scorer import AnomalyScorer


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

    def test_data_loader_can_infer_numeric_features_without_explicit_config(self):
        config = load_config()
        config["features"] = {
            "mode": "infer",
            "exclude_columns": ["manual_ignore"],
            "label_overrides": {"numeric_b": "Numeric B"},
        }
        config["pipeline"]["non_feature_columns"] = ["entity_code"]

        frame = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "snapshot_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "entity_code": ["E1", "E2", "E3"],
                "manual_ignore": [99, 98, 97],
                "numeric_a": [1.0, 2.0, 3.0],
                "numeric_b": ["4.0", "5.0", "6.0"],
                "categorical_x": ["A", "B", "C"],
                "created_at": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            }
        )

        loader = DataLoader(config)
        validated = loader.validate_data(frame)

        self.assertEqual(loader.feature_names, ["numeric_a", "numeric_b"])
        self.assertTrue({"numeric_a", "numeric_b"}.issubset(validated.columns))
        self.assertNotIn("manual_ignore", loader.feature_names)
        self.assertNotIn("entity_code", loader.feature_names)
        self.assertNotIn("created_at", loader.feature_names)

    def test_models_and_scorer_can_run_with_inferred_features(self):
        config = load_config()
        config["features"] = {
            "mode": "infer",
            "exclude_columns": [],
            "label_overrides": {"feature_b": "Feature B"},
        }
        config["preprocessing"]["hard_bounds"]["rules"] = []
        config["preprocessing"]["missing"]["feature_strategies"] = {}

        feature_frame = pd.DataFrame(
            {
                "feature_a": [0.1, 0.2, 0.3, 0.4, 0.5],
                "feature_b": [10.0, 11.0, 10.5, 12.0, 9.5],
                "feature_c": [1.0, 0.0, 1.0, 0.0, 1.0],
            }
        )
        model = AnomalyModels(config)
        model.fit(feature_frame)

        scoring_frame = feature_frame.copy()
        scoring_frame["customer_id"] = [f"C{i}" for i in range(len(scoring_frame))]
        scoring_frame["snapshot_date"] = pd.date_range("2026-01-01", periods=len(scoring_frame), freq="D")

        scorer = AnomalyScorer(config, model, metadata={"run_id": "run-1"})
        scored = scorer.score(scoring_frame)

        self.assertEqual(model.feature_names, ["feature_a", "feature_b", "feature_c"])
        self.assertEqual(len(scored), len(scoring_frame))
        self.assertIn("reason_1", scored.columns)
        self.assertIn("run_id", scored.columns)

    def test_data_loader_accepts_configured_categorical_features(self):
        config = load_config()
        config["features"] = {
            "mode": "infer",
            "exclude_columns": [],
            "categorical": {
                "default_include": False,
                "per_feature": {
                    "risk_band": {
                        "include": True,
                        "transforms": ["ordinal", "is_unseen"],
                        "order": ["low", "medium", "high"],
                    }
                },
            },
        }
        config["preprocessing"]["hard_bounds"]["rules"] = []
        config["preprocessing"]["missing"]["feature_strategies"] = {}

        frame = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "snapshot_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "numeric_a": [1.0, 2.0, 3.0],
                "risk_band": ["low", "medium", "high"],
            }
        )

        loader = DataLoader(config)
        validated = loader.validate_data(frame)

        self.assertIn("risk_band", loader.feature_names)
        self.assertEqual(validated.loc[0, "risk_band"], "low")

    def test_categorical_transforms_generate_expected_model_features(self):
        config = load_config()
        config["features"] = {
            "mode": "infer",
            "exclude_columns": [],
            "categorical": {
                "default_include": False,
                "per_feature": {
                    "risk_band": {
                        "include": True,
                        "transforms": ["ordinal", "is_unseen"],
                        "order": ["low", "medium", "high"],
                    },
                    "channel": {
                        "include": True,
                        "transforms": ["one_hot", "rarity"],
                    },
                },
            },
        }
        config["preprocessing"]["hard_bounds"]["rules"] = []
        config["preprocessing"]["missing"]["feature_strategies"] = {}

        train_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C2", "C2", "C3"],
                "snapshot_date": pd.date_range("2026-01-01", periods=5, freq="D"),
                "numeric_a": [1.0, 2.0, 2.5, 1.5, 3.0],
                "risk_band": ["low", "medium", "medium", "high", "low"],
                "channel": ["web", "atm", "web", "branch", "atm"],
            }
        )
        scoring_df = pd.DataFrame(
            {
                "customer_id": ["C4"],
                "snapshot_date": [pd.Timestamp("2026-01-06")],
                "numeric_a": [2.2],
                "risk_band": ["high"],
                "channel": ["mobile"],
            }
        )

        model = AnomalyModels(config)
        model.fit(train_df)

        self.assertIn("risk_band__ordinal", model.feature_names)
        self.assertIn("risk_band__is_unseen", model.feature_names)
        self.assertIn("channel__rarity", model.feature_names)
        self.assertTrue(any(name.startswith("channel__oh__") for name in model.feature_names))
        md_features = model.feature_selection_summary()["branch_features"]["mahalanobis"]
        self.assertIn("risk_band__ordinal", md_features)
        self.assertIn("channel__rarity", md_features)
        self.assertNotIn("risk_band__is_unseen", md_features)
        self.assertFalse(any(name.startswith("channel__oh__") for name in md_features))

        scorer = AnomalyScorer(config, model, metadata={"run_id": "run-cat"})
        scored = scorer.score(scoring_df)

        self.assertEqual(len(scored), 1)
        self.assertIn("reason_1", scored.columns)
        self.assertEqual(scored.loc[0, "run_id"], "run-cat")

    def test_raw_type_overrides_lock_binary_inference_from_reference_schema(self):
        config = load_config()
        features = get_feature_list(config)
        config.setdefault("features", {})["raw_type_overrides"] = {"overlimit_count_4w": "continuous"}

        frame = pd.DataFrame([{feature: 1.0 for feature in features} for _ in range(5)])
        frame["overlimit_count_4w"] = [0.0, 1.0, 0.0, 1.0, 0.0]

        preprocessor = FeaturePreprocessor(config, features)
        preprocessor.fit(frame)
        summary = preprocessor.summarize()

        self.assertEqual(
            summary["feature_registry"]["overlimit_count_4w"]["raw_type"],
            "continuous",
        )


if __name__ == "__main__":
    unittest.main()
