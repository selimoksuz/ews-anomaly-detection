from __future__ import annotations

import unittest

import pandas as pd

from engine.business_features import FAZ1_BASE_FEATURES
from engine.ticari_orta_faz1_demo import TicariOrtaFaz1DemoBuilder


class TicariOrtaFaz1DemoBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = TicariOrtaFaz1DemoBuilder()

    def test_native_frame_contains_today_and_missing_values(self):
        frame = self.builder.build_native_frame(
            num_customers=24,
            num_snapshots=12,
            end_date="2026-04-15",
            seed=123,
        )
        self.assertEqual(pd.Timestamp("2026-04-15"), pd.to_datetime(frame["snapshot_date"]).max())
        self.assertTrue(frame[["pos_monthly_volume", "fs_ebitda_cumulative", "ifrs9_behavioral_pd"]].isna().any().any())

    def test_derived_frame_contains_base_and_history_columns(self):
        native = self.builder.build_native_frame(
            num_customers=20,
            num_snapshots=52,
            end_date="2026-04-15",
            seed=456,
        )
        derived = self.builder.build_derived_frame(native)
        self.assertLess(len(derived), len(native))
        self.assertGreater(pd.to_datetime(derived["snapshot_date"]).min(), pd.to_datetime(native["snapshot_date"]).min())
        for feature in FAZ1_BASE_FEATURES:
            self.assertIn(feature, derived.columns)
            self.assertIn(f"{feature}__delta_1", derived.columns)
            self.assertIn(f"{feature}__self_zscore_6", derived.columns)
        self.assertIn("pos_volume_change__trend_slope_6", derived.columns)
        self.assertIn("net_sales_change__population_percentile", derived.columns)
        self.assertIn("net_sales_change__vs_population_median_delta", derived.columns)

    def test_outcomes_are_binary(self):
        native = self.builder.build_native_frame(
            num_customers=20,
            num_snapshots=52,
            end_date="2026-04-15",
            seed=789,
        )
        derived = self.builder.build_derived_frame(native)
        outcomes = self.builder.build_outcomes_frame(derived, seed=790)
        self.assertTrue(set(outcomes["label_30dpd_8w"].unique()).issubset({0, 1}))
        self.assertTrue(set(outcomes["label_default_12m"].unique()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
