import unittest

import pandas as pd

from engine.config_loader import load_config
from engine.quality import QualityGateError, QualityManager


class QualityManagerTests(unittest.TestCase):
    def test_native_quality_can_fail_on_duplicates_and_staleness(self):
        config = load_config()
        config["quality"]["native"]["freshness"]["max_stale_share_fail"] = 0.20
        manager = QualityManager(config)

        frame = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C2", "C3"],
                "snapshot_date": pd.to_datetime(["2026-04-30", "2026-04-30", "2026-04-30", "2026-04-30"]),
                "segment": ["SEG_A"] * 4,
                "fs_last_update_date": pd.to_datetime(["2025-01-31", "2025-01-31", "2025-02-28", "2026-03-31"]),
                "metric_a": [1.0, 2.0, None, 4.0],
                "metric_b": [1.0, 1.0, 1.0, 999.0],
            }
        )

        report = manager.evaluate(
            frame,
            dataset_name="native_scope",
            stage="development",
            rule_key="native",
            feature_columns=["metric_a", "metric_b"],
        )

        self.assertEqual(report["status"], "fail")
        self.assertGreater(report["duplicate_key_count"], 0)
        self.assertIn("fs_last_update_date", report["freshness"])
        with self.assertRaises(QualityGateError) as ctx:
            manager.enforce(report, stage="development")
        message = str(ctx.exception)
        self.assertIn("duplicate_keys", message)
        self.assertIn("observed=", message)
        lines = QualityManager.format_report_lines(report)
        joined = "\n".join(lines)
        self.assertIn("failing checks", joined)
        self.assertIn("rows=", joined)

    def test_future_fs_dates_do_not_fail_by_default(self):
        config = load_config()
        config["quality"]["native"]["min_rows_warn"] = 1
        config["quality"]["native"]["min_unique_customers_warn"] = 1
        manager = QualityManager(config)
        frame = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "snapshot_date": pd.to_datetime(["2025-09-30"] * 3),
                "segment": ["SEG_A"] * 3,
                "fs_last_update_date": pd.to_datetime(["2025-10-15", "2025-11-01", "2025-09-15"]),
                "metric_a": [1.0, 1.1, 0.9],
                "metric_b": [10.0, 11.0, 9.5],
            }
        )

        report = manager.evaluate(
            frame,
            dataset_name="native_full",
            stage="development",
            rule_key="native",
            feature_columns=["metric_a", "metric_b"],
        )

        freshness = report["freshness"].get("fs_last_update_date")
        self.assertIsNotNone(freshness)
        self.assertGreater(freshness["future_date_share"], 0)
        self.assertEqual(freshness["status"], "pass")
        self.assertEqual(report["status"], "pass")

    def test_derived_quality_passes_for_clean_frame(self):
        config = load_config()
        config["quality"]["derived"]["min_rows_warn"] = 1
        config["quality"]["derived"]["min_unique_customers_warn"] = 1
        manager = QualityManager(config)
        frame = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3", "C4"],
                "snapshot_date": pd.to_datetime(["2026-04-30"] * 4),
                "segment": ["SEG_A"] * 4,
                "feature_a": [1.0, 1.2, 0.9, 1.1],
                "feature_b": [10.0, 10.5, 9.7, 10.1],
            }
        )

        report = manager.evaluate(
            frame,
            dataset_name="derived_scope",
            stage="live_scoring",
            rule_key="derived",
            feature_columns=["feature_a", "feature_b"],
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["duplicate_key_count"], 0)
        self.assertAlmostEqual(report["avg_feature_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
