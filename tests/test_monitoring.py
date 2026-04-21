import unittest

from engine.config_loader import load_config
from engine.monitoring import MonitoringManager


class MonitoringHistoryTests(unittest.TestCase):
    def test_build_history_row_flattens_payload_and_extras(self):
        manager = MonitoringManager(load_config())

        class FakeRun:
            run_id = "score-live-SEG-20260421-aa"
            run_type = "score-live"
            segment = "SEG_A"
            created_at = "2026-04-21T10:00:00Z"

        run_info = {
            "run_id": FakeRun.run_id,
            "run_type": FakeRun.run_type,
            "segment": FakeRun.segment,
            "status": "completed",
            "started_at": FakeRun.created_at,
            "model_version": "model-123",
            "scope_snapshot": "2026-04-30",
            "monitoring_path": "runtime/runs/fake/monitoring/monitoring.json",
        }
        payload = {
            "input": {"rows": 180, "unique_customers": 180, "snapshots": 1, "avg_feature_missing_ratio": 0.03},
            "scores": {
                "band_share": {"NORMAL": 0.6, "SARI": 0.2, "TURUNCU": 0.15, "KIRMIZI": 0.05},
                "anomaly_score": {"mean": 50.1, "median": 49.9, "p95": 92.0, "p99": 98.5},
                "ae_score": {"mean": 50.0, "median": 49.5, "p95": 93.0, "p99": 98.7},
                "if_score": {"mean": 50.2, "median": 49.8, "p95": 92.5, "p99": 98.4},
                "md_score": {"mean": 49.9, "median": 49.7, "p95": 92.1, "p99": 98.6},
            },
            "quality": {
                "materialization": {
                    "native_full": {"status": "pass", "avg_feature_coverage": 0.99, "max_outlier_share": 0.015},
                    "native_scope": {"status": "pass"},
                    "derived_full": {"status": "warn", "avg_feature_coverage": 0.96, "max_outlier_share": 0.07},
                    "derived_scope": {"status": "warn"},
                }
            },
        }
        extras_stability = {"oot": {"metrics": {"ensemble_score": {"ks_stat": 0.08, "mean_ratio": 1.02}}}}
        row = manager.build_history_row(
            run_info=run_info,
            payload=payload,
            extras={
                "stability": extras_stability,
                "supervised": {
                    "precision_at_top_percent": 0.93,
                    "recall_at_top_percent": 0.12,
                    "f1_at_top_percent": 0.21,
                    "lift_at_top_percent": 1.08,
                },
                "weights": {"autoencoder": 0.3, "isolation_forest": 0.3, "mahalanobis": 0.4},
            },
        )

        self.assertEqual(row["run_id"], FakeRun.run_id)
        self.assertEqual(row["run_type"], "score-live")
        self.assertEqual(row["input_rows"], 180)
        self.assertAlmostEqual(row["band_kirmizi"], 0.05)
        self.assertAlmostEqual(row["score_p99"], 98.5)
        self.assertEqual(row["quality_native_full"], "PASS")
        self.assertEqual(row["quality_derived_full"], "WARN")
        self.assertAlmostEqual(row["native_avg_coverage"], 0.99)
        self.assertAlmostEqual(row["derived_max_outlier"], 0.07)
        self.assertAlmostEqual(row["stability_oot_ks"], 0.08)
        self.assertAlmostEqual(row["supervised_precision"], 0.93)
        self.assertAlmostEqual(row["weight_ae"], 0.3)

    def test_build_history_row_tolerates_missing_sections(self):
        manager = MonitoringManager(load_config())
        row = manager.build_history_row(
            run_info={"run_id": "rid", "run_type": "tune-weights", "segment": "SEG_A", "status": "completed"},
            payload=None,
            extras=None,
        )
        self.assertEqual(row["run_id"], "rid")
        self.assertIsNone(row["band_kirmizi"])
        self.assertIsNone(row["quality_native_full"])
        self.assertIsNone(row["stability_oot_ks"])


if __name__ == "__main__":
    unittest.main()
