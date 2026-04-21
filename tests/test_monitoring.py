import json
import unittest

import numpy as np
import pandas as pd

from engine.config_loader import load_config
from engine.monitoring import MonitoringManager


class MonitoringHistoryTests(unittest.TestCase):
    def test_summarize_scores_includes_shape_and_buckets(self):
        manager = MonitoringManager(load_config())
        rng = np.random.default_rng(42)
        values = np.clip(rng.normal(loc=50, scale=20, size=2000), 0, 100)
        frame = pd.DataFrame({
            "alert_band": ["NORMAL"] * 1800 + ["KIRMIZI"] * 200,
            "anomaly_score": values,
            "ae_score": values,
            "if_score": values,
            "md_score": values,
        })
        summary = manager.summarize_scores(frame)
        anomaly = summary["anomaly_score"]
        self.assertIn("skew", anomaly)
        self.assertIn("kurtosis", anomaly)
        buckets = summary["score_buckets"]
        self.assertAlmostEqual(sum(buckets.values()), 1.0, places=4)
        self.assertEqual(len(buckets), 10)

    def test_compute_psi_positive_for_shifted_distribution(self):
        stable = {"000_010": 0.1, "010_020": 0.3, "020_030": 0.3, "030_040": 0.3}
        shifted = {"000_010": 0.4, "010_020": 0.3, "020_030": 0.2, "030_040": 0.1}
        psi = MonitoringManager.compute_psi(shifted, stable)
        self.assertGreater(psi, 0.1)
        unchanged = MonitoringManager.compute_psi(stable, stable)
        self.assertLess(unchanged, 1e-4)

    def test_build_history_row_new_p0_p1_fields(self):
        manager = MonitoringManager(load_config())

        class FakeRun:
            run_id = "score-live-SEG-20260421-xx"
            run_type = "score-live"
            segment = "SEG_A"
            created_at = "2026-04-21T10:00:00Z"

        run_info = {
            "run_id": FakeRun.run_id,
            "run_type": FakeRun.run_type,
            "segment": FakeRun.segment,
            "status": "completed",
            "started_at": "2026-04-21T10:00:00Z",
            "finished_at": "2026-04-21T10:02:30Z",
            "model_version": "model-abc",
            "monitoring_path": "/tmp/monitoring.json",
        }
        payload = {
            "input": {"rows": 180, "unique_customers": 180, "snapshots": 1, "avg_feature_missing_ratio": 0.02},
            "scores": {
                "band_share": {"NORMAL": 0.6, "SARI": 0.2, "TURUNCU": 0.15, "KIRMIZI": 0.05},
                "anomaly_score": {"mean": 50.0, "median": 49.5, "p95": 92.0, "p99": 98.0, "skew": 0.5, "kurtosis": 2.1},
                "score_buckets": {"000_010": 0.1, "010_020": 0.2, "020_030": 0.3, "030_040": 0.4},
            },
            "quality": {
                "materialization": {
                    "native_full": {
                        "status": "pass",
                        "avg_feature_coverage": 0.99,
                        "max_outlier_share": 0.01,
                        "freshness": {
                            "fs_last_update_date": {"max_age_days_observed": 180}
                        },
                    },
                    "derived_full": {"status": "pass", "avg_feature_coverage": 0.96, "max_outlier_share": 0.04},
                }
            },
        }
        stability = {
            "test": {"metrics": {"ensemble_score": {"ks_stat": 0.05, "mean_ratio": 1.02}}},
            "calibration": {"metrics": {"ensemble_score": {"ks_stat": 0.06, "mean_ratio": 1.03}}},
            "oot": {"metrics": {"ensemble_score": {"ks_stat": 0.09, "mean_ratio": 1.10}}},
        }
        extras = {
            "stability": stability,
            "calibration": {"rows": 720, "monotonic": True},
            "supervised": {"precision_at_top_percent": 0.9, "recall_at_top_percent": 0.1, "f1_at_top_percent": 0.18, "lift_at_top_percent": 1.08},
            "weights": {"autoencoder": 0.3, "isolation_forest": 0.3, "mahalanobis": 0.4},
            "score_psi_vs_prev": 0.12,
            "band_persistence_kirmizi": 0.75,
            "dominant_reason": {"feature": "bank_debt_to_turnover", "share": 0.42},
            "result_row_count": 180,
        }

        row = manager.build_history_row(run_info=run_info, payload=payload, extras=extras)

        self.assertAlmostEqual(row["duration_seconds"], 150.0, places=1)
        self.assertEqual(row["freshness_max_age_days"], 180)
        self.assertAlmostEqual(row["score_psi_vs_prev"], 0.12)
        self.assertAlmostEqual(row["band_persistence_kirmizi"], 0.75)
        self.assertEqual(row["dominant_reason_feature"], "bank_debt_to_turnover")
        self.assertAlmostEqual(row["dominant_reason_share"], 0.42)
        self.assertEqual(row["result_row_count"], 180)
        self.assertAlmostEqual(row["stability_test_ks"], 0.05)
        self.assertAlmostEqual(row["stability_cal_mean_ratio"], 1.03)
        self.assertAlmostEqual(row["stability_oot_ks"], 0.09)
        self.assertEqual(row["calibration_rows"], 720)
        self.assertEqual(row["calibration_monotonic"], 1)
        self.assertAlmostEqual(row["score_skew"], 0.5)
        buckets_dict = json.loads(row["score_buckets"])
        self.assertIn("000_010", buckets_dict)

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
