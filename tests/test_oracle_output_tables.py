import unittest

import pandas as pd

from llm.llm_anomaly import attach_ml_companion_scores, select_ml_balanced_customer_ids
from llm.oracle_output import prepare_llm_feature_frame, prepare_llm_reason_frame, prepare_llm_result_frame


class OracleOutputTableTests(unittest.TestCase):
    def test_prepare_llm_result_reason_and_feature_frames(self):
        decisions = [
            {
                "mono_id": "m1",
                "cohort_dt": "2025-12-31",
                "is_anomaly": True,
                "anomaly_type": "PEER_UYUMSUZLUGU",
                "risk_level": "YUKSEK",
                "anomaly_score": 0.82,
                "ml_ensemble_score": 96.4,
                "ml_is_anomaly": True,
                "ml_alert_band": "KIRMIZI",
                "ml_if_score": 94.2,
                "ml_residual_score": 99.1,
                "ml_autoencoder_score": 95.3,
                "reason_summary": "Musteri history bozulmasi peer tarafinda da destekleniyor.",
                "reason_1": "Musteri gecmisine gore risk yonunde sapma",
                "reason_1_weight": 0.55,
                "reason_2": "Trend kirilmasi",
                "reason_2_weight": 0.3,
                "reason_3": "Peer destekleyici sapma",
                "reason_3_weight": 0.15,
                "recommended_action": "Manuel incele",
                "evidence_data_quality": {
                    "coverage_ratio": 0.95,
                    "missing_feature_count": 1,
                    "customer_history_periods": 6,
                    "caveat": "Bir finansal term stale olabilir.",
                },
                "main_reasons": [
                    {
                        "feature": "bank_risk_to_assets",
                        "evidence": "current=0.72 peer=0.28 peer_z=4.2",
                        "interpretation": "Peer grubuna gore riskli ayrisma var.",
                    }
                ],
                "evidence_features": [
                    {
                        "name": "bank_risk_to_assets",
                        "dictionary": {
                            "label": "Banka risk / varlik",
                            "category": "bank_risk",
                            "formula": "bank_total_risk / toplam_varlik_ttr",
                            "source_columns": ["bank_total_risk", "toplam_varlik_ttr"],
                            "risk_direction": "HIGHER_IS_RISKY",
                        },
                        "current_value": 0.72,
                        "previous_value": 0.31,
                        "change_pct": 132.2,
                        "history": {
                            "period_count": 6,
                            "median": 0.28,
                            "p25": 0.2,
                            "p75": 0.34,
                            "robust_scale": 0.1,
                            "rolling_3m_median": 0.33,
                            "rolling_6m_median": 0.28,
                            "rolling_12m_median": 0.27,
                        },
                        "trend": {
                            "slope_6m": 0.08,
                            "slope_12m": 0.04,
                            "trend_break_flag": True,
                            "trend_note": "Cari deger tarihsel robust banda gore kirilim gosteriyor.",
                        },
                        "seasonality": {
                            "month_of_year": 12,
                            "same_month_last_year_value": 0.22,
                            "yoy_change_pct": 227.2,
                            "same_month_customer_median": 0.24,
                            "same_month_customer_z": 4.8,
                            "seasonal_peer_median": 0.25,
                            "seasonal_peer_z": 5.1,
                        },
                        "peer": {
                            "peer_median": 0.28,
                            "peer_z": 4.2,
                            "peer_support": 82,
                            "peer_definition_level": "AY_SEGMENT_SIZE",
                            "peer_quality": "KABUL_EDILEBILIR",
                        },
                        "data_quality": {"missing_flag": False, "history_periods": 6},
                        "snapshot_series": {"window_periods": 6, "customer": [], "peer": []},
                    }
                ],
            }
        ]

        results = prepare_llm_result_frame(
            decisions,
            run_id="llm_test",
            llm_model="gpt-oss-20b",
            evidence_source="oracle_input",
        )
        reasons = prepare_llm_reason_frame(decisions, run_id="llm_test")
        features = prepare_llm_feature_frame(decisions, run_id="llm_test")

        self.assertEqual(results.loc[0, "ml_ensemble_score"], 96.4)
        self.assertNotIn("ml_anomaly_score", results.columns)
        self.assertEqual(results.loc[0, "ml_autoencoder_score"], 95.3)
        self.assertEqual(results.loc[0, "llm_confidence"], 0.82)
        self.assertIn("same_month_z=4.8", results.loc[0, "seasonality_assessment"])
        self.assertIn("trend_break=1", results.loc[0, "trend_assessment"])
        self.assertIn("peer_z=4.2", results.loc[0, "peer_assessment"])
        self.assertIn("coverage_ratio=0.95", results.loc[0, "caveat"])
        self.assertNotIn("evidence_features", results.loc[0, "raw_response"])
        self.assertNotIn("evidence_data_quality", results.loc[0, "raw_response"])
        self.assertEqual(reasons.loc[0, "feature_name"], "bank_risk_to_assets")
        self.assertEqual(features.loc[0, "feature_name"], "bank_risk_to_assets")
        self.assertEqual(features.loc[0, "peer_z"], 4.2)
        self.assertAlmostEqual(features.loc[0, "history_z"], 4.4)
        self.assertEqual(features.loc[0, "trend_break_flag"], 1)

    def test_ml_balanced_customer_selection(self):
        frame = pd.DataFrame(
            [
                {"mono_id": f"A{i}", "cohort_dt": "2026-05-31", "ensemble_score": 99 - i, "alert_band": "KIRMIZI"}
                for i in range(8)
            ]
            + [
                {"mono_id": f"N{i}", "cohort_dt": "2026-05-31", "ensemble_score": i, "alert_band": "NORMAL"}
                for i in range(8)
            ]
        )

        customer_ids, selected = select_ml_balanced_customer_ids(
            frame,
            total_customers=12,
        )

        self.assertEqual(customer_ids[:6], ["A0", "A1", "A2", "A3", "A4", "A5"])
        self.assertEqual(customer_ids[6:], ["N0", "N1", "N2", "N3", "N4", "N5"])
        self.assertEqual(selected["selection_bucket"].tolist().count("ML_HIGH_ANOMALY"), 6)
        self.assertEqual(selected["selection_bucket"].tolist().count("ML_NORMAL_REFERENCE"), 6)
        self.assertEqual(selected["selection_score_column"].unique().tolist(), ["ensemble_score"])

    def test_ml_balanced_customer_selection_uses_best_available_score_column(self):
        frame = pd.DataFrame(
            [
                {"mono_id": f"A{i}", "cohort_dt": "2026-05-31", "autoencoder_score": 99 - i, "alert_band": "TURUNCU"}
                for i in range(3)
            ]
            + [
                {"mono_id": f"N{i}", "cohort_dt": "2026-05-31", "autoencoder_score": i, "alert_band": "NORMAL"}
                for i in range(3)
            ]
        )

        customer_ids, selected = select_ml_balanced_customer_ids(frame, total_customers=4)

        self.assertEqual(customer_ids, ["A0", "A1", "N0", "N1"])
        self.assertEqual(selected["selection_score_column"].unique().tolist(), ["autoencoder_score"])
        self.assertEqual(selected["selection_model"].unique().tolist(), ["autoencoder"])

    def test_ml_companion_scores_do_not_write_duplicate_anomaly_alias(self):
        decisions = [{"mono_id": "C1", "cohort_dt": "2026-05-31"}]
        evidence = [{"mono_id": "C1", "cohort_dt": "2026-05-31"}]
        score_frame = pd.DataFrame(
            [
                {
                    "mono_id": "C1",
                    "cohort_dt": "2026-05-31",
                    "ensemble_score": 91.2,
                    "if_score": 88.0,
                    "residual_score": 90.0,
                    "autoencoder_score": 95.0,
                    "alert_band": "SARI",
                }
            ]
        )

        enriched = attach_ml_companion_scores(
            decisions,
            evidence,
            table_key="multivar_input",
            scoring_month="2026-05-31",
            max_train_rows=10,
            output_path="runtime/llm/test.jsonl",
            score_frame=score_frame,
            score_summary={"scores_path": "memory", "scored_rows": 1},
        )

        self.assertEqual(enriched[0]["ml_ensemble_score"], 91.2)
        self.assertNotIn("ml_anomaly_score", enriched[0])


if __name__ == "__main__":
    unittest.main()
