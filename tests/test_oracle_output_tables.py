import unittest

import pandas as pd

from llm.llm_anomaly import select_ml_balanced_customer_ids
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
                "ml_anomaly_score": 96.4,
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
        self.assertEqual(results.loc[0, "ml_autoencoder_score"], 95.3)
        self.assertNotIn("evidence_features", results.loc[0, "raw_response"])
        self.assertEqual(reasons.loc[0, "feature_name"], "bank_risk_to_assets")
        self.assertEqual(features.loc[0, "feature_name"], "bank_risk_to_assets")
        self.assertEqual(features.loc[0, "peer_z"], 4.2)
        self.assertAlmostEqual(features.loc[0, "history_z"], 4.4)
        self.assertEqual(features.loc[0, "trend_break_flag"], 1)

    def test_ml_balanced_customer_selection(self):
        frame = pd.DataFrame(
            [
                {"mono_id": f"A{i}", "cohort_dt": "2026-05-31", "ensemble_score": 99 - i, "alert_band": "KIRMIZI"}
                for i in range(7)
            ]
            + [
                {"mono_id": f"N{i}", "cohort_dt": "2026-05-31", "ensemble_score": i, "alert_band": "NORMAL"}
                for i in range(8)
            ]
        )

        customer_ids, selected = select_ml_balanced_customer_ids(
            frame,
            total_customers=10,
            anomaly_customers=5,
        )

        self.assertEqual(customer_ids[:5], ["A0", "A1", "A2", "A3", "A4"])
        self.assertEqual(customer_ids[5:], ["N0", "N1", "N2", "N3", "N4"])
        self.assertEqual(selected["selection_bucket"].tolist().count("ML_HIGH_ANOMALY"), 5)
        self.assertEqual(selected["selection_bucket"].tolist().count("ML_NORMAL_REFERENCE"), 5)


if __name__ == "__main__":
    unittest.main()
