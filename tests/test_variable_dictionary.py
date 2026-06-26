import unittest

import pandas as pd

from engine.multivar_anomaly import build_feature_frame, evaluate_feature_formula
from engine.variable_dictionary import (
    final_llm_include_features,
    generated_feature_names,
    raw_variable_groups,
    raw_variable_metadata,
)
from llm.evidence_builder import is_allowed_llm_feature


class VariableDictionaryTests(unittest.TestCase):
    def test_dictionary_groups_and_final_feature_policy_are_loaded(self):
        groups = raw_variable_groups()
        raw_metadata = raw_variable_metadata()

        self.assertIn("current_snapshot", groups)
        self.assertIn("l1_term_financial", groups)
        self.assertIn("q_financial", groups)
        self.assertIn("kkb", groups)
        self.assertIn("pd_rating", groups)
        self.assertIn("internal_other", groups)
        self.assertEqual(raw_metadata["rating_group"]["role"], "direct_rating_signal_allowed")
        self.assertIn("memzuc_limit_utilization", generated_feature_names())
        self.assertIn("rating_group", final_llm_include_features())

    def test_generated_features_are_built_from_dictionary_formulas(self):
        frame = pd.DataFrame(
            [
                {
                    "cohort_dt": "2026-05-31",
                    "mono_id": "C1",
                    "bank_total_risk": 200.0,
                    "memzuc_total_risk": 400.0,
                    "memzuc_total_limit": 800.0,
                    "memzuc_st_mt_cash_risk": 100.0,
                    "toplam_varlik_ttr": 1000.0,
                    "fs_net_sales_cumulative_l1y": 500.0,
                    "fs_trade_receivables_l1y": 50.0,
                    "fs_notes_receivable_l1y": 25.0,
                    "equity_l1y": 300.0,
                    "gunceltkn_dgr": 20.0,
                    "gunceltbe_dgr": 10.0,
                }
            ]
        )

        features = build_feature_frame(frame, [column for column in frame.columns if column not in {"cohort_dt", "mono_id"}])

        self.assertAlmostEqual(features.loc[0, "memzuc_limit_utilization"], 0.5)
        self.assertAlmostEqual(features.loc[0, "bank_to_memzuc_risk_ratio"], 0.5)
        self.assertAlmostEqual(features.loc[0, "internal_tkn_tbe_ratio"], 2.0)
        self.assertNotIn("q_debt_to_sales", features.columns)
        self.assertNotIn("pd_ratio", features.columns)

    def test_formula_engine_supports_arithmetic_for_yaml_trials(self):
        frame = pd.DataFrame({"a": [10.0], "b": [2.0], "c": [3.0]})

        result = evaluate_feature_formula("(a / b) * c + 1", frame)

        self.assertAlmostEqual(result.iloc[0], 16.0)

    def test_llm_feature_gate_uses_dictionary_include_exclude(self):
        self.assertTrue(is_allowed_llm_feature("memzuc_limit_utilization"))
        self.assertTrue(is_allowed_llm_feature("rating_group"))
        self.assertFalse(is_allowed_llm_feature("irb_rating_pd"))
        self.assertFalse(is_allowed_llm_feature("q_debt_to_sales"))


if __name__ == "__main__":
    unittest.main()
