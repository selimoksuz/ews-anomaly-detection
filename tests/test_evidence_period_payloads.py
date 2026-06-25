import unittest

import pandas as pd

from engine.multivar_anomaly import ID_COLUMN, TIME_COLUMN
from llm.evidence_builder import EvidenceConfig, build_evidence_packages_from_prepared_windows


def raw_row(customer_id: str, cohort_dt: str, risk: float, assets: float, segment: str = "4002") -> dict:
    return {
        ID_COLUMN: customer_id,
        TIME_COLUMN: cohort_dt,
        "musteri_segment": segment,
        "cst_sector": "IMALAT",
        "bank_total_risk": risk,
        "kred_top_am_risk": risk,
        "memzuc_total_risk": risk * 1.3,
        "memzuc_total_limit": risk * 2.5,
        "memzuc_st_mt_cash_risk": risk * 0.4,
        "toplam_varlik_ttr": assets,
        "fs_net_sales_cumulative_l1y": assets * 0.8,
        "fs_trade_receivables_l1y": assets * 0.2,
        "fs_notes_receivable_l1y": assets * 0.1,
        "fs_net_profit_cumulative_l1y": assets * 0.05,
        "equity_l1y": assets * 0.3,
        "irb_rating_pd": risk / max(assets, 1.0),
        "irb_model_pd": risk / max(assets * 1.1, 1.0),
        "rating_group": 5,
        "gunceltkn_dgr": risk * 0.08,
        "gunceltbe_dgr": risk * 0.12,
    }


class EvidencePeriodPayloadTests(unittest.TestCase):
    def test_selected_customer_history_rows_feed_single_scoring_package(self):
        train_df = pd.DataFrame(
            [
                raw_row("C1", "2026-03-31", 100, 1000),
                raw_row("C1", "2026-04-30", 120, 1000),
                raw_row("C2", "2026-03-31", 70, 900),
                raw_row("C2", "2026-04-30", 80, 900),
                raw_row("C3", "2026-03-31", 60, 800),
                raw_row("C3", "2026-04-30", 65, 800),
            ]
        )
        selected_history_df = pd.DataFrame(
            [
                raw_row("C1", "2026-03-31", 100, 1000),
                raw_row("C1", "2026-04-30", 120, 1000),
            ]
        )
        score_df = pd.DataFrame(
            [
                raw_row("C1", "2026-05-31", 160, 1000),
                raw_row("C2", "2026-05-31", 90, 900),
            ]
        )
        numeric_columns = [
            "bank_total_risk",
            "kred_top_am_risk",
            "memzuc_total_risk",
            "memzuc_total_limit",
            "memzuc_st_mt_cash_risk",
            "toplam_varlik_ttr",
            "fs_net_sales_cumulative_l1y",
            "fs_trade_receivables_l1y",
            "fs_notes_receivable_l1y",
            "fs_net_profit_cumulative_l1y",
            "equity_l1y",
            "irb_rating_pd",
            "irb_model_pd",
            "rating_group",
            "gunceltkn_dgr",
            "gunceltbe_dgr",
        ]

        packages = build_evidence_packages_from_prepared_windows(
            train_df=train_df,
            score_df=score_df,
            selected_history_df=selected_history_df,
            prior_df=pd.DataFrame(),
            numeric_source_columns=numeric_columns,
            scoring_month=pd.Timestamp("2026-05-31"),
            selected_customer_ids=["C1"],
            config=EvidenceConfig(scoring_month="2026-05-31", max_customers=1, top_features=3),
        )

        self.assertEqual(len(packages), 1)
        self.assertEqual(packages[0]["cohort_dt"], "2026-05-31")
        self.assertEqual(packages[0]["mono_id"], "C1")
        self.assertEqual(packages[0]["data_quality"]["customer_history_periods"], 2)
        self.assertTrue(packages[0]["decision_contract"]["scoring_month_only"])
        self.assertTrue(all(item["features"] for item in packages))
        self.assertTrue(
            any((feature.get("history") or {}).get("period_count") == 2 for feature in packages[0]["features"])
        )
        for feature in packages[0]["features"]:
            series = feature.get("snapshot_series") or {}
            self.assertEqual([item["cohort_dt"] for item in series.get("customer", [])], ["2026-03-31", "2026-04-30", "2026-05-31"])
            self.assertEqual([item["cohort_dt"] for item in series.get("peer", [])], ["2026-03-31", "2026-04-30", "2026-05-31"])
            self.assertTrue(series["customer"][-1]["is_current_snapshot"])
            self.assertTrue(series["peer"][-1]["peer_available"])


if __name__ == "__main__":
    unittest.main()
