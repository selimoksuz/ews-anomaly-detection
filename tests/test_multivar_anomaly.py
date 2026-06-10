import tempfile
import unittest
from pathlib import Path

import pandas as pd

from engine.multivar_anomaly import (
    EXCLUDED_FEATURE_COLUMNS,
    assign_operational_bands,
    operational_band_thresholds,
    run_multivar_anomaly,
)


class MultivarAnomalyTests(unittest.TestCase):
    def test_local_multivar_run_writes_scores_and_excludes_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "anomaly_multivar.csv"
            rows = []
            for month in ("31.01.2025", "28.02.2025", "31.03.2025"):
                for idx in range(12):
                    multiplier = 8 if month == "31.03.2025" and idx == 0 else 1
                    rows.append(
                        {
                            "COHORT_DT": month,
                            "MONO_ID": f"C_{idx:03d}",
                            "MUSTERI_SEGMENT": 4001,
                            "BILANCO_FLG": 1,
                            "CST_SECTOR": "SECTOR_A",
                            "CST_NACE_CODE": "NACE_A",
                            "CST_NACE_CODE_ID": 1000,
                            "BANK_TOTAL_RISK": 1000 * multiplier + idx,
                            "FINANCIAL_TERM_L1Y": "31.12.2024",
                            "FS_NET_SALES_CUMULATIVE_L1Y": 5000 + idx,
                            "FS_TRADE_RECEIVABLES_L1Y": 400 + idx,
                            "FS_NOTES_RECEIVABLE_L1Y": 20,
                            "SUPHELI_TICARI_ALACAKLAR_L1Y": 0,
                            "EQUITY_L1Y": 2000 - (100 * multiplier if idx == 0 else 0),
                            "FS_NET_PROFIT_CUMULATIVE_L1Y": 300,
                            "FINANCIAL_TERM_Q": "30.09.2024",
                            "ANNUALIZATION_Q": 1.333,
                            "FS_NET_SALES_CUMULATIVE_Q": 4500 + idx,
                            "FS_EBITDA_CUMULATIVE_Q": 600,
                            "FS_NET_PROFIT_CUMULATIVE_Q": 250,
                            "FS_TRADE_RECEIVABLES_Q": 380,
                            "FS_NOTES_RECEIVABLE_Q": 15,
                            "SUPHELI_ALACAKLAR_Q": 0,
                            "FS_EQUITY_Q": 1900,
                            "MEMZUC_TOTAL_RISK": 900 * multiplier + idx,
                            "MEMZUC_TOTAL_LIMIT": 3000,
                            "MEMZUC_ST_MT_CASH_RISK": 500 * multiplier,
                            "IRB_RATING_PD": 0.01 * multiplier,
                            "IRB_MODEL_PD": 0.02 * multiplier,
                            "RATING_GROUP": 3,
                            "TOPLAM_VARLIK_TTR": 10000,
                            "REF_DONEM_ID": 202503,
                            "GUNCELTKN_DGR": 700,
                            "GUNCELTBE_DGR": 15,
                            "KKBGUNCELSORGU_NO": 123,
                            "YUKLEME_ZMN": "01.04.2025 09:00:00",
                        }
                    )

            pd.DataFrame(rows).to_csv(input_path, index=False)
            summary = run_multivar_anomaly(
                input_path=input_path,
                output_dir=tmp_path / "out",
                max_train_rows=100,
                chunk_size=10,
                n_estimators=20,
            )

            self.assertEqual(summary["scoring_month"], "2025-03-31")
            self.assertEqual(summary["scored_rows"], 12)
            self.assertTrue(Path(summary["scores_path"]).exists())
            self.assertTrue(Path(summary["top_path"]).exists())
            self.assertFalse(EXCLUDED_FEATURE_COLUMNS.intersection(summary["selected_features"]))

            scores = pd.read_csv(summary["scores_path"])
            self.assertIn("reason_1", scores.columns)
            self.assertIn("alert_type", scores.columns)
            self.assertIn("review_queue", scores.columns)
            self.assertGreaterEqual(scores.loc[0, "anomaly_score"], scores.loc[1, "anomaly_score"])

    def test_operational_bands_use_top_tail_thresholds(self):
        scores = list(range(100))
        thresholds = operational_band_thresholds(scores)
        bands = assign_operational_bands(scores, thresholds)

        self.assertEqual(bands.count("KIRMIZI"), 1)
        self.assertLessEqual(bands.count("TURUNCU"), 2)
        self.assertGreaterEqual(thresholds["kirmizi"], 97.5)
        self.assertGreaterEqual(thresholds["turuncu"], 95.0)
        self.assertGreaterEqual(thresholds["sari"], 90.0)


if __name__ == "__main__":
    unittest.main()
