import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from engine.output_writer import OutputWriter


class OutputWriterTests(unittest.TestCase):
    @patch("engine.output_writer.OracleConnector")
    def test_rewrite_same_snapshot_deletes_before_insert(self, connector_cls):
        config = {
            "pipeline": {
                "id_column": "customer_id",
                "time_column": "snapshot_date",
            },
            "sources": {
                "outputs": {
                    "backend": "oracle",
                    "oracle": {
                        "results_table_key": "results",
                        "details_table_key": "details",
                        "full_effects_table_key": "full_effects",
                    },
                }
            },
            "scoring": {
                "top_n_reasons": 3,
                "persist_full_feature_effects": False,
            },
        }
        writer = OutputWriter(config)

        connector = MagicMock()
        connector.delete_scored_scope.return_value = {
            "results": 2,
            "details": 4,
            "full_effects": 0,
        }
        connector.write_results.return_value = 2
        connector.write_details.return_value = 1
        connector.write_full_effects.return_value = 0
        connector_cls.return_value.__enter__.return_value = connector

        results = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "snapshot_date": [pd.Timestamp("2026-04-14"), pd.Timestamp("2026-04-14")],
                "anomaly_score": [91.2, 75.4],
                "alert_band": ["KIRMIZI", "TURUNCU"],
                "detay": [
                    {"f1": {"label": "F1", "beklenen": 1, "gerceklesen": 2, "degisim_pct": 100, "katki_pct": 20}},
                    {"f1": {"label": "F1", "beklenen": 1, "gerceklesen": 1.5, "degisim_pct": 50, "katki_pct": 10}},
                ],
            }
        )

        summary = writer.write(results, pd.Timestamp("2026-04-14"), run_id="run-1", segment="ALL")

        connector.delete_scored_scope.assert_called_once_with(
            snapshot_date=pd.Timestamp("2026-04-14"),
            start_date=None,
            end_date=None,
            segment="ALL",
        )
        method_names = [call[0] for call in connector.method_calls]
        self.assertLess(method_names.index("delete_scored_scope"), method_names.index("write_results"))
        self.assertEqual(summary["deleted_results"], 2)
        self.assertEqual(summary["deleted_details"], 4)
        self.assertEqual(summary["inserted_results"], 2)


if __name__ == "__main__":
    unittest.main()
