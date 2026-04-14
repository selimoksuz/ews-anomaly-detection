import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from engine.source_loader import SourceLoader


class SourceLoaderOracleQueryTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "pipeline": {
                "id_column": "customer_id",
                "time_column": "snapshot_date",
            },
            "sources": {
                "input_features": {
                    "backend": "oracle",
                    "oracle": {"table": "input_features"},
                }
            },
            "oracle": {
                "schema": "ZT_VAR2",
                "tables": {
                    "input_features": "EWS_INPUT_FEATURES",
                },
            },
        }
        self.loader = SourceLoader(self.config)

    @patch("engine.source_loader.OracleConnector")
    def test_latest_snapshot_query_is_segment_aware(self, connector_cls):
        connector = MagicMock()
        connector.schema = "ZT_VAR2"
        connector._qualified_table_name.return_value = "ZT_VAR2.EWS_INPUT_FEATURES"
        connector._read_query.return_value = pd.DataFrame(
            {"snapshot_date": [pd.Timestamp("2026-04-14")]}
        )
        connector_cls.return_value.__enter__.return_value = connector

        self.loader.load_frame(
            "input_features",
            latest_snapshot=True,
            segment_column="segment",
            segment_value="VIP",
        )

        sql, params = connector._read_query.call_args.args
        self.assertIn(
            "TRUNC(SNAPSHOT_DATE) = (SELECT MAX(TRUNC(SNAPSHOT_DATE)) FROM ZT_VAR2.EWS_INPUT_FEATURES WHERE SEGMENT = :segment_value)",
            sql,
        )
        self.assertIn("SEGMENT = :segment_value", sql)
        self.assertEqual(params["segment_value"], "VIP")

    @patch("engine.source_loader.OracleConnector")
    def test_date_range_query_uses_day_level_bounds(self, connector_cls):
        connector = MagicMock()
        connector.schema = "ZT_VAR2"
        connector._qualified_table_name.return_value = "ZT_VAR2.EWS_INPUT_FEATURES"
        connector._read_query.return_value = pd.DataFrame(
            {"snapshot_date": [pd.Timestamp("2026-04-14")]}
        )
        connector_cls.return_value.__enter__.return_value = connector

        self.loader.load_frame(
            "input_features",
            start_date="2026-04-10",
            end_date="2026-04-14",
        )

        sql, params = connector._read_query.call_args.args
        self.assertIn("TRUNC(SNAPSHOT_DATE) >= TRUNC(:start_date)", sql)
        self.assertIn("TRUNC(SNAPSHOT_DATE) <= TRUNC(:end_date)", sql)
        self.assertEqual(pd.Timestamp(params["start_date"]).date().isoformat(), "2026-04-10")
        self.assertEqual(pd.Timestamp(params["end_date"]).date().isoformat(), "2026-04-14")


if __name__ == "__main__":
    unittest.main()
