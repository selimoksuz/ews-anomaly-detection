import warnings
import unittest

import pandas as pd

from llm import evidence_builder


class EvidenceBuilderWarningTests(unittest.TestCase):
    def test_customer_snapshot_series_does_not_concat_empty_history_frame(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", FutureWarning)
            series = evidence_builder.customer_snapshot_series(
                scoring_month=pd.Timestamp("2026-05-31"),
                current=123.0,
                history_dates=pd.Series([], dtype="datetime64[ns]"),
                history_series=pd.Series([], dtype="float64"),
                series_periods=6,
            )

        self.assertEqual(series, [{"cohort_dt": "2026-05-31", "value": 123.0, "is_current_snapshot": True}])
        self.assertFalse([item for item in captured if issubclass(item.category, FutureWarning)])


if __name__ == "__main__":
    unittest.main()
