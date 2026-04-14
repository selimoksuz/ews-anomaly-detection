import unittest

import pandas as pd

from engine.lifecycle import LifecycleManager
from engine.windowing import WindowResolver


class WindowResolverTests(unittest.TestCase):
    def test_relative_windows_resolve_latest_oot_and_calibration_with_shared_history_for_train_test(self):
        config = {
            "development": {
                "windows": {
                    "mode": "relative_periods",
                    "relative": {
                        "calibration_periods": 2,
                        "oot_periods": 2,
                        "test_size": 0.25,
                    },
                }
            }
        }
        snapshots = pd.date_range("2026-01-01", periods=8, freq="7D")

        resolver = WindowResolver(config)
        windows = resolver.resolve(snapshots)

        self.assertEqual(windows["train"].start, pd.Timestamp("2026-01-01"))
        self.assertEqual(windows["train"].end, pd.Timestamp("2026-02-05"))
        self.assertEqual(windows["test"].start, pd.Timestamp("2026-01-01"))
        self.assertEqual(windows["test"].end, pd.Timestamp("2026-02-05"))
        self.assertEqual(windows["calibration"].start, pd.Timestamp("2026-02-12"))
        self.assertEqual(windows["calibration"].end, pd.Timestamp("2026-02-19"))
        self.assertEqual(windows["oot"].start, pd.Timestamp("2026-02-12"))
        self.assertEqual(windows["oot"].end, pd.Timestamp("2026-02-19"))

    def test_relative_history_split_preserves_each_snapshot_in_train_and_test(self):
        manager = LifecycleManager.__new__(LifecycleManager)
        manager.id_column = "customer_id"
        manager.time_column = "snapshot_date"
        config = {
            "development": {
                "windows": {
                    "relative": {
                        "test_size": 0.25,
                        "split_seed": 42,
                    }
                }
            }
        }
        frame = pd.DataFrame(
            {
                "customer_id": [f"C{i:02d}" for i in range(12)],
                "snapshot_date": list(pd.to_datetime(["2026-01-01"] * 6 + ["2026-01-08"] * 6)),
                "feature_x": list(range(12)),
            }
        )

        train_frame, test_frame = manager._split_train_test_frame(frame, config)

        self.assertEqual(len(train_frame), 8)
        self.assertEqual(len(test_frame), 4)
        self.assertEqual(set(train_frame["snapshot_date"].astype(str)), set(test_frame["snapshot_date"].astype(str)))


if __name__ == "__main__":
    unittest.main()
