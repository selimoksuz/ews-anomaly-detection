import unittest

import pandas as pd

from engine.windowing import WindowResolver


class WindowResolverTests(unittest.TestCase):
    def test_relative_windows_resolve_in_reverse_chronological_blocks(self):
        config = {
            "development": {
                "windows": {
                    "mode": "relative_periods",
                    "relative": {
                        "train_periods": 4,
                        "dev_periods": 2,
                        "calibration_periods": 1,
                        "oot_periods": 1,
                    },
                }
            }
        }
        snapshots = pd.date_range("2026-01-01", periods=8, freq="7D")

        resolver = WindowResolver(config)
        windows = resolver.resolve(snapshots)

        self.assertEqual(windows["train"].start, pd.Timestamp("2026-01-01"))
        self.assertEqual(windows["train"].end, pd.Timestamp("2026-01-22"))
        self.assertEqual(windows["dev"].start, pd.Timestamp("2026-01-29"))
        self.assertEqual(windows["dev"].end, pd.Timestamp("2026-02-05"))
        self.assertEqual(windows["calibration"].start, pd.Timestamp("2026-02-12"))
        self.assertEqual(windows["oot"].start, pd.Timestamp("2026-02-19"))


if __name__ == "__main__":
    unittest.main()
