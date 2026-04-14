import unittest

import pandas as pd

from engine.config_loader import get_feature_list, load_config
from engine.sampling import TrainSampler
from tests.helpers import make_feature_frame


class SamplingTests(unittest.TestCase):
    def _build_frame(self, rows: int = 200) -> tuple[dict, pd.DataFrame, list[str]]:
        config = load_config()
        config["development"]["sampling"] = {
            "enabled": True,
            "activate_if_rows_gt": 0,
            "max_rows": 60,
            "tail_z_threshold": 3.5,
            "random_seed": 42,
            "validation": {
                "max_snapshot_share_delta": 0.15,
                "max_tail_share_delta": 0.15,
                "max_missing_share_delta": 0.15,
                "max_feature_missing_delta": 0.10,
                "max_feature_ks": 0.35,
                "fallback_to_full_on_fail": True,
            },
        }

        frame = make_feature_frame(rows, seed=123)
        frame["snapshot_date"] = pd.date_range("2026-01-01", periods=10, freq="D").repeat(rows // 10)
        frame.loc[::11, "txn_amount_weekly"] = None
        frame.loc[::17, "checking_balance"] = None
        frame.loc[[5, 35, 95], "outstanding_balance"] = frame["outstanding_balance"].max() * 25
        feature_names = get_feature_list(config)
        return config, frame, feature_names

    def test_train_sampler_applies_and_validates_representative_sample(self):
        config, frame, feature_names = self._build_frame()
        sampler = TrainSampler(config, id_column="customer_id", time_column="snapshot_date")

        result = sampler.sample(frame, feature_names=feature_names, window_name="train")

        self.assertEqual(result.report["status"], "applied")
        self.assertEqual(len(result.frame), 60)
        self.assertEqual(result.report["sampled_rows"], 60)
        self.assertEqual(result.report["used_rows"], 60)
        self.assertLessEqual(
            result.report["validation"]["max_snapshot_share_delta"],
            config["development"]["sampling"]["validation"]["max_snapshot_share_delta"],
        )
        self.assertIn("checks", result.report["validation"])

    def test_train_sampler_falls_back_to_full_frame_when_validation_fails(self):
        config, frame, feature_names = self._build_frame()
        config["development"]["sampling"]["validation"]["max_feature_ks"] = 0.0
        sampler = TrainSampler(config, id_column="customer_id", time_column="snapshot_date")

        result = sampler.sample(frame, feature_names=feature_names, window_name="train")

        self.assertEqual(result.report["status"], "validation_failed_fallback")
        self.assertEqual(len(result.frame), len(frame))
        self.assertEqual(result.report["used_rows"], len(frame))


if __name__ == "__main__":
    unittest.main()
