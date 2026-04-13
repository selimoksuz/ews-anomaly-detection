import unittest

import numpy as np

from engine.calibration import ScoreCalibrator
from engine.config_loader import get_feature_list, load_config
from engine.models import AnomalyModels
from scripts.generate_data import generate_training_data


class CalibrationTests(unittest.TestCase):
    def test_empirical_calibration_maps_raw_scores_into_score_range(self):
        config = load_config()
        features = get_feature_list(config)
        train_df = generate_training_data(n=600, seed=42)
        train_df = train_df[train_df["split_flag"] == "TRAIN"].reset_index(drop=True)

        model = AnomalyModels(config)
        model.fit(train_df[features].fillna(0).values)
        X = model.transform(train_df[features].fillna(0).values)

        calibrator = ScoreCalibrator(config)
        artifact = calibrator.fit(
            {
                "ae_raw": model.raw_ae_scores(X),
                "if_raw": model.raw_if_scores(X),
                "md_raw": model.raw_md_scores(X),
            },
            model_version="MODEL-1",
            segment="ALL",
            window={"rows": len(train_df)},
        )
        calibrated = calibrator.apply(
            {
                "ae_raw": model.raw_ae_scores(X),
                "if_raw": model.raw_if_scores(X),
                "md_raw": model.raw_md_scores(X),
            },
            artifact,
        )

        for column in ("ae_cal", "if_cal", "md_cal"):
            self.assertTrue(np.all(calibrated[column] >= 0.0))
            self.assertTrue(np.all(calibrated[column] <= 100.0))

        order = np.argsort(model.raw_ae_scores(X))
        ae_sorted = calibrated["ae_cal"][order]
        self.assertTrue(np.all(np.diff(ae_sorted) >= -1e-9))


if __name__ == "__main__":
    unittest.main()
