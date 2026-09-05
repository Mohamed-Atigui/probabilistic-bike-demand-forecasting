import unittest

import numpy as np

from bikeshare_forecast.data import generate_demo_data
from bikeshare_forecast.features import FEATURE_COLUMNS, build_features, chronological_split


class FeatureTests(unittest.TestCase):
    def test_features_use_past_targets_only(self):
        raw = generate_demo_data(n_hours=24 * 40, seed=7)
        featured = build_features(raw)
        first = featured.iloc[0]
        original_idx = 168
        self.assertEqual(first["demand_lag_24"], raw.loc[original_idx - 24, "demand"])
        self.assertEqual(first["demand_lag_168"], raw.loc[original_idx - 168, "demand"])
        self.assertTrue(np.isclose(first["demand_rolling_24"], raw.loc[original_idx - 24:original_idx - 1, "demand"].mean()))
        self.assertEqual(featured[FEATURE_COLUMNS].isna().sum().sum(), 0)


    def test_split_is_strictly_chronological(self):
        featured = build_features(generate_demo_data(n_hours=24 * 60))
        train, calibration, test = chronological_split(featured)
        self.assertLess(train["timestamp"].max(), calibration["timestamp"].min())
        self.assertLess(calibration["timestamp"].max(), test["timestamp"].min())


if __name__ == "__main__":
    unittest.main()
