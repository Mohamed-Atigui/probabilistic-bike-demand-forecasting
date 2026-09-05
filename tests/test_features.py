import unittest

import numpy as np
import pandas as pd

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

    def test_lags_are_matched_by_timestamp_when_an_hour_is_missing(self):
        raw = generate_demo_data(n_hours=24 * 40, seed=7)
        missing_timestamp = raw.loc[200, "timestamp"]
        irregular = raw.drop(index=200).reset_index(drop=True)
        featured = build_features(irregular).set_index("timestamp")

        affected_24h = missing_timestamp + pd.Timedelta(hours=24)
        unaffected = missing_timestamp + pd.Timedelta(hours=25)

        self.assertNotIn(affected_24h, featured.index)
        self.assertEqual(
            featured.loc[unaffected, "demand_lag_24"],
            irregular.set_index("timestamp").loc[unaffected - pd.Timedelta(hours=24), "demand"],
        )

    def test_rolling_mean_uses_the_previous_24_clock_hours(self):
        raw = generate_demo_data(n_hours=24 * 40, seed=7)
        raw = raw.drop(index=200).reset_index(drop=True)
        timestamp = raw.loc[250, "timestamp"]
        featured = build_features(raw).set_index("timestamp")
        expected = raw.loc[
            (raw["timestamp"] >= timestamp - pd.Timedelta(hours=24))
            & (raw["timestamp"] < timestamp),
            "demand",
        ].mean()

        self.assertTrue(np.isclose(featured.loc[timestamp, "demand_rolling_24"], expected))


    def test_split_is_strictly_chronological(self):
        featured = build_features(generate_demo_data(n_hours=24 * 60))
        train, calibration, test = chronological_split(featured)
        self.assertLess(train["timestamp"].max(), calibration["timestamp"].min())
        self.assertLess(calibration["timestamp"].max(), test["timestamp"].min())


if __name__ == "__main__":
    unittest.main()
