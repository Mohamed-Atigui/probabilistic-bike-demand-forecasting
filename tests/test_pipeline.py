import tempfile
import unittest
from pathlib import Path

from bikeshare_forecast.train import run_experiment


class PipelineTests(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metrics = run_experiment(tmp_path, n_hours=24 * 75)
            self.assertGreater(metrics["model"]["mae"], 0)
            self.assertGreaterEqual(metrics["prediction_interval_90"]["coverage"], 0.75)
            self.assertLessEqual(metrics["prediction_interval_90"]["coverage"], 1.0)
            self.assertTrue((tmp_path / "metrics.json").exists())
            self.assertTrue((tmp_path / "model.joblib").exists())
            self.assertTrue((tmp_path / "forecast.png").exists())


if __name__ == "__main__":
    unittest.main()
