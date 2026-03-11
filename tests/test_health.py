import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astock_prob.config import AppConfig, ConfidenceThresholdsConfig, ModelConfig, PathsConfig, ReportingConfig
from astock_prob.modeling.health import build_model_health
from astock_prob.modeling.pipeline import TrainingArtifacts


class ModelHealthTests(unittest.TestCase):
    def test_health_downgrades_on_low_samples_and_high_drift(self) -> None:
        config = AppConfig(
            symbols=["002202", "601727"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            horizon_days=20,
            return_grid=[-0.1, 0.0, 0.1],
            touch_thresholds=[0.05, 0.1],
            provider="csv",
            benchmark_symbol="000300",
            sector_proxy_symbols={},
            paths=PathsConfig(Path("."), Path("."), Path("."), Path(".")),
            model=ModelConfig(calibration_min_samples=20),
            reporting=ReportingConfig(),
            project_root=Path("."),
            target_symbols=["002202", "601727"],
            confidence_thresholds=ConfidenceThresholdsConfig(ece_warn=0.08, sample_warn=120, drift_warn=1.0),
        )
        artifacts = TrainingArtifacts(
            feature_columns=["f1", "f2"],
            state_columns=["f1"],
            terminal_model=None,
            touch_models={},
            bootstrap_baseline=None,
            terminal_weights=(1.0, 0.0, 0.0),
            touch_weights={},
            terminal_scaler=None,
            touch_calibrators={},
            metadata={"feature_medians": {"f1": 0.0, "f2": 0.0}, "feature_scales": {"f1": 1.0, "f2": 1.0}},
        )
        dataset = pd.DataFrame(
            {
                "symbol": ["002202", "601727"],
                "date": [pd.Timestamp("2024-12-31"), pd.Timestamp("2024-12-31")],
                "f1": [5.0, 0.1],
                "f2": [4.0, 0.1],
            }
        )
        terminal_df = pd.DataFrame({"symbol": ["002202", "601727"], "as_of_date": ["2024-12-31", "2024-12-31"]})
        touch_df = pd.DataFrame({"symbol": ["002202", "601727"]})
        training_metrics = {"validation_samples": 20, "touch_ece_mean": 0.12}
        health = build_model_health(artifacts, dataset, terminal_df, touch_df, training_metrics, training_metrics, config)
        self.assertEqual(health["overall_confidence_flag"], "low")
        self.assertEqual(health["symbols"]["002202"]["confidence_flag"], "low")


if __name__ == "__main__":
    unittest.main()
