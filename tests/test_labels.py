import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astock_prob.config import AppConfig, ModelConfig, PathsConfig, ReportingConfig
from astock_prob.labels.generator import build_labeled_frame, touch_column_name


class LabelGenerationTests(unittest.TestCase):
    def test_terminal_and_touch_labels(self) -> None:
        config = AppConfig(
            symbols=["AAA"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            horizon_days=3,
            return_grid=[-0.1, 0.0, 0.1],
            touch_thresholds=[0.05, 0.1],
            provider="csv",
            benchmark_symbol="000300",
            sector_proxy_symbols={},
            paths=PathsConfig(Path("."), Path("."), Path("."), Path(".")),
            model=ModelConfig(),
            reporting=ReportingConfig(),
            project_root=Path("."),
        )
        feature_frame = pd.DataFrame(
            {
                "symbol": ["AAA"] * 5,
                "date": pd.date_range("2024-01-01", periods=5, freq="D"),
                "close": [10.0, 10.5, 10.8, 11.2, 11.0],
                "high": [10.2, 10.7, 11.1, 11.4, 11.1],
                "low": [9.8, 10.2, 10.6, 10.9, 10.8],
            }
        )
        labeled = build_labeled_frame(feature_frame, config)
        first_row = labeled.iloc[0]
        self.assertEqual(first_row["terminal_bucket"], ">=10%")
        self.assertEqual(first_row[touch_column_name(0.05, "up")], 1.0)
        self.assertEqual(first_row[touch_column_name(0.1, "up")], 1.0)
        self.assertEqual(first_row[touch_column_name(0.05, "down")], 0.0)


if __name__ == "__main__":
    unittest.main()
