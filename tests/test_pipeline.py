import json
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astock_prob.config import AppConfig
from astock_prob.data.ingestion import fetch_all_data
from astock_prob.data.providers import build_provider
from astock_prob.modeling.health import apply_confidence_flags, build_model_health
from astock_prob.modeling.pipeline import predict_dataset, train_models
from astock_prob.pipeline import build_dataset, latest_rows


class PipelineIntegrationTests(unittest.TestCase):
    def test_end_to_end_global_pipeline_with_csv_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            source_root.mkdir(parents=True, exist_ok=True)
            self._write_fixture_data(source_root)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "symbols": ["002202", "601727"],
                        "target_symbols": ["002202", "601727"],
                        "start_date": "2022-01-03",
                        "end_date": "2023-12-29",
                        "horizon_days": 20,
                        "return_grid": [-0.2, -0.1, 0.0, 0.1, 0.2],
                        "touch_thresholds": [0.05, 0.1],
                        "provider": "csv",
                        "benchmark_symbol": "000300",
                        "sector_proxy_symbols": {"002202": "399102", "601727": "000300"},
                        "training_universe": {
                            "mode": "seed_only",
                            "seed_symbols": ["002202", "601727", "300001", "300002", "300003", "300004", "300005", "300006"],
                            "sectors": ["风电"],
                            "max_symbols": 20
                        },
                        "universe_filters": {
                            "min_listing_days": 180,
                            "min_turnover": 2000000,
                            "exclude_st": True,
                            "exclude_suspended": True
                        },
                        "refresh_schedule": {"daily_close_time": "15:15", "daily_run_time": "15:25"},
                        "retrain_schedule": {"weekly_day": "Saturday", "monthly_full_refresh_day": 1},
                        "confidence_thresholds": {"ece_warn": 0.08, "sample_warn": 60, "drift_warn": 2.5},
                        "paths": {
                            "cache_dir": "artifacts/cache",
                            "dataset_dir": "artifacts/datasets",
                            "model_dir": "artifacts/models",
                            "report_dir": "artifacts/reports"
                        },
                        "model": {
                            "random_seed": 7,
                            "gbm_paths": 200,
                            "bootstrap_neighbors": 20,
                            "bootstrap_min_history": 30,
                            "validation_start": "2022-09-01",
                            "test_start": "2023-07-01",
                            "temperature_grid": [0.8, 1.0, 1.2],
                            "ensemble_weight_step": 0.25,
                            "calibration_min_samples": 20,
                            "calibration_bins": 4
                        },
                        "reporting": {"format": "both"}
                    }
                ),
                encoding="utf-8",
            )
            config = AppConfig.from_file(config_path)
            provider = build_provider(config.provider, str(source_root))
            fetch_all_data(config, provider)
            dataset = build_dataset(config)
            self.assertGreater(dataset["symbol"].nunique(), 2)
            self.assertNotIn("300998", dataset["symbol"].unique())
            self.assertNotIn("300999", dataset["symbol"].unique())
            artifacts, metrics = train_models(dataset, config)
            latest = latest_rows(dataset, config.effective_target_symbols)
            self.assertEqual(sorted(latest["symbol"].tolist()), ["002202", "601727"])
            terminal_df, touch_df = predict_dataset(artifacts, latest, config)
            health = build_model_health(artifacts, dataset, terminal_df, touch_df, metrics, metrics, config)
            terminal_df, touch_df = apply_confidence_flags(terminal_df, touch_df, health)
            self.assertFalse(terminal_df.empty)
            self.assertFalse(touch_df.empty)
            self.assertIn("confidence_flag", terminal_df.columns)
            term_sums = terminal_df.groupby(["symbol", "as_of_date"])["terminal_prob"].sum()
            self.assertTrue(np.allclose(term_sums.to_numpy(), 1.0, atol=1e-6))
            for (_, _, direction), group in touch_df.groupby(["symbol", "as_of_date", "direction"]):
                ordered = group.assign(
                    threshold=group["return_bucket"].str.replace("%", "", regex=False).str.replace("+", "", regex=False).str.replace("-", "", regex=False).astype(int)
                ).sort_values("threshold")["touch_prob"].to_numpy()
                self.assertTrue(np.all(np.diff(ordered) <= 1e-6))

    def _write_fixture_data(self, source_root: Path) -> None:
        dates = pd.bdate_range("2022-01-03", "2023-12-29")
        benchmark = self._make_price_frame("000300", dates, base=4000.0, drift=0.0002, scale=5.0, volume_base=5_000_000)
        sector = self._make_price_frame("399102", dates, base=3200.0, drift=0.00025, scale=6.0, volume_base=3_000_000)
        benchmark.to_csv(source_root / "000300_benchmark.csv", index=False)
        sector.to_csv(source_root / "399102_benchmark.csv", index=False)
        symbols = ["002202", "601727", "300001", "300002", "300003", "300004", "300005", "300006"]
        for idx, symbol in enumerate(symbols):
            price = self._make_price_frame(symbol, dates, base=18.0 + idx * 3.0, drift=0.00035 + idx * 0.00002, scale=0.18 + idx * 0.01)
            valuation = pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "pe_ttm": 20 + np.sin(np.linspace(0, 8, len(dates))) * 5 + idx,
                    "pb": 2 + np.cos(np.linspace(0, 6, len(dates))) * 0.3,
                    "ps_ttm": 1.5 + np.sin(np.linspace(0, 4, len(dates))) * 0.2,
                    "dividend_yield": 1.2 + idx * 0.1,
                }
            )
            quarter_dates = pd.date_range("2021-12-31", "2023-12-31", freq="QE")
            financial = pd.DataFrame(
                {
                    "symbol": symbol,
                    "announce_date": quarter_dates + pd.Timedelta(days=25),
                    "report_date": quarter_dates,
                    "revenue_yoy": np.linspace(0.05, 0.18, len(quarter_dates)),
                    "net_profit_yoy": np.linspace(0.03, 0.15, len(quarter_dates)),
                    "roe": np.linspace(0.08, 0.14, len(quarter_dates)),
                    "debt_ratio": np.linspace(0.55, 0.48, len(quarter_dates)),
                }
            )
            price.to_csv(source_root / f"{symbol}_price.csv", index=False)
            valuation.to_csv(source_root / f"{symbol}_valuation.csv", index=False)
            financial.to_csv(source_root / f"{symbol}_financial.csv", index=False)
        short_dates = pd.bdate_range("2023-07-03", "2023-12-29")
        self._make_price_frame("300998", short_dates, base=15.0, drift=0.0003, scale=0.15).to_csv(source_root / "300998_price.csv", index=False)
        self._make_price_frame("300998", short_dates, base=15.0, drift=0.0003, scale=0.15)[["date", "symbol"]].assign(pe_ttm=20, pb=2).to_csv(source_root / "300998_valuation.csv", index=False)
        pd.DataFrame({"symbol": ["300998"], "announce_date": [pd.Timestamp("2023-08-31")], "report_date": [pd.Timestamp("2023-06-30")], "revenue_yoy": [0.1], "net_profit_yoy": [0.1], "roe": [0.1], "debt_ratio": [0.5]}).to_csv(source_root / "300998_financial.csv", index=False)
        low_turnover = self._make_price_frame("300999", dates, base=12.0, drift=0.0002, scale=0.12, volume_base=10_000)
        low_turnover.to_csv(source_root / "300999_price.csv", index=False)
        low_turnover[["date", "symbol"]].assign(pe_ttm=18, pb=1.5).to_csv(source_root / "300999_valuation.csv", index=False)
        pd.DataFrame({"symbol": ["300999"], "announce_date": [pd.Timestamp("2022-08-31")], "report_date": [pd.Timestamp("2022-06-30")], "revenue_yoy": [0.08], "net_profit_yoy": [0.07], "roe": [0.09], "debt_ratio": [0.45]}).to_csv(source_root / "300999_financial.csv", index=False)

    def _make_price_frame(self, symbol: str, dates: pd.DatetimeIndex, base: float, drift: float, scale: float, volume_base: float = 1_000_000) -> pd.DataFrame:
        t = np.arange(len(dates))
        close = base + drift * 1000 * t + np.sin(t / 12.0) * scale * 8 + np.cos(t / 7.0) * scale * 3
        open_price = close * (1.0 + np.sin(t / 9.0) * 0.002)
        high = np.maximum(open_price, close) * 1.01
        low = np.minimum(open_price, close) * 0.99
        volume = volume_base + (np.sin(t / 5.0) + 1.5) * volume_base * 0.12
        amount = volume * close
        turnover = 2.0 + np.cos(t / 10.0) * 0.3
        return pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
                "turnover": turnover,
            }
        )


if __name__ == "__main__":
    unittest.main()
