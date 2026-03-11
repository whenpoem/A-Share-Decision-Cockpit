from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from astock_prob.config import AppConfig


def build_labeled_frame(feature_frame: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    frame = feature_frame.sort_values(["symbol", "date"]).copy()
    future_close = frame.groupby("symbol")["close"].shift(-config.horizon_days)
    frame["future_return_60"] = future_close / frame["close"] - 1.0
    future_high = _future_window_stat(frame, "high", config.horizon_days, reducer="max")
    future_low = _future_window_stat(frame, "low", config.horizon_days, reducer="min")
    frame["future_high_60"] = future_high
    frame["future_low_60"] = future_low
    frame["terminal_bucket"] = pd.cut(
        frame["future_return_60"],
        bins=[-np.inf] + list(config.bucket_edges) + [np.inf],
        labels=config.bucket_labels,
        right=False,
    )
    bucket_map = {label: idx for idx, label in enumerate(config.bucket_labels)}
    frame["terminal_bucket_id"] = frame["terminal_bucket"].map(bucket_map)
    for threshold in config.touch_thresholds:
        up_col = touch_column_name(threshold, "up")
        down_col = touch_column_name(threshold, "down")
        frame[up_col] = (frame["future_high_60"] >= frame["close"] * (1.0 + threshold)).astype(float)
        frame[down_col] = (frame["future_low_60"] <= frame["close"] * (1.0 - threshold)).astype(float)
    frame["is_trainable"] = frame["future_return_60"].notna().astype(int)
    return frame


def touch_column_name(threshold: float, direction: str) -> str:
    pct = int(round(threshold * 100))
    return f"touch_{direction}_{pct}"


def terminal_bucket_index(config: AppConfig) -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(config.bucket_labels)}


def _future_window_stat(frame: pd.DataFrame, column: str, horizon: int, reducer: str) -> pd.Series:
    results = []
    for _, symbol_frame in frame.groupby("symbol", sort=False):
        series = symbol_frame[column].to_numpy(dtype=float)
        values = np.full(series.shape[0], np.nan)
        for idx in range(series.shape[0]):
            end = min(idx + horizon + 1, series.shape[0])
            window = series[idx + 1:end]
            if len(window) == 0:
                continue
            values[idx] = np.max(window) if reducer == "max" else np.min(window)
        results.append(pd.Series(values, index=symbol_frame.index))
    return pd.concat(results).sort_index()
