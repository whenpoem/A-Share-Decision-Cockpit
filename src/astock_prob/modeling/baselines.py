from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from astock_prob.config import AppConfig
from astock_prob.features.engineering import feature_columns
from astock_prob.labels.generator import touch_column_name


@dataclass
class GbmMonteCarloBaseline:
    config: AppConfig

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        n_classes = len(self.config.bucket_labels)
        terminal_probs = np.zeros((len(frame), n_classes), dtype=float)
        touch_probs: Dict[str, np.ndarray] = {
            touch_column_name(threshold, direction): np.zeros(len(frame), dtype=float)
            for threshold in self.config.touch_thresholds
            for direction in ("up", "down")
        }
        for row_idx, row in enumerate(frame.itertuples(index=False)):
            mu_value = getattr(row, "ewma_mu", 0.08)
            sigma_value = getattr(row, "ewma_sigma", 0.3)
            mu = float(mu_value) if pd.notna(mu_value) else 0.08
            sigma = float(sigma_value) if pd.notna(sigma_value) else 0.3
            sigma = max(min(sigma, 1.5), 0.05)
            rng = np.random.default_rng(self.config.model.random_seed + row_idx)
            increments = rng.normal(
                loc=(mu - 0.5 * sigma**2) / 252.0,
                scale=sigma / np.sqrt(252.0),
                size=(self.config.model.gbm_paths, self.config.horizon_days),
            )
            rel_paths = np.exp(np.cumsum(increments, axis=1))
            terminal_returns = rel_paths[:, -1] - 1.0
            bucket_edges = np.array([-np.inf] + list(self.config.bucket_edges) + [np.inf])
            bucket_ids = np.digitize(terminal_returns, bucket_edges[1:-1], right=False)
            terminal_probs[row_idx] = np.bincount(bucket_ids, minlength=n_classes) / len(bucket_ids)
            path_max = rel_paths.max(axis=1)
            path_min = rel_paths.min(axis=1)
            for threshold in self.config.touch_thresholds:
                touch_probs[touch_column_name(threshold, "up")][row_idx] = float((path_max >= 1.0 + threshold).mean())
                touch_probs[touch_column_name(threshold, "down")][row_idx] = float((path_min <= 1.0 - threshold).mean())
        return terminal_probs, touch_probs


@dataclass
class HistoricalBootstrapBaseline:
    config: AppConfig
    state_columns: List[str] = field(default_factory=list)
    train_features_: np.ndarray | None = None
    train_terminal_: np.ndarray | None = None
    train_touches_: Dict[str, np.ndarray] = field(default_factory=dict)
    medians_: pd.Series | None = None
    scales_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "HistoricalBootstrapBaseline":
        self.state_columns = choose_state_columns(frame)
        state_frame = frame[self.state_columns]
        self.medians_ = state_frame.median(numeric_only=True).fillna(0.0)
        filled = state_frame.fillna(self.medians_)
        self.scales_ = filled.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        self.train_features_ = ((filled - self.medians_) / self.scales_).to_numpy(dtype=float)
        self.train_terminal_ = frame["terminal_bucket_id"].to_numpy(dtype=int)
        self.train_touches_ = {
            touch_column_name(threshold, direction): frame[touch_column_name(threshold, direction)].to_numpy(dtype=float)
            for threshold in self.config.touch_thresholds
            for direction in ("up", "down")
        }
        return self

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        if self.train_features_ is None or self.train_terminal_ is None:
            raise RuntimeError("HistoricalBootstrapBaseline must be fitted before predict.")
        n_classes = len(self.config.bucket_labels)
        state_frame = frame[self.state_columns].fillna(self.medians_)
        query = ((state_frame - self.medians_) / self.scales_).to_numpy(dtype=float)
        terminal_probs = np.zeros((len(frame), n_classes), dtype=float)
        touch_probs: Dict[str, np.ndarray] = {
            key: np.zeros(len(frame), dtype=float)
            for key in self.train_touches_
        }
        for row_idx, row in enumerate(query):
            distances = np.linalg.norm(self.train_features_ - row, axis=1)
            neighbors = np.argsort(distances)[: self.config.model.bootstrap_neighbors]
            neighbor_dist = distances[neighbors]
            weights = 1.0 / (neighbor_dist + 1e-6)
            weights /= weights.sum()
            terminal_probs[row_idx] = _weighted_class_probs(
                self.train_terminal_[neighbors],
                weights,
                n_classes,
            )
            for key, values in self.train_touches_.items():
                touch_probs[key][row_idx] = float(np.dot(weights, values[neighbors]))
        return terminal_probs, touch_probs


def choose_state_columns(frame: pd.DataFrame) -> List[str]:
    preferred = [
        "return_5",
        "return_20",
        "return_60",
        "volatility_20",
        "volatility_60",
        "dist_ma_20",
        "dist_ma_60",
        "macd_hist",
        "rsi_14",
        "kdj_j",
        "volume_ratio_20",
        "turnover_ratio_20",
        "excess_return_20",
        "sector_excess_20",
        "ewma_mu",
        "ewma_sigma",
        "pb_pct_252",
        "pe_ttm_pct_252",
        "revenue_yoy",
        "net_profit_yoy",
        "roe",
        "debt_ratio",
    ]
    available = [col for col in preferred if col in frame.columns]
    if available:
        return available
    return feature_columns(frame)[:10]


def _weighted_class_probs(labels: np.ndarray, weights: np.ndarray, n_classes: int) -> np.ndarray:
    probs = np.zeros(n_classes, dtype=float)
    for label, weight in zip(labels, weights):
        probs[int(label)] += weight
    total = probs.sum()
    if total == 0:
        probs[:] = 1.0 / n_classes
    return probs / probs.sum()
