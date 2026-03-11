from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


EPSILON = 1e-12


def save_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = payload
    if is_dataclass(payload):
        serializable = asdict(payload)
    target.write_text(json.dumps(serializable, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def annualize_return(daily_return: pd.Series, periods_per_year: int = 252) -> pd.Series:
    return daily_return.rolling(periods_per_year).mean() * periods_per_year


def annualize_volatility(daily_return: pd.Series, window: int, periods_per_year: int = 252) -> pd.Series:
    return daily_return.rolling(window).std(ddof=0) * math.sqrt(periods_per_year)


def safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def max_drawdown(close: pd.Series, window: int) -> pd.Series:
    rolling_peak = close.rolling(window).max()
    return close / rolling_peak - 1.0


def exponential_weighted_mu_sigma(log_returns: pd.Series, halflife: int = 20) -> pd.DataFrame:
    ewma_mu = log_returns.ewm(halflife=halflife, adjust=False).mean() * 252
    ewma_sigma = log_returns.ewm(halflife=halflife, adjust=False).std(bias=False) * math.sqrt(252)
    return pd.DataFrame({"ewma_mu": ewma_mu, "ewma_sigma": ewma_sigma})


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.clip(exp_values.sum(axis=1, keepdims=True), EPSILON, None)


def multiclass_log_loss(y_true: Sequence[int], probs: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    clipped = np.clip(probs, EPSILON, 1.0)
    row_index = np.arange(y_true_arr.shape[0])
    return float(-np.mean(np.log(clipped[row_index, y_true_arr])))


def brier_score(y_true: Sequence[int], probs: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    probs_arr = np.asarray(probs, dtype=float)
    return float(np.mean((probs_arr - y_true_arr) ** 2))


def expected_calibration_error(y_true: Sequence[int], probs: Sequence[float], bins: int = 10) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    probs_arr = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (probs_arr >= left) & (probs_arr <= right)
        else:
            mask = (probs_arr >= left) & (probs_arr < right)
        if not np.any(mask):
            continue
        avg_conf = probs_arr[mask].mean()
        avg_acc = y_true_arr[mask].mean()
        ece += np.abs(avg_conf - avg_acc) * mask.mean()
    return float(ece)


def ranked_probability_score(y_true: Sequence[int], probs: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true_arr)), y_true_arr] = 1.0
    cum_probs = np.cumsum(probs, axis=1)
    cum_true = np.cumsum(one_hot, axis=1)
    return float(np.mean(np.sum((cum_probs - cum_true) ** 2, axis=1) / (n_classes - 1)))


def ensure_probability_matrix(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, EPSILON, None)
    return clipped / clipped.sum(axis=1, keepdims=True)


def month_starts(dates: Iterable[pd.Timestamp]) -> List[pd.Timestamp]:
    ordered = pd.Series(pd.to_datetime(list(dates))).sort_values().drop_duplicates()
    return ordered.groupby(ordered.dt.to_period("M")).min().tolist()


def format_probability(value: float) -> str:
    return f"{value:.2%}"


def to_serializable_frame(frame: pd.DataFrame) -> List[Dict[str, object]]:
    return frame.replace({np.nan: None}).to_dict(orient="records")
