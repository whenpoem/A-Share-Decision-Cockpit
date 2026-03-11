from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from astock_prob.config import AppConfig
from astock_prob.data.ingestion import load_symbol_raw_data
from astock_prob.data.storage import load_frame, save_frame
from astock_prob.data.universe import resolve_symbol_universe
from astock_prob.features.engineering import build_feature_frame
from astock_prob.labels.generator import build_labeled_frame
from astock_prob.utils import save_json


def build_dataset(config: AppConfig) -> pd.DataFrame:
    symbols = resolve_symbol_universe(config)
    target_set = set(config.effective_target_symbols)
    frames = []
    skipped: dict[str, str] = {}
    for symbol in symbols:
        try:
            payload = load_symbol_raw_data(config, symbol)
            feature_frame = build_feature_frame(symbol, payload, config)
        except Exception as exc:
            skipped[symbol] = str(exc)
            continue
        labeled = build_labeled_frame(feature_frame, config)
        labeled["is_target"] = int(symbol in target_set)
        frames.append(labeled)
    if not frames:
        raise ValueError("No symbol data available to build the modeling dataset.")
    dataset = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    dataset["symbol"] = dataset["symbol"].astype(str).str.zfill(6)
    dataset = _augment_cross_sectional_features(dataset)
    dataset = _apply_universe_filters(dataset, config)
    save_frame(dataset, dataset_path(config))
    save_json(config.paths.dataset_dir / "dataset_build_summary.json", _dataset_summary(dataset, skipped))
    return dataset


def load_or_build_dataset(config: AppConfig) -> pd.DataFrame:
    path = dataset_path(config)
    if path.exists():
        frame = load_frame(path)
        if "symbol" in frame.columns:
            frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        for column in ("date", "announce_date", "report_date"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column])
        return frame
    return build_dataset(config)


def latest_rows(dataset: pd.DataFrame, target_symbols: list[str] | None = None) -> pd.DataFrame:
    target_set = set(target_symbols or [])
    working = dataset.copy()
    if "is_target" in working.columns:
        working = working[working["is_target"] == 1]
    elif target_set:
        working = working[working["symbol"].isin(target_set)]
    latest_idx = working.groupby("symbol")["date"].idxmax()
    return working.loc[latest_idx].sort_values("symbol").reset_index(drop=True)


def dataset_path(config: AppConfig) -> Path:
    return config.paths.dataset_dir / "modeling_dataset.csv"


def clone_config(config: AppConfig, *, validation_start: str | None = None, test_start: str | None = None) -> AppConfig:
    cloned = deepcopy(config)
    if validation_start is not None:
        cloned.model.validation_start = validation_start
    if test_start is not None:
        cloned.model.test_start = test_start
    return cloned


def _augment_cross_sectional_features(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    group = frame.groupby("date", observed=False)
    for col in ("return_20", "return_60", "volatility_20", "amount", "turnover", "ewma_sigma"):
        if col not in frame.columns:
            continue
        median_col = f"peer_median_{col}"
        rank_col = f"peer_rank_{col}"
        diff_col = f"peer_excess_{col}"
        frame[median_col] = group[col].transform("median")
        frame[diff_col] = frame[col] - frame[median_col]
        frame[rank_col] = group[col].rank(pct=True)
    if "return_20" in frame.columns:
        positive = (frame["return_20"] > 0).astype(float)
        frame["market_breadth_20"] = positive.groupby(frame["date"]).transform("mean")
        frame["market_trend_20"] = frame.groupby("date", observed=False)["return_20"].transform("mean")
    if "volatility_20" in frame.columns:
        frame["market_volatility_state"] = frame.groupby("date", observed=False)["volatility_20"].transform("mean")
    frame["earnings_season_flag"] = frame["date"].dt.month.isin([1, 3, 4, 7, 8, 10]).astype(float)
    frame["month_of_year"] = frame["date"].dt.month.astype(float)
    frame["weekday"] = frame["date"].dt.dayofweek.astype(float)
    return frame


def _apply_universe_filters(dataset: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    frame = dataset.copy()
    target_set = set(config.effective_target_symbols)
    stats = frame.groupby("symbol", observed=False).agg(
        listing_days=("close", "count"),
        avg_amount=("amount", "mean"),
        last_date=("date", "max"),
    )
    stats["eligible"] = (
        (stats["listing_days"] >= config.universe_filters.min_listing_days)
        & (stats["avg_amount"].fillna(0.0) >= config.universe_filters.min_turnover)
    )
    latest_date = frame["date"].max()
    if config.universe_filters.exclude_suspended:
        stats["eligible"] = stats["eligible"] & ((latest_date - stats["last_date"]).dt.days <= 10)
    eligible_symbols = set(stats.index[stats["eligible"]].tolist()) | target_set
    frame["is_universe_member"] = frame["symbol"].isin(eligible_symbols).astype(int)
    filtered = frame[frame["symbol"].isin(eligible_symbols)].copy()
    filtered["is_target"] = filtered["symbol"].isin(target_set).astype(int)
    return filtered.reset_index(drop=True)


def _dataset_summary(dataset: pd.DataFrame, skipped: dict[str, str]) -> dict[str, object]:
    return {
        "rows": int(len(dataset)),
        "symbols": int(dataset["symbol"].nunique()),
        "target_symbols": sorted(dataset.loc[dataset["is_target"] == 1, "symbol"].astype(str).unique().tolist()) if "is_target" in dataset.columns else [],
        "training_symbols": sorted(dataset.loc[dataset["is_target"] == 0, "symbol"].astype(str).unique().tolist()) if "is_target" in dataset.columns else [],
        "skipped_symbols": skipped,
    }
