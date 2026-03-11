from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from astock_prob.config import AppConfig
from astock_prob.modeling.pipeline import TrainingArtifacts


def build_model_health(
    artifacts: TrainingArtifacts,
    dataset: pd.DataFrame,
    terminal_df: pd.DataFrame,
    touch_df: pd.DataFrame,
    training_metrics: Dict[str, object],
    recent_quality: Dict[str, float],
    config: AppConfig,
) -> Dict[str, object]:
    feature_names = artifacts.feature_columns
    medians = pd.Series(artifacts.metadata.get("feature_medians", {}), dtype=float)
    scales = pd.Series(artifacts.metadata.get("feature_scales", {}), dtype=float).replace(0.0, 1.0)
    thresholds = config.confidence_thresholds
    symbol_health: Dict[str, Dict[str, object]] = {}
    overall_rank = 0
    for symbol in config.effective_target_symbols:
        latest = dataset.loc[dataset["symbol"] == symbol].sort_values("date").tail(1)
        if latest.empty:
            symbol_health[symbol] = {"confidence_flag": "low", "reasons": ["missing_latest_row"], "drift_score": None, "feature_missing_ratio": 1.0}
            overall_rank = max(overall_rank, 2)
            continue
        feature_slice = latest[feature_names]
        missing_ratio = float(feature_slice.isna().mean(axis=1).iloc[0])
        aligned = feature_slice.iloc[0].reindex(feature_names)
        drift_score = _drift_score(aligned, medians.reindex(feature_names), scales.reindex(feature_names))
        reasons = []
        rank = 0
        touch_ece = float(recent_quality.get("touch_ece_mean", training_metrics.get("touch_ece_mean", 0.0)))
        validation_samples = int(training_metrics.get("validation_samples", 0))
        if missing_ratio > 0.35:
            reasons.append("feature_missing_ratio_high")
            rank = max(rank, 2)
        elif missing_ratio > 0.15:
            reasons.append("feature_missing_ratio_warn")
            rank = max(rank, 1)
        if touch_ece > thresholds.ece_warn * 1.5:
            reasons.append("recent_touch_ece_high")
            rank = max(rank, 2)
        elif touch_ece > thresholds.ece_warn:
            reasons.append("recent_touch_ece_warn")
            rank = max(rank, 1)
        if validation_samples < thresholds.sample_warn:
            reasons.append("validation_samples_low")
            rank = max(rank, 1 if validation_samples >= thresholds.sample_warn / 2 else 2)
        if drift_score > thresholds.drift_warn * 1.5:
            reasons.append("feature_drift_high")
            rank = max(rank, 2)
        elif drift_score > thresholds.drift_warn:
            reasons.append("feature_drift_warn")
            rank = max(rank, 1)
        symbol_health[symbol] = {
            "confidence_flag": _rank_to_flag(rank),
            "reasons": reasons,
            "drift_score": drift_score,
            "feature_missing_ratio": missing_ratio,
        }
        overall_rank = max(overall_rank, rank)
    return {
        "as_of_date": terminal_df["as_of_date"].iloc[0] if not terminal_df.empty else None,
        "overall_confidence_flag": _rank_to_flag(overall_rank),
        "recent_quality": recent_quality,
        "training_validation_samples": int(training_metrics.get("validation_samples", 0)),
        "symbols": symbol_health,
    }


def apply_confidence_flags(terminal_df: pd.DataFrame, touch_df: pd.DataFrame, health: Dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    terminal = terminal_df.copy()
    touch = touch_df.copy()
    symbol_map = {symbol: payload.get("confidence_flag", "unknown") for symbol, payload in health.get("symbols", {}).items()}
    terminal["confidence_flag"] = terminal["symbol"].map(symbol_map).fillna(health.get("overall_confidence_flag", "unknown"))
    touch["confidence_flag"] = touch["symbol"].map(symbol_map).fillna(health.get("overall_confidence_flag", "unknown"))
    return terminal, touch


def load_recent_quality(config: AppConfig) -> Dict[str, float]:
    path = config.paths.report_dir / "backtest_monthly_metrics.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    metric_columns = [col for col in ["terminal_log_loss", "terminal_rps", "touch_brier_mean", "touch_ece_mean"] if col in frame.columns]
    latest = frame.tail(3)
    return {col: float(latest[col].mean()) for col in metric_columns}


def save_model_health(config: AppConfig, payload: Dict[str, object]) -> Path:
    from astock_prob.utils import save_json

    path = config.paths.report_dir / "daily_model_health.json"
    save_json(path, payload)
    return path


def _rank_to_flag(rank: int) -> str:
    if rank >= 2:
        return "low"
    if rank == 1:
        return "medium"
    return "high"


def _drift_score(row: pd.Series, medians: pd.Series, scales: pd.Series) -> float:
    aligned = row.astype(float)
    med = medians.astype(float).replace({np.nan: 0.0})
    scale = scales.astype(float).replace({0.0: 1.0}).fillna(1.0)
    z = ((aligned.fillna(med) - med) / scale).replace([np.inf, -np.inf], np.nan).abs()
    return float(z.dropna().mean()) if not z.dropna().empty else 0.0
