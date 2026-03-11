from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from astock_prob.config import AppConfig
from astock_prob.labels.generator import touch_column_name
from astock_prob.modeling.pipeline import predict_dataset, train_models
from astock_prob.pipeline import clone_config
from astock_prob.utils import brier_score, expected_calibration_error, month_starts, multiclass_log_loss, ranked_probability_score, save_json


@dataclass
class BacktestResult:
    terminal_predictions: pd.DataFrame
    touch_predictions: pd.DataFrame
    monthly_metrics: pd.DataFrame
    touch_task_metrics: pd.DataFrame
    summary_metrics: Dict[str, float]


def run_walkforward_backtest(dataset: pd.DataFrame, config: AppConfig) -> BacktestResult:
    test_frame = dataset[(dataset["date"] >= config.model.test_start) & (dataset["is_trainable"] == 1) & (dataset["is_universe_member"] == 1)].copy()
    prediction_dates = month_starts(test_frame["date"])
    all_terminal: List[pd.DataFrame] = []
    all_touch: List[pd.DataFrame] = []
    monthly_rows: List[Dict[str, object]] = []
    task_rows: List[Dict[str, object]] = []
    for month_idx, score_start in enumerate(prediction_dates):
        next_month = prediction_dates[month_idx + 1] if month_idx + 1 < len(prediction_dates) else None
        fold_config = clone_config(config, test_start=score_start.date().isoformat())
        month_slice = dataset[(dataset["date"] >= score_start) & (dataset["is_trainable"] == 1) & (dataset["is_universe_member"] == 1)].copy()
        if next_month is not None:
            month_slice = month_slice[month_slice["date"] < next_month]
        if month_slice.empty:
            continue
        try:
            train_source = dataset[dataset["date"] < score_start].copy()
            artifacts, _ = train_models(train_source, fold_config)
        except Exception:
            continue
        terminal_df, touch_df = predict_dataset(artifacts, month_slice, config)
        terminal_df["month_start"] = score_start.date().isoformat()
        touch_df["month_start"] = score_start.date().isoformat()
        all_terminal.append(terminal_df)
        all_touch.append(touch_df)
        month_metrics, task_metrics = _fold_metrics(score_start, terminal_df, touch_df, config)
        monthly_rows.append(month_metrics)
        task_rows.extend(task_metrics)
    terminal_predictions = pd.concat(all_terminal, ignore_index=True) if all_terminal else pd.DataFrame()
    touch_predictions = pd.concat(all_touch, ignore_index=True) if all_touch else pd.DataFrame()
    monthly_metrics = pd.DataFrame(monthly_rows)
    touch_task_metrics = pd.DataFrame(task_rows)
    summary = {
        "terminal_log_loss_mean": float(monthly_metrics["terminal_log_loss"].mean()) if not monthly_metrics.empty else np.nan,
        "terminal_rps_mean": float(monthly_metrics["terminal_rps"].mean()) if not monthly_metrics.empty else np.nan,
        "touch_brier_mean": float(monthly_metrics["touch_brier_mean"].mean()) if not monthly_metrics.empty else np.nan,
        "touch_ece_mean": float(monthly_metrics["touch_ece_mean"].mean()) if not monthly_metrics.empty else np.nan,
    }
    return BacktestResult(
        terminal_predictions=terminal_predictions,
        touch_predictions=touch_predictions,
        monthly_metrics=monthly_metrics,
        touch_task_metrics=touch_task_metrics,
        summary_metrics=summary,
    )


def save_backtest_outputs(result: BacktestResult, config: AppConfig) -> Dict[str, Path]:
    outputs = {
        "terminal_predictions": _save(result.terminal_predictions, config.paths.report_dir / "backtest_terminal_predictions.csv"),
        "touch_predictions": _save(result.touch_predictions, config.paths.report_dir / "backtest_touch_predictions.csv"),
        "monthly_metrics": _save(result.monthly_metrics, config.paths.report_dir / "backtest_monthly_metrics.csv"),
        "touch_task_metrics": _save(result.touch_task_metrics, config.paths.report_dir / "backtest_touch_task_metrics.csv"),
    }
    save_json(config.paths.report_dir / "backtest_summary.json", result.summary_metrics)
    return outputs


def _fold_metrics(score_start: pd.Timestamp, terminal_df: pd.DataFrame, touch_df: pd.DataFrame, config: AppConfig) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    if terminal_df.empty:
        return (
            {
                "month_start": score_start.date().isoformat(),
                "terminal_log_loss": np.nan,
                "terminal_rps": np.nan,
                "touch_brier_mean": np.nan,
                "touch_ece_mean": np.nan,
                "samples": 0,
            },
            [],
        )
    terminal_wide = terminal_df.pivot_table(index=["symbol", "as_of_date", "actual_bucket"], columns="return_bucket", values="ensemble_prob").reset_index()
    ordered_probs = terminal_wide[config.bucket_labels].to_numpy(dtype=float)
    actual_bucket_map = {label: idx for idx, label in enumerate(config.bucket_labels)}
    y_true = terminal_wide["actual_bucket"].map(actual_bucket_map).to_numpy(dtype=int)
    touch_scores = []
    touch_eces = []
    task_rows: List[Dict[str, object]] = []
    for threshold in config.touch_thresholds:
        for direction in ("up", "down"):
            key = touch_column_name(threshold, direction)
            subset = touch_df[touch_df["touch_key"] == key]
            if subset.empty:
                continue
            brier = brier_score(subset["actual_touch"].astype(float), subset["ensemble_prob"].astype(float))
            ece = expected_calibration_error(
                subset["actual_touch"].astype(float),
                subset["ensemble_prob"].astype(float),
                bins=config.model.calibration_bins,
            )
            touch_scores.append(brier)
            touch_eces.append(ece)
            task_rows.append(
                {
                    "month_start": score_start.date().isoformat(),
                    "touch_key": key,
                    "direction": direction,
                    "threshold": int(threshold * 100),
                    "samples": int(len(subset)),
                    "positive_rate": float(subset["actual_touch"].astype(float).mean()),
                    "brier": brier,
                    "ece": ece,
                }
            )
    return (
        {
            "month_start": score_start.date().isoformat(),
            "terminal_log_loss": multiclass_log_loss(y_true, ordered_probs),
            "terminal_rps": ranked_probability_score(y_true, ordered_probs),
            "touch_brier_mean": float(np.mean(touch_scores)) if touch_scores else np.nan,
            "touch_ece_mean": float(np.mean(touch_eces)) if touch_eces else np.nan,
            "samples": int(len(terminal_wide)),
        },
        task_rows,
    )


def _save(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path
