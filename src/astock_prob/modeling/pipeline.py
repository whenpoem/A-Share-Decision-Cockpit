from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from astock_prob.config import AppConfig
from astock_prob.features.engineering import feature_columns
from astock_prob.labels.generator import touch_column_name
from astock_prob.modeling.baselines import GbmMonteCarloBaseline, HistoricalBootstrapBaseline, choose_state_columns
from astock_prob.modeling.calibration import BinaryCalibrator, TemperatureScaler, fit_binary_calibrator
from astock_prob.modeling.constraints import enforce_monotonic_touch_probabilities
from astock_prob.modeling.ensemble import blend_binary, blend_multiclass, fit_binary_weights, fit_multiclass_weights
from astock_prob.modeling.fallback_models import SklearnCompatibleWrapper
from astock_prob.utils import brier_score, expected_calibration_error, multiclass_log_loss, ranked_probability_score, save_json


@dataclass
class TrainingArtifacts:
    feature_columns: List[str]
    state_columns: List[str]
    terminal_model: object
    touch_models: Dict[str, object]
    bootstrap_baseline: HistoricalBootstrapBaseline
    terminal_weights: tuple[float, float, float]
    touch_weights: Dict[str, tuple[float, float, float]]
    terminal_scaler: TemperatureScaler
    touch_calibrators: Dict[str, BinaryCalibrator]
    metadata: Dict[str, object] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TrainingArtifacts":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def train_models(dataset: pd.DataFrame, config: AppConfig) -> tuple[TrainingArtifacts, Dict[str, float]]:
    model_frame = dataset.loc[(dataset["is_trainable"] == 1) & (dataset["is_universe_member"] == 1)].copy()
    model_frame = model_frame.dropna(subset=["terminal_bucket_id"])
    train_frame = model_frame[model_frame["date"] < config.model.validation_start].copy()
    val_frame = model_frame[(model_frame["date"] >= config.model.validation_start) & (model_frame["date"] < config.model.test_start)].copy()
    if len(train_frame) < config.model.bootstrap_min_history or val_frame.empty:
        raise ValueError("Not enough training or validation data to fit the project models.")
    features = feature_columns(model_frame)
    state_cols = choose_state_columns(model_frame)
    terminal_model = _fit_terminal_model(train_frame, features, config)
    touch_models = _fit_touch_models(train_frame, features, config)
    gbm_baseline = GbmMonteCarloBaseline(config)
    bootstrap_baseline = HistoricalBootstrapBaseline(config).fit(train_frame)
    val_terminal_gbm, val_touch_gbm = gbm_baseline.predict(val_frame)
    val_terminal_bootstrap, val_touch_bootstrap = bootstrap_baseline.predict(val_frame)
    val_terminal_ml = terminal_model.predict_proba(val_frame[features])
    terminal_weights = fit_multiclass_weights(
        [val_terminal_gbm, val_terminal_bootstrap, val_terminal_ml],
        val_frame["terminal_bucket_id"].to_numpy(dtype=int),
        config.model.ensemble_weight_step,
    )
    val_terminal_blended = blend_multiclass([val_terminal_gbm, val_terminal_bootstrap, val_terminal_ml], terminal_weights)
    terminal_scaler = TemperatureScaler(config.model.temperature_grid).fit(
        val_terminal_blended,
        val_frame["terminal_bucket_id"].to_numpy(dtype=int),
    )
    touch_weights: Dict[str, tuple[float, float, float]] = {}
    touch_calibrators: Dict[str, BinaryCalibrator] = {}
    touch_task_metrics: Dict[str, Dict[str, float | int | str]] = {}
    for threshold in config.touch_thresholds:
        for direction in ("up", "down"):
            key = touch_column_name(threshold, direction)
            y_val = val_frame[key].to_numpy(dtype=int)
            ml_probs = touch_models[key].predict_proba(val_frame[features])[:, -1]
            weight = fit_binary_weights(
                [val_touch_gbm[key], val_touch_bootstrap[key], ml_probs],
                y_val,
                config.model.ensemble_weight_step,
            )
            touch_weights[key] = weight
            blended = blend_binary([val_touch_gbm[key], val_touch_bootstrap[key], ml_probs], weight)
            calibrator = fit_binary_calibrator(blended, y_val, config.model.calibration_min_samples)
            touch_calibrators[key] = calibrator
            calibrated = calibrator.transform(blended)
            touch_task_metrics[key] = {
                "samples": int(len(y_val)),
                "positive_rate": float(np.mean(y_val)) if len(y_val) else 0.0,
                "brier": brier_score(y_val, calibrated),
                "ece": expected_calibration_error(y_val, calibrated, bins=config.model.calibration_bins),
                "calibrator": calibrator.method,
            }
    combined_train = pd.concat([train_frame, val_frame], ignore_index=True)
    final_terminal_model = _fit_terminal_model(combined_train, features, config)
    final_touch_models = _fit_touch_models(combined_train, features, config)
    final_bootstrap = HistoricalBootstrapBaseline(config).fit(combined_train)
    feature_medians = combined_train[features].median(numeric_only=True).fillna(0.0)
    feature_scales = combined_train[features].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    artifacts = TrainingArtifacts(
        feature_columns=features,
        state_columns=state_cols,
        terminal_model=final_terminal_model,
        touch_models=final_touch_models,
        bootstrap_baseline=final_bootstrap,
        terminal_weights=terminal_weights,
        touch_weights=touch_weights,
        terminal_scaler=terminal_scaler,
        touch_calibrators=touch_calibrators,
        metadata={
            "bucket_labels": config.bucket_labels,
            "touch_thresholds": config.touch_thresholds,
            "validation_start": config.model.validation_start,
            "test_start": config.model.test_start,
            "model_version": "0.2.0",
            "training_samples": int(len(combined_train)),
            "validation_samples": int(len(val_frame)),
            "universe_symbols": int(combined_train["symbol"].nunique()),
            "feature_medians": {key: float(value) for key, value in feature_medians.to_dict().items()},
            "feature_scales": {key: float(value) for key, value in feature_scales.to_dict().items()},
            "target_symbols": config.effective_target_symbols,
        },
    )
    metrics = evaluate_on_validation(
        val_frame,
        val_terminal_gbm,
        val_terminal_bootstrap,
        val_terminal_ml,
        val_touch_gbm,
        val_touch_bootstrap,
        touch_models,
        features,
        terminal_weights,
        touch_weights,
        terminal_scaler,
        touch_calibrators,
        config,
    )
    metrics["training_samples"] = int(len(combined_train))
    metrics["validation_samples"] = int(len(val_frame))
    metrics["universe_symbols"] = int(combined_train["symbol"].nunique())
    metrics["touch_task_metrics"] = touch_task_metrics
    return artifacts, metrics


def predict_dataset(artifacts: TrainingArtifacts, frame: pd.DataFrame, config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    gbm_baseline = GbmMonteCarloBaseline(config)
    terminal_gbm, touch_gbm = gbm_baseline.predict(frame)
    terminal_bootstrap, touch_bootstrap = artifacts.bootstrap_baseline.predict(frame)
    terminal_ml = artifacts.terminal_model.predict_proba(frame[artifacts.feature_columns])
    terminal_blend = blend_multiclass([terminal_gbm, terminal_bootstrap, terminal_ml], artifacts.terminal_weights)
    terminal_final = artifacts.terminal_scaler.transform(terminal_blend)
    terminal_rows = []
    for row_idx, row in enumerate(frame.itertuples(index=False)):
        for bucket_idx, bucket_label in enumerate(config.bucket_labels):
            terminal_rows.append(
                {
                    "symbol": getattr(row, "symbol"),
                    "as_of_date": pd.Timestamp(getattr(row, "date")).date().isoformat(),
                    "horizon_days": config.horizon_days,
                    "return_bucket": bucket_label,
                    "terminal_prob": float(terminal_final[row_idx, bucket_idx]),
                    "gbm_prob": float(terminal_gbm[row_idx, bucket_idx]),
                    "bootstrap_prob": float(terminal_bootstrap[row_idx, bucket_idx]),
                    "ml_prob": float(terminal_ml[row_idx, bucket_idx]),
                    "ensemble_prob": float(terminal_final[row_idx, bucket_idx]),
                    "confidence_flag": "unknown",
                    "actual_bucket": getattr(row, "terminal_bucket", None),
                    "model_version": artifacts.metadata.get("model_version", "0.2.0"),
                    "data_cutoff": pd.Timestamp(getattr(row, "date")).date().isoformat(),
                }
            )
    touch_rows = []
    for row_idx, row in enumerate(frame.itertuples(index=False)):
        raw_probs: Dict[str, float] = {}
        ml_prob_cache: Dict[str, float] = {}
        row_frame = frame.iloc[[row_idx]][artifacts.feature_columns]
        for threshold in config.touch_thresholds:
            for direction in ("up", "down"):
                key = touch_column_name(threshold, direction)
                ml_prob = artifacts.touch_models[key].predict_proba(row_frame)[:, -1][0]
                ml_prob_cache[key] = float(ml_prob)
                blended = blend_binary(
                    [
                        np.array([touch_gbm[key][row_idx]]),
                        np.array([touch_bootstrap[key][row_idx]]),
                        np.array([ml_prob]),
                    ],
                    artifacts.touch_weights[key],
                )[0]
                calibrated = float(artifacts.touch_calibrators[key].transform(np.array([blended]))[0])
                raw_probs[key] = calibrated
        adjusted_up = enforce_monotonic_touch_probabilities(raw_probs, config.touch_thresholds, "up")
        adjusted_down = enforce_monotonic_touch_probabilities(raw_probs, config.touch_thresholds, "down")
        adjusted_probs = {**adjusted_up, **adjusted_down}
        for threshold in config.touch_thresholds:
            for direction in ("up", "down"):
                key = touch_column_name(threshold, direction)
                touch_rows.append(
                    {
                        "symbol": getattr(row, "symbol"),
                        "as_of_date": pd.Timestamp(getattr(row, "date")).date().isoformat(),
                        "horizon_days": config.horizon_days,
                        "return_bucket": f"{'+' if direction == 'up' else '-'}{int(threshold * 100)}%",
                        "touch_key": key,
                        "direction": direction,
                        "gbm_prob": float(touch_gbm[key][row_idx]),
                        "bootstrap_prob": float(touch_bootstrap[key][row_idx]),
                        "ml_prob": float(ml_prob_cache[key]),
                        "ensemble_prob": float(adjusted_probs[key]),
                        "touch_prob": float(adjusted_probs[key]),
                        "confidence_flag": "unknown",
                        "actual_touch": getattr(row, key, None),
                        "model_version": artifacts.metadata.get("model_version", "0.2.0"),
                        "data_cutoff": pd.Timestamp(getattr(row, "date")).date().isoformat(),
                    }
                )
    return pd.DataFrame(terminal_rows), pd.DataFrame(touch_rows)


def evaluate_on_validation(
    val_frame: pd.DataFrame,
    terminal_gbm: np.ndarray,
    terminal_bootstrap: np.ndarray,
    terminal_ml: np.ndarray,
    touch_gbm: Dict[str, np.ndarray],
    touch_bootstrap: Dict[str, np.ndarray],
    touch_models: Dict[str, object],
    features: List[str],
    terminal_weights: tuple[float, float, float],
    touch_weights: Dict[str, tuple[float, float, float]],
    terminal_scaler: TemperatureScaler,
    touch_calibrators: Dict[str, BinaryCalibrator],
    config: AppConfig,
) -> Dict[str, float]:
    y_terminal = val_frame["terminal_bucket_id"].to_numpy(dtype=int)
    terminal_blend = blend_multiclass([terminal_gbm, terminal_bootstrap, terminal_ml], terminal_weights)
    terminal_final = terminal_scaler.transform(terminal_blend)
    metrics = {
        "terminal_log_loss": multiclass_log_loss(y_terminal, terminal_final),
        "terminal_rps": ranked_probability_score(y_terminal, terminal_final),
        "terminal_gbm_log_loss": multiclass_log_loss(y_terminal, terminal_gbm),
    }
    brier_values = []
    ece_values = []
    for threshold in config.touch_thresholds:
        for direction in ("up", "down"):
            key = touch_column_name(threshold, direction)
            y_val = val_frame[key].to_numpy(dtype=int)
            ml_probs = touch_models[key].predict_proba(val_frame[features])[:, -1]
            blended = blend_binary([touch_gbm[key], touch_bootstrap[key], ml_probs], touch_weights[key])
            calibrated = touch_calibrators[key].transform(blended)
            brier_values.append(brier_score(y_val, calibrated))
            ece_values.append(expected_calibration_error(y_val, calibrated, bins=config.model.calibration_bins))
    metrics["touch_brier_mean"] = float(np.mean(brier_values))
    metrics["touch_ece_mean"] = float(np.mean(ece_values))
    return metrics


def save_training_outputs(
    artifacts: TrainingArtifacts,
    metrics: Dict[str, float],
    config: AppConfig,
    model_name: str = "latest.pkl",
) -> Path:
    model_path = artifacts.save(config.paths.model_dir / model_name)
    save_json(config.paths.model_dir / "latest_metrics.json", metrics)
    return model_path


def _fit_terminal_model(train_frame: pd.DataFrame, features: List[str], config: AppConfig) -> SklearnCompatibleWrapper:
    return SklearnCompatibleWrapper(
        "multiclass",
        random_seed=config.model.random_seed,
        expected_classes=np.arange(len(config.bucket_labels)),
    ).fit(
        train_frame[features],
        train_frame["terminal_bucket_id"].astype(int),
    )


def _fit_touch_models(train_frame: pd.DataFrame, features: List[str], config: AppConfig) -> Dict[str, SklearnCompatibleWrapper]:
    return {
        touch_column_name(threshold, direction): SklearnCompatibleWrapper(
            "binary",
            random_seed=config.model.random_seed,
            expected_classes=np.array([0, 1]),
        ).fit(
            train_frame[features],
            train_frame[touch_column_name(threshold, direction)].astype(int),
        )
        for threshold in config.touch_thresholds
        for direction in ("up", "down")
    }
