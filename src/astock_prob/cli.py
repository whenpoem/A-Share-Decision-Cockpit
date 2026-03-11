from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import sleep

import pandas as pd

from astock_prob.backtest.walkforward import run_walkforward_backtest, save_backtest_outputs
from astock_prob.config import AppConfig
from astock_prob.data.ingestion import fetch_all_data
from astock_prob.data.providers import build_provider
from astock_prob.data.universe import resolve_symbol_universe
from astock_prob.modeling.health import apply_confidence_flags, build_model_health, load_recent_quality, save_model_health
from astock_prob.modeling.pipeline import TrainingArtifacts, predict_dataset, save_training_outputs, train_models
from astock_prob.pipeline import build_dataset, latest_rows, load_or_build_dataset
from astock_prob.reporting.report import generate_backtest_report, generate_prediction_report
from astock_prob.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share three-month probability research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_name in ("refresh-data", "fetch-data"):
        fetch_parser = subparsers.add_parser(command_name, help="Fetch and cache raw market data.")
        fetch_parser.add_argument("--config", default="configs/default.json", help="Path to the JSON configuration file.")
        fetch_parser.add_argument("--source-root", default=None, help="Optional source directory for CSV provider.")
    for command_name in ("train-global", "train"):
        train_parser = subparsers.add_parser(command_name, help="Train the global probability models.")
        train_parser.add_argument("--config", default="configs/default.json", help="Path to the JSON configuration file.")
    backtest_parser = subparsers.add_parser("backtest", help="Run a walk-forward backtest and emit reports.")
    backtest_parser.add_argument("--config", default="configs/default.json", help="Path to the JSON configuration file.")
    for command_name in ("score-daily",):
        predict_parser = subparsers.add_parser(command_name, help="Generate latest daily prediction CSV outputs.")
        predict_parser.add_argument("--config", default="configs/default.json", help="Path to the JSON configuration file.")
    for command_name in ("report-daily",):
        report_parser = subparsers.add_parser(command_name, help="Generate the latest daily prediction report.")
        report_parser.add_argument("--config", default="configs/default.json", help="Path to the JSON configuration file.")
    predict_alias = subparsers.add_parser("predict", help="Alias for score-daily + report-daily.")
    predict_alias.add_argument("--config", default="configs/default.json", help="Path to the JSON configuration file.")
    run_daily_parser = subparsers.add_parser("run-daily-pipeline", help="Refresh, retrain if needed, score, and report.")
    run_daily_parser.add_argument("--config", default="configs/live_run.json", help="Path to the JSON configuration file.")
    run_daily_parser.add_argument("--source-root", default=None, help="Optional source directory for CSV provider.")
    run_daily_parser.add_argument("--fetch-retries", type=int, default=5, help="Maximum full fetch retries before aborting.")
    run_daily_parser.add_argument("--retry-wait", type=int, default=30, help="Seconds to wait between retries.")
    run_daily_parser.add_argument("--force-train", action="store_true", help="Force retraining during the daily pipeline.")
    run_live_alias = subparsers.add_parser("run-live", help="Alias for run-daily-pipeline.")
    run_live_alias.add_argument("--config", default="configs/live_run.json", help="Path to the JSON configuration file.")
    run_live_alias.add_argument("--source-root", default=None, help="Optional source directory for CSV provider.")
    run_live_alias.add_argument("--fetch-retries", type=int, default=5, help="Maximum full fetch retries before aborting.")
    run_live_alias.add_argument("--retry-wait", type=int, default=30, help="Seconds to wait between retries.")
    run_live_alias.add_argument("--force-train", action="store_true", help="Force retraining during the daily pipeline.")
    args = parser.parse_args()
    config = AppConfig.from_file(args.config)

    if args.command in {"refresh-data", "fetch-data"}:
        provider = build_provider(config.provider, getattr(args, "source_root", None))
        outputs = fetch_all_data(config, provider)
        save_json(config.paths.report_dir / "fetch_outputs.json", {key: str(value) for key, value in outputs.items()})
        return
    if args.command in {"train-global", "train"}:
        dataset = build_dataset(config)
        artifacts, metrics = train_models(dataset, config)
        save_training_outputs(artifacts, metrics, config)
        return
    if args.command == "backtest":
        _run_backtest(config)
        return
    if args.command == "score-daily":
        _score_daily(config)
        return
    if args.command == "report-daily":
        _report_daily(config)
        return
    if args.command == "predict":
        _score_daily(config)
        _report_daily(config)
        return
    if args.command in {"run-daily-pipeline", "run-live"}:
        provider = build_provider(config.provider, getattr(args, "source_root", None))
        attempt = 0
        while attempt < args.fetch_retries:
            attempt += 1
            print(f"[daily] fetch attempt {attempt}/{args.fetch_retries}")
            outputs = fetch_all_data(config, provider)
            save_json(config.paths.report_dir / "fetch_outputs.json", {key: str(value) for key, value in outputs.items()})
            if _critical_price_files_ready(config):
                print("[daily] fetch stage complete")
                break
            if attempt >= args.fetch_retries:
                raise RuntimeError("Critical target price files are still missing after all fetch retries.")
            print(f"[daily] waiting {args.retry_wait} seconds before retry")
            sleep(args.retry_wait)
        dataset = build_dataset(config)
        if args.force_train or _should_retrain_today(config):
            artifacts, metrics = train_models(dataset, config)
            save_training_outputs(artifacts, metrics, config)
            _run_backtest(config, dataset=dataset)
        _score_daily(config, dataset=dataset)
        _report_daily(config)


def _run_backtest(config: AppConfig, dataset: pd.DataFrame | None = None) -> None:
    working = dataset if dataset is not None else load_or_build_dataset(config)
    result = run_walkforward_backtest(working, config)
    save_backtest_outputs(result, config)
    generate_backtest_report(
        result.terminal_predictions,
        result.touch_predictions,
        result.monthly_metrics,
        result.touch_task_metrics,
        result.summary_metrics,
        config,
    )


def _score_daily(config: AppConfig, dataset: pd.DataFrame | None = None) -> None:
    working = dataset if dataset is not None else load_or_build_dataset(config)
    artifacts = TrainingArtifacts.load(config.paths.model_dir / "latest.pkl")
    latest = latest_rows(working, config.effective_target_symbols)
    terminal_df, touch_df = predict_dataset(artifacts, latest, config)
    recent_quality = load_recent_quality(config)
    training_metrics = _load_json(config.paths.model_dir / "latest_metrics.json")
    health = build_model_health(artifacts, working, terminal_df, touch_df, training_metrics, recent_quality, config)
    terminal_df, touch_df = apply_confidence_flags(terminal_df, touch_df, health)
    terminal_df.to_csv(config.paths.report_dir / "daily_terminal_predictions.csv", index=False, encoding="utf-8-sig")
    touch_df.to_csv(config.paths.report_dir / "daily_touch_predictions.csv", index=False, encoding="utf-8-sig")
    terminal_df.to_csv(config.paths.report_dir / "latest_terminal_predictions.csv", index=False, encoding="utf-8-sig")
    touch_df.to_csv(config.paths.report_dir / "latest_touch_predictions.csv", index=False, encoding="utf-8-sig")
    save_model_health(config, health)


def _report_daily(config: AppConfig) -> None:
    terminal_df = pd.read_csv(config.paths.report_dir / "daily_terminal_predictions.csv")
    touch_df = pd.read_csv(config.paths.report_dir / "daily_touch_predictions.csv")
    health = _load_json(config.paths.report_dir / "daily_model_health.json")
    recent_quality = load_recent_quality(config)
    generate_prediction_report(terminal_df, touch_df, config, output_stem="daily_report", health=health, recent_quality=recent_quality)
    generate_prediction_report(terminal_df, touch_df, config, output_stem="latest_prediction", health=health, recent_quality=recent_quality)


def _critical_price_files_ready(config: AppConfig) -> bool:
    raw_root = config.paths.cache_dir / "raw"
    required = [raw_root / symbol / "price.csv" for symbol in config.effective_target_symbols]
    required.append(raw_root / "benchmarks" / f"{config.benchmark_symbol}.csv")
    for path in required:
        if not path.exists() or path.stat().st_size <= 8:
            return False
    return True


def _should_retrain_today(config: AppConfig) -> bool:
    model_path = config.paths.model_dir / "latest.pkl"
    if not model_path.exists():
        return True
    today = config.latest_as_of_date
    if today.strftime("%A").lower() == config.retrain_schedule.weekly_day.lower():
        return True
    return today.day == config.retrain_schedule.monthly_full_refresh_day


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
