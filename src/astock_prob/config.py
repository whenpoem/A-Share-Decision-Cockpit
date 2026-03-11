from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PathsConfig:
    cache_dir: Path
    dataset_dir: Path
    model_dir: Path
    report_dir: Path

    @classmethod
    def from_dict(cls, payload: Dict[str, str], root: Path) -> "PathsConfig":
        return cls(
            cache_dir=(root / payload["cache_dir"]).resolve(),
            dataset_dir=(root / payload["dataset_dir"]).resolve(),
            model_dir=(root / payload["model_dir"]).resolve(),
            report_dir=(root / payload["report_dir"]).resolve(),
        )

    def ensure_dirs(self) -> None:
        for path in (self.cache_dir, self.dataset_dir, self.model_dir, self.report_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingUniverseConfig:
    mode: str = "seed_only"
    seed_symbols: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)
    max_symbols: int = 200

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "TrainingUniverseConfig":
        return cls(
            mode=str(payload.get("mode", "seed_only")),
            seed_symbols=[str(item).zfill(6) for item in payload.get("seed_symbols", [])],
            sectors=[str(item) for item in payload.get("sectors", [])],
            max_symbols=int(payload.get("max_symbols", 200)),
        )


@dataclass
class UniverseFiltersConfig:
    min_listing_days: int = 250
    min_turnover: float = 30_000_000.0
    exclude_st: bool = True
    exclude_suspended: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "UniverseFiltersConfig":
        return cls(
            min_listing_days=int(payload.get("min_listing_days", 250)),
            min_turnover=float(payload.get("min_turnover", 30_000_000.0)),
            exclude_st=bool(payload.get("exclude_st", True)),
            exclude_suspended=bool(payload.get("exclude_suspended", True)),
        )


@dataclass
class RefreshScheduleConfig:
    daily_close_time: str = "15:15"
    daily_run_time: str = "15:25"

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "RefreshScheduleConfig":
        return cls(
            daily_close_time=str(payload.get("daily_close_time", "15:15")),
            daily_run_time=str(payload.get("daily_run_time", "15:25")),
        )


@dataclass
class RetrainScheduleConfig:
    weekly_day: str = "Saturday"
    monthly_full_refresh_day: int = 1

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "RetrainScheduleConfig":
        return cls(
            weekly_day=str(payload.get("weekly_day", "Saturday")),
            monthly_full_refresh_day=int(payload.get("monthly_full_refresh_day", 1)),
        )


@dataclass
class ConfidenceThresholdsConfig:
    ece_warn: float = 0.08
    sample_warn: int = 120
    drift_warn: float = 2.5

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ConfidenceThresholdsConfig":
        return cls(
            ece_warn=float(payload.get("ece_warn", 0.08)),
            sample_warn=int(payload.get("sample_warn", 120)),
            drift_warn=float(payload.get("drift_warn", 2.5)),
        )


@dataclass
class ModelConfig:
    random_seed: int = 42
    gbm_paths: int = 2000
    bootstrap_neighbors: int = 60
    bootstrap_min_history: int = 120
    validation_start: str = "2023-01-01"
    test_start: str = "2024-01-01"
    temperature_grid: List[float] = field(default_factory=lambda: [0.6, 0.8, 1.0, 1.2, 1.5, 2.0])
    ensemble_weight_step: float = 0.1
    calibration_min_samples: int = 200
    calibration_bins: int = 5

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ModelConfig":
        return cls(
            random_seed=int(payload.get("random_seed", 42)),
            gbm_paths=int(payload.get("gbm_paths", 2000)),
            bootstrap_neighbors=int(payload.get("bootstrap_neighbors", 60)),
            bootstrap_min_history=int(payload.get("bootstrap_min_history", 120)),
            validation_start=str(payload.get("validation_start", "2023-01-01")),
            test_start=str(payload.get("test_start", "2024-01-01")),
            temperature_grid=[float(item) for item in payload.get("temperature_grid", [0.6, 0.8, 1.0, 1.2, 1.5, 2.0])],
            ensemble_weight_step=float(payload.get("ensemble_weight_step", 0.1)),
            calibration_min_samples=int(payload.get("calibration_min_samples", 200)),
            calibration_bins=int(payload.get("calibration_bins", 5)),
        )


@dataclass
class ReportingConfig:
    format: str = "both"

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ReportingConfig":
        return cls(format=str(payload.get("format", "both")))


@dataclass
class AppConfig:
    symbols: List[str]
    start_date: str
    end_date: Optional[str]
    horizon_days: int
    return_grid: List[float]
    touch_thresholds: List[float]
    provider: str
    benchmark_symbol: str
    sector_proxy_symbols: Dict[str, str]
    paths: PathsConfig
    model: ModelConfig
    reporting: ReportingConfig
    project_root: Path
    target_symbols: List[str] = field(default_factory=list)
    training_universe: TrainingUniverseConfig = field(default_factory=TrainingUniverseConfig)
    universe_filters: UniverseFiltersConfig = field(default_factory=UniverseFiltersConfig)
    refresh_schedule: RefreshScheduleConfig = field(default_factory=RefreshScheduleConfig)
    retrain_schedule: RetrainScheduleConfig = field(default_factory=RetrainScheduleConfig)
    confidence_thresholds: ConfidenceThresholdsConfig = field(default_factory=ConfidenceThresholdsConfig)

    @classmethod
    def from_file(cls, config_path: str | Path) -> "AppConfig":
        config_file = Path(config_path).resolve()
        payload = json.loads(config_file.read_text(encoding="utf-8"))
        root = config_file.parent.parent
        symbols = [str(item).zfill(6) for item in payload.get("symbols", payload.get("target_symbols", []))]
        target_symbols = [str(item).zfill(6) for item in payload.get("target_symbols", symbols)]
        app = cls(
            symbols=symbols,
            start_date=payload["start_date"],
            end_date=payload.get("end_date"),
            horizon_days=int(payload["horizon_days"]),
            return_grid=[float(item) for item in payload["return_grid"]],
            touch_thresholds=[float(item) for item in payload["touch_thresholds"]],
            provider=str(payload.get("provider", "akshare")),
            benchmark_symbol=str(payload.get("benchmark_symbol", "000300")).zfill(6),
            sector_proxy_symbols={str(key).zfill(6): str(value).zfill(6) for key, value in payload.get("sector_proxy_symbols", {}).items()},
            paths=PathsConfig.from_dict(payload["paths"], root),
            model=ModelConfig.from_dict(payload.get("model", {})),
            reporting=ReportingConfig.from_dict(payload.get("reporting", {})),
            project_root=root.resolve(),
            target_symbols=target_symbols,
            training_universe=TrainingUniverseConfig.from_dict(payload.get("training_universe", {})),
            universe_filters=UniverseFiltersConfig.from_dict(payload.get("universe_filters", {})),
            refresh_schedule=RefreshScheduleConfig.from_dict(payload.get("refresh_schedule", {})),
            retrain_schedule=RetrainScheduleConfig.from_dict(payload.get("retrain_schedule", {})),
            confidence_thresholds=ConfidenceThresholdsConfig.from_dict(payload.get("confidence_thresholds", {})),
        )
        app.paths.ensure_dirs()
        return app

    @property
    def end_date_or_today(self) -> str:
        if self.end_date:
            return self.end_date
        return date.today().isoformat()

    @property
    def bucket_edges(self) -> List[float]:
        return sorted(self.return_grid)

    @property
    def bucket_labels(self) -> List[str]:
        edges = self.bucket_edges
        labels = [f"<{edges[0]:.0%}"]
        for left, right in zip(edges[:-1], edges[1:]):
            labels.append(f"[{left:.0%},{right:.0%})")
        labels.append(f">={edges[-1]:.0%}")
        return labels

    @property
    def latest_as_of_date(self) -> datetime:
        return datetime.fromisoformat(self.end_date_or_today)

    @property
    def all_seed_symbols(self) -> List[str]:
        ordered = list(dict.fromkeys(self.symbols + self.target_symbols + self.training_universe.seed_symbols))
        return [str(item).zfill(6) for item in ordered]

    @property
    def effective_target_symbols(self) -> List[str]:
        if self.target_symbols:
            return [str(item).zfill(6) for item in self.target_symbols]
        return [str(item).zfill(6) for item in self.symbols]
