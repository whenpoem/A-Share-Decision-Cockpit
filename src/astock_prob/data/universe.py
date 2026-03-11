from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from astock_prob.config import AppConfig
from astock_prob.data.providers import BaseDataProvider
from astock_prob.utils import save_json


def resolve_symbol_universe(config: AppConfig, provider: BaseDataProvider | None = None, use_cache: bool = True) -> List[str]:
    cached = load_universe_snapshot(config) if use_cache else {}
    if cached.get("symbols"):
        return [str(symbol).zfill(6) for symbol in cached["symbols"]]
    if provider is None:
        return config.all_seed_symbols
    discovered = provider.list_training_candidates(
        seed_symbols=config.all_seed_symbols,
        sector_keywords=config.training_universe.sectors,
        max_symbols=config.training_universe.max_symbols,
        exclude_st=config.universe_filters.exclude_st,
    )
    symbols = list(dict.fromkeys(config.effective_target_symbols + [str(symbol).zfill(6) for symbol in discovered]))
    save_universe_snapshot(
        config,
        {
            "target_symbols": config.effective_target_symbols,
            "symbols": symbols,
            "training_universe_mode": config.training_universe.mode,
            "sector_keywords": config.training_universe.sectors,
            "max_symbols": config.training_universe.max_symbols,
        },
    )
    return symbols


def load_universe_snapshot(config: AppConfig) -> Dict[str, object]:
    path = universe_snapshot_path(config)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_universe_snapshot(config: AppConfig, payload: Dict[str, object]) -> Path:
    path = universe_snapshot_path(config)
    save_json(path, payload)
    return path


def universe_snapshot_path(config: AppConfig) -> Path:
    return config.paths.cache_dir / "universe_snapshot.json"
