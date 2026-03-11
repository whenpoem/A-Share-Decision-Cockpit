from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, Optional

import pandas as pd

from astock_prob.config import AppConfig
from astock_prob.data.providers import BaseDataProvider
from astock_prob.data.storage import load_frame, save_frame
from astock_prob.data.universe import resolve_symbol_universe
from astock_prob.utils import save_json


def fetch_all_data(config: AppConfig, provider: BaseDataProvider) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    status: Dict[str, Dict[str, object]] = {}
    start_date = config.start_date
    end_date = config.end_date_or_today
    raw_root = config.paths.cache_dir / "raw"
    benchmark_root = raw_root / "benchmarks"
    symbols = resolve_symbol_universe(config, provider, use_cache=False)
    for symbol in symbols:
        symbol_root = raw_root / symbol
        outputs[f"{symbol}:price"] = _fetch_and_save(
            label=f"{symbol}:price",
            fetcher=lambda symbol=symbol: provider.fetch_price_history(symbol, start_date, end_date),
            path=symbol_root / "price.csv",
            status=status,
            critical=True,
        )
        outputs[f"{symbol}:valuation"] = _fetch_and_save(
            label=f"{symbol}:valuation",
            fetcher=lambda symbol=symbol: provider.fetch_valuation_history(symbol),
            path=symbol_root / "valuation.csv",
            status=status,
        )
        outputs[f"{symbol}:financial"] = _fetch_and_save(
            label=f"{symbol}:financial",
            fetcher=lambda symbol=symbol: provider.fetch_financial_history(symbol),
            path=symbol_root / "financial.csv",
            status=status,
        )
        sector_symbol = config.sector_proxy_symbols.get(symbol)
        if sector_symbol:
            outputs[f"{symbol}:sector"] = _fetch_and_save(
                label=f"{symbol}:sector:{sector_symbol}",
                fetcher=lambda sector_symbol=sector_symbol: provider.fetch_benchmark_history(sector_symbol, start_date, end_date),
                path=benchmark_root / f"{sector_symbol}.csv",
                status=status,
            )
    outputs["benchmark"] = _fetch_and_save(
        label=f"benchmark:{config.benchmark_symbol}",
        fetcher=lambda: provider.fetch_benchmark_history(config.benchmark_symbol, start_date, end_date),
        path=benchmark_root / f"{config.benchmark_symbol}.csv",
        status=status,
        critical=True,
    )
    save_json(config.paths.report_dir / "fetch_status.json", status)
    return outputs


def load_symbol_raw_data(config: AppConfig, symbol: str) -> Dict[str, pd.DataFrame]:
    raw_root = config.paths.cache_dir / "raw"
    symbol_root = raw_root / symbol
    benchmark_root = raw_root / "benchmarks"
    payload = {
        "price": _safe_load(symbol_root / "price.csv", parse_dates=["date"]),
        "valuation": _safe_load(symbol_root / "valuation.csv", parse_dates=["date"]),
        "financial": _safe_load(symbol_root / "financial.csv", parse_dates=["announce_date", "report_date"]),
        "benchmark": _safe_load(benchmark_root / f"{config.benchmark_symbol}.csv", parse_dates=["date"]),
    }
    sector_symbol = config.sector_proxy_symbols.get(symbol)
    if sector_symbol:
        payload["sector"] = _safe_load(benchmark_root / f"{sector_symbol}.csv", parse_dates=["date"])
    else:
        payload["sector"] = pd.DataFrame()
    return payload


def _safe_load(path: Path, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return load_frame(path, parse_dates=parse_dates)


def _fetch_and_save(
    label: str,
    fetcher,
    path: Path,
    status: Dict[str, Dict[str, object]],
    critical: bool = False,
) -> Path:
    print(f"[fetch] start {label}")
    started = perf_counter()
    try:
        frame = fetcher()
        if frame is None:
            frame = pd.DataFrame()
        elapsed = round(perf_counter() - started, 2)
        rows = int(len(frame))
        if rows == 0 and path.exists() and path.stat().st_size > 8:
            status[label] = {
                "status": "empty_preserved",
                "rows": 0,
                "seconds": elapsed,
                "path": str(path),
                "critical": critical,
            }
            print(f"[fetch] empty {label} seconds={elapsed} preserved_existing={path}")
            return path
        target = save_frame(frame, path)
        status[label] = {
            "status": "ok",
            "rows": rows,
            "seconds": elapsed,
            "path": str(target),
            "critical": critical,
        }
        print(f"[fetch] done {label} rows={rows} seconds={elapsed}")
        return target
    except Exception as exc:
        elapsed = round(perf_counter() - started, 2)
        if path.exists() and path.stat().st_size > 8:
            status[label] = {
                "status": "error_preserved",
                "rows": 0,
                "seconds": elapsed,
                "path": str(path),
                "critical": critical,
                "error": str(exc),
            }
            print(f"[fetch] error {label} seconds={elapsed} preserved_existing={path} error={exc}")
            return path
        target = save_frame(pd.DataFrame(), path)
        status[label] = {
            "status": "error",
            "rows": 0,
            "seconds": elapsed,
            "path": str(target),
            "critical": critical,
            "error": str(exc),
        }
        print(f"[fetch] error {label} seconds={elapsed} error={exc}")
        return target
