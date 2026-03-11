from __future__ import annotations

import numpy as np
import pandas as pd

from astock_prob.config import AppConfig
from astock_prob.utils import (
    annualize_volatility,
    exponential_weighted_mu_sigma,
    max_drawdown,
    safe_pct_change,
)


def build_feature_frame(symbol: str, payload: dict[str, pd.DataFrame], config: AppConfig) -> pd.DataFrame:
    price = payload["price"].copy()
    if price.empty:
        raise ValueError(f"No price data cached for {symbol}. Run fetch-data first.")
    price = price.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    price["date"] = pd.to_datetime(price["date"])
    price["symbol"] = symbol
    price["log_return_1"] = np.log(price["close"]).diff()
    price["return_1"] = safe_pct_change(price["close"], 1)
    price["return_5"] = safe_pct_change(price["close"], 5)
    price["return_10"] = safe_pct_change(price["close"], 10)
    price["return_20"] = safe_pct_change(price["close"], 20)
    price["return_60"] = safe_pct_change(price["close"], 60)
    price["volatility_20"] = annualize_volatility(price["return_1"], 20)
    price["volatility_60"] = annualize_volatility(price["return_1"], 60)
    ewma = exponential_weighted_mu_sigma(price["log_return_1"].fillna(0.0), halflife=20)
    price = pd.concat([price, ewma], axis=1)
    for window in (5, 10, 20, 30, 60, 120):
        ma_col = f"ma_{window}"
        price[ma_col] = price["close"].rolling(window).mean()
        price[f"dist_{ma_col}"] = price["close"] / price[ma_col] - 1.0
        price[f"volume_ratio_{window}"] = price["volume"] / price["volume"].rolling(window).mean()
        if "turnover" in price.columns:
            price[f"turnover_ratio_{window}"] = price["turnover"] / price["turnover"].rolling(window).mean()
    price["drawdown_20"] = max_drawdown(price["close"], 20)
    price["drawdown_60"] = max_drawdown(price["close"], 60)
    price["gap_open"] = price["open"] / price["close"].shift(1) - 1.0
    price["amplitude_ratio"] = (price["high"] - price["low"]) / price["close"].shift(1)
    price["atr_14"] = _average_true_range(price, 14)
    macd_frame = _macd(price["close"])
    price = pd.concat([price, macd_frame], axis=1)
    price["rsi_14"] = _rsi(price["close"], 14)
    kdj_frame = _kdj(price)
    price = pd.concat([price, kdj_frame], axis=1)
    benchmark = payload.get("benchmark", pd.DataFrame())
    if not benchmark.empty:
        benchmark = benchmark[["date", "close"]].rename(columns={"close": "benchmark_close"})
        benchmark["date"] = pd.to_datetime(benchmark["date"])
        benchmark["benchmark_return_1"] = safe_pct_change(benchmark["benchmark_close"], 1)
        benchmark["benchmark_return_20"] = safe_pct_change(benchmark["benchmark_close"], 20)
        benchmark["benchmark_return_60"] = safe_pct_change(benchmark["benchmark_close"], 60)
        benchmark["benchmark_volatility_20"] = annualize_volatility(benchmark["benchmark_return_1"], 20)
        benchmark["benchmark_ma_20"] = benchmark["benchmark_close"].rolling(20).mean()
        benchmark["benchmark_dist_ma_20"] = benchmark["benchmark_close"] / benchmark["benchmark_ma_20"] - 1.0
        price = price.merge(benchmark, on="date", how="left")
        price["excess_return_20"] = price["return_20"] - price["benchmark_return_20"]
        price["excess_return_60"] = price["return_60"] - price["benchmark_return_60"]
    sector = payload.get("sector", pd.DataFrame())
    if not sector.empty:
        sector = sector[["date", "close"]].rename(columns={"close": "sector_close"})
        sector["date"] = pd.to_datetime(sector["date"])
        sector["sector_return_20"] = safe_pct_change(sector["sector_close"], 20)
        sector["sector_return_60"] = safe_pct_change(sector["sector_close"], 60)
        price = price.merge(sector, on="date", how="left")
        price["sector_excess_20"] = price["return_20"] - price["sector_return_20"]
        price["sector_excess_60"] = price["return_60"] - price["sector_return_60"]
    valuation = payload.get("valuation", pd.DataFrame())
    if not valuation.empty:
        if "symbol" in valuation.columns:
            valuation["symbol"] = valuation["symbol"].astype(str).str.zfill(len(symbol))
        valuation["date"] = pd.to_datetime(valuation["date"])
        valuation_cols = [col for col in valuation.columns if col not in {"symbol"}]
        price = price.merge(valuation[valuation_cols], on="date", how="left")
        for col in ("pe", "pe_ttm", "pb", "ps", "ps_ttm", "dividend_yield"):
            if col in price.columns:
                price[f"{col}_pct_252"] = price[col].rolling(252).rank(pct=True)
    financial = payload.get("financial", pd.DataFrame())
    if not financial.empty:
        financial = financial.sort_values("announce_date").copy()
        if "symbol" in financial.columns:
            financial["symbol"] = financial["symbol"].astype(str).str.zfill(len(symbol))
        financial["announce_date"] = pd.to_datetime(financial["announce_date"])
        price = pd.merge_asof(
            price.sort_values("date"),
            financial,
            left_on="date",
            right_on="announce_date",
            by="symbol",
            direction="backward",
        )
    feature_frame = price.replace([np.inf, -np.inf], np.nan)
    feature_frame["symbol"] = symbol
    feature_frame["date"] = pd.to_datetime(feature_frame["date"])
    feature_frame["horizon_days"] = config.horizon_days
    return feature_frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "symbol",
        "date",
        "announce_date",
        "report_date",
        "horizon_days",
        "close",
        "open",
        "high",
        "low",
        "volume",
        "amount",
        "pct_chg",
        "change",
        "future_return_60",
        "future_high_60",
        "future_low_60",
        "terminal_bucket_id",
        "terminal_bucket",
        "is_trainable",
        "is_target",
        "is_universe_member",
    }
    return [
        col
        for col in frame.columns
        if col not in excluded
        and not col.startswith("touch_")
        and not col.startswith("path_")
        and pd.api.types.is_numeric_dtype(frame[col])
    ]


def _average_true_range(frame: pd.DataFrame, window: int) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    dif = fast_ema - slow_ema
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return pd.DataFrame({"macd_dif": dif, "macd_dea": dea, "macd_hist": hist})


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _kdj(frame: pd.DataFrame, window: int = 9) -> pd.DataFrame:
    lowest_low = frame["low"].rolling(window).min()
    highest_high = frame["high"].rolling(window).max()
    rsv = (frame["close"] - lowest_low) / (highest_high - lowest_low)
    k = rsv.ewm(com=2, adjust=False).mean() * 100.0
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j})
