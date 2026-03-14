from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from engine.config import Settings
from engine.storage.db import StateStore
from engine.types import FinancialSnapshot, PriorSignal


@dataclass
class MarketSnapshot:
    bars: dict[str, pd.DataFrame]
    financials: dict[str, FinancialSnapshot]
    priors: list[PriorSignal]
    as_of_date: datetime


class BaseMarketProvider:
    def list_symbols(self, limit: int) -> list[dict[str, str]]:
        raise NotImplementedError

    def fetch_price_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_financial_snapshot(self, symbol: str) -> FinancialSnapshot:
        raise NotImplementedError


class SampleMarketProvider(BaseMarketProvider):
    def __init__(self, symbols: Optional[Iterable[str]] = None) -> None:
        self.symbols = list(symbols or [])

    def list_symbols(self, limit: int) -> list[dict[str, str]]:
        watchlist = self.symbols or [
            "000001",
            "000333",
            "000651",
            "000858",
            "002415",
            "002594",
            "300750",
            "600036",
            "600519",
            "601318",
            "601398",
            "601899",
        ]
        sectors = [
            "银行",
            "家电",
            "白酒",
            "消费电子",
            "新能源车",
            "电池",
            "半导体",
            "券商",
            "资源",
            "保险",
            "石化",
            "有色",
        ]
        return [
            {"symbol": code, "name": f"样本{code[-3:]}", "sector": sectors[idx % len(sectors)]}
            for idx, code in enumerate(watchlist[:limit])
        ]

    def fetch_price_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.bdate_range(start_date, end_date)
        seed = int(symbol[-3:])
        rng = np.random.default_rng(seed)
        t = np.arange(len(dates))
        base = 12 + (seed % 50)
        trend = np.sin(t / 18.0 + seed / 17.0) * 0.9 + np.cos(t / 7.0) * 0.35
        noise = rng.normal(0.0, 0.15, size=len(dates)).cumsum()
        close = np.maximum(base + trend + noise, 2.0)
        open_price = close * (1.0 + rng.normal(0, 0.004, size=len(dates)))
        high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.003, 0.018, size=len(dates)))
        low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.003, 0.018, size=len(dates)))
        volume = rng.integers(5_000_000, 30_000_000, size=len(dates))
        amount = volume * close
        return pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
                "turnover": rng.uniform(1.0, 6.0, size=len(dates)),
            }
        )

    def fetch_financial_snapshot(self, symbol: str) -> FinancialSnapshot:
        seed = int(symbol[-3:])
        return FinancialSnapshot(
            symbol=symbol,
            as_of_date=datetime.utcnow() - timedelta(days=20),
            revenue_yoy=((seed % 17) - 6) / 25.0,
            net_profit_yoy=((seed % 21) - 8) / 22.0,
            roe=0.08 + (seed % 12) / 100.0,
            debt_ratio=0.25 + (seed % 14) / 20.0,
        )


class AkshareMarketProvider(BaseMarketProvider):
    def __init__(self) -> None:
        import akshare as ak

        self.ak = ak

    def list_symbols(self, limit: int) -> list[dict[str, str]]:
        raw = self.ak.stock_zh_a_spot_em()
        frame = raw.rename(
            columns={
                "代码": "symbol",
                "名称": "name",
                "总市值": "market_cap",
                "所属行业": "sector",
            }
        )
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        if "sector" not in frame.columns:
            frame["sector"] = "未知"
        frame = frame[~frame["name"].astype(str).str.contains("ST", case=False, na=False)]
        frame = frame.sort_values("market_cap", ascending=False).head(limit)
        return frame[["symbol", "name", "sector"]].to_dict(orient="records")

    def fetch_price_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raw = self.ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq",
        )
        frame = raw.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover",
            }
        )
        frame["date"] = pd.to_datetime(frame["date"])
        frame["symbol"] = symbol
        return frame[["date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover"]]

    def fetch_financial_snapshot(self, symbol: str) -> FinancialSnapshot:
        candidates = [
            ("stock_financial_abstract_ths", {"symbol": symbol, "indicator": "按报告期"}),
            ("stock_financial_analysis_indicator", {"symbol": symbol}),
        ]
        for func_name, kwargs in candidates:
            func = getattr(self.ak, func_name, None)
            if func is None:
                continue
            try:
                raw = func(**kwargs)
            except Exception:
                continue
            if raw is None or raw.empty:
                continue
            row = raw.iloc[-1]
            report_date = pd.to_datetime(row.iloc[0], errors="coerce")
            return FinancialSnapshot(
                symbol=symbol,
                as_of_date=(report_date.to_pydatetime() if not pd.isna(report_date) else datetime.utcnow()),
                revenue_yoy=float(pd.to_numeric(row.iloc[6], errors="coerce") or 0.0),
                net_profit_yoy=float(pd.to_numeric(row.iloc[2], errors="coerce") or 0.0),
                roe=float(pd.to_numeric(row.iloc[14], errors="coerce") or 0.0),
                debt_ratio=float(pd.to_numeric(row.iloc[15], errors="coerce") or 0.0),
            )
        return SampleMarketProvider([symbol]).fetch_financial_snapshot(symbol)


class MarketService:
    def __init__(self, settings: Settings, store: StateStore) -> None:
        self.settings = settings
        self.store = store
        self.provider = self._build_provider()

    def _build_provider(self) -> BaseMarketProvider:
        if self.settings.market_provider == "sample":
            return SampleMarketProvider(self.settings.default_watchlist)
        try:
            return AkshareMarketProvider()
        except Exception:
            if not self.settings.fallback_to_sample_market:
                raise
            return SampleMarketProvider(self.settings.default_watchlist)

    def refresh_market(
        self,
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        watchlist: Optional[list[str]] = None,
    ) -> MarketSnapshot:
        end_value = end_date or datetime.utcnow().date().isoformat()
        provider_name = type(self.provider).__name__
        try:
            symbols = self.provider.list_symbols(self.settings.market_universe_size)
        except Exception as exc:
            raise RuntimeError(
                f"[market:{provider_name}] list_symbols failed for end_date={end_value}: {exc}"
            ) from exc
        if watchlist:
            normalized = {symbol.zfill(6) for symbol in watchlist}
            symbols = [item for item in symbols if item["symbol"] in normalized]
            existing = {item["symbol"] for item in symbols}
            for symbol in sorted(normalized - existing):
                symbols.append({"symbol": symbol, "name": symbol, "sector": "Unknown"})
        bars: dict[str, pd.DataFrame] = {}
        financials: dict[str, FinancialSnapshot] = {}
        for meta in symbols:
            symbol = meta["symbol"]
            try:
                frame = self.provider.fetch_price_history(symbol, start_date, end_value)
            except Exception as exc:
                raise RuntimeError(
                    f"[market:{provider_name}] fetch_price_history failed for symbol={symbol}, "
                    f"start_date={start_date}, end_date={end_value}: {exc}"
                ) from exc
            frame = frame.sort_values("date").reset_index(drop=True)
            frame["name"] = meta["name"]
            frame["sector"] = meta["sector"]
            bars[symbol] = frame
            try:
                financials[symbol] = self.provider.fetch_financial_snapshot(symbol)
            except Exception as exc:
                raise RuntimeError(
                    f"[market:{provider_name}] fetch_financial_snapshot failed for symbol={symbol}: {exc}"
                ) from exc
            self.store.upsert_symbol(symbol, meta["name"], meta["sector"])
            self._save_frame(self.settings.market_storage / f"{symbol}.parquet", frame)
        priors, as_of_date = build_prior_signals(bars, financials)
        return MarketSnapshot(bars=bars, financials=financials, priors=priors, as_of_date=as_of_date)

    @staticmethod
    def _save_frame(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def build_prior_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    financials: dict[str, FinancialSnapshot],
) -> tuple[list[PriorSignal], datetime]:
    feature_rows = []
    latest_date = max(frame["date"].max() for frame in bars_by_symbol.values())
    market_returns = []
    market_vols = []
    for symbol, frame in bars_by_symbol.items():
        working = frame.copy().sort_values("date")
        working["ret_1"] = working["close"].pct_change()
        working["ret_5"] = working["close"].pct_change(5)
        working["ret_20"] = working["close"].pct_change(20)
        working["vol_20"] = working["ret_1"].rolling(20).std()
        working["amount_20"] = working["amount"].rolling(20).mean()
        working["drawdown_20"] = working["close"] / working["close"].rolling(20).max() - 1.0
        working["breakout_ratio"] = working["close"] / working["high"].rolling(20).max() - 1.0
        latest = working.iloc[-1]
        market_returns.append(float(latest.get("ret_20", 0.0) or 0.0))
        market_vols.append(float(latest.get("vol_20", 0.0) or 0.0))
        feature_rows.append(
            {
                "symbol": symbol,
                "name": latest["name"],
                "sector": latest["sector"],
                "date": latest["date"],
                "latest_close": float(latest["close"]),
                "ret_5": float(latest.get("ret_5", 0.0) or 0.0),
                "ret_20": float(latest.get("ret_20", 0.0) or 0.0),
                "vol_20": float(latest.get("vol_20", 0.0) or 0.0),
                "amount_20": float(latest.get("amount_20", 0.0) or 0.0),
                "drawdown_20": float(latest.get("drawdown_20", 0.0) or 0.0),
                "breakout_ratio": float(latest.get("breakout_ratio", 0.0) or 0.0),
                "roe": financials[symbol].roe,
                "net_profit_yoy": financials[symbol].net_profit_yoy,
                "revenue_yoy": financials[symbol].revenue_yoy,
                "debt_ratio": financials[symbol].debt_ratio,
            }
        )
    feature_frame = pd.DataFrame(feature_rows)
    regime = classify_market_regime(_safe_mean(market_returns), _safe_mean(market_vols))
    for column in ["ret_5", "ret_20", "vol_20", "amount_20", "drawdown_20", "breakout_ratio", "roe", "net_profit_yoy", "revenue_yoy", "debt_ratio"]:
        feature_frame[f"{column}_z"] = _zscore(feature_frame[column])
    priors = []
    for row in feature_frame.itertuples(index=False):
        trend = _sigmoid(0.95 * row.ret_20_z + 0.55 * row.roe_z + 0.45 * row.net_profit_yoy_z)
        reversal = _sigmoid((-0.75 * row.ret_5_z) + (-0.60 * row.drawdown_20_z) + 0.35 * row.amount_20_z)
        breakout = _sigmoid(1.10 * row.breakout_ratio_z + 0.45 * row.ret_20_z)
        downside = _sigmoid(0.80 * row.vol_20_z + 0.85 * (-row.drawdown_20_z) + 0.35 * row.debt_ratio_z)
        liquidity = _sigmoid(1.15 * row.amount_20_z)
        event_sensitivity = _sigmoid(0.7 * abs(row.ret_5_z) + 0.65 * row.vol_20_z)
        regime_alignment = _regime_alignment(regime, trend, reversal, downside)
        prior_long = np.clip(
            0.30 * trend
            + 0.20 * reversal
            + 0.25 * breakout
            + 0.15 * liquidity
            + 0.10 * regime_alignment
            - 0.20 * downside,
            0.0,
            1.0,
        )
        prior_avoid = np.clip(0.55 * downside + 0.20 * (1.0 - liquidity) + 0.25 * event_sensitivity, 0.0, 1.0)
        priors.append(
            PriorSignal(
                symbol=row.symbol,
                name=row.name,
                sector=row.sector,
                as_of_date=row.date.to_pydatetime(),
                latest_close=float(row.latest_close),
                trend_score=float(trend),
                reversal_score=float(reversal),
                breakout_score=float(breakout),
                downside_risk_score=float(downside),
                liquidity_score=float(liquidity),
                event_sensitivity_score=float(event_sensitivity),
                regime_alignment_score=float(regime_alignment),
                prior_long_score=float(prior_long),
                prior_avoid_score=float(prior_avoid),
                market_regime=regime,
            )
        )
    priors.sort(key=lambda item: item.prior_long_score, reverse=True)
    return priors, latest_date.to_pydatetime()


def classify_market_regime(avg_return: float, avg_vol: float) -> str:
    if avg_return > 0.05 and avg_vol < 0.035:
        return "risk_on"
    if avg_return < -0.03 or avg_vol > 0.055:
        return "risk_off"
    return "neutral"


def _regime_alignment(regime: str, trend: float, reversal: float, downside: float) -> float:
    if regime == "risk_on":
        return float(np.clip(0.7 * trend + 0.2 * reversal - 0.2 * downside, 0.0, 1.0))
    if regime == "risk_off":
        return float(np.clip(0.6 * (1.0 - downside) + 0.2 * reversal, 0.0, 1.0))
    return float(np.clip(0.5 * trend + 0.3 * reversal + 0.2 * (1.0 - downside), 0.0, 1.0))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def _safe_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.isnan(array).all():
        return 0.0
    return float(np.nanmean(array))
