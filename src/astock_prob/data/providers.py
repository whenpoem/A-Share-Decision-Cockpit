from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Optional

import numpy as np
import pandas as pd


class BaseDataProvider(ABC):
    def list_training_candidates(self, seed_symbols: list[str], sector_keywords: list[str], max_symbols: int, exclude_st: bool) -> list[str]:
        return list(dict.fromkeys(seed_symbols))

    @abstractmethod
    def fetch_price_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_benchmark_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_valuation_history(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_financial_history(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError


class CsvDataProvider(BaseDataProvider):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _load(self, relative_path: str, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
        full_path = self.root / relative_path
        if not full_path.exists():
            return pd.DataFrame()
        return pd.read_csv(full_path, parse_dates=parse_dates or [])

    def fetch_price_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        frame = self._load(f"{symbol}_price.csv", parse_dates=["date"])
        if frame.empty:
            return frame
        if "symbol" in frame.columns:
            frame["symbol"] = _standardize_code(frame["symbol"], symbol)
        return frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)].reset_index(drop=True)

    def fetch_benchmark_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        frame = self._load(f"{symbol}_benchmark.csv", parse_dates=["date"])
        if frame.empty:
            return frame
        return frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)].reset_index(drop=True)

    def fetch_valuation_history(self, symbol: str) -> pd.DataFrame:
        frame = self._load(f"{symbol}_valuation.csv", parse_dates=["date"])
        if "symbol" in frame.columns:
            frame["symbol"] = _standardize_code(frame["symbol"], symbol)
        return frame

    def fetch_financial_history(self, symbol: str) -> pd.DataFrame:
        frame = self._load(f"{symbol}_financial.csv", parse_dates=["announce_date", "report_date"])
        if "symbol" in frame.columns:
            frame["symbol"] = _standardize_code(frame["symbol"], symbol)
        return frame

    def list_training_candidates(self, seed_symbols: list[str], sector_keywords: list[str], max_symbols: int, exclude_st: bool) -> list[str]:
        universe_path = self.root / "universe_members.csv"
        if universe_path.exists():
            frame = pd.read_csv(universe_path)
            if "sector" in frame.columns and sector_keywords:
                mask = frame["sector"].astype(str).apply(lambda value: any(keyword in value for keyword in sector_keywords))
                frame = frame.loc[mask]
            if "name" in frame.columns and exclude_st:
                frame = frame.loc[~frame["name"].astype(str).str.contains("ST", case=False, na=False)]
            if "symbol" in frame.columns:
                members = frame["symbol"].astype(str).str.zfill(6).tolist()
            else:
                members = []
        else:
            members = [path.name.replace("_price.csv", "") for path in self.root.glob("*_price.csv")]
        ordered = list(dict.fromkeys(seed_symbols + members))
        return ordered[:max_symbols] if max_symbols > 0 else ordered


class AkshareDataProvider(BaseDataProvider):
    def __init__(self) -> None:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("AkShare is not installed. Run `python -m pip install akshare`.") from exc
        self.ak = ak

    def fetch_price_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        chunks = self._fetch_stock_hist_chunks(symbol, start_date, end_date)
        if not chunks:
            return pd.DataFrame()
        frame = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        return frame

    def fetch_benchmark_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        candidates = [
            ("index_zh_a_hist", {"symbol": symbol, "period": "daily", "start_date": self._date_compact(start_date), "end_date": self._date_compact(end_date)}),
            ("stock_zh_index_daily_em", {"symbol": self._benchmark_prefix(symbol)}),
        ]
        for func_name, kwargs in candidates:
            func = getattr(self.ak, func_name, None)
            if func is None:
                continue
            try:
                raw = func(**kwargs)
            except Exception:
                continue
            frame = _normalize_hist_frame(raw, symbol=symbol, date_range=(start_date, end_date), with_code_column=False)
            if not frame.empty:
                return frame
        return pd.DataFrame()

    def fetch_valuation_history(self, symbol: str) -> pd.DataFrame:
        primary = getattr(self.ak, "stock_a_indicator_lg", None)
        if primary is not None:
            try:
                raw = primary(symbol=symbol)
            except Exception:
                raw = pd.DataFrame()
            frame = _normalize_primary_valuation(raw, symbol)
            if not frame.empty:
                return frame
        fallback = getattr(self.ak, "stock_value_em", None)
        if fallback is None:
            return pd.DataFrame()
        try:
            raw = fallback(symbol=symbol)
        except Exception:
            return pd.DataFrame()
        return _normalize_fallback_valuation(raw, symbol)

    def fetch_financial_history(self, symbol: str) -> pd.DataFrame:
        candidates = [
            ("stock_financial_abstract_ths", {"symbol": symbol, "indicator": "\u6309\u62a5\u544a\u671f"}),
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
            frame = _normalize_financial_frame(raw, symbol)
            if not frame.empty:
                return frame
        return pd.DataFrame()

    def list_training_candidates(self, seed_symbols: list[str], sector_keywords: list[str], max_symbols: int, exclude_st: bool) -> list[str]:
        members: list[str] = list(seed_symbols)
        if sector_keywords:
            board_func = getattr(self.ak, "stock_board_industry_name_em", None)
            cons_func = getattr(self.ak, "stock_board_industry_cons_em", None)
            if board_func is not None and cons_func is not None:
                try:
                    board_frame = board_func()
                except Exception:
                    board_frame = pd.DataFrame()
                board_names = _match_board_names(board_frame, sector_keywords)
                for board_name in board_names:
                    try:
                        cons = cons_func(symbol=board_name)
                    except Exception:
                        continue
                    members.extend(_extract_symbols_from_constituents(cons, exclude_st))
                    if max_symbols > 0 and len(set(members)) >= max_symbols:
                        break
        ordered = list(dict.fromkeys(symbol.zfill(6) for symbol in members if str(symbol).isdigit()))
        return ordered[:max_symbols] if max_symbols > 0 else ordered

    @staticmethod
    def _date_compact(value: str) -> str:
        return value.replace("-", "")

    @staticmethod
    def _benchmark_prefix(symbol: str) -> str:
        if symbol.startswith(("0", "3")):
            return f"sz{symbol}"
        return f"sh{symbol}"

    def _fetch_stock_hist_chunks(self, symbol: str, start_date: str, end_date: str) -> list[pd.DataFrame]:
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        cursor = start
        chunks: list[pd.DataFrame] = []
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=365), end)
            raw = pd.DataFrame()
            for attempt in range(3):
                try:
                    raw = self.ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=cursor.strftime("%Y%m%d"),
                        end_date=chunk_end.strftime("%Y%m%d"),
                        adjust="qfq",
                    )
                    break
                except Exception:
                    if attempt == 2:
                        raw = pd.DataFrame()
                    else:
                        sleep(1.0)
            normalized = _normalize_hist_frame(
                raw,
                symbol=symbol,
                date_range=(cursor.isoformat(), chunk_end.isoformat()),
                with_code_column=True,
            )
            if not normalized.empty:
                chunks.append(normalized)
            cursor = chunk_end + timedelta(days=1)
        return chunks


def build_provider(name: str, source_root: Optional[str] = None) -> BaseDataProvider:
    lowered = name.lower()
    if lowered == "csv":
        if not source_root:
            raise ValueError("CSV provider requires `source_root`.")
        return CsvDataProvider(source_root)
    if lowered == "akshare":
        return AkshareDataProvider()
    raise ValueError(f"Unsupported provider: {name}")


def _normalize_hist_frame(
    raw: pd.DataFrame,
    *,
    symbol: str,
    date_range: tuple[str, str],
    with_code_column: bool,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    rename_candidates = {
        "\u65e5\u671f": "date",
        "\u5f00\u76d8": "open",
        "\u6536\u76d8": "close",
        "\u6700\u9ad8": "high",
        "\u6700\u4f4e": "low",
        "\u6210\u4ea4\u91cf": "volume",
        "\u6210\u4ea4\u989d": "amount",
        "\u632f\u5e45": "amplitude",
        "\u6da8\u8dcc\u5e45": "pct_chg",
        "\u6da8\u8dcc\u989d": "change",
        "\u6362\u624b\u7387": "turnover",
        "\u80a1\u7968\u4ee3\u7801": "raw_symbol",
    }
    frame = frame.rename(columns=rename_candidates)
    if {"date", "open", "high", "low", "close"}.issubset(frame.columns):
        normalized = frame.copy()
    else:
        columns = list(frame.columns)
        if with_code_column and len(columns) >= 12:
            position_map = {
                "date": 0,
                "open": 2,
                "close": 3,
                "high": 4,
                "low": 5,
                "volume": 6,
                "amount": 7,
                "amplitude": 8,
                "pct_chg": 9,
                "change": 10,
                "turnover": 11,
            }
        elif len(columns) >= 7:
            position_map = {
                "date": 0,
                "open": 1,
                "close": 2,
                "high": 3,
                "low": 4,
                "volume": 5,
                "amount": 6,
            }
        else:
            return pd.DataFrame()
        normalized = pd.DataFrame({target: frame.iloc[:, idx] for target, idx in position_map.items() if idx < len(columns)})
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["symbol"] = symbol
    for col in ("open", "high", "low", "close", "volume", "amount", "amplitude", "pct_chg", "change", "turnover"):
        if col in normalized.columns:
            normalized[col] = _parse_numeric_series(normalized[col])
    start_date, end_date = date_range
    normalized = normalized.dropna(subset=["date", "close"])
    normalized = normalized[(normalized["date"] >= pd.Timestamp(start_date)) & (normalized["date"] <= pd.Timestamp(end_date))]
    keep_cols = [
        col
        for col in ["date", "symbol", "open", "high", "low", "close", "volume", "amount", "amplitude", "pct_chg", "change", "turnover"]
        if col in normalized.columns
    ]
    return normalized[keep_cols].sort_values("date").reset_index(drop=True)


def _normalize_primary_valuation(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    rename_candidates = {
        "trade_date": "date",
        "date": "date",
        "pe": "pe",
        "pe_ttm": "pe_ttm",
        "pb": "pb",
        "ps": "ps",
        "ps_ttm": "ps_ttm",
        "dv_ratio": "dividend_yield",
        "total_mv": "total_market_value",
        "circ_mv": "float_market_value",
    }
    frame = frame.rename(columns=rename_candidates)
    if "date" not in frame.columns:
        frame["date"] = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    else:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = symbol
    for col in ("pe", "pe_ttm", "pb", "ps", "ps_ttm", "dividend_yield", "total_market_value", "float_market_value"):
        if col in frame.columns:
            frame[col] = _parse_numeric_series(frame[col])
    keep_cols = [col for col in ["date", "symbol", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dividend_yield", "total_market_value", "float_market_value"] if col in frame.columns]
    return frame[keep_cols].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _normalize_fallback_valuation(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    columns = list(frame.columns)
    normalized = pd.DataFrame(
        {
            "date": pd.to_datetime(frame.iloc[:, 0], errors="coerce"),
            "symbol": symbol,
        }
    )
    if len(columns) > 3:
        normalized["total_market_value"] = _parse_numeric_series(frame.iloc[:, 3])
    if len(columns) > 4:
        normalized["float_market_value"] = _parse_numeric_series(frame.iloc[:, 4])
    if len(columns) > 7:
        normalized["pe_ttm"] = _parse_numeric_series(frame.iloc[:, 7])
    if len(columns) > 8:
        normalized["pe"] = _parse_numeric_series(frame.iloc[:, 8])
    if len(columns) > 9:
        normalized["pb"] = _parse_numeric_series(frame.iloc[:, 9])
    if len(columns) > 12:
        normalized["ps"] = _parse_numeric_series(frame.iloc[:, 12])
    return normalized.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _normalize_financial_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    columns = list(frame.columns)
    if len(columns) < 10:
        return pd.DataFrame()
    report_date = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    normalized = pd.DataFrame(
        {
            "symbol": symbol,
            "report_date": report_date,
            "announce_date": report_date.map(_estimate_announce_date),
            "net_profit": _parse_numeric_series(frame.iloc[:, 1]) if len(columns) > 1 else np.nan,
            "net_profit_yoy": _parse_numeric_series(frame.iloc[:, 2]) if len(columns) > 2 else np.nan,
            "revenue": _parse_numeric_series(frame.iloc[:, 5]) if len(columns) > 5 else np.nan,
            "revenue_yoy": _parse_numeric_series(frame.iloc[:, 6]) if len(columns) > 6 else np.nan,
            "gross_margin": _parse_numeric_series(frame.iloc[:, 13]) if len(columns) > 13 else np.nan,
            "roe": _parse_numeric_series(frame.iloc[:, 14]) if len(columns) > 14 else np.nan,
            "current_ratio": _parse_numeric_series(frame.iloc[:, 20]) if len(columns) > 20 else np.nan,
            "quick_ratio": _parse_numeric_series(frame.iloc[:, 21]) if len(columns) > 21 else np.nan,
            "debt_ratio": _parse_numeric_series(frame.iloc[:, 24]) if len(columns) > 24 else np.nan,
        }
    )
    return normalized.dropna(subset=["report_date"]).sort_values("announce_date").reset_index(drop=True)


def _estimate_announce_date(report_date: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(report_date):
        return pd.NaT
    month_day = (report_date.month, report_date.day)
    if month_day == (3, 31):
        return report_date + pd.Timedelta(days=30)
    if month_day == (6, 30):
        return report_date + pd.Timedelta(days=62)
    if month_day == (9, 30):
        return report_date + pd.Timedelta(days=31)
    if month_day == (12, 31):
        return report_date + pd.Timedelta(days=120)
    return report_date + pd.Timedelta(days=45)


def _standardize_code(series: pd.Series, symbol: str) -> pd.Series:
    return series.astype(str).str.zfill(len(symbol))


def _match_board_names(frame: pd.DataFrame, sector_keywords: list[str]) -> list[str]:
    if frame is None or frame.empty:
        return []
    names = []
    for _, row in frame.iterrows():
        row_values = [str(value) for value in row.tolist()]
        board_name = next((value for value in row_values if any(keyword in value for keyword in sector_keywords)), None)
        if board_name is None:
            continue
        if any(keyword in board_name for keyword in sector_keywords):
            names.append(board_name)
    return list(dict.fromkeys(names))


def _extract_symbols_from_constituents(frame: pd.DataFrame, exclude_st: bool) -> list[str]:
    if frame is None or frame.empty:
        return []
    output: list[str] = []
    for _, row in frame.iterrows():
        values = [str(value).strip() for value in row.tolist()]
        symbol = next((value for value in values if value.isdigit() and len(value) in {5, 6}), None)
        if symbol is None:
            continue
        name = next((value for value in values if value and not value.replace(".", "", 1).isdigit()), "")
        if exclude_st and "ST" in name.upper():
            continue
        output.append(symbol.zfill(6))
    return output


def _parse_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"--": np.nan, "nan": np.nan, "None": np.nan, "False": np.nan, "True": np.nan, "": np.nan})
    multiplier = pd.Series(1.0, index=cleaned.index)
    multiplier = multiplier.mask(cleaned.str.endswith("亿", na=False), 1e8)
    multiplier = multiplier.mask(cleaned.str.endswith("万", na=False), 1e4)
    percent_mask = cleaned.str.endswith("%", na=False)
    numeric = cleaned.str.replace(",", "", regex=False)
    numeric = numeric.str.replace("亿", "", regex=False).str.replace("万", "", regex=False).str.replace("%", "", regex=False)
    values = pd.to_numeric(numeric, errors="coerce")
    values = values * multiplier
    values = values.mask(percent_mask, values / 100.0)
    return values
