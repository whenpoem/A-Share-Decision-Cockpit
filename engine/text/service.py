from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from engine.config import Settings
from engine.storage.db import StateStore
from engine.types import FinancialSnapshot, PositionState, PriorSignal, SymbolEventPack, TextEvent


class BaseTextProvider:
    provider_name = "base"

    def fetch_events(self, symbol: str, as_of_date: datetime) -> list[TextEvent]:
        raise NotImplementedError


class NullTextProvider(BaseTextProvider):
    provider_name = "derived-only"

    def fetch_events(self, symbol: str, as_of_date: datetime) -> list[TextEvent]:
        return []


class AkshareTextProvider(BaseTextProvider):
    provider_name = "akshare"

    def __init__(self, settings: Settings) -> None:
        import akshare as ak

        self.ak = ak
        self.settings = settings

    def fetch_events(self, symbol: str, as_of_date: datetime) -> list[TextEvent]:
        events: list[TextEvent] = []
        events.extend(self._fetch_news(symbol, as_of_date))
        events.extend(self._fetch_announcements(symbol, as_of_date))
        return events

    def _fetch_news(self, symbol: str, as_of_date: datetime) -> list[TextEvent]:
        try:
            frame = self.ak.stock_news_em(symbol=symbol)
        except Exception:
            return []
        if frame is None or frame.empty:
            return []
        frame = frame.copy()
        frame["发布时间"] = pd.to_datetime(frame["发布时间"], errors="coerce")
        frame = frame.dropna(subset=["发布时间"])
        frame = frame[frame["发布时间"] <= as_of_date]
        frame = frame.sort_values("发布时间", ascending=False).head(
            self.settings.max_news_per_symbol
        )
        events: list[TextEvent] = []
        for row in frame.itertuples(index=False):
            title = _clean_text(getattr(row, "新闻标题", ""))
            content = _clean_text(getattr(row, "新闻内容", "")) or title
            source = _clean_text(getattr(row, "文章来源", "")) or "eastmoney"
            url = _clean_text(getattr(row, "新闻链接", ""))
            events.append(
                build_event(
                    symbol,
                    published_at=row.发布时间.to_pydatetime(),
                    source_type="news",
                    source_name=f"eastmoney:{source}",
                    title=title,
                    content=content,
                    url=url,
                    importance_hint=0.62,
                    sentiment_hint=0.0,
                )
            )
        return events

    def _fetch_announcements(self, symbol: str, as_of_date: datetime) -> list[TextEvent]:
        start_date = (as_of_date - timedelta(days=self.settings.text_lookback_days)).strftime(
            "%Y%m%d"
        )
        end_date = as_of_date.strftime("%Y%m%d")
        try:
            frame = self.ak.stock_zh_a_disclosure_report_cninfo(
                symbol=symbol,
                market="沪深京",
                start_date=start_date,
                end_date=end_date,
            )
        except Exception:
            return []
        if frame is None or frame.empty:
            return []
        frame = frame.copy()
        frame["公告时间"] = pd.to_datetime(frame["公告时间"], errors="coerce")
        frame = frame.dropna(subset=["公告时间"])
        frame = frame[frame["公告时间"] <= as_of_date]
        frame = frame.sort_values("公告时间", ascending=False).head(
            self.settings.max_announcements_per_symbol
        )
        events: list[TextEvent] = []
        for row in frame.itertuples(index=False):
            title = _clean_text(getattr(row, "公告标题", ""))
            url = _clean_text(getattr(row, "公告链接", ""))
            content = f"公告标题: {title}. Refer to the linked disclosure for the full filing."
            events.append(
                build_event(
                    symbol,
                    published_at=row.公告时间.to_pydatetime(),
                    source_type="announcement",
                    source_name="cninfo:disclosure",
                    title=title,
                    content=content,
                    url=url,
                    importance_hint=0.75,
                    sentiment_hint=0.0,
                )
            )
        return events


class TextService:
    def __init__(
        self,
        settings: Settings,
        store: StateStore,
        provider: Optional[BaseTextProvider] = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.provider = provider or self._build_provider()

    def refresh_text(
        self,
        priors: list[PriorSignal],
        financials: dict[str, FinancialSnapshot],
        bars: dict[str, pd.DataFrame],
        positions: Optional[dict[str, PositionState]] = None,
    ) -> list[TextEvent]:
        if not priors:
            return []
        as_of_date = max(item.as_of_date for item in priors)
        focus_symbols = self._focus_symbols(priors, positions or {})
        events = self._load_seed_events()
        for symbol in sorted(focus_symbols):
            try:
                events.extend(self.provider.fetch_events(symbol, as_of_date))
            except Exception:
                if not self.settings.text_fallback_to_derived:
                    raise
        events.extend(self._build_derived_events(priors, financials, bars, focus_symbols))
        deduped = dedupe_events(events)
        self._write_events(deduped, as_of_date)
        self.store.replace_text_events([event.model_dump(mode="json") for event in deduped])
        return deduped

    def build_event_packs(
        self,
        as_of_date: datetime,
        priors: list[PriorSignal],
        financials: dict[str, FinancialSnapshot],
        positions: dict[str, PositionState],
    ) -> list[SymbolEventPack]:
        candidate_symbols = self._focus_symbols(priors, positions)
        packs = []
        for prior in priors:
            if prior.symbol not in candidate_symbols:
                continue
            raw_events = self.store.list_events(
                prior.symbol, as_of_date.isoformat(), self.settings.max_events_per_symbol
            )
            events = [TextEvent.model_validate(payload) for payload in raw_events]
            packs.append(
                SymbolEventPack(
                    as_of_date=as_of_date,
                    symbol=prior.symbol,
                    prior=prior,
                    financial_snapshot=financials.get(prior.symbol),
                    events=events,
                    position=positions.get(prior.symbol),
                    market_regime=prior.market_regime,
                )
            )
        return packs

    def _build_provider(self) -> BaseTextProvider:
        if self.settings.text_provider == "derived":
            return NullTextProvider()
        if self.settings.text_provider == "akshare":
            try:
                return AkshareTextProvider(self.settings)
            except Exception:
                if not self.settings.text_fallback_to_derived:
                    raise
        return NullTextProvider()

    def _focus_symbols(
        self,
        priors: list[PriorSignal],
        positions: dict[str, PositionState],
    ) -> set[str]:
        candidate_symbols = {item.symbol for item in priors[: self.settings.candidate_pool_size]}
        candidate_symbols |= {
            item.symbol
            for item in sorted(priors, key=lambda item: item.prior_avoid_score, reverse=True)[
                : self.settings.avoid_pool_size
            ]
        }
        candidate_symbols |= set(positions.keys())
        return candidate_symbols

    def _load_seed_events(self) -> list[TextEvent]:
        events: list[TextEvent] = []
        for path in sorted(self.settings.text_storage.glob("*.jsonl")):
            if path.name.startswith("events_"):
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    events.append(TextEvent.model_validate(json.loads(line)))
        return events

    def _build_derived_events(
        self,
        priors: list[PriorSignal],
        financials: dict[str, FinancialSnapshot],
        bars: dict[str, pd.DataFrame],
        focus_symbols: set[str],
    ) -> list[TextEvent]:
        derived: list[TextEvent] = []
        for prior in priors:
            if prior.symbol not in focus_symbols:
                continue
            frame = bars[prior.symbol].sort_values("date")
            latest = frame.iloc[-1]
            prev = frame.iloc[-2] if len(frame) > 1 else latest
            date_value = latest["date"].to_pydatetime()
            previous_close = float(prev["close"]) if float(prev["close"]) else 0.0
            daily_move = (
                float(latest["close"]) / previous_close - 1.0 if previous_close else 0.0
            )
            if abs(daily_move) >= 0.04:
                direction = "up" if daily_move > 0 else "down"
                derived.append(
                    build_event(
                        prior.symbol,
                        date_value,
                        "news",
                        "price-monitor",
                        f"{prior.name} moved {direction} {daily_move:.2%} in one session",
                        (
                            f"Latest session move was {daily_move:.2%}. "
                            "Short-term sentiment and order flow need confirmation."
                        ),
                        importance_hint=min(abs(daily_move) * 8, 1.0),
                        sentiment_hint=daily_move,
                    )
                )
            financial = financials.get(prior.symbol)
            if financial:
                if financial.net_profit_yoy >= 0.15 or financial.revenue_yoy >= 0.12:
                    derived.append(
                        build_event(
                            prior.symbol,
                            financial.as_of_date,
                            "filing",
                            "financial-summary",
                            f"{prior.name} shows improving financial momentum",
                            (
                                f"Revenue yoy={financial.revenue_yoy:.1%}, "
                                f"net profit yoy={financial.net_profit_yoy:.1%}."
                            ),
                            importance_hint=0.72,
                            sentiment_hint=0.55,
                        )
                    )
                if financial.debt_ratio >= 0.65:
                    derived.append(
                        build_event(
                            prior.symbol,
                            financial.as_of_date,
                            "announcement",
                            "balance-sheet",
                            f"{prior.name} has elevated leverage pressure",
                            f"Debt ratio is approximately {financial.debt_ratio:.1%}.",
                            importance_hint=0.64,
                            sentiment_hint=-0.45,
                        )
                    )
            if prior.prior_avoid_score >= 0.70:
                derived.append(
                    build_event(
                        prior.symbol,
                        date_value,
                        "news",
                        "risk-engine",
                        f"{prior.name} has high prior risk",
                        (
                            f"downside_risk_score={prior.downside_risk_score:.2f}, "
                            f"prior_avoid_score={prior.prior_avoid_score:.2f}."
                        ),
                        importance_hint=0.67,
                        sentiment_hint=-0.25,
                    )
                )
        return derived

    def _write_events(self, events: list[TextEvent], as_of_date: datetime) -> None:
        path = self.settings.text_storage / f"events_{as_of_date.date().isoformat()}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(event.model_dump_json() + "\n")


def build_event(
    symbol: str,
    published_at: datetime,
    source_type: str,
    source_name: str,
    title: str,
    content: str,
    *,
    url: str = "",
    importance_hint: float,
    sentiment_hint: float,
) -> TextEvent:
    event_id = hashlib.sha1(
        f"{symbol}|{published_at.isoformat()}|{source_name}|{title}".encode("utf-8")
    ).hexdigest()[:20]
    return TextEvent(
        event_id=event_id,
        symbol=symbol,
        published_at=published_at,
        source_type=source_type,
        source_name=source_name,
        title=title,
        content=content,
        url=url,
        importance_hint=float(importance_hint),
        sentiment_hint=float(sentiment_hint),
    )


def dedupe_events(events: list[TextEvent]) -> list[TextEvent]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[TextEvent] = []
    for event in sorted(events, key=lambda item: item.published_at):
        key = (event.symbol, event.source_name, event.title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()
