from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from engine.config import Settings
from engine.storage.db import StateStore
from engine.types import FinancialSnapshot, PositionState, PriorSignal, SymbolEventPack, TextEvent

ROUTINE_EVENT_PATTERNS = (
    "股东大会",
    "董事会",
    "监事会",
    "章程",
    "任职资格",
    "权益分派",
    "分红实施",
    "回购进展",
    "回购报告书",
    "注册资本",
    "闲置募集资金",
    "治理",
    "议案",
    "审议通过",
    "聘任",
)

POSITIVE_CATALYST_PATTERNS = (
    "业绩预增",
    "业绩快报",
    "中标",
    "重大合同",
    "大额订单",
    "回购注销",
    "增持",
    "提价",
    "扩产",
    "扭亏",
    "超预期",
    "新产品",
    "投产",
)

NEGATIVE_CATALYST_PATTERNS = (
    "减持",
    "诉讼",
    "处罚",
    "亏损",
    "下修",
    "终止",
    "违约",
    "爆雷",
    "监管",
    "停产",
    "商誉",
    "问询",
)


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
    cninfo_detail_endpoint = "http://www.cninfo.com.cn/new/announcement/bulletin_detail"
    cninfo_static_base = "http://static.cninfo.com.cn"

    def __init__(self, settings: Settings) -> None:
        import akshare as ak

        self.ak = ak
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/132.0.0.0 Safari/537.36"
                ),
                "Accept": "*/*",
            }
        )

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
        published_col = _pick_column(frame, "发布时间", "published_at")
        title_col = _pick_column(frame, "新闻标题", "title")
        content_col = _pick_column(frame, "新闻内容", "content")
        source_col = _pick_column(frame, "文章来源", "source")
        url_col = _pick_column(frame, "新闻链接", "url")
        frame[published_col] = pd.to_datetime(frame[published_col], errors="coerce")
        frame = frame.dropna(subset=[published_col])
        frame = frame[frame[published_col] <= as_of_date]
        frame = frame.sort_values(published_col, ascending=False).head(
            self.settings.max_news_per_symbol
        )
        events: list[TextEvent] = []
        for row in frame.itertuples(index=False):
            title = _clean_text(getattr(row, title_col))
            content = _clean_text(getattr(row, content_col)) or title
            source = _clean_text(getattr(row, source_col)) or "eastmoney"
            url = _clean_text(getattr(row, url_col))
            published_at = getattr(row, published_col).to_pydatetime()
            events.append(
                build_event(
                    symbol,
                    published_at=published_at,
                    source_type="news",
                    source_name=f"eastmoney:{source}",
                    title=title,
                    content=content,
                    url=url,
                    importance_hint=_importance_from_text(title, content, base=0.56),
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
        title_col = _pick_column(frame, "公告标题", "title")
        url_col = _pick_column(frame, "公告链接", "url")
        time_col = _pick_column(frame, "公告时间", "published_at")
        frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
        frame = frame.dropna(subset=[time_col])
        frame = frame[frame[time_col] <= as_of_date]
        frame = frame.sort_values(time_col, ascending=False).head(
            self.settings.max_announcements_per_symbol
        )
        events: list[TextEvent] = []
        for row in frame.itertuples(index=False):
            title = _clean_text(getattr(row, title_col))
            detail_url = _clean_text(getattr(row, url_col))
            published_at = getattr(row, time_col).to_pydatetime()
            detail = self._resolve_announcement_detail(symbol, title, published_at, detail_url)
            content = detail["snippet"] or (
                f"Announcement title: {title}. Full body extraction was unavailable."
            )
            events.append(
                build_event(
                    symbol,
                    published_at=published_at,
                    source_type="announcement",
                    source_name="cninfo:disclosure",
                    title=title,
                    content=content,
                    url=detail.get("file_url") or detail_url,
                    importance_hint=_importance_from_text(title, content, base=0.68),
                    sentiment_hint=0.0,
                )
            )
        return events

    def _resolve_announcement_detail(
        self,
        symbol: str,
        title: str,
        published_at: datetime,
        detail_url: str,
    ) -> dict[str, str]:
        announcement_id = _parse_announcement_id(detail_url)
        if not announcement_id:
            return {"snippet": f"Announcement title: {title}.", "file_url": detail_url}
        cached = self._read_cached_announcement(announcement_id)
        if cached is not None:
            return cached
        detail = self._fetch_announcement_detail(symbol, announcement_id, published_at)
        if detail is None:
            payload = {
                "announcement_id": announcement_id,
                "detail_url": detail_url,
                "file_url": "",
                "pdf_path": "",
                "text_path": "",
                "snippet": f"Announcement title: {title}. Full body extraction was unavailable.",
            }
            self._write_cached_announcement(announcement_id, payload, "")
            return payload
        file_url = detail.get("file_url", "")
        full_text = _clean_text(detail.get("announcement_content", ""))
        pdf_path = ""
        if file_url:
            pdf_path = str(self._pdf_cache_path(announcement_id))
            if not self._pdf_cache_path(announcement_id).exists():
                pdf_bytes = self._download_pdf(file_url)
                if pdf_bytes:
                    self._pdf_cache_path(announcement_id).write_bytes(pdf_bytes)
            if self._pdf_cache_path(announcement_id).exists():
                extracted = self._extract_pdf_text(self._pdf_cache_path(announcement_id))
                if extracted:
                    full_text = extracted
        if not full_text:
            full_text = f"Announcement title: {title}."
        snippet = _build_snippet(full_text, self.settings.max_announcement_body_chars)
        payload = {
            "announcement_id": announcement_id,
            "detail_url": detail_url,
            "file_url": file_url or detail_url,
            "pdf_path": pdf_path if self._pdf_cache_path(announcement_id).exists() else "",
            "text_path": str(self._text_cache_path(announcement_id)),
            "snippet": snippet,
        }
        self._write_cached_announcement(announcement_id, payload, full_text)
        return payload

    def _fetch_announcement_detail(
        self,
        symbol: str,
        announcement_id: str,
        published_at: datetime,
    ) -> Optional[dict[str, str]]:
        announce_time = published_at.date().isoformat()
        flag_guess = _cninfo_flag(symbol)
        for flag in (flag_guess, not flag_guess):
            try:
                response = self.session.post(
                    self.cninfo_detail_endpoint,
                    params={
                        "announceId": announcement_id,
                        "flag": str(flag).lower(),
                        "announceTime": announce_time,
                    },
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception:
                continue
            announcement = payload.get("announcement") or {}
            file_url = payload.get("fileUrl") or ""
            if announcement or file_url:
                return {
                    "announcement_id": announcement_id,
                    "announcement_content": _clean_text(
                        announcement.get("announcementContent", "")
                    ),
                    "file_url": file_url,
                }
        return None

    def _download_pdf(self, file_url: str) -> bytes:
        try:
            response = self.session.get(file_url, timeout=30)
            response.raise_for_status()
        except Exception:
            return b""
        return response.content if response.content[:4] == b"%PDF" else b""

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        try:
            from pypdf import PdfReader
        except Exception:
            return ""
        try:
            reader = PdfReader(str(pdf_path))
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                text = _normalize_pdf_text(text)
                if text:
                    pages.append(text)
            return "\n\n".join(pages).strip()
        except Exception:
            return ""

    def _read_cached_announcement(self, announcement_id: str) -> Optional[dict[str, str]]:
        meta_path = self._meta_cache_path(announcement_id)
        if not meta_path.exists():
            return None
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        text_path = Path(payload.get("text_path", ""))
        if not text_path.exists():
            return None
        full_text = text_path.read_text(encoding="utf-8")
        payload["snippet"] = _build_snippet(
            full_text, self.settings.max_announcement_body_chars
        )
        return payload

    def _write_cached_announcement(
        self,
        announcement_id: str,
        payload: dict[str, str],
        full_text: str,
    ) -> None:
        text_path = self._text_cache_path(announcement_id)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(full_text, encoding="utf-8")
        meta_path = self._meta_cache_path(announcement_id)
        meta_path.write_text(
            json.dumps(
                {
                    **payload,
                    "cached_at": datetime.utcnow().isoformat(timespec="seconds"),
                    "text_length": len(full_text),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _meta_cache_path(self, announcement_id: str) -> Path:
        return self.settings.disclosure_storage / f"{announcement_id}.json"

    def _pdf_cache_path(self, announcement_id: str) -> Path:
        return self.settings.disclosure_pdf_storage / f"{announcement_id}.pdf"

    def _text_cache_path(self, announcement_id: str) -> Path:
        return self.settings.disclosure_text_storage / f"{announcement_id}.txt"


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
            except Exception as exc:
                if not self.settings.text_fallback_to_derived:
                    provider_name = getattr(self.provider, "provider_name", type(self.provider).__name__)
                    raise RuntimeError(
                        f"[text:{provider_name}] fetch_events failed for symbol={symbol}, "
                        f"as_of_date={as_of_date.isoformat()}: {exc}"
                    ) from exc
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
                prior.symbol,
                as_of_date.isoformat(),
                max(self.settings.max_events_per_symbol * 4, 16),
            )
            events = [TextEvent.model_validate(payload) for payload in raw_events]
            events = self._rank_events(events, as_of_date)
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

    def _rank_events(self, events: list[TextEvent], as_of_date: datetime) -> list[TextEvent]:
        if not events:
            return []
        ranked = sorted(
            events,
            key=lambda item: self._event_priority(item, as_of_date),
            reverse=True,
        )
        filtered = [
            item
            for item in ranked
            if self._event_priority(item, as_of_date) >= 0.22 or item.source_name in {"financial-summary", "risk-engine"}
        ]
        if not filtered:
            filtered = ranked[: self.settings.max_events_per_symbol]
        return filtered[: self.settings.max_events_per_symbol]

    @staticmethod
    def _event_priority(event: TextEvent, as_of_date: datetime) -> float:
        age_days = max((as_of_date - event.published_at).total_seconds() / 86400.0, 0.0)
        recency = max(0.0, 1.0 - min(age_days / 20.0, 1.0))
        source_bonus = 0.06 if event.source_type == "filing" else 0.03 if event.source_type == "announcement" else 0.0
        return (
            0.62 * float(event.importance_hint)
            + 0.23 * recency
            + 0.10 * abs(float(event.sentiment_hint))
            + source_bonus
        )

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


def _importance_from_text(title: str, content: str, *, base: float) -> float:
    text = f"{title} {content}".lower()
    score = base
    if any(pattern.lower() in text for pattern in ROUTINE_EVENT_PATTERNS):
        score -= 0.28
    if any(pattern.lower() in text for pattern in POSITIVE_CATALYST_PATTERNS):
        score += 0.24
    if any(pattern.lower() in text for pattern in NEGATIVE_CATALYST_PATTERNS):
        score += 0.18
    return float(max(0.12, min(score, 0.95)))


def _pick_column(frame: pd.DataFrame, *candidates: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"missing expected columns: {candidates}")


def _parse_announcement_id(detail_url: str) -> str:
    if not detail_url:
        return ""
    parsed = urlparse(detail_url)
    query = parse_qs(parsed.query)
    return str(query.get("announcementId", [""])[0])


def _cninfo_flag(symbol: str) -> bool:
    return symbol.startswith(("000", "001", "002", "003", "300", "301"))


def _build_snippet(text: str, max_chars: int) -> str:
    text = _normalize_pdf_text(text)
    if len(text) <= max_chars:
        return text
    paragraphs = [item.strip() for item in re.split(r"\n{2,}", text) if item.strip()]
    parts: list[str] = []
    total = 0
    for paragraph in paragraphs:
        remaining = max_chars - total
        if remaining <= 0:
            break
        chunk = paragraph[:remaining]
        parts.append(chunk)
        total += len(chunk) + 2
    snippet = "\n\n".join(parts).strip()
    return snippet[:max_chars].rstrip() + "..."


def _normalize_pdf_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()
