from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from engine.config import Settings
from engine.storage.db import StateStore
from engine.text.service import AkshareTextProvider, BaseTextProvider, TextService, build_event, dedupe_events
from engine.types import FinancialSnapshot, PositionState, PriorSignal


def test_dedupe_events_removes_same_symbol_source_and_title() -> None:
    now = datetime(2026, 3, 14, 15, 30)
    first = build_event(
        "000001",
        now,
        "news",
        "mock",
        "duplicate title",
        "content a",
        importance_hint=0.4,
        sentiment_hint=0.1,
    )
    second = build_event(
        "000001",
        now + timedelta(minutes=1),
        "news",
        "mock",
        "duplicate title",
        "content b",
        importance_hint=0.5,
        sentiment_hint=0.2,
    )
    third = build_event(
        "000001",
        now + timedelta(minutes=2),
        "announcement",
        "mock-2",
        "another title",
        "content c",
        importance_hint=0.5,
        sentiment_hint=-0.2,
    )

    deduped = dedupe_events([second, first, third])

    assert len(deduped) == 2
    assert deduped[0].title == "duplicate title"
    assert deduped[1].title == "another title"


class RecordingProvider(BaseTextProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch_events(self, symbol: str, as_of_date: datetime):
        self.calls.append(symbol)
        return [
            build_event(
                symbol,
                as_of_date - timedelta(hours=1),
                "news",
                "stub",
                f"{symbol} event",
                "provider event",
                importance_hint=0.5,
                sentiment_hint=0.0,
            )
        ]


def test_refresh_text_fetches_only_focus_symbols(tmp_path) -> None:
    settings = _settings_for_tmpdir(tmp_path)
    settings.candidate_pool_size = 1
    settings.avoid_pool_size = 1
    store = StateStore(settings.db_path)
    provider = RecordingProvider()
    service = TextService(settings, store, provider=provider)

    priors = [
        _prior("000001", "Alpha", long_score=0.90, avoid_score=0.10),
        _prior("000002", "Beta", long_score=0.20, avoid_score=0.85),
        _prior("000003", "Gamma", long_score=0.10, avoid_score=0.05),
    ]
    positions = {
        "000003": PositionState(
            symbol="000003",
            quantity=100,
            avg_cost=10.0,
            last_price=10.5,
            market_value=1050.0,
            industry="Tech",
            acquired_at=datetime(2026, 3, 1, 9, 30),
            peak_price=11.0,
            stop_loss_pct=0.08,
            take_profit_pct=0.18,
            time_stop_days=20,
        )
    }

    service.refresh_text(
        priors,
        {item.symbol: _financial(item.symbol) for item in priors},
        {item.symbol: _bars(item.symbol, item.as_of_date) for item in priors},
        positions,
    )

    assert set(provider.calls) == {"000001", "000002", "000003"}
    assert store.list_events("000001", priors[0].as_of_date.isoformat(), 10)
    assert store.list_events("000002", priors[0].as_of_date.isoformat(), 10)


class FakeAkshareTextProvider(AkshareTextProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ak = _FakeAkshare()

    def _fetch_announcement_detail(
        self,
        symbol: str,
        announcement_id: str,
        published_at: datetime,
    ) -> dict[str, str] | None:
        return {
            "announcement_id": announcement_id,
            "announcement_content": "",
            "file_url": f"https://example.com/{announcement_id}.pdf",
        }

    def _download_pdf(self, file_url: str) -> bytes:
        return b"%PDF-1.4 fake"

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        return "Line one.\n\nLine two.\n\nLine three."


def test_akshare_provider_extracts_and_caches_announcement_body(tmp_path) -> None:
    settings = _settings_for_tmpdir(tmp_path)
    settings.max_news_per_symbol = 3
    settings.max_announcements_per_symbol = 3
    settings.max_announcement_body_chars = 40
    provider = FakeAkshareTextProvider(settings)
    as_of_date = datetime(2026, 3, 14, 15, 0)

    events = provider.fetch_events("000001", as_of_date)

    assert len(events) == 2
    announcement = next(item for item in events if item.source_type == "announcement")
    assert "Line one." in announcement.content
    assert announcement.url.endswith(".pdf")
    assert (settings.disclosure_storage / "1223456789.json").exists()
    assert (settings.disclosure_pdf_storage / "1223456789.pdf").exists()
    assert (settings.disclosure_text_storage / "1223456789.txt").exists()

    second_pass = provider.fetch_events("000001", as_of_date)
    second_announcement = next(item for item in second_pass if item.source_type == "announcement")
    assert second_announcement.content == announcement.content


class _FakeAkshare:
    def stock_news_em(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "关键词": symbol,
                    "新闻标题": "recent news",
                    "新闻内容": "recent content",
                    "发布时间": "2026-03-14 09:10:00",
                    "文章来源": "Eastmoney",
                    "新闻链接": "https://example.com/news",
                },
                {
                    "关键词": symbol,
                    "新闻标题": "future news",
                    "新闻内容": "future content",
                    "发布时间": "2026-03-15 09:10:00",
                    "文章来源": "Eastmoney",
                    "新闻链接": "https://example.com/future",
                },
            ]
        )

    def stock_zh_a_disclosure_report_cninfo(
        self,
        symbol: str,
        market: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "代码": symbol,
                    "简称": "Ping An",
                    "公告标题": "annual report summary",
                    "公告时间": "2026-03-13 18:00:00",
                    "公告链接": "https://example.com/detail?announcementId=1223456789",
                },
                {
                    "代码": symbol,
                    "简称": "Ping An",
                    "公告标题": "future notice",
                    "公告时间": "2026-03-15 18:00:00",
                    "公告链接": "https://example.com/detail?announcementId=1223999999",
                },
            ]
        )


def _settings_for_tmpdir(tmp_path) -> Settings:
    settings = Settings.load(Path(__file__).resolve().parents[1])
    settings.storage_root = tmp_path
    settings.market_storage = tmp_path / "market"
    settings.text_storage = tmp_path / "text"
    settings.disclosure_storage = settings.text_storage / "disclosures"
    settings.disclosure_pdf_storage = settings.disclosure_storage / "pdf"
    settings.disclosure_text_storage = settings.disclosure_storage / "text"
    settings.artifact_storage = tmp_path / "artifacts"
    settings.db_path = tmp_path / "state.db"
    settings.text_provider = "derived"
    for path in (
        settings.storage_root,
        settings.market_storage,
        settings.text_storage,
        settings.disclosure_storage,
        settings.disclosure_pdf_storage,
        settings.disclosure_text_storage,
        settings.artifact_storage,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return settings


def _prior(symbol: str, name: str, *, long_score: float, avoid_score: float) -> PriorSignal:
    return PriorSignal(
        symbol=symbol,
        name=name,
        sector="Banks",
        as_of_date=datetime(2026, 3, 14, 15, 0),
        latest_close=12.5,
        trend_score=0.6,
        reversal_score=0.4,
        breakout_score=0.5,
        downside_risk_score=0.3,
        liquidity_score=0.7,
        event_sensitivity_score=0.4,
        regime_alignment_score=0.6,
        prior_long_score=long_score,
        prior_avoid_score=avoid_score,
        market_regime="neutral",
    )


def _financial(symbol: str) -> FinancialSnapshot:
    return FinancialSnapshot(
        symbol=symbol,
        as_of_date=datetime(2026, 3, 1, 18, 0),
        revenue_yoy=0.10,
        net_profit_yoy=0.12,
        roe=0.11,
        debt_ratio=0.35,
    )


def _bars(symbol: str, as_of_date: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": as_of_date - timedelta(days=1),
                "symbol": symbol,
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "volume": 1_000_000,
                "amount": 10_000_000,
                "turnover": 1.2,
            },
            {
                "date": as_of_date,
                "symbol": symbol,
                "open": 10.1,
                "high": 10.7,
                "low": 10.0,
                "close": 10.6,
                "volume": 1_100_000,
                "amount": 11_000_000,
                "turnover": 1.3,
            },
        ]
    )
