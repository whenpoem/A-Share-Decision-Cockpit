from __future__ import annotations

from pathlib import Path

from engine.config import Settings
from engine.market.service import BaseMarketProvider, MarketService, SampleMarketProvider
from engine.storage.db import StateStore


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings.load(Path(__file__).resolve().parents[1])
    settings.system_storage_root = tmp_path
    settings.system_db_path = tmp_path / "system.db"
    settings.storage_root = tmp_path / "paper"
    settings.market_storage = settings.storage_root / "market"
    settings.text_storage = settings.storage_root / "text"
    settings.disclosure_storage = settings.text_storage / "disclosures"
    settings.disclosure_pdf_storage = settings.disclosure_storage / "pdf"
    settings.disclosure_text_storage = settings.disclosure_storage / "text"
    settings.artifact_storage = settings.storage_root / "artifacts"
    settings.db_path = settings.storage_root / "state.db"
    for path in (
        settings.system_storage_root,
        settings.storage_root,
        settings.market_storage,
        settings.text_storage,
        settings.disclosure_storage,
        settings.disclosure_pdf_storage,
        settings.disclosure_text_storage,
        settings.artifact_storage,
    ):
        path.mkdir(parents=True, exist_ok=True)
    settings.market_provider = "sample"
    return settings


class GuardedProvider(BaseMarketProvider):
    def __init__(self, symbols: list[str]) -> None:
        self.sample = SampleMarketProvider(symbols)
        self.list_calls = 0

    def list_symbols(self, limit: int) -> list[dict[str, str]]:
        self.list_calls += 1
        raise AssertionError("list_symbols should not be called")

    def fetch_price_history(self, symbol: str, start_date: str, end_date: str):
        return self.sample.fetch_price_history(symbol, start_date, end_date)

    def fetch_financial_snapshot(self, symbol: str):
        return self.sample.fetch_financial_snapshot(symbol)


def test_refresh_market_prefers_watchlist_over_remote_universe(tmp_path) -> None:
    settings = make_settings(tmp_path)
    store = StateStore(settings.db_path)
    service = MarketService(settings, store)
    provider = GuardedProvider(["000001", "600519"])
    service.provider = provider

    snapshot = service.refresh_market(
        start_date="2025-01-01",
        end_date="2025-01-31",
        watchlist=["000001", "600519"],
    )

    assert provider.list_calls == 0
    assert sorted(snapshot.bars.keys()) == ["000001", "600519"]


def test_refresh_market_prefers_cached_universe_before_remote_universe(tmp_path) -> None:
    settings = make_settings(tmp_path)
    store = StateStore(settings.db_path)
    store.upsert_symbol("000001", "平安银行", "银行")
    store.upsert_symbol("600519", "贵州茅台", "白酒")
    service = MarketService(settings, store)
    provider = GuardedProvider(["000001", "600519"])
    service.provider = provider

    snapshot = service.refresh_market(
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    assert provider.list_calls == 0
    assert sorted(snapshot.bars.keys()) == ["000001", "600519"]
