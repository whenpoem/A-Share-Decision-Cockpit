from __future__ import annotations

from datetime import datetime
from pathlib import Path

from engine.agents.providers import LLMCallRecord, LLMUnavailableError, _extract_json_object
from engine.agents.service import DecisionAgent, ResearchAgent
from engine.config import Settings
from engine.types import PriorSignal, SymbolEventPack


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
    return settings


class FailingChain:
    def generate(self, system_prompt: str, user_prompt: str, schema):
        raise LLMUnavailableError(
            "deepseek: schema validation failed: missing field",
            call_records=[
                LLMCallRecord(
                    provider_name="deepseek",
                    success=False,
                    detail="deepseek schema validation failed: missing field trade_intents",
                )
            ],
        )


def test_extract_json_object_handles_markdown_wrapper() -> None:
    payload = "analysis first\n```json\n{\"a\":1,\"b\":\"ok\"}\n```\nextra"
    assert _extract_json_object(payload) == '{"a":1,"b":"ok"}'


def test_research_agent_fallback_keeps_failure_details(tmp_path) -> None:
    settings = make_settings(tmp_path)
    agent = ResearchAgent(settings, FailingChain())
    pack = SymbolEventPack(
        as_of_date=datetime(2025, 6, 26),
        symbol="000001",
        prior=PriorSignal(
            symbol="000001",
            name="平安银行",
            sector="银行",
            as_of_date=datetime(2025, 6, 26),
            latest_close=10.0,
            trend_score=0.5,
            reversal_score=0.5,
            breakout_score=0.5,
            downside_risk_score=0.5,
            liquidity_score=0.5,
            event_sensitivity_score=0.5,
            regime_alignment_score=0.5,
            prior_long_score=0.5,
            prior_avoid_score=0.5,
            market_regime="neutral",
        ),
        financial_snapshot=None,
        events=[],
        position=None,
        market_regime="neutral",
        memory_context={},
    )

    result = agent.run(pack)

    assert result.degrade_mode is True
    assert result.call_records
    assert "deepseek schema validation failed" in result.payload.risks[0]
    assert "deepseek" in result.payload.summary


def test_decision_agent_fallback_includes_failure_reason(tmp_path) -> None:
    settings = make_settings(tmp_path)
    agent = DecisionAgent(settings, FailingChain())

    result = agent.run(
        as_of_date=datetime(2025, 6, 26),
        market_view="neutral",
        cards=[],
        positions_payload=[],
        priors_payload=[],
        memory_payload={},
    )

    assert result.degrade_mode is True
    assert result.call_records
    assert "deepseek schema validation failed" in result.payload.rationale
    assert result.payload.provider_name == "system-fallback"
