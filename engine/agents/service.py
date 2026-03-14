from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from engine.agents.providers import LLMUnavailableError, ProviderChain
from engine.config import Settings
from engine.types import ResearchCard, SymbolEventPack, TradeIntent, TradeIntentSet


def _json_block(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _record_value(record: Any, field: str) -> Any:
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


def _failure_details(call_records: list[Any]) -> list[str]:
    return [
        detail
        for record in call_records
        if not _record_value(record, "success")
        for detail in [_record_value(record, "detail")]
        if detail
    ]


@dataclass
class AgentResult:
    payload: object
    degrade_mode: bool
    call_records: list[dict[str, str]]


class ResearchAgent:
    def __init__(self, settings: Settings, chain: ProviderChain) -> None:
        self.settings = settings
        self.chain = chain

    def run(self, pack: SymbolEventPack) -> AgentResult:
        system_prompt = (
            "You are an A-share research analyst. "
            "Read price priors, text events, financials, position state, and memory context. "
            "Return only JSON that matches the ResearchCard schema. "
            "If evidence is weak or conflicting, use stance=neutral and event_quality=weak or mixed. "
            "Output one JSON object only. Do not add markdown fences or prose outside JSON."
        )
        user_prompt = _json_block(
            {
                "symbol": pack.symbol,
                "market_regime": pack.market_regime,
                "prior": pack.prior.model_dump(mode="json"),
                "financial_snapshot": (
                    pack.financial_snapshot.model_dump(mode="json")
                    if pack.financial_snapshot
                    else None
                ),
                "events": [event.model_dump(mode="json") for event in pack.events],
                "position": pack.position.model_dump(mode="json") if pack.position else None,
                "memory_context": pack.memory_context,
                "output_rules": {
                    "require_json_only": True,
                    "single_json_object_only": True,
                    "no_markdown": True,
                    "confidence_range": [0.0, 1.0],
                    "max_evidence_items": 4,
                    "max_driver_items": 4,
                    "max_risk_items": 4,
                },
            }
        )
        try:
            card, records = self.chain.generate(system_prompt, user_prompt, ResearchCard)
            card.provider_name = next(
                (record.provider_name for record in records if record.success),
                "system",
            )
            return AgentResult(
                payload=card,
                degrade_mode=False,
                call_records=[record.__dict__ for record in records],
            )
        except LLMUnavailableError as exc:
            details = _failure_details(getattr(exc, "call_records", []) or [])
            fallback_card = ResearchCard(
                symbol=pack.symbol,
                stance="neutral",
                confidence=0.0,
                summary=f"LLM unavailable: {exc}",
                evidence=[event.title for event in pack.events[:2]],
                event_quality="weak",
                drivers=["No structured text research was produced, so new entries stay blocked."],
                risks=details[:2] or ["LLM unavailable"],
                invalidators=["Re-run research when an LLM provider is available."],
                holding_horizon_days=5,
                suggested_action_bias="hold" if pack.position else "avoid",
                provider_name="system-fallback",
            )
            return AgentResult(
                payload=fallback_card,
                degrade_mode=True,
                call_records=[record.__dict__ for record in getattr(exc, "call_records", [])],
            )


class DecisionAgent:
    def __init__(self, settings: Settings, chain: ProviderChain) -> None:
        self.settings = settings
        self.chain = chain

    def run(
        self,
        as_of_date: datetime,
        market_view: str,
        cards: list[ResearchCard],
        positions_payload: list[dict[str, Any]],
        priors_payload: list[dict[str, Any]],
        memory_payload: dict[str, Any],
    ) -> AgentResult:
        system_prompt = (
            "You are an A-share portfolio manager. "
            "Read research cards, existing positions, priors, and portfolio memory. "
            "Return only JSON that matches the TradeIntentSet schema. "
            "Every trade intent must include target_weight, stop_loss_pct, take_profit_pct, "
            "time_stop_days, and evidence_count. Use higher cash when uncertainty is elevated. "
            "Output one JSON object only. Do not add markdown fences or prose outside JSON."
        )
        user_prompt = _json_block(
            {
                "as_of_date": as_of_date,
                "market_view": market_view,
                "cards": [card.model_dump(mode="json") for card in cards],
                "positions": positions_payload,
                "priors": priors_payload,
                "memory": memory_payload,
                "constraints": {
                    "max_position_weight": self.settings.max_position_weight,
                    "max_gross_exposure": self.settings.max_gross_exposure,
                    "cash_bias_when_uncertain": True,
                },
                "output_rules": {
                    "single_json_object_only": True,
                    "no_markdown": True,
                    "allow_empty_trade_intents_when_no_edge": True,
                },
            }
        )
        try:
            decision, records = self.chain.generate(
                system_prompt,
                user_prompt,
                TradeIntentSet,
            )
            decision.provider_name = next(
                (record.provider_name for record in records if record.success),
                "system",
            )
            return AgentResult(
                payload=decision,
                degrade_mode=False,
                call_records=[record.__dict__ for record in records],
            )
        except LLMUnavailableError as exc:
            details = _failure_details(getattr(exc, "call_records", []) or [])
            fallback = TradeIntentSet(
                as_of_date=as_of_date,
                market_view=f"{market_view} / degraded",
                cash_target=1.0,
                trade_intents=[
                    TradeIntent(
                        symbol=position["symbol"],
                        action="hold",
                        target_weight=min(
                            position.get("weight", 0.0),
                            self.settings.max_position_weight,
                        ),
                        max_weight=self.settings.max_position_weight,
                        confidence=0.0,
                        thesis="LLM unavailable, risk-only mode",
                        holding_horizon_days=5,
                        stop_loss_pct=self.settings.stop_loss_pct,
                        take_profit_pct=self.settings.take_profit_pct,
                        time_stop_days=self.settings.time_stop_days,
                        evidence_count=0,
                    )
                    for position in positions_payload
                ],
                rejected_symbols=[],
                portfolio_risks=details[:3] or ["LLM unavailable"],
                decision_confidence=0.0,
                rationale=(
                    "DeepSeek did not return an accepted TradeIntentSet. "
                    + (" | ".join(details[:2]) if details else "Keep cash high and disable new entries.")
                ),
                provider_name="system-fallback",
            )
            return AgentResult(
                payload=fallback,
                degrade_mode=True,
                call_records=[record.__dict__ for record in getattr(exc, "call_records", [])],
            )
