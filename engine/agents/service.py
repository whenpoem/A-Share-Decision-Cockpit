from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

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


class PartialTradeIntentSet(BaseModel):
    as_of_date: datetime | None = None
    market_view: str | None = None
    cash_target: float | None = None
    trade_intents: list[TradeIntent] = Field(default_factory=list)
    rejected_symbols: list[str] = Field(default_factory=list)
    portfolio_risks: list[str] = Field(default_factory=list)
    decision_confidence: float | None = None
    rationale: str | None = None
    provider_name: str = "system"

    @field_validator("rejected_symbols", mode="before")
    @classmethod
    def _coerce_rejected_symbols(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        normalized: list[str] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item.zfill(6) if item.isdigit() else item)
                continue
            if isinstance(item, int):
                normalized.append(str(item).zfill(6))
                continue
            if isinstance(item, dict):
                symbol = item.get("symbol") or item.get("code")
                if symbol is None:
                    continue
                symbol_text = str(symbol)
                normalized.append(symbol_text.zfill(6) if symbol_text.isdigit() else symbol_text)
        return normalized


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _finalize_trade_intent_set(
    payload: PartialTradeIntentSet,
    *,
    as_of_date: datetime,
    market_view: str,
) -> TradeIntentSet:
    trade_intents = payload.trade_intents or []
    inferred_cash_target = 1.0
    if trade_intents:
        target_sum = sum(max(intent.target_weight, 0.0) for intent in trade_intents)
        inferred_cash_target = max(0.0, 1.0 - min(target_sum, 1.0))
    cash_target = _clamp_unit(
        payload.cash_target if payload.cash_target is not None else inferred_cash_target
    )
    if payload.decision_confidence is not None:
        decision_confidence = _clamp_unit(payload.decision_confidence)
    elif trade_intents:
        decision_confidence = _clamp_unit(
            sum(intent.confidence for intent in trade_intents) / max(len(trade_intents), 1)
        )
    else:
        decision_confidence = 0.15
    rationale = payload.rationale
    if not rationale:
        rationale = (
            "No high-conviction trade intents were returned; keep a defensive posture."
            if not trade_intents
            else "Decision payload was partially repaired with runtime defaults."
        )
    return TradeIntentSet(
        as_of_date=payload.as_of_date or as_of_date,
        market_view=payload.market_view or market_view,
        cash_target=cash_target,
        trade_intents=trade_intents,
        rejected_symbols=payload.rejected_symbols,
        portfolio_risks=payload.portfolio_risks,
        decision_confidence=decision_confidence,
        rationale=rationale,
        provider_name=payload.provider_name,
    )


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
            "Even when there are no trades, you must still return all top-level keys: "
            "as_of_date, market_view, cash_target, trade_intents, rejected_symbols, "
            "portfolio_risks, decision_confidence, rationale, provider_name. "
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
                    "required_top_level_keys": [
                        "as_of_date",
                        "market_view",
                        "cash_target",
                        "trade_intents",
                        "rejected_symbols",
                        "portfolio_risks",
                        "decision_confidence",
                        "rationale",
                        "provider_name",
                    ],
                    "empty_trade_template": {
                        "as_of_date": str(as_of_date),
                        "market_view": market_view,
                        "cash_target": 1.0,
                        "trade_intents": [],
                        "rejected_symbols": [],
                        "portfolio_risks": [],
                        "decision_confidence": 0.2,
                        "rationale": "No high-conviction setup today.",
                        "provider_name": "deepseek",
                    },
                },
            }
        )
        try:
            raw_decision, records = self.chain.generate(
                system_prompt,
                user_prompt,
                PartialTradeIntentSet,
            )
            decision = _finalize_trade_intent_set(
                raw_decision,
                as_of_date=as_of_date,
                market_view=market_view,
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
