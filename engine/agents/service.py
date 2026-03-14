from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from engine.agents.providers import LLMUnavailableError, ProviderChain
from engine.config import Settings
from engine.types import ResearchCard, SymbolEventPack, TradeIntent, TradeIntentSet


def _json_block(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


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
            "你是A股研究员。只能根据给定材料输出 JSON，不要输出自然语言解释。"
            "如果证据不足，要明确标成 weak 和 neutral。"
        )
        user_prompt = _json_block(
            {
                "symbol": pack.symbol,
                "market_regime": pack.market_regime,
                "prior": pack.prior.model_dump(mode="json"),
                "financial_snapshot": pack.financial_snapshot.model_dump(mode="json")
                if pack.financial_snapshot
                else None,
                "events": [event.model_dump(mode="json") for event in pack.events],
                "position": pack.position.model_dump(mode="json") if pack.position else None,
            }
        )
        try:
            card, records = self.chain.generate(system_prompt, user_prompt, ResearchCard)
            card.provider_name = next((record.provider_name for record in records if record.success), "system")
            return AgentResult(payload=card, degrade_mode=False, call_records=[record.__dict__ for record in records])
        except LLMUnavailableError as exc:
            fallback_card = ResearchCard(
                symbol=pack.symbol,
                stance="neutral",
                confidence=0.0,
                summary=f"LLM unavailable: {exc}",
                evidence=[event.title for event in pack.events[:2]],
                event_quality="weak",
                drivers=["未能完成结构化研究，禁止新增仓位"],
                risks=["LLM unavailable"],
                invalidators=["等待下一次成功研究"],
                holding_horizon_days=5,
                suggested_action_bias="hold" if pack.position else "avoid",
                provider_name="system-fallback",
            )
            return AgentResult(payload=fallback_card, degrade_mode=True, call_records=[])


class DecisionAgent:
    def __init__(self, settings: Settings, chain: ProviderChain) -> None:
        self.settings = settings
        self.chain = chain

    def run(
        self,
        as_of_date: datetime,
        market_view: str,
        cards: list[ResearchCard],
        positions_payload: list[dict],
        priors_payload: list[dict],
    ) -> AgentResult:
        system_prompt = (
            "你是A股组合经理。只能输出 JSON。"
            "可以建议买卖，但必须给出 target_weight、stop_loss_pct、time_stop_days 和 evidence_count。"
            "如果证据不足或市场风险过大，优先提高现金比例。"
        )
        user_prompt = _json_block(
            {
                "as_of_date": as_of_date,
                "market_view": market_view,
                "cards": [card.model_dump(mode="json") for card in cards],
                "positions": positions_payload,
                "priors": priors_payload,
                "constraints": {
                    "max_position_weight": self.settings.max_position_weight,
                    "max_gross_exposure": self.settings.max_gross_exposure,
                    "cash_bias_when_uncertain": True,
                },
            }
        )
        try:
            decision, records = self.chain.generate(system_prompt, user_prompt, TradeIntentSet)
            decision.provider_name = next((record.provider_name for record in records if record.success), "system")
            return AgentResult(payload=decision, degrade_mode=False, call_records=[record.__dict__ for record in records])
        except LLMUnavailableError:
            fallback = TradeIntentSet(
                as_of_date=as_of_date,
                market_view=f"{market_view} / degraded",
                cash_target=1.0,
                trade_intents=[
                    TradeIntent(
                        symbol=position["symbol"],
                        action="hold",
                        target_weight=min(position.get("weight", 0.0), self.settings.max_position_weight),
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
                portfolio_risks=["LLM unavailable"],
                decision_confidence=0.0,
                rationale="Both DeepSeek and Qwen failed. Disable new entries and keep risk-only mode.",
                provider_name="system-fallback",
            )
            return AgentResult(payload=fallback, degrade_mode=True, call_records=[])

