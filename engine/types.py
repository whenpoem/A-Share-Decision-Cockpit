from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class FinancialSnapshot(BaseModel):
    symbol: str
    as_of_date: datetime
    revenue_yoy: float = 0.0
    net_profit_yoy: float = 0.0
    roe: float = 0.0
    debt_ratio: float = 0.0


class TextEvent(BaseModel):
    event_id: str
    symbol: str
    published_at: datetime
    source_type: Literal["news", "announcement", "filing"]
    source_name: str
    title: str
    content: str
    url: str = ""
    importance_hint: float = 0.0
    sentiment_hint: float = 0.0


class PriorSignal(BaseModel):
    symbol: str
    name: str
    sector: str
    as_of_date: datetime
    latest_close: float
    trend_score: float
    reversal_score: float
    breakout_score: float
    downside_risk_score: float
    liquidity_score: float
    event_sensitivity_score: float
    regime_alignment_score: float
    prior_long_score: float
    prior_avoid_score: float
    market_regime: Literal["risk_on", "neutral", "risk_off"]


class PositionState(BaseModel):
    symbol: str
    quantity: int
    avg_cost: float
    last_price: float
    market_value: float
    industry: str
    acquired_at: datetime
    peak_price: float
    stop_loss_pct: float
    take_profit_pct: float
    time_stop_days: int


class SymbolEventPack(BaseModel):
    as_of_date: datetime
    symbol: str
    prior: PriorSignal
    financial_snapshot: Optional[FinancialSnapshot] = None
    events: List[TextEvent] = Field(default_factory=list)
    position: Optional[PositionState] = None
    market_regime: Literal["risk_on", "neutral", "risk_off"]


class ResearchCard(BaseModel):
    symbol: str
    stance: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    evidence: List[str] = Field(default_factory=list)
    event_quality: Literal["verified", "mixed", "weak"]
    drivers: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    invalidators: List[str] = Field(default_factory=list)
    holding_horizon_days: int = 10
    suggested_action_bias: Literal["open_long", "trim", "hold", "avoid"] = "hold"
    provider_name: str = "system"


class TradeIntent(BaseModel):
    symbol: str
    action: Literal["buy", "sell", "reduce", "hold"]
    target_weight: float = Field(ge=0.0, le=1.0)
    max_weight: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str
    holding_horizon_days: int = 10
    stop_loss_pct: float = Field(ge=0.0, le=0.5)
    take_profit_pct: float = Field(ge=0.0, le=1.0)
    time_stop_days: int = 20
    evidence_count: int = 0


class TradeIntentSet(BaseModel):
    as_of_date: datetime
    market_view: str
    cash_target: float = Field(ge=0.0, le=1.0)
    trade_intents: List[TradeIntent] = Field(default_factory=list)
    rejected_symbols: List[str] = Field(default_factory=list)
    portfolio_risks: List[str] = Field(default_factory=list)
    decision_confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    provider_name: str = "system"


class RiskDecision(BaseModel):
    symbol: str
    requested_action: str
    requested_weight: float
    approved_weight: float
    final_status: Literal["approved", "clipped", "delayed", "rejected"]
    risk_flags: List[str] = Field(default_factory=list)
    rationale: str = ""


class RiskReview(BaseModel):
    as_of_date: datetime
    approved: List[RiskDecision] = Field(default_factory=list)
    clipped: List[RiskDecision] = Field(default_factory=list)
    delayed: List[RiskDecision] = Field(default_factory=list)
    rejected: List[RiskDecision] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


class OrderRequest(BaseModel):
    symbol: str
    action: Literal["buy", "sell"]
    quantity: int
    limit_price: float
    target_weight: float


class FillEvent(BaseModel):
    symbol: str
    action: Literal["buy", "sell"]
    quantity: int
    price: float
    fees: float
    filled_at: datetime
    status: Literal["filled", "blocked", "pending"]
    notes: List[str] = Field(default_factory=list)


class DailyRunSummary(BaseModel):
    run_id: str
    as_of_date: datetime
    status: Literal["ok", "degraded", "error"]
    stage: str
    degrade_mode: bool = False
    candidate_symbols: List[str] = Field(default_factory=list)
    approved_symbols: List[str] = Field(default_factory=list)
    clipped_symbols: List[str] = Field(default_factory=list)
    rejected_symbols: List[str] = Field(default_factory=list)
    fills: List[FillEvent] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    cash_balance: float = 0.0
    ending_nav: float = 0.0
