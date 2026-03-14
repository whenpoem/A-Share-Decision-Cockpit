from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


Mode = Literal["backtest", "paper", "live"]
TaskStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
ModeStatus = Literal["idle", "running", "waiting_approval", "connected", "disabled", "error"]
ApprovalStatus = Literal[
    "pending_approval",
    "approved",
    "rejected",
    "expired",
    "sent",
    "filled",
    "blocked",
    "cancelled",
]


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


class PositionMemory(BaseModel):
    symbol: str
    is_open: bool = True
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    initial_thesis: str = ""
    current_thesis: str = ""
    invalidators: list[str] = Field(default_factory=list)
    holding_days: int = 0
    last_research_summary: str = ""
    last_decision_action: str = "hold"
    entry_reason: str = ""
    exit_reason: str = ""
    mfe: float = 0.0
    mae: float = 0.0
    last_price: float = 0.0
    avg_cost: float = 0.0
    quantity: int = 0


class SymbolEventPack(BaseModel):
    as_of_date: datetime
    symbol: str
    prior: PriorSignal
    financial_snapshot: Optional[FinancialSnapshot] = None
    events: list[TextEvent] = Field(default_factory=list)
    position: Optional[PositionState] = None
    market_regime: Literal["risk_on", "neutral", "risk_off"]
    memory_context: dict[str, Any] = Field(default_factory=dict)


class ResearchCard(BaseModel):
    symbol: str
    stance: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    evidence: list[str] = Field(default_factory=list)
    event_quality: Literal["verified", "mixed", "weak"]
    drivers: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    invalidators: list[str] = Field(default_factory=list)
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
    trade_intents: list[TradeIntent] = Field(default_factory=list)
    rejected_symbols: list[str] = Field(default_factory=list)
    portfolio_risks: list[str] = Field(default_factory=list)
    decision_confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    provider_name: str = "system"


class RiskDecision(BaseModel):
    symbol: str
    requested_action: str
    requested_weight: float
    approved_weight: float
    final_status: Literal["approved", "clipped", "delayed", "rejected"]
    risk_flags: list[str] = Field(default_factory=list)
    rationale: str = ""


class RiskReview(BaseModel):
    as_of_date: datetime
    approved: list[RiskDecision] = Field(default_factory=list)
    clipped: list[RiskDecision] = Field(default_factory=list)
    delayed: list[RiskDecision] = Field(default_factory=list)
    rejected: list[RiskDecision] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class OrderRequest(BaseModel):
    symbol: str
    action: Literal["buy", "sell"]
    quantity: int
    limit_price: float
    target_weight: float
    run_id: str = ""
    mode: Mode = "paper"
    ticket_id: str = ""
    rationale: str = ""


class FillEvent(BaseModel):
    symbol: str
    action: Literal["buy", "sell"]
    quantity: int
    price: float
    fees: float
    filled_at: datetime
    status: Literal["filled", "blocked", "pending"]
    notes: list[str] = Field(default_factory=list)
    order_ref: str = ""
    realized_pnl: float = 0.0


class DailyRunSummary(BaseModel):
    run_id: str
    mode: Mode
    as_of_date: datetime
    status: Literal["ok", "degraded", "error"]
    stage: str
    degrade_mode: bool = False
    candidate_symbols: list[str] = Field(default_factory=list)
    approved_symbols: list[str] = Field(default_factory=list)
    clipped_symbols: list[str] = Field(default_factory=list)
    rejected_symbols: list[str] = Field(default_factory=list)
    pending_approval_symbols: list[str] = Field(default_factory=list)
    fills: list[FillEvent] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    cash_balance: float = 0.0
    ending_nav: float = 0.0


class BacktestConfig(BaseModel):
    start_date: str
    end_date: str
    initial_capital: float = Field(gt=0.0)
    watchlist: list[str] = Field(default_factory=list)


class BacktestRunSummary(BaseModel):
    run_id: str
    mode: Mode = "backtest"
    start_date: str
    end_date: str
    initial_capital: float
    status: str
    equity_curve: list[dict[str, Any]] = Field(default_factory=list)
    drawdown_curve: list[dict[str, Any]] = Field(default_factory=list)
    daily_positions: list[dict[str, Any]] = Field(default_factory=list)
    orders: list[dict[str, Any]] = Field(default_factory=list)
    fills: list[dict[str, Any]] = Field(default_factory=list)
    daily_risk_reviews: list[dict[str, Any]] = Field(default_factory=list)
    daily_decisions: list[dict[str, Any]] = Field(default_factory=list)
    trade_log: list[dict[str, Any]] = Field(default_factory=list)
    memory_timeline: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class DecisionJournalEntry(BaseModel):
    id: str
    run_id: str
    as_of_date: datetime
    symbol: str
    stage: Literal["research", "decision", "risk", "execution"]
    payload: dict[str, Any] = Field(default_factory=dict)


class PortfolioMemory(BaseModel):
    as_of_date: Optional[datetime] = None
    market_regime: str = "neutral"
    consecutive_failures: int = 0
    last_large_derisking_reason: str = ""
    current_nav: float = 0.0
    peak_nav: float = 0.0
    risk_flags: list[str] = Field(default_factory=list)


class ApprovalTicket(BaseModel):
    ticket_id: str
    run_id: str
    mode: Mode
    symbol: str
    side: Literal["buy", "sell"]
    target_weight: float
    planned_quantity: int
    limit_price: float
    reason: str
    risk_flags: list[str] = Field(default_factory=list)
    status: ApprovalStatus = "pending_approval"
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    broker_order_id: str = ""


class BrokerAccountSnapshot(BaseModel):
    provider: str
    available: bool
    connected: bool
    account_id: str = ""
    cash: float = 0.0
    equity: float = 0.0
    market_value: float = 0.0
    buying_power: float = 0.0
    message: str = ""


class BrokerOrderSnapshot(BaseModel):
    broker_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: int
    limit_price: float
    status: str
    submitted_at: datetime
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    ticket_id: str = ""
    notes: list[str] = Field(default_factory=list)


class TaskRun(BaseModel):
    task_id: str
    task_type: str
    mode: Optional[Mode] = None
    status: TaskStatus
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    created_at: datetime
    updated_at: datetime


class ModeState(BaseModel):
    mode: Mode
    status: ModeStatus = "idle"
    active: bool = False
    last_run_id: str = ""
    last_task_id: str = ""
    approval_count: int = 0
    broker_connected: bool = False
    live_ready: bool = False
    message: str = ""
    updated_at: datetime

