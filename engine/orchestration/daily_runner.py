from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional
from uuid import uuid4

import pandas as pd

from engine.agents.providers import DeepSeekProvider, ProviderChain
from engine.agents.service import DecisionAgent, ResearchAgent
from engine.config import Settings
from engine.execution.broker import build_broker_adapter
from engine.market.service import MarketService, MarketSnapshot, build_prior_signals
from engine.memory.service import MemoryService
from engine.risk.service import RiskService
from engine.runtime.tasks import TaskManager
from engine.sim.service import SimulationService
from engine.storage.db import StateStore
from engine.storage.system_store import SystemStore
from engine.text.service import TextService
from engine.types import (
    ApprovalTicket,
    BacktestConfig,
    BacktestRunSummary,
    BrokerAccountSnapshot,
    DailyRunSummary,
    FillEvent,
    Mode,
    ModeState,
    OrderRequest,
    PositionState,
    PriorSignal,
    ResearchCard,
    RiskDecision,
    RiskReview,
    TaskRun,
    TradeIntentSet,
)

ProgressCallback = Callable[[float, str], None]


@dataclass
class ModeRuntime:
    mode: Mode
    settings: Settings
    store: StateStore
    market_service: MarketService
    text_service: TextService
    research_agent: ResearchAgent
    decision_agent: DecisionAgent
    risk_service: RiskService
    sim_service: SimulationService
    memory_service: MemoryService


@dataclass
class CycleBundle:
    summary: DailyRunSummary
    market_snapshot: MarketSnapshot
    research_cards: list[ResearchCard]
    decision: TradeIntentSet
    risk_review: RiskReview
    orders: list[OrderRequest]
    fills: list[FillEvent]
    updated_positions: dict[str, PositionState]
    ending_cash: float
    approval_tickets: list[ApprovalTicket]


def _noop_progress(_: float, __: str) -> None:
    return None


class DailyRunner:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings.load()
        self.system_store = SystemStore(self.settings.system_db_path)
        self.task_manager = TaskManager(self.system_store)
        self._mode_runtimes: dict[Mode, ModeRuntime] = {}
        self.live_adapter = build_broker_adapter(self.settings.live_broker)
        for mode in ("backtest", "paper", "live"):
            self._ensure_mode_state(mode)

    def _runtime(self, mode: Mode) -> ModeRuntime:
        cached = self._mode_runtimes.get(mode)
        if cached is not None:
            return cached
        mode_settings = self.settings.for_mode(mode)
        store = StateStore(mode_settings.db_path)
        provider_chain = ProviderChain(
            DeepSeekProvider(mode_settings.primary_provider),
        )
        runtime = ModeRuntime(
            mode=mode,
            settings=mode_settings,
            store=store,
            market_service=MarketService(mode_settings, store),
            text_service=TextService(mode_settings, store),
            research_agent=ResearchAgent(mode_settings, provider_chain),
            decision_agent=DecisionAgent(mode_settings, provider_chain),
            risk_service=RiskService(mode_settings),
            sim_service=SimulationService(mode_settings),
            memory_service=MemoryService(store),
        )
        self._mode_runtimes[mode] = runtime
        return runtime

    def _default_mode_state(self, mode: Mode) -> ModeState:
        live_ready = bool(self.settings.live_broker.enabled) if mode == "live" else True
        status = "disabled" if mode == "live" and not live_ready else "idle"
        return ModeState(
            mode=mode,
            status=status,
            active=False,
            live_ready=live_ready,
            updated_at=datetime.utcnow(),
        )

    def _ensure_mode_state(self, mode: Mode) -> ModeState:
        payload = self.system_store.load_mode_state(mode)
        if payload:
            return ModeState.model_validate(payload)
        state = self._default_mode_state(mode)
        self.system_store.save_mode_state(mode, state.model_dump(mode="json"))
        return state

    def _set_mode_state(self, mode: Mode, **updates: Any) -> ModeState:
        payload = self.system_store.load_mode_state(mode) or self._default_mode_state(mode).model_dump(mode="json")
        state = ModeState.model_validate(payload)
        for key, value in updates.items():
            setattr(state, key, value)
        state.updated_at = datetime.utcnow()
        self.system_store.save_mode_state(mode, state.model_dump(mode="json"))
        return state

    def _load_positions(self, runtime: ModeRuntime, mode: Mode) -> dict[str, PositionState]:
        if mode == "live":
            try:
                broker_positions = self.live_adapter.positions_snapshot()
            except Exception:
                broker_positions = {}
            if broker_positions:
                runtime.store.replace_positions(
                    [position.model_dump(mode="json") for position in broker_positions.values()]
                )
                return broker_positions
        return {
            row["symbol"]: PositionState.model_validate(row)
            for row in runtime.store.list_positions()
        }

    def _cash_balance(self, runtime: ModeRuntime, mode: Mode) -> float:
        if mode == "live":
            try:
                account = self.live_adapter.account_snapshot()
            except Exception:
                account = BrokerAccountSnapshot(
                    provider=self.settings.live_broker.provider,
                    available=False,
                    connected=False,
                    message="Broker account unavailable.",
                )
            if account.connected or account.available:
                return account.cash
        return runtime.store.load_memory("portfolio_cash").get(
            "value",
            runtime.settings.default_capital,
        )

    def _load_cached_bars(self, runtime: ModeRuntime, symbols: list[str]) -> dict[str, pd.DataFrame]:
        bars: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            path = runtime.settings.market_storage / f"{symbol}.parquet"
            if not path.exists():
                continue
            bars[symbol] = pd.read_parquet(path)
        return bars

    def refresh_market(
        self,
        mode: Mode = "paper",
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        watchlist: Optional[list[str]] = None,
    ) -> MarketSnapshot:
        runtime = self._runtime(mode)
        return runtime.market_service.refresh_market(
            start_date=start_date,
            end_date=end_date,
            watchlist=watchlist,
        )

    def _position_payload(
        self,
        runtime: ModeRuntime,
        positions: dict[str, PositionState],
        cash_balance: float,
    ) -> tuple[list[dict[str, Any]], float]:
        nav = self._portfolio_nav(positions) + cash_balance
        if nav <= 0:
            return [], runtime.settings.default_capital
        payload = []
        for position in positions.values():
            payload.append(
                {
                    "symbol": position.symbol,
                    "weight": position.market_value / nav,
                    "avg_cost": position.avg_cost,
                    "last_price": position.last_price,
                    "market_value": position.market_value,
                    "industry": position.industry,
                }
            )
        return payload, nav

    @staticmethod
    def _portfolio_nav(positions: dict[str, PositionState]) -> float:
        return sum(position.market_value for position in positions.values())

    def _build_approval_tickets(
        self,
        mode: Mode,
        run_id: str,
        as_of_date: datetime,
        review: RiskReview,
        orders: list[OrderRequest],
    ) -> list[ApprovalTicket]:
        review_map: dict[tuple[str, str], RiskDecision] = {}
        for bucket in (review.approved, review.clipped, review.delayed, review.rejected):
            for item in bucket:
                review_map[(item.symbol, item.requested_action)] = item
        tickets: list[ApprovalTicket] = []
        now = datetime.utcnow()
        for order in orders:
            ticket_id = uuid4().hex[:12]
            decision = review_map.get((order.symbol, order.action))
            tickets.append(
                ApprovalTicket(
                    ticket_id=ticket_id,
                    run_id=run_id,
                    mode=mode,
                    symbol=order.symbol,
                    side=order.action,
                    target_weight=order.target_weight,
                    planned_quantity=order.quantity,
                    limit_price=order.limit_price,
                    reason=order.rationale or (decision.rationale if decision else ""),
                    risk_flags=list(decision.risk_flags if decision else []),
                    status="pending_approval",
                    created_at=as_of_date,
                    updated_at=now,
                    expires_at=None,
                )
            )
        return tickets

    def _persist_run_bundle(
        self,
        runtime: ModeRuntime,
        bundle: CycleBundle,
    ) -> None:
        run_id = bundle.summary.run_id
        as_of_str = bundle.summary.as_of_date.isoformat()
        runtime.store.save_research_cards(
            run_id,
            as_of_str,
            [card.model_dump(mode="json") for card in bundle.research_cards],
        )
        runtime.store.save_portfolio_decision(
            run_id,
            as_of_str,
            bundle.decision.model_dump(mode="json"),
        )
        runtime.store.save_risk_review(
            run_id,
            as_of_str,
            bundle.risk_review.model_dump(mode="json"),
        )
        runtime.store.save_orders(
            run_id,
            [
                {
                    **order.model_dump(mode="json"),
                    "order_ref": order.ticket_id or f"{run_id}-{order.symbol}-{order.action}",
                    "status": "pending_approval" if bundle.approval_tickets else "prepared",
                    "notes": [],
                }
                for order in bundle.orders
            ],
        )
        if bundle.fills:
            runtime.store.save_fills(
                run_id,
                [fill.model_dump(mode="json") for fill in bundle.fills],
            )
        runtime.store.replace_positions(
            [position.model_dump(mode="json") for position in bundle.updated_positions.values()]
        )
        runtime.store.save_daily_snapshot(
            run_id,
            as_of_str,
            {
                "priors": [item.model_dump(mode="json") for item in bundle.market_snapshot.priors],
                "summary": bundle.summary.model_dump(mode="json"),
            },
        )
        runtime.store.save_run(
            run_id,
            as_of_str,
            bundle.summary.status,
            bundle.summary.stage,
            bundle.summary.degrade_mode,
            bundle.summary.model_dump(mode="json"),
        )
        if bundle.approval_tickets:
            runtime.store.save_approval_tickets(
                [ticket.model_dump(mode="json") for ticket in bundle.approval_tickets]
            )
        artifact_path = runtime.settings.artifact_storage / f"{run_id}.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "summary": bundle.summary.model_dump(mode="json"),
                    "cards": [card.model_dump(mode="json") for card in bundle.research_cards],
                    "decision": bundle.decision.model_dump(mode="json"),
                    "risk_review": bundle.risk_review.model_dump(mode="json"),
                    "approvals": [ticket.model_dump(mode="json") for ticket in bundle.approval_tickets],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def _execute_cycle(
        self,
        runtime: ModeRuntime,
        mode: Mode,
        run_id: str,
        market_snapshot: MarketSnapshot,
        run_at: datetime,
        positions: dict[str, PositionState],
        cash_balance: float,
        *,
        auto_execute: bool,
        progress: ProgressCallback,
        execution_bars: Optional[dict[str, pd.DataFrame]] = None,
    ) -> CycleBundle:
        progress(0.25, "refreshing text")
        positions = runtime.sim_service.mark_to_market(run_at, market_snapshot.bars, positions)
        runtime.text_service.refresh_text(
            market_snapshot.priors,
            market_snapshot.financials,
            market_snapshot.bars,
            positions,
        )
        progress(0.40, "building event packs")
        event_packs = runtime.text_service.build_event_packs(
            as_of_date=run_at,
            priors=market_snapshot.priors,
            financials=market_snapshot.financials,
            positions=positions,
        )
        enriched_packs = [
            pack.model_copy(
                update={
                    "memory_context": runtime.memory_service.symbol_context(
                        pack.symbol,
                        positions.get(pack.symbol),
                    )
                }
            )
            for pack in event_packs
        ]
        research_cards: list[ResearchCard] = []
        call_records: list[dict[str, Any]] = []
        degrade_mode = False
        for pack in enriched_packs:
            result = runtime.research_agent.run(pack)
            card = result.payload
            assert isinstance(card, ResearchCard)
            runtime.memory_service.record_research(run_id, run_at, card)
            research_cards.append(card)
            call_records.extend(result.call_records)
            degrade_mode = degrade_mode or result.degrade_mode

        progress(0.58, "running decision agent")
        positions_payload, nav = self._position_payload(runtime, positions, cash_balance)
        decision_result = runtime.decision_agent.run(
            as_of_date=run_at,
            market_view=market_snapshot.priors[0].market_regime if market_snapshot.priors else "neutral",
            cards=research_cards,
            positions_payload=positions_payload,
            priors_payload=[pack.prior.model_dump(mode="json") for pack in enriched_packs],
            memory_payload=runtime.memory_service.portfolio_context(
                list({*positions.keys(), *(pack.symbol for pack in enriched_packs)})
            ),
        )
        decision = decision_result.payload
        assert isinstance(decision, TradeIntentSet)
        runtime.memory_service.record_decision(run_id, run_at, decision)
        call_records.extend(decision_result.call_records)
        degrade_mode = degrade_mode or decision_result.degrade_mode

        progress(0.72, "running risk engine")
        prior_map = {item.symbol: item for item in market_snapshot.priors}
        peak_nav = runtime.store.load_memory("portfolio_peak_nav").get(
            "value",
            nav or runtime.settings.default_capital,
        )
        risk_review = runtime.risk_service.review(
            run_at,
            decision,
            prior_map,
            positions,
            nav,
            peak_nav,
        )
        runtime.memory_service.record_risk(run_id, run_at, risk_review)
        orders = runtime.sim_service.prepare_orders(
            risk_review,
            prior_map,
            positions,
            max(nav, runtime.settings.default_capital),
        )
        for order in orders:
            matching = next(
                (
                    item
                    for item in decision.trade_intents
                    if item.symbol == order.symbol
                    and item.action in {order.action, "reduce" if order.action == "sell" else order.action}
                ),
                None,
            )
            order.rationale = matching.thesis if matching else ""

        fills: list[FillEvent] = []
        updated_positions = positions
        ending_cash = cash_balance
        approval_tickets: list[ApprovalTicket] = []
        progress(0.86, "building execution plan")
        if auto_execute:
            fills, updated_positions, ending_cash = runtime.sim_service.submit_orders(
                run_at,
                orders,
                execution_bars or market_snapshot.bars,
                positions,
                prior_map,
                cash_balance,
            )
            updated_nav = self._portfolio_nav(updated_positions) + ending_cash
            runtime.store.save_memory("portfolio_cash", {"value": ending_cash})
            runtime.store.save_memory("portfolio_peak_nav", {"value": max(float(peak_nav), updated_nav)})
            runtime.memory_service.update_after_execution(
                run_id=run_id,
                as_of_date=run_at,
                positions=updated_positions,
                fills=fills,
                market_regime=decision.market_view,
                nav=updated_nav,
                peak_nav=max(float(peak_nav), updated_nav),
                risk_flags=risk_review.risk_flags,
            )
        else:
            approval_tickets = self._build_approval_tickets(
                mode,
                run_id,
                run_at,
                risk_review,
                orders,
            )
            ticket_map = {(ticket.symbol, ticket.side): ticket.ticket_id for ticket in approval_tickets}
            for order in orders:
                order.ticket_id = ticket_map.get((order.symbol, order.action), "")

        updated_nav = self._portfolio_nav(updated_positions) + ending_cash
        summary = DailyRunSummary(
            run_id=run_id,
            mode=mode,
            as_of_date=run_at,
            status="degraded" if degrade_mode else "ok",
            stage="waiting_approval" if approval_tickets else "completed",
            degrade_mode=degrade_mode,
            candidate_symbols=[pack.symbol for pack in enriched_packs],
            approved_symbols=[item.symbol for item in risk_review.approved],
            clipped_symbols=[item.symbol for item in risk_review.clipped],
            rejected_symbols=[item.symbol for item in risk_review.rejected],
            pending_approval_symbols=[ticket.symbol for ticket in approval_tickets],
            fills=fills,
            risk_flags=risk_review.risk_flags,
            notes=[
                record.get("detail", "")
                for record in call_records
                if not record.get("success", True) and record.get("detail")
            ],
            cash_balance=ending_cash,
            ending_nav=updated_nav,
        )
        return CycleBundle(
            summary=summary,
            market_snapshot=market_snapshot,
            research_cards=research_cards,
            decision=decision,
            risk_review=risk_review,
            orders=orders,
            fills=fills,
            updated_positions=updated_positions,
            ending_cash=ending_cash,
            approval_tickets=approval_tickets,
        )

    def run_cycle(self, mode: Mode = "paper", as_of_date: Optional[str] = None) -> DailyRunSummary:
        runtime = self._runtime(mode)
        self._set_mode_state(mode, status="running", active=True, message="Running cycle")
        market_snapshot = runtime.market_service.refresh_market()
        run_at = market_snapshot.as_of_date if as_of_date is None else datetime.fromisoformat(as_of_date)
        run_id = uuid4().hex[:12]
        positions = self._load_positions(runtime, mode)
        cash_balance = self._cash_balance(runtime, mode)
        bundle = self._execute_cycle(
            runtime,
            mode,
            run_id,
            market_snapshot,
            run_at,
            positions,
            cash_balance,
            auto_execute=False if mode in {"paper", "live"} else True,
            progress=_noop_progress,
        )
        self._persist_run_bundle(runtime, bundle)
        self._set_mode_state(
            mode,
            status="waiting_approval" if bundle.approval_tickets else "idle",
            active=False,
            last_run_id=run_id,
            approval_count=len(runtime.store.list_approval_tickets("pending_approval")),
            broker_connected=self.live_adapter.account_snapshot().connected if mode == "live" else False,
            message="Cycle completed",
        )
        return bundle.summary

    def run_daily(self, as_of_date: Optional[str] = None) -> DailyRunSummary:
        return self.run_cycle("paper", as_of_date)

    def run_backtest(self, config: BacktestConfig, progress: ProgressCallback = _noop_progress) -> BacktestRunSummary:
        runtime = self._runtime("backtest")
        runtime.store.reset_runtime_state()
        run_id = f"bt-{uuid4().hex[:10]}"
        progress(0.05, "loading market history")
        snapshot = runtime.market_service.refresh_market(
            start_date=config.start_date,
            end_date=config.end_date,
            watchlist=config.watchlist or None,
        )
        runtime.store.save_memory("portfolio_cash", {"value": config.initial_capital})
        runtime.store.save_memory("portfolio_peak_nav", {"value": config.initial_capital})
        all_dates = sorted(
            {
                item.to_pydatetime()
                for frame in snapshot.bars.values()
                for item in frame["date"].drop_duplicates()
                if config.start_date <= item.date().isoformat() <= config.end_date
            }
        )
        equity_curve: list[dict[str, Any]] = []
        drawdown_curve: list[dict[str, Any]] = []
        daily_positions: list[dict[str, Any]] = []
        daily_decisions: list[dict[str, Any]] = []
        daily_risk_reviews: list[dict[str, Any]] = []
        trade_log: list[dict[str, Any]] = []
        memory_timeline: list[dict[str, Any]] = []
        positions: dict[str, PositionState] = {}
        cash_balance = config.initial_capital
        peak_nav = config.initial_capital
        for index, current_date in enumerate(all_dates):
            progress(0.10 + 0.80 * ((index + 1) / max(len(all_dates), 1)), f"backtesting {current_date.date().isoformat()}")
            sliced_bars = {
                symbol: frame[frame["date"] <= pd.Timestamp(current_date)].copy()
                for symbol, frame in snapshot.bars.items()
            }
            sliced_bars = {symbol: frame for symbol, frame in sliced_bars.items() if not frame.empty}
            priors, _ = build_prior_signals(sliced_bars, snapshot.financials)
            daily_snapshot = MarketSnapshot(
                bars=sliced_bars,
                financials=snapshot.financials,
                priors=priors,
                as_of_date=current_date,
            )
            bundle = self._execute_cycle(
                runtime,
                "backtest",
                f"{run_id}-{current_date:%Y%m%d}",
                daily_snapshot,
                current_date,
                positions,
                cash_balance,
                auto_execute=True,
                progress=_noop_progress,
                execution_bars=snapshot.bars,
            )
            self._persist_run_bundle(runtime, bundle)
            positions = bundle.updated_positions
            cash_balance = bundle.ending_cash
            peak_nav = max(peak_nav, bundle.summary.ending_nav)
            runtime.store.save_memory("portfolio_peak_nav", {"value": peak_nav})
            equity_curve.append({"as_of_date": current_date.date().isoformat(), "nav": bundle.summary.ending_nav})
            drawdown_curve.append(
                {
                    "as_of_date": current_date.date().isoformat(),
                    "drawdown": 0.0 if peak_nav <= 0 else max(0.0, 1.0 - bundle.summary.ending_nav / peak_nav),
                }
            )
            daily_positions.append(
                {
                    "as_of_date": current_date.date().isoformat(),
                    "positions": [position.model_dump(mode="json") for position in positions.values()],
                    "cash": cash_balance,
                }
            )
            daily_decisions.append(
                {
                    "as_of_date": current_date.date().isoformat(),
                    "run_id": bundle.summary.run_id,
                    "decision": bundle.decision.model_dump(mode="json"),
                }
            )
            daily_risk_reviews.append(
                {
                    "as_of_date": current_date.date().isoformat(),
                    "run_id": bundle.summary.run_id,
                    "risk_review": bundle.risk_review.model_dump(mode="json"),
                }
            )
            trade_log.extend([fill.model_dump(mode="json") for fill in bundle.fills if fill.status == "filled"])
            memory_timeline.extend(runtime.store.list_portfolio_memory(limit=1))
        metrics = self._backtest_metrics(config.initial_capital, equity_curve, trade_log)
        summary = BacktestRunSummary(
            run_id=run_id,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            status="completed",
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            daily_positions=daily_positions,
            orders=runtime.store.list_orders(),
            fills=runtime.store.list_fills(),
            daily_risk_reviews=daily_risk_reviews,
            daily_decisions=daily_decisions,
            trade_log=trade_log,
            memory_timeline=memory_timeline,
            metrics=metrics,
        )
        runtime.store.save_backtest_run(
            run_id,
            config.start_date,
            config.end_date,
            "completed",
            summary.model_dump(mode="json"),
        )
        artifact_path = runtime.settings.artifact_storage / f"{run_id}.json"
        artifact_path.write_text(
            json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self._set_mode_state(
            "backtest",
            status="idle",
            active=False,
            last_run_id=run_id,
            approval_count=0,
            message="Backtest completed",
        )
        return summary

    def _backtest_metrics(
        self,
        initial_capital: float,
        equity_curve: list[dict[str, Any]],
        trade_log: list[dict[str, Any]],
    ) -> dict[str, float]:
        if not equity_curve:
            return {}
        navs = pd.Series([item["nav"] for item in equity_curve], dtype=float)
        returns = navs.pct_change().fillna(0.0)
        total_return = navs.iloc[-1] / initial_capital - 1.0 if initial_capital else 0.0
        annual_return = (1.0 + total_return) ** (252 / max(len(navs), 1)) - 1.0 if len(navs) > 1 else total_return
        annual_vol = returns.std(ddof=0) * (252**0.5)
        downside = returns[returns < 0]
        sortino = 0.0
        if not downside.empty and downside.std(ddof=0) > 0:
            sortino = returns.mean() / downside.std(ddof=0) * (252**0.5)
        rolling_peak = navs.cummax()
        max_drawdown = (1.0 - navs / rolling_peak).max() if not rolling_peak.empty else 0.0
        realized = [float(item.get("realized_pnl", 0.0)) for item in trade_log if item.get("action") == "sell"]
        wins = [value for value in realized if value > 0]
        losses = [value for value in realized if value < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        win_rate = len(wins) / len(realized) if realized else 0.0
        fee_ratio = sum(float(item.get("fees", 0.0)) for item in trade_log) / max(navs.iloc[-1], 1.0)
        benchmark_return = returns.mean() * len(navs) if len(navs) > 1 else 0.0
        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_vol),
            "max_drawdown": float(max_drawdown),
            "sortino": float(sortino),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_holding_days": 0.0,
            "daily_turnover": 0.0,
            "fee_ratio": float(fee_ratio),
            "benchmark_excess_return": float(total_return - benchmark_return),
        }

    def run_backtest_task(self, config: BacktestConfig) -> TaskRun:
        task = self.task_manager.submit(
            "run_backtest",
            "backtest",
            lambda progress: self.run_backtest(config, progress).model_dump(mode="json"),
        )
        self._set_mode_state(
            "backtest",
            status="running",
            active=True,
            last_task_id=task["task_id"],
            message="Running backtest",
        )
        return TaskRun.model_validate(task)

    def run_paper_cycle_task(self, as_of_date: Optional[str] = None) -> TaskRun:
        task = self.task_manager.submit(
            "run_paper_cycle",
            "paper",
            lambda progress: self.run_cycle("paper", as_of_date).model_dump(mode="json"),
        )
        self._set_mode_state(
            "paper",
            status="running",
            active=True,
            last_task_id=task["task_id"],
            message="Running paper cycle",
        )
        return TaskRun.model_validate(task)

    def run_live_cycle_task(self, as_of_date: Optional[str] = None) -> TaskRun:
        task = self.task_manager.submit(
            "run_live_cycle",
            "live",
            lambda progress: self.run_cycle("live", as_of_date).model_dump(mode="json"),
        )
        self._set_mode_state(
            "live",
            status="running",
            active=True,
            last_task_id=task["task_id"],
            message="Running live cycle",
        )
        return TaskRun.model_validate(task)

    def start_live_session(self) -> TaskRun:
        task = self.task_manager.submit(
            "start_live_session",
            "live",
            lambda progress: self._start_live_session(progress),
        )
        self._set_mode_state(
            "live",
            status="running",
            active=True,
            last_task_id=task["task_id"],
            message="Connecting live broker",
        )
        return TaskRun.model_validate(task)

    def _start_live_session(self, progress: ProgressCallback) -> dict[str, Any]:
        progress(0.30, "connecting broker")
        snapshot = self.live_adapter.connect()
        self._set_mode_state(
            "live",
            status="connected" if snapshot.connected else ("disabled" if not snapshot.available else "error"),
            active=snapshot.connected,
            broker_connected=snapshot.connected,
            live_ready=snapshot.available,
            message=snapshot.message,
        )
        return snapshot.model_dump(mode="json")

    def stop_live_session(self) -> TaskRun:
        task = self.task_manager.submit(
            "stop_live_session",
            "live",
            lambda progress: self._stop_live_session(progress),
        )
        return TaskRun.model_validate(task)

    def _stop_live_session(self, progress: ProgressCallback) -> dict[str, Any]:
        progress(0.50, "disconnecting broker")
        self.live_adapter.disconnect()
        state = self._set_mode_state(
            "live",
            status="idle" if self.settings.live_broker.enabled else "disabled",
            active=False,
            broker_connected=False,
            message="Live broker disconnected",
        )
        return state.model_dump(mode="json")

    def rebuild_memory(self, mode: Mode) -> TaskRun:
        task = self.task_manager.submit(
            "rebuild_memory",
            mode,
            lambda progress: self._rebuild_memory(mode, progress),
        )
        return TaskRun.model_validate(task)

    def _rebuild_memory(self, mode: Mode, progress: ProgressCallback) -> dict[str, Any]:
        progress(0.50, "rebuilding memory")
        runtime = self._runtime(mode)
        runtime.memory_service.rebuild()
        return {
            "mode": mode,
            "position_memories": runtime.store.list_position_memories(),
            "portfolio_memory": runtime.store.latest_portfolio_memory(),
        }

    def approve_ticket(self, mode: Mode, ticket_id: str) -> dict[str, Any]:
        runtime = self._runtime(mode)
        ticket = runtime.store.get_approval_ticket(ticket_id)
        if ticket is None:
            raise ValueError(f"Unknown ticket: {ticket_id}")
        if ticket.get("status") != "pending_approval":
            return ticket
        if mode == "paper":
            result = self._approve_paper_ticket(runtime, ticket)
        else:
            result = self._approve_live_ticket(runtime, ticket)
        remaining = len(runtime.store.list_approval_tickets("pending_approval"))
        self._set_mode_state(
            mode,
            status="waiting_approval" if remaining else "idle",
            approval_count=remaining,
            active=False if mode == "paper" else self.live_adapter.account_snapshot().connected,
            broker_connected=self.live_adapter.account_snapshot().connected if mode == "live" else False,
            message="Approval updated",
        )
        return result

    def _approve_paper_ticket(self, runtime: ModeRuntime, ticket: dict[str, Any]) -> dict[str, Any]:
        snapshot = runtime.store.load_daily_snapshot(ticket["run_id"])
        priors = {
            item["symbol"]: PriorSignal.model_validate(item)
            for item in snapshot.get("priors", [])
        }
        bars = self._load_cached_bars(runtime, [ticket["symbol"]])
        positions = self._load_positions(runtime, "paper")
        cash_balance = self._cash_balance(runtime, "paper")
        order = OrderRequest(
            symbol=ticket["symbol"],
            action=ticket["side"],
            quantity=ticket["planned_quantity"],
            limit_price=ticket["limit_price"],
            target_weight=ticket["target_weight"],
            run_id=ticket["run_id"],
            mode="paper",
            ticket_id=ticket["ticket_id"],
            rationale=ticket.get("reason", ""),
        )
        fills, updated_positions, ending_cash = runtime.sim_service.submit_orders(
            datetime.fromisoformat(ticket["created_at"]),
            [order],
            bars,
            positions,
            priors,
            cash_balance,
        )
        runtime.store.save_fills(ticket["run_id"], [fill.model_dump(mode="json") for fill in fills])
        runtime.store.replace_positions(
            [position.model_dump(mode="json") for position in updated_positions.values()]
        )
        runtime.store.save_memory("portfolio_cash", {"value": ending_cash})
        nav = self._portfolio_nav(updated_positions) + ending_cash
        peak_nav = max(runtime.store.load_memory("portfolio_peak_nav").get("value", nav), nav)
        runtime.store.save_memory("portfolio_peak_nav", {"value": peak_nav})
        runtime.memory_service.update_after_execution(
            run_id=ticket["run_id"],
            as_of_date=datetime.fromisoformat(ticket["created_at"]),
            positions=updated_positions,
            fills=fills,
            market_regime=snapshot.get("summary", {}).get("market_view", "neutral"),
            nav=nav,
            peak_nav=peak_nav,
            risk_flags=snapshot.get("summary", {}).get("risk_flags", []),
        )
        fill_status = fills[0].status if fills else "approved"
        runtime.store.update_approval_ticket(
            ticket["ticket_id"],
            status="filled" if fill_status == "filled" else ("blocked" if fill_status == "blocked" else "approved"),
        )
        runtime.store.update_order_status(ticket["ticket_id"], fill_status)
        return {
            "ticket": runtime.store.get_approval_ticket(ticket["ticket_id"]),
            "fills": [fill.model_dump(mode="json") for fill in fills],
            "portfolio": self.current_portfolio("paper"),
        }

    def _approve_live_ticket(self, runtime: ModeRuntime, ticket: dict[str, Any]) -> dict[str, Any]:
        account = self.live_adapter.account_snapshot()
        if not account.connected:
            raise RuntimeError("Live broker is not connected.")
        order = OrderRequest(
            symbol=ticket["symbol"],
            action=ticket["side"],
            quantity=ticket["planned_quantity"],
            limit_price=ticket["limit_price"],
            target_weight=ticket["target_weight"],
            run_id=ticket["run_id"],
            mode="live",
            ticket_id=ticket["ticket_id"],
            rationale=ticket.get("reason", ""),
        )
        broker_order = self.live_adapter.submit_order(order)
        runtime.store.update_approval_ticket(
            ticket["ticket_id"],
            status="sent" if broker_order.status not in {"filled", "blocked"} else broker_order.status,
            broker_order_id=broker_order.broker_order_id,
        )
        runtime.store.update_order_status(ticket["ticket_id"], broker_order.status, broker_order.notes)
        fills = self.live_adapter.fills_since()
        if fills:
            runtime.store.save_fills(ticket["run_id"], [fill.model_dump(mode="json") for fill in fills])
            positions = self.live_adapter.positions_snapshot()
            runtime.store.replace_positions([position.model_dump(mode="json") for position in positions.values()])
            live_account = self.live_adapter.account_snapshot()
            runtime.store.save_memory("portfolio_cash", {"value": live_account.cash})
            runtime.memory_service.update_after_execution(
                run_id=ticket["run_id"],
                as_of_date=datetime.utcnow(),
                positions=positions,
                fills=fills,
                market_regime="live",
                nav=live_account.equity,
                peak_nav=max(runtime.store.load_memory("portfolio_peak_nav").get("value", live_account.equity), live_account.equity),
                risk_flags=[],
            )
        return {
            "ticket": runtime.store.get_approval_ticket(ticket["ticket_id"]),
            "broker_order": broker_order.model_dump(mode="json"),
            "account": self.live_account(),
        }

    def reject_ticket(self, mode: Mode, ticket_id: str) -> dict[str, Any]:
        runtime = self._runtime(mode)
        ticket = runtime.store.update_approval_ticket(ticket_id, status="rejected")
        if ticket is None:
            raise ValueError(f"Unknown ticket: {ticket_id}")
        runtime.store.update_order_status(ticket_id, "rejected")
        remaining = len(runtime.store.list_approval_tickets("pending_approval"))
        self._set_mode_state(
            mode,
            status="waiting_approval" if remaining else "idle",
            approval_count=remaining,
            message="Ticket rejected",
        )
        return ticket

    def dashboard_summary(self, mode: Mode = "paper") -> dict[str, Any]:
        runtime = self._runtime(mode)
        latest = runtime.store.latest_run()
        portfolio = self.current_portfolio(mode)
        approvals = runtime.store.list_approval_tickets("pending_approval") if mode in {"paper", "live"} else []
        traffic_light = "green"
        if approvals:
            traffic_light = "amber"
        if latest and latest.get("degrade_mode"):
            traffic_light = "red"
        return {
            "mode": mode,
            "run": latest,
            "portfolio": portfolio,
            "signals": self.today_signals(mode)["cards"],
            "risk": {
                "traffic_light": traffic_light,
                "flags": latest["summary"].get("risk_flags", []) if latest else [],
            },
            "approvals": approvals,
        }

    def today_signals(self, mode: Mode = "paper") -> dict[str, Any]:
        runtime = self._runtime(mode)
        latest = runtime.store.latest_run()
        if latest is None:
            return {"run": None, "cards": [], "decision": None, "risk": None}
        return {
            "run": latest,
            "cards": runtime.store.list_cards_for_run(latest["run_id"]),
            "decision": runtime.store.load_decision_for_run(latest["run_id"]),
            "risk": runtime.store.load_risk_for_run(latest["run_id"]),
        }

    def current_portfolio(self, mode: Mode = "paper") -> dict[str, Any]:
        runtime = self._runtime(mode)
        positions = runtime.store.list_positions()
        if mode == "live":
            account = self.live_adapter.account_snapshot()
            return {
                "positions": positions,
                "nav": account.equity,
                "cash": account.cash,
                "gross_exposure": account.market_value / account.equity if account.equity else 0.0,
            }
        cash_balance = runtime.store.load_memory("portfolio_cash").get("value", runtime.settings.default_capital)
        nav = sum(position["market_value"] for position in positions) + cash_balance
        return {
            "positions": positions,
            "nav": nav,
            "cash": cash_balance,
            "gross_exposure": sum(position["market_value"] for position in positions) / nav if nav else 0.0,
        }

    def run_details(self, mode: Mode, run_id: str) -> dict[str, Any]:
        runtime = self._runtime(mode)
        latest = runtime.store.fetch_one("SELECT summary_json FROM daily_runs WHERE run_id = ?", (run_id,))
        if latest is None:
            return {"run_id": run_id, "found": False}
        return {
            "run_id": run_id,
            "summary": json.loads(latest["summary_json"]),
            "cards": runtime.store.list_cards_for_run(run_id),
            "decision": runtime.store.load_decision_for_run(run_id),
            "risk": runtime.store.load_risk_for_run(run_id),
            "orders": runtime.store.list_orders(run_id),
            "fills": runtime.store.list_fills(run_id),
        }

    def backtest_runs(self) -> list[dict[str, Any]]:
        runtime = self._runtime("backtest")
        return runtime.store.list_backtest_runs()

    def backtest_run_details(self, run_id: str) -> dict[str, Any]:
        runtime = self._runtime("backtest")
        payload = runtime.store.load_backtest_run(run_id)
        if payload is None:
            return {"run_id": run_id, "found": False}
        return payload

    def backtest_summary(self) -> dict[str, Any]:
        runs = self.backtest_runs()
        latest = runs[0]["summary"] if runs else None
        return {
            "runs": len(runs),
            "latest": latest,
        }

    def memory_snapshot(self, mode: Mode = "paper") -> dict[str, Any]:
        runtime = self._runtime(mode)
        return {
            "positions": runtime.store.list_position_memories(),
            "portfolio": runtime.store.latest_portfolio_memory(),
            "journal": runtime.store.list_decision_journal(limit=100),
        }

    def paper_account(self) -> dict[str, Any]:
        return self.current_portfolio("paper")

    def live_account(self) -> dict[str, Any]:
        return self.live_adapter.account_snapshot().model_dump(mode="json")

    def live_approval_queue(self) -> list[dict[str, Any]]:
        return self._runtime("live").store.list_approval_tickets()

    def paper_approval_queue(self) -> list[dict[str, Any]]:
        return self._runtime("paper").store.list_approval_tickets()

    def status_system(self) -> dict[str, Any]:
        return {
            "modes": [self._ensure_mode_state(mode).model_dump(mode="json") for mode in ("backtest", "paper", "live")],
            "tasks": self.task_manager.list(20),
            "live_account": self.live_account(),
        }

    def status_tasks(self) -> list[dict[str, Any]]:
        return self.task_manager.list(50)

    def status_mode(self, mode: Mode) -> dict[str, Any]:
        runtime = self._runtime(mode)
        state = self._ensure_mode_state(mode).model_dump(mode="json")
        return {
            "state": state,
            "latest_run": runtime.store.latest_run(),
            "approvals": runtime.store.list_approval_tickets("pending_approval") if mode in {"paper", "live"} else [],
            "task_history": [task for task in self.task_manager.list(50) if task.get("mode") == mode],
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "system": self.status_system(),
            "paper": self.status_mode("paper"),
            "live": self.status_mode("live"),
            "backtest": self.status_mode("backtest"),
        }
