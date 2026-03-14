from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from engine.storage.db import StateStore
from engine.types import (
    FillEvent,
    PortfolioMemory,
    PositionMemory,
    PositionState,
    ResearchCard,
    RiskReview,
    TradeIntentSet,
)


class MemoryService:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def symbol_context(self, symbol: str, position: PositionState | None) -> dict:
        memory = self.store.load_position_memory(symbol)
        journal = self.store.list_decision_journal(limit=6, symbol=symbol)
        latest = journal[:3]
        return {
            "position_memory": memory,
            "recent_journal": latest,
            "position_state": position.model_dump(mode="json") if position else None,
        }

    def portfolio_context(self, symbols: list[str]) -> dict:
        portfolio_memory = self.store.latest_portfolio_memory()
        holdings = []
        for symbol in symbols:
            memory = self.store.load_position_memory(symbol)
            if memory:
                holdings.append(
                    {
                        "symbol": symbol,
                        "current_thesis": memory.get("current_thesis", ""),
                        "holding_days": memory.get("holding_days", 0),
                        "invalidators": memory.get("invalidators", []),
                        "last_action": memory.get("last_decision_action", "hold"),
                    }
                )
        return {
            "portfolio_memory": portfolio_memory,
            "position_memories": holdings,
            "recent_decisions": self.store.list_decision_journal(limit=12),
        }

    def record_research(self, run_id: str, as_of_date: datetime, card: ResearchCard) -> None:
        self.store.append_decision_journal(
            entry_id=uuid4().hex,
            run_id=run_id,
            as_of_date=as_of_date.isoformat(),
            symbol=card.symbol,
            stage="research",
            payload=card.model_dump(mode="json"),
        )
        existing = PositionMemory.model_validate(self.store.load_position_memory(card.symbol) or {"symbol": card.symbol})
        existing.last_research_summary = card.summary
        if not existing.initial_thesis and card.summary:
            existing.initial_thesis = card.summary
        if card.invalidators:
            existing.invalidators = card.invalidators
        self.store.upsert_position_memory(card.symbol, existing.model_dump(mode="json"))

    def record_decision(self, run_id: str, as_of_date: datetime, decision: TradeIntentSet) -> None:
        for intent in decision.trade_intents:
            self.store.append_decision_journal(
                entry_id=uuid4().hex,
                run_id=run_id,
                as_of_date=as_of_date.isoformat(),
                symbol=intent.symbol,
                stage="decision",
                payload=intent.model_dump(mode="json"),
            )
            existing = PositionMemory.model_validate(
                self.store.load_position_memory(intent.symbol) or {"symbol": intent.symbol}
            )
            if intent.thesis:
                if not existing.initial_thesis:
                    existing.initial_thesis = intent.thesis
                existing.current_thesis = intent.thesis
            existing.last_decision_action = intent.action
            self.store.upsert_position_memory(intent.symbol, existing.model_dump(mode="json"))

    def record_risk(self, run_id: str, as_of_date: datetime, review: RiskReview) -> None:
        for bucket_name, bucket in (
            ("approved", review.approved),
            ("clipped", review.clipped),
            ("delayed", review.delayed),
            ("rejected", review.rejected),
        ):
            for item in bucket:
                self.store.append_decision_journal(
                    entry_id=uuid4().hex,
                    run_id=run_id,
                    as_of_date=as_of_date.isoformat(),
                    symbol=item.symbol,
                    stage="risk",
                    payload={
                        "bucket": bucket_name,
                        **item.model_dump(mode="json"),
                    },
                )

    def update_after_execution(
        self,
        run_id: str,
        as_of_date: datetime,
        positions: dict[str, PositionState],
        fills: list[FillEvent],
        market_regime: str,
        nav: float,
        peak_nav: float,
        risk_flags: list[str],
    ) -> None:
        fill_by_symbol = {fill.symbol: fill for fill in fills if fill.status == "filled"}
        known_symbols = {item["symbol"] for item in self.store.list_position_memories()} | set(positions.keys())
        for symbol in sorted(known_symbols):
            current = PositionMemory.model_validate(
                self.store.load_position_memory(symbol) or {"symbol": symbol}
            )
            position = positions.get(symbol)
            fill = fill_by_symbol.get(symbol)
            if position is not None:
                current.is_open = True
                current.opened_at = current.opened_at or position.acquired_at
                current.closed_at = None
                current.holding_days = max(
                    0,
                    (as_of_date.date() - current.opened_at.date()).days if current.opened_at else 0,
                )
                current.quantity = position.quantity
                current.avg_cost = position.avg_cost
                current.last_price = position.last_price
                current.mfe = max(current.mfe, (position.last_price / position.avg_cost) - 1.0 if position.avg_cost else 0.0)
                current.mae = min(current.mae, (position.last_price / position.avg_cost) - 1.0 if position.avg_cost else 0.0)
                if fill and fill.action == "buy" and fill.quantity > 0:
                    current.entry_reason = current.current_thesis or current.last_research_summary
            else:
                current.is_open = False
                current.quantity = 0
                current.closed_at = as_of_date
                if fill and fill.action == "sell" and fill.quantity > 0:
                    current.exit_reason = ",".join(fill.notes) or "position_closed"
            self.store.upsert_position_memory(symbol, current.model_dump(mode="json"))
            if fill:
                self.store.append_decision_journal(
                    entry_id=uuid4().hex,
                    run_id=run_id,
                    as_of_date=as_of_date.isoformat(),
                    symbol=symbol,
                    stage="execution",
                    payload=fill.model_dump(mode="json"),
                )

        portfolio_memory = PortfolioMemory.model_validate(
            self.store.latest_portfolio_memory()
            or {"as_of_date": as_of_date, "market_regime": market_regime}
        )
        portfolio_memory.as_of_date = as_of_date
        portfolio_memory.market_regime = market_regime
        portfolio_memory.current_nav = nav
        portfolio_memory.peak_nav = max(portfolio_memory.peak_nav, peak_nav, nav)
        portfolio_memory.risk_flags = risk_flags
        if any("drawdown" in flag or "degraded" in flag for flag in risk_flags):
            portfolio_memory.last_large_derisking_reason = ",".join(risk_flags)
        if any(fill.realized_pnl < 0 for fill in fills):
            portfolio_memory.consecutive_failures += 1
        elif any(fill.realized_pnl > 0 for fill in fills):
            portfolio_memory.consecutive_failures = 0
        self.store.append_portfolio_memory(
            as_of_date.isoformat(),
            portfolio_memory.model_dump(mode="json"),
        )

    def rebuild(self) -> None:
        positions = {row["symbol"]: PositionState.model_validate(row) for row in self.store.list_positions()}
        latest = self.store.latest_run()
        if latest is None:
            return
        summary = latest["summary"]
        fills = [
            FillEvent.model_validate(fill)
            for fill in summary.get("fills", [])
        ]
        self.update_after_execution(
            run_id=latest["run_id"],
            as_of_date=datetime.fromisoformat(latest["as_of_date"]),
            positions=positions,
            fills=fills,
            market_regime=summary.get("market_view", "neutral"),
            nav=float(summary.get("ending_nav", 0.0)),
            peak_nav=float(self.store.load_memory("portfolio_peak_nav").get("value", 0.0)),
            risk_flags=summary.get("risk_flags", []),
        )

