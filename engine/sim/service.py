from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from engine.config import Settings
from engine.types import FillEvent, OrderRequest, PositionState, PriorSignal, RiskReview


class SimulationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prepare_orders(
        self,
        review: RiskReview,
        priors: dict[str, PriorSignal],
        positions: dict[str, PositionState],
        nav: float,
    ) -> list[OrderRequest]:
        orders: list[OrderRequest] = []
        for bucket in (review.approved, review.clipped):
            for decision in bucket:
                prior = priors[decision.symbol]
                current_weight = positions[decision.symbol].market_value / nav if decision.symbol in positions and nav > 0 else 0.0
                diff_weight = decision.approved_weight - current_weight
                if abs(diff_weight) < 0.005:
                    continue
                action = "buy" if diff_weight > 0 else "sell"
                budget = nav * abs(diff_weight)
                quantity = int(budget / prior.latest_close / 100) * 100
                if quantity <= 0:
                    continue
                orders.append(
                    OrderRequest(
                        symbol=decision.symbol,
                        action=action,
                        quantity=quantity,
                        limit_price=prior.latest_close,
                        target_weight=decision.approved_weight,
                    )
                )
        return orders

    def submit_orders(
        self,
        as_of_date: datetime,
        orders: list[OrderRequest],
        bars: dict[str, pd.DataFrame],
        positions: dict[str, PositionState],
        priors: dict[str, PriorSignal],
        starting_cash: float,
    ) -> tuple[list[FillEvent], dict[str, PositionState], float]:
        updated_positions = {symbol: position.model_copy(deep=True) for symbol, position in positions.items()}
        fills: list[FillEvent] = []
        cash_balance = starting_cash
        for order in orders:
            frame = bars[order.symbol].sort_values("date").reset_index(drop=True)
            fill = self._execute_order(as_of_date, order, frame, updated_positions.get(order.symbol), cash_balance)
            fills.append(fill)
            if fill.status != "filled":
                continue
            position = updated_positions.get(order.symbol)
            if order.action == "buy":
                cash_balance -= (fill.price * fill.quantity) + fill.fees
                previous_qty = position.quantity if position else 0
                previous_cost = position.avg_cost if position else 0.0
                new_qty = previous_qty + fill.quantity
                new_cost = ((previous_qty * previous_cost) + (fill.quantity * fill.price)) / max(new_qty, 1)
                updated_positions[order.symbol] = PositionState(
                    symbol=order.symbol,
                    quantity=new_qty,
                    avg_cost=new_cost,
                    last_price=fill.price,
                    market_value=new_qty * fill.price,
                    industry=priors[order.symbol].sector,
                    acquired_at=position.acquired_at if position else as_of_date,
                    peak_price=max(fill.price, position.peak_price if position else fill.price),
                    stop_loss_pct=self.settings.stop_loss_pct,
                    take_profit_pct=self.settings.take_profit_pct,
                    time_stop_days=self.settings.time_stop_days,
                )
            else:
                cash_balance += (fill.price * fill.quantity) - fill.fees
                if position is None:
                    continue
                new_qty = max(0, position.quantity - fill.quantity)
                if new_qty == 0:
                    updated_positions.pop(order.symbol, None)
                else:
                    updated_positions[order.symbol] = PositionState(
                        symbol=order.symbol,
                        quantity=new_qty,
                        avg_cost=position.avg_cost,
                        last_price=fill.price,
                        market_value=new_qty * fill.price,
                        industry=position.industry,
                        acquired_at=position.acquired_at,
                        peak_price=max(position.peak_price, fill.price),
                        stop_loss_pct=position.stop_loss_pct,
                        take_profit_pct=position.take_profit_pct,
                        time_stop_days=position.time_stop_days,
                    )
        return fills, updated_positions, max(0.0, cash_balance)

    def sync_positions(self, positions: dict[str, PositionState]) -> dict[str, PositionState]:
        return positions

    def sync_fills(self, fills: list[FillEvent]) -> list[FillEvent]:
        return fills

    def _execute_order(
        self,
        as_of_date: datetime,
        order: OrderRequest,
        frame: pd.DataFrame,
        current_position: Optional[PositionState],
        cash_balance: float,
    ) -> FillEvent:
        current_idx = frame.index[frame["date"] == pd.Timestamp(as_of_date)]
        if len(current_idx) == 0 or current_idx[0] + 1 >= len(frame):
            return FillEvent(
                symbol=order.symbol,
                action=order.action,
                quantity=order.quantity,
                price=order.limit_price,
                fees=0.0,
                filled_at=as_of_date,
                status="pending",
                notes=["next_trading_day_missing"],
            )
        next_bar = frame.iloc[current_idx[0] + 1]
        prev_close = float(frame.iloc[current_idx[0]]["close"])
        if current_position and order.action == "sell" and current_position.acquired_at.date() >= as_of_date.date():
            return FillEvent(
                symbol=order.symbol,
                action=order.action,
                quantity=0,
                price=float(next_bar["open"]),
                fees=0.0,
                filled_at=next_bar["date"].to_pydatetime(),
                status="blocked",
                notes=["t_plus_1"],
            )
        if float(next_bar["high"]) == float(next_bar["low"]) and float(next_bar["open"]) >= prev_close * 1.095 and order.action == "buy":
            return FillEvent(
                symbol=order.symbol,
                action=order.action,
                quantity=0,
                price=float(next_bar["open"]),
                fees=0.0,
                filled_at=next_bar["date"].to_pydatetime(),
                status="blocked",
                notes=["one_word_limit_up"],
            )
        if float(next_bar["high"]) == float(next_bar["low"]) and float(next_bar["open"]) <= prev_close * 0.905 and order.action == "sell":
            return FillEvent(
                symbol=order.symbol,
                action=order.action,
                quantity=0,
                price=float(next_bar["open"]),
                fees=0.0,
                filled_at=next_bar["date"].to_pydatetime(),
                status="blocked",
                notes=["one_word_limit_down"],
            )
        price = float(next_bar["open"]) * (1 + (self.settings.slippage_bps / 10000.0 if order.action == "buy" else -self.settings.slippage_bps / 10000.0))
        quantity = order.quantity
        fees = price * quantity * self.settings.fee_rate
        if order.action == "sell":
            fees += price * quantity * self.settings.stamp_tax_rate
        if order.action == "buy":
            affordable_lots = int(cash_balance / max((price * 100) * (1 + self.settings.fee_rate), 1)) * 100
            if affordable_lots <= 0:
                return FillEvent(
                    symbol=order.symbol,
                    action=order.action,
                    quantity=0,
                    price=price,
                    fees=0.0,
                    filled_at=next_bar["date"].to_pydatetime(),
                    status="blocked",
                    notes=["insufficient_cash"],
                )
            quantity = min(quantity, affordable_lots)
            fees = price * quantity * self.settings.fee_rate
        return FillEvent(
            symbol=order.symbol,
            action=order.action,
            quantity=quantity,
            price=price,
            fees=fees,
            filled_at=next_bar["date"].to_pydatetime(),
            status="filled",
            notes=[],
        )
