from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from engine.config import LiveBrokerConfig
from engine.execution.interfaces import ExecutionAdapter
from engine.types import BrokerAccountSnapshot, BrokerOrderSnapshot, FillEvent, OrderRequest, PositionState


@dataclass
class DisabledBrokerAdapter(ExecutionAdapter):
    config: LiveBrokerConfig

    def connect(self) -> BrokerAccountSnapshot:
        return BrokerAccountSnapshot(
            provider=self.config.provider,
            available=False,
            connected=False,
            account_id=self.config.account_id,
            message="Live trading is disabled.",
        )

    def disconnect(self) -> None:
        return None

    def account_snapshot(self) -> BrokerAccountSnapshot:
        return self.connect()

    def positions_snapshot(self) -> dict[str, PositionState]:
        return {}

    def open_orders(self) -> list[BrokerOrderSnapshot]:
        return []

    def submit_order(self, order_request: OrderRequest) -> BrokerOrderSnapshot:
        raise RuntimeError("Live broker is disabled.")

    def cancel_order(self, broker_order_id: str) -> BrokerOrderSnapshot | None:
        return None

    def fills_since(self, cursor: str = "") -> list[FillEvent]:
        return []


@dataclass
class MockBrokerAdapter(ExecutionAdapter):
    config: LiveBrokerConfig
    connected: bool = False
    cash: float = 1_000_000.0
    orders: dict[str, BrokerOrderSnapshot] = field(default_factory=dict)
    fills: list[FillEvent] = field(default_factory=list)
    positions: dict[str, PositionState] = field(default_factory=dict)

    def connect(self) -> BrokerAccountSnapshot:
        self.connected = True
        return self.account_snapshot()

    def disconnect(self) -> None:
        self.connected = False

    def account_snapshot(self) -> BrokerAccountSnapshot:
        market_value = sum(position.market_value for position in self.positions.values())
        equity = self.cash + market_value
        return BrokerAccountSnapshot(
            provider=self.config.provider,
            available=True,
            connected=self.connected,
            account_id=self.config.account_id or "mock-live",
            cash=self.cash,
            equity=equity,
            market_value=market_value,
            buying_power=self.cash,
            message="Mock live adapter",
        )

    def positions_snapshot(self) -> dict[str, PositionState]:
        return {symbol: position.model_copy(deep=True) for symbol, position in self.positions.items()}

    def open_orders(self) -> list[BrokerOrderSnapshot]:
        return list(self.orders.values())

    def submit_order(self, order_request: OrderRequest) -> BrokerOrderSnapshot:
        if not self.connected:
            raise RuntimeError("Mock broker is not connected.")
        broker_order_id = uuid4().hex[:16]
        submitted_at = datetime.utcnow()
        order = BrokerOrderSnapshot(
            broker_order_id=broker_order_id,
            symbol=order_request.symbol,
            side=order_request.action,
            quantity=order_request.quantity,
            limit_price=order_request.limit_price,
            status="sent",
            submitted_at=submitted_at,
            filled_quantity=order_request.quantity,
            average_fill_price=order_request.limit_price,
            ticket_id=order_request.ticket_id,
            notes=["mock_fill"],
        )
        self.orders[broker_order_id] = order
        position = self.positions.get(order_request.symbol)
        if order_request.action == "buy":
            previous_qty = position.quantity if position else 0
            previous_cost = position.avg_cost if position else 0.0
            new_qty = previous_qty + order_request.quantity
            new_cost = (
                ((previous_qty * previous_cost) + (order_request.quantity * order_request.limit_price))
                / max(new_qty, 1)
            )
            self.cash = max(0.0, self.cash - (order_request.quantity * order_request.limit_price))
            self.positions[order_request.symbol] = PositionState(
                symbol=order_request.symbol,
                quantity=new_qty,
                avg_cost=new_cost,
                last_price=order_request.limit_price,
                market_value=new_qty * order_request.limit_price,
                industry=position.industry if position else "Unknown",
                acquired_at=position.acquired_at if position else submitted_at,
                peak_price=max(order_request.limit_price, position.peak_price if position else order_request.limit_price),
                stop_loss_pct=position.stop_loss_pct if position else 0.08,
                take_profit_pct=position.take_profit_pct if position else 0.18,
                time_stop_days=position.time_stop_days if position else 20,
            )
        else:
            if position is not None:
                sold_qty = min(position.quantity, order_request.quantity)
                self.cash += sold_qty * order_request.limit_price
                remaining = position.quantity - sold_qty
                if remaining <= 0:
                    self.positions.pop(order_request.symbol, None)
                else:
                    position.quantity = remaining
                    position.last_price = order_request.limit_price
                    position.market_value = remaining * order_request.limit_price
        self.fills.append(
            FillEvent(
                symbol=order_request.symbol,
                action=order_request.action,
                quantity=order_request.quantity,
                price=order_request.limit_price,
                fees=0.0,
                filled_at=submitted_at,
                status="filled",
                notes=["mock_fill"],
                order_ref=broker_order_id,
            )
        )
        return order

    def cancel_order(self, broker_order_id: str) -> BrokerOrderSnapshot | None:
        order = self.orders.get(broker_order_id)
        if order is None:
            return None
        order.status = "cancelled"
        return order

    def fills_since(self, cursor: str = "") -> list[FillEvent]:
        return list(self.fills)


@dataclass
class QmtReadyBrokerAdapter(ExecutionAdapter):
    config: LiveBrokerConfig
    connected: bool = False

    def connect(self) -> BrokerAccountSnapshot:
        self.connected = False
        return BrokerAccountSnapshot(
            provider=self.config.provider,
            available=bool(self.config.enabled),
            connected=False,
            account_id=self.config.account_id,
            message="QMT/Ptrade adapter contract is ready. Enable local SDK wiring to trade live.",
        )

    def disconnect(self) -> None:
        self.connected = False

    def account_snapshot(self) -> BrokerAccountSnapshot:
        return self.connect()

    def positions_snapshot(self) -> dict[str, PositionState]:
        return {}

    def open_orders(self) -> list[BrokerOrderSnapshot]:
        return []

    def submit_order(self, order_request: OrderRequest) -> BrokerOrderSnapshot:
        raise RuntimeError("QMT/Ptrade adapter is feature-gated until local SDK wiring is enabled.")

    def cancel_order(self, broker_order_id: str) -> BrokerOrderSnapshot | None:
        return None

    def fills_since(self, cursor: str = "") -> list[FillEvent]:
        return []


def build_broker_adapter(config: LiveBrokerConfig) -> ExecutionAdapter:
    if not config.enabled:
        return DisabledBrokerAdapter(config)
    if config.provider == "mock":
        return MockBrokerAdapter(config)
    return QmtReadyBrokerAdapter(config)
