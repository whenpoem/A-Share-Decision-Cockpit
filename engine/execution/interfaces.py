from __future__ import annotations

from typing import Protocol

from engine.types import BrokerAccountSnapshot, BrokerOrderSnapshot, FillEvent, OrderRequest, PositionState


class ExecutionAdapter(Protocol):
    def connect(self) -> BrokerAccountSnapshot:
        ...

    def disconnect(self) -> None:
        ...

    def account_snapshot(self) -> BrokerAccountSnapshot:
        ...

    def positions_snapshot(self) -> dict[str, PositionState]:
        ...

    def open_orders(self) -> list[BrokerOrderSnapshot]:
        ...

    def submit_order(self, order_request: OrderRequest) -> BrokerOrderSnapshot:
        ...

    def cancel_order(self, broker_order_id: str) -> BrokerOrderSnapshot | None:
        ...

    def fills_since(self, cursor: str = "") -> list[FillEvent]:
        ...

