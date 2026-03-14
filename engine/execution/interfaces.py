from __future__ import annotations

from typing import Protocol

from engine.types import FillEvent, OrderRequest, PositionState


class ExecutionAdapter(Protocol):
    def prepare_orders(self, trade_intents) -> list[OrderRequest]:
        ...

    def submit_orders(self, order_requests: list[OrderRequest]) -> list[FillEvent]:
        ...

    def sync_positions(self) -> dict[str, PositionState]:
        ...

    def sync_fills(self) -> list[FillEvent]:
        ...

