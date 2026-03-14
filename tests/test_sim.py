from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from engine.config import Settings
from engine.sim.service import SimulationService
from engine.types import OrderRequest, PositionState


def test_simulation_blocks_t_plus_one_sell() -> None:
    settings = Settings.load(Path(__file__).resolve().parents[1])
    service = SimulationService(settings)
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-14", "2026-03-15"]),
            "open": [10.0, 10.1],
            "high": [10.5, 10.3],
            "low": [9.8, 9.9],
            "close": [10.2, 10.0],
        }
    )
    order = OrderRequest(symbol="000001", action="sell", quantity=100, limit_price=10.2, target_weight=0.0)
    position = PositionState(
        symbol="000001",
        quantity=100,
        avg_cost=10.0,
        last_price=10.2,
        market_value=1020.0,
        industry="银行",
        acquired_at=datetime(2026, 3, 14, 9, 35),
        peak_price=10.2,
        stop_loss_pct=0.08,
        take_profit_pct=0.18,
        time_stop_days=20,
    )

    fill = service._execute_order(datetime(2026, 3, 14), order, bars, position, 1_000_000.0)

    assert fill.status == "blocked"
    assert "t_plus_1" in fill.notes
