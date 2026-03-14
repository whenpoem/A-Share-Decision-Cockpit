from __future__ import annotations

from datetime import datetime
from pathlib import Path

from engine.config import Settings
from engine.risk.service import RiskService
from engine.types import PriorSignal, TradeIntent, TradeIntentSet


def test_risk_service_rejects_buy_without_any_evidence() -> None:
    settings = Settings.load(Path(__file__).resolve().parents[1])
    service = RiskService(settings)
    prior = PriorSignal(
        symbol="000001",
        name="样本001",
        sector="银行",
        as_of_date=datetime(2026, 3, 14),
        latest_close=12.0,
        trend_score=0.7,
        reversal_score=0.4,
        breakout_score=0.5,
        downside_risk_score=0.3,
        liquidity_score=0.9,
        event_sensitivity_score=0.2,
        regime_alignment_score=0.6,
        prior_long_score=0.72,
        prior_avoid_score=0.21,
        market_regime="neutral",
    )
    decision = TradeIntentSet(
        as_of_date=datetime(2026, 3, 14),
        market_view="neutral",
        cash_target=0.3,
        trade_intents=[
            TradeIntent(
                symbol="000001",
                action="buy",
                target_weight=0.08,
                max_weight=0.10,
                confidence=0.8,
                thesis="看多",
                holding_horizon_days=10,
                stop_loss_pct=0.08,
                take_profit_pct=0.18,
                time_stop_days=20,
                evidence_count=0,
            )
        ],
        decision_confidence=0.7,
        rationale="测试",
    )

    review = service.review(
        as_of_date=datetime(2026, 3, 14),
        decision=decision,
        priors={"000001": prior},
        positions={},
        latest_nav=1_000_000,
        peak_nav=1_000_000,
    )

    assert not review.approved
    assert review.rejected[0].symbol == "000001"
    assert "insufficient_evidence" in review.rejected[0].risk_flags
