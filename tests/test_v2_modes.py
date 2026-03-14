from __future__ import annotations

import time
from pathlib import Path

from engine.agents.service import AgentResult
from engine.config import Settings
from engine.orchestration.daily_runner import DailyRunner
from engine.types import BacktestConfig, RiskDecision, RiskReview, TradeIntent, TradeIntentSet


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings.load(Path(__file__).resolve().parents[1])
    settings.system_storage_root = tmp_path
    settings.system_db_path = tmp_path / "system.db"
    settings.storage_root = tmp_path / "paper"
    settings.market_storage = settings.storage_root / "market"
    settings.text_storage = settings.storage_root / "text"
    settings.disclosure_storage = settings.text_storage / "disclosures"
    settings.disclosure_pdf_storage = settings.disclosure_storage / "pdf"
    settings.disclosure_text_storage = settings.disclosure_storage / "text"
    settings.artifact_storage = settings.storage_root / "artifacts"
    settings.db_path = settings.storage_root / "state.db"
    for path in (
        settings.system_storage_root,
        settings.storage_root,
        settings.market_storage,
        settings.text_storage,
        settings.disclosure_storage,
        settings.disclosure_pdf_storage,
        settings.disclosure_text_storage,
        settings.artifact_storage,
    ):
        path.mkdir(parents=True, exist_ok=True)
    settings.market_provider = "sample"
    settings.text_provider = "derived"
    settings.primary_provider.api_key = ""
    settings.fallback_provider.api_key = ""
    return settings


def wait_for_task(runner: DailyRunner, task_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        tasks = runner.status_tasks()
        match = next((task for task in tasks if task["task_id"] == task_id), None)
        if match and match["status"] in {"completed", "failed"}:
            return match
        time.sleep(0.1)
    raise AssertionError(f"Task {task_id} did not finish in time.")


def force_buy_decision(runner: DailyRunner, mode: str, symbol: str) -> None:
    runtime = runner._runtime(mode)  # type: ignore[arg-type]

    def fake_run(**kwargs):
        return AgentResult(
            payload=TradeIntentSet(
                as_of_date=kwargs["as_of_date"],
                market_view=kwargs["market_view"],
                cash_target=0.2,
                trade_intents=[
                    TradeIntent(
                        symbol=symbol,
                        action="buy",
                        target_weight=0.08,
                        max_weight=0.10,
                        confidence=0.7,
                        thesis="Deterministic test long",
                        holding_horizon_days=10,
                        stop_loss_pct=0.08,
                        take_profit_pct=0.18,
                        time_stop_days=20,
                        evidence_count=3,
                    )
                ],
                rejected_symbols=[],
                portfolio_risks=[],
                decision_confidence=0.7,
                rationale="Deterministic test decision",
                provider_name="test",
            ),
            degrade_mode=False,
            call_records=[],
        )

    runtime.decision_agent.run = fake_run  # type: ignore[method-assign]

    def fake_review(as_of_date, decision, priors, positions, latest_nav, peak_nav):
        return RiskReview(
            as_of_date=as_of_date,
            approved=[
                RiskDecision(
                    symbol=symbol,
                    requested_action="buy",
                    requested_weight=0.08,
                    approved_weight=0.08,
                    final_status="approved",
                    risk_flags=[],
                    rationale="Deterministic test approval",
                )
            ],
            clipped=[],
            delayed=[],
            rejected=[],
            risk_flags=[],
        )

    runtime.risk_service.review = fake_review  # type: ignore[method-assign]


def test_backtest_run_produces_walk_forward_summary(tmp_path) -> None:
    settings = make_settings(tmp_path)
    runner = DailyRunner(settings)

    summary = runner.run_backtest(
        BacktestConfig(
            start_date="2025-01-01",
            end_date="2025-03-31",
            initial_capital=500_000.0,
            watchlist=["000001", "000333", "600519"],
        )
    )

    assert summary.status == "completed"
    assert summary.equity_curve
    assert runner.backtest_runs()[0]["run_id"] == summary.run_id
    assert runner.status_mode("backtest")["state"]["last_run_id"] == summary.run_id


def test_paper_cycle_creates_approvals_and_memory(tmp_path) -> None:
    settings = make_settings(tmp_path)
    runner = DailyRunner(settings)
    force_buy_decision(runner, "paper", "000001")

    summary = runner.run_daily()
    queue = runner.paper_approval_queue()

    assert summary.stage == "waiting_approval"
    assert queue

    result = runner.approve_ticket("paper", queue[0]["ticket_id"])
    memory = runner.memory_snapshot("paper")

    assert result["ticket"]["status"] in {"filled", "blocked", "approved"}
    assert memory["journal"]
    assert any(item["symbol"] == queue[0]["symbol"] for item in memory["positions"])


def test_live_mock_mode_supports_connection_and_approval_flow(tmp_path) -> None:
    settings = make_settings(tmp_path)
    settings.live_broker.enabled = True
    settings.live_broker.provider = "mock"
    runner = DailyRunner(settings)
    force_buy_decision(runner, "live", "000001")

    start_task = runner.start_live_session()
    wait_for_task(runner, start_task.task_id)
    live_account = runner.live_account()
    assert live_account["connected"] is True

    summary = runner.run_cycle("live")
    queue = runner.live_approval_queue()

    assert summary.stage == "waiting_approval"
    assert queue

    result = runner.approve_ticket("live", queue[0]["ticket_id"])
    assert result["ticket"]["status"] in {"sent", "filled"}
    assert result["account"]["connected"] is True
