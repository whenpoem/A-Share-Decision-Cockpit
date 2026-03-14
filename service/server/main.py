from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine.config import Settings
from engine.orchestration.daily_runner import DailyRunner
from engine.types import BacktestConfig, Mode


class DailyRunRequest(BaseModel):
    as_of_date: Optional[str] = None


class RefreshMarketRequest(BaseModel):
    mode: Mode = "paper"
    start_date: str = "2023-01-01"
    end_date: Optional[str] = None
    watchlist: list[str] = Field(default_factory=list)


class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    initial_capital: float = 1_000_000.0
    watchlist: list[str] = Field(default_factory=list)


class MemoryRequest(BaseModel):
    mode: Mode = "paper"


settings = Settings.load()
runner = DailyRunner(settings)
app = FastAPI(title="A-Share Decision Cockpit API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/data/refresh-market")
def refresh_market(request: RefreshMarketRequest) -> dict:
    snapshot = runner.refresh_market(
        mode=request.mode,
        start_date=request.start_date,
        end_date=request.end_date,
        watchlist=request.watchlist or None,
    )
    return {
        "mode": request.mode,
        "as_of_date": snapshot.as_of_date.isoformat(),
        "symbols": [item.symbol for item in snapshot.priors],
        "market_regime": snapshot.priors[0].market_regime if snapshot.priors else "neutral",
    }


@app.post("/api/control/run-backtest")
def run_backtest(request: BacktestRequest) -> dict:
    task = runner.run_backtest_task(
        BacktestConfig(
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            watchlist=request.watchlist,
        )
    )
    return {"task": task.model_dump(mode="json")}


@app.post("/api/control/run-paper-cycle")
def run_paper_cycle(request: DailyRunRequest) -> dict:
    task = runner.run_paper_cycle_task(request.as_of_date)
    return {"task": task.model_dump(mode="json")}


@app.post("/api/control/run-live-cycle")
def run_live_cycle(request: DailyRunRequest) -> dict:
    task = runner.run_live_cycle_task(request.as_of_date)
    return {"task": task.model_dump(mode="json")}


@app.post("/api/control/start-live-session")
def start_live_session() -> dict:
    task = runner.start_live_session()
    return {"task": task.model_dump(mode="json")}


@app.post("/api/control/stop-live-session")
def stop_live_session() -> dict:
    task = runner.stop_live_session()
    return {"task": task.model_dump(mode="json")}


@app.post("/api/control/rebuild-memory")
def rebuild_memory(request: MemoryRequest) -> dict:
    task = runner.rebuild_memory(request.mode)
    return {"task": task.model_dump(mode="json")}


@app.get("/api/status/system")
def status_system() -> dict:
    return runner.status_system()


@app.get("/api/status/tasks")
def status_tasks() -> list[dict]:
    return runner.status_tasks()


@app.get("/api/status/mode/{mode}")
def status_mode(mode: Literal["backtest", "paper", "live"]) -> dict:
    return runner.status_mode(mode)


@app.get("/api/backtest/runs")
def backtest_runs() -> dict:
    return {"runs": runner.backtest_runs()}


@app.get("/api/backtest/runs/{run_id}")
def backtest_run_details(run_id: str) -> dict:
    return runner.backtest_run_details(run_id)


@app.get("/api/paper/account")
def paper_account() -> dict:
    return runner.paper_account()


@app.get("/api/live/account")
def live_account() -> dict:
    return runner.live_account()


@app.get("/api/live/approval-queue")
def live_approval_queue() -> dict:
    return {"tickets": runner.live_approval_queue()}


@app.get("/api/paper/approval-queue")
def paper_approval_queue() -> dict:
    return {"tickets": runner.paper_approval_queue()}


@app.post("/api/live/approve/{ticket_id}")
def live_approve(ticket_id: str) -> dict:
    return runner.approve_ticket("live", ticket_id)


@app.post("/api/live/reject/{ticket_id}")
def live_reject(ticket_id: str) -> dict:
    return runner.reject_ticket("live", ticket_id)


@app.post("/api/paper/approve/{ticket_id}")
def paper_approve(ticket_id: str) -> dict:
    return runner.approve_ticket("paper", ticket_id)


@app.post("/api/paper/reject/{ticket_id}")
def paper_reject(ticket_id: str) -> dict:
    return runner.reject_ticket("paper", ticket_id)


@app.get("/api/dashboard/summary")
def dashboard_summary(mode: Mode = Query("paper")) -> dict:
    return runner.dashboard_summary(mode)


@app.get("/api/signals/today")
def today_signals(mode: Mode = Query("paper")) -> dict:
    return runner.today_signals(mode)


@app.get("/api/portfolio/current")
def portfolio_current(mode: Mode = Query("paper")) -> dict:
    return runner.current_portfolio(mode)


@app.get("/api/runs/{mode}/{run_id}")
def run_details(mode: Literal["backtest", "paper", "live"], run_id: str) -> dict:
    return runner.run_details(mode, run_id)


@app.get("/api/memory/{mode}")
def memory_snapshot(mode: Literal["backtest", "paper", "live"]) -> dict:
    return runner.memory_snapshot(mode)


@app.get("/api/diagnostics")
def diagnostics() -> dict:
    return runner.diagnostics()


@app.get("/api/backtest/summary")
def backtest_summary() -> dict:
    return runner.backtest_summary()


@app.post("/api/sim/run-daily")
def compat_run_daily(request: DailyRunRequest) -> dict:
    summary = runner.run_daily(request.as_of_date)
    return {"summary": summary.model_dump(mode="json")}


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(runner.status_system())
            await asyncio.sleep(1.0)
    except Exception:
        await websocket.close()


def run() -> None:
    uvicorn.run("service.server.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
