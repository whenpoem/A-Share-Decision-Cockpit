from __future__ import annotations

from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.config import Settings
from engine.orchestration.daily_runner import DailyRunner


class DailyRunRequest(BaseModel):
    as_of_date: Optional[str] = None


class RefreshMarketRequest(BaseModel):
    start_date: str = "2023-01-01"
    end_date: Optional[str] = None


settings = Settings.load()
runner = DailyRunner(settings)
app = FastAPI(title="A-Share Local Decision Workstation", version="0.1.0")
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
    snapshot = runner.refresh_market(start_date=request.start_date, end_date=request.end_date)
    return {
        "as_of_date": snapshot.as_of_date.isoformat(),
        "symbols": [item.symbol for item in snapshot.priors],
        "market_regime": snapshot.priors[0].market_regime if snapshot.priors else "neutral",
    }


@app.post("/api/data/refresh-text")
def refresh_text() -> dict:
    snapshot = runner.refresh_market()
    events = runner.refresh_text(snapshot)
    return {"events": len(events), "symbols": sorted({event.symbol for event in events})}


@app.post("/api/research/run")
def run_research(request: DailyRunRequest) -> dict:
    summary = runner.run_daily(request.as_of_date)
    signals = runner.today_signals()
    return {"summary": summary.model_dump(mode="json"), "cards": signals["cards"]}


@app.post("/api/decision/run")
def run_decision(request: DailyRunRequest) -> dict:
    summary = runner.run_daily(request.as_of_date)
    signals = runner.today_signals()
    return {"summary": summary.model_dump(mode="json"), "decision": signals["decision"]}


@app.post("/api/risk/run")
def run_risk(request: DailyRunRequest) -> dict:
    summary = runner.run_daily(request.as_of_date)
    signals = runner.today_signals()
    return {"summary": summary.model_dump(mode="json"), "risk": signals["risk"]}


@app.post("/api/sim/run-daily")
def run_daily(request: DailyRunRequest) -> dict:
    summary = runner.run_daily(request.as_of_date)
    return {"summary": summary.model_dump(mode="json")}


@app.get("/api/dashboard/summary")
def dashboard_summary() -> dict:
    return runner.dashboard_summary()


@app.get("/api/signals/today")
def today_signals() -> dict:
    return runner.today_signals()


@app.get("/api/portfolio/current")
def portfolio_current() -> dict:
    return runner.current_portfolio()


@app.get("/api/runs/{run_id}")
def run_details(run_id: str) -> dict:
    return runner.run_details(run_id)


@app.get("/api/backtest/summary")
def backtest_summary() -> dict:
    return runner.backtest_summary()


def run() -> None:
    uvicorn.run("service.server.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
