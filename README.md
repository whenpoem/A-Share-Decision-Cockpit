# A-Share Decision Cockpit

[中文说明](README.zh-CN.md)

A local A-share AI decision workstation with three operating modes:

- `Backtest`: daily walk-forward backtesting
- `Paper`: approval-based simulated trading
- `Live-ready`: manual-approval live workflow with a mock broker and a QMT/Ptrade adapter contract

The system combines structured market priors, real text ingestion, LLM research and decision agents, deterministic risk control, continuous thesis memory, and a React control cockpit.

## What Is Included

- `FastAPI` backend control plane
- `React + Vite` cockpit frontend
- `DeepSeek` as the primary LLM provider
- A deterministic risk shell around all trade intents
- A-share trading rules in simulation:
  - `T+1`
  - 100-share lot size
  - limit-up / limit-down blocking
  - fees and stamp tax
- position memory, thesis memory, decision journal, and portfolio memory
- task tracking, mode state tracking, approval queues, and diagnostics

## V2 Status

The current repository is at a usable `v2` baseline.

Implemented:

- three-mode workflow:
  - `Backtest`
  - `Paper`
  - `Live-ready`
- daily walk-forward backtesting with persisted results
- approval-based paper trading
- live session control, approval flow, and broker status handling
- mock live broker for local end-to-end testing
- `QMT/Ptrade` adapter contract behind a feature gate
- market data refresh with `baostock`, `akshare` financial fallback, and a sample fallback
- real A-share text ingestion:
  - Eastmoney stock news
  - CNINFO disclosure metadata
  - announcement PDF download, text extraction, and local cache
- `ResearchAgent` and `DecisionAgent` with schema-validated JSON outputs
- deterministic risk review with `approved / clipped / delayed / rejected`
- memory persistence for holdings, thesis evolution, and portfolio state
- frontend cockpit for control, approvals, backtests, signals, memory, and diagnostics

Current boundary:

- real QMT/Ptrade order routing still needs local broker SDK wiring on your machine
- the `Live-ready` path already includes the API, state machine, approval queue, mock execution, and adapter contract
- the public text coverage is still daily-research oriented, not intraday news infrastructure

## Architecture

```text
service/server      FastAPI API and WebSocket status feed
service/frontend    React + Vite cockpit
engine/market       A-share market data and prior signals
engine/text         Text ingestion and disclosure caching
engine/agents       ResearchAgent / DecisionAgent / LLM providers
engine/risk         Deterministic risk review
engine/sim          Daily A-share simulation and mark-to-market
engine/memory       Position memory / thesis memory / journal
engine/execution    Mock broker and live-ready adapter contract
engine/runtime      Task manager
engine/storage      SQLite stores for mode data and system state
storage/backtest    Backtest mode artifacts and state
storage/paper       Paper mode artifacts and state
storage/live        Live mode artifacts and state
tests/              Backend tests
```

## Decision Flow

1. Refresh market data and compute prior signals.
2. Select the focus universe from long candidates, avoid candidates, and current positions.
3. Pull news and disclosure events for focus symbols.
4. Build per-symbol `ResearchCard` outputs.
5. Aggregate them into a portfolio-level `TradeIntentSet`.
6. Run deterministic risk checks.
7. Route the result by mode:
   - `Backtest`: execute automatically in the daily simulator
   - `Paper`: place orders into the approval queue
   - `Live-ready`: place orders into the live approval queue
8. Persist runs, journals, positions, approvals, orders, fills, and artifacts.

## Data Sources

The current setup uses:

- `baostock` for A-share symbols and daily bars
- `akshare` as the fallback for financial snapshots
- `akshare.stock_news_em(symbol)` for Eastmoney stock news
- `akshare.stock_zh_a_disclosure_report_cninfo(...)` for CNINFO disclosure metadata

Announcement handling stores:

- `storage/<mode>/text/disclosures/<announcement_id>.json`
- `storage/<mode>/text/disclosures/pdf/<announcement_id>.pdf`
- `storage/<mode>/text/disclosures/text/<announcement_id>.txt`

Long filings are truncated before entering the agent event stream. Full extracted text stays in the local cache.

## Environment

Create and activate the dedicated Conda environment:

```powershell
conda env create -f environment.yml
conda activate ashare-agent
python -m pip install -r requirements.txt
```

Verified backend runtime:

- Python `3.11`

## Configuration

Set environment variables directly or load them from `.env.example`.

Common variables:

```powershell
$env:DEEPSEEK_API_KEY="your-key"
$env:ASHARE_MARKET_PROVIDER="baostock"
$env:ASHARE_TEXT_PROVIDER="akshare"
$env:ASHARE_STORAGE_ROOT="D:\bug\storage"
```

Live-related variables:

```powershell
$env:ASHARE_LIVE_ENABLED="true"
$env:ASHARE_LIVE_BROKER="mock"
$env:ASHARE_LIVE_ACCOUNT_ID="demo-account"
```

Useful options:

- `DEEPSEEK_MODEL`
- `ASHARE_DEFAULT_WATCHLIST`
- `ASHARE_BLACKLIST_SYMBOLS`
- `ASHARE_QMT_TERMINAL_PATH`
- `ASHARE_QMT_USER`
- `ASHARE_QMT_PASSWORD`

If DeepSeek is unavailable or returns unusable structured output, the system falls back to risk-only behavior and blocks new entries.

## Running

Backend:

```powershell
python -m service.server.main
```

Frontend:

```powershell
cd service/frontend
npm install
npm run dev
```

Default local addresses:

- API: `http://127.0.0.1:8000`
- Cockpit: `http://127.0.0.1:5173`
- Status WebSocket: `ws://127.0.0.1:8000/ws/status`

## How To Use

### 1. Backtest

- Open the cockpit
- Switch mode to `BACKTEST`
- Go to the `Backtest` page
- Set:
  - `start date`
  - `end date`
  - `initial capital`
  - optional `watchlist`
- Run the backtest
- Review:
  - task status
  - metrics
  - equity curve
  - daily decision timeline

### 2. Paper Trading

- Switch mode to `PAPER`
- Run a paper cycle from `Control Center`
- Open `Paper Trading`
- Review pending approvals
- Approve or reject each ticket
- Check updated positions, memory, and diagnostics

### 3. Live-ready

- Set live-related environment variables
- Start a live session from `Control Center`
- Open `Live Trading`
- Review broker status and approval queue
- Approve tickets manually

With `ASHARE_LIVE_BROKER=mock`, the full approval-to-fill loop works locally.

With a real broker path, the adapter contract is ready and the local SDK wiring is still required.

## API Overview

Core endpoints:

- `POST /api/control/run-backtest`
- `POST /api/control/run-paper-cycle`
- `POST /api/control/run-live-cycle`
- `POST /api/control/start-live-session`
- `POST /api/control/stop-live-session`
- `POST /api/control/rebuild-memory`
- `GET /api/status/system`
- `GET /api/status/tasks`
- `GET /api/status/mode/{mode}`
- `GET /api/backtest/runs`
- `GET /api/backtest/runs/{run_id}`
- `GET /api/paper/account`
- `GET /api/live/account`
- `GET /api/paper/approval-queue`
- `GET /api/live/approval-queue`
- `POST /api/paper/approve/{ticket_id}`
- `POST /api/paper/reject/{ticket_id}`
- `POST /api/live/approve/{ticket_id}`
- `POST /api/live/reject/{ticket_id}`
- `GET /api/dashboard/summary?mode=paper|live|backtest`
- `GET /api/signals/today?mode=paper|live|backtest`
- `GET /api/memory/{mode}`
- `WS /ws/status`

Compatibility endpoint kept for older scripts:

- `POST /api/sim/run-daily`

## Testing

Backend:

```powershell
D:\miniconda\envs\ashare-agent\python.exe -m pytest -q
```

Frontend build:

```powershell
cd service/frontend
npm run build
```

Latest verification:

- backend: `9 passed`
- frontend build: passed

## Roadmap

Next high-value steps:

- local QMT/Ptrade SDK wiring on the live adapter
- richer A-share text coverage and source ranking
- regime-aware throttling beyond the current rule set
- backtest result drill-down charts in the cockpit
- stronger execution audit views for live mode

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
