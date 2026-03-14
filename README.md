# A-Share Local Decision Workstation

[Chinese](README.zh-CN.md)

This repository has been rebuilt from the old probability-research project into a local A-share decision workstation inspired by the `AI-Trader` flow:

- `market -> text -> research -> decision -> risk -> simulation`
- `FastAPI` backend
- `React + Vite` cockpit
- `DeepSeek` primary LLM with `Qwen` fallback
- deterministic risk shell around LLM trade intents
- daily A-share simulation with `T+1`, lot-size, limit-up/limit-down, fees, and stamp tax

It is designed for local single-user research and paper trading, not for social signals or copy trading.

## Current Status

The current `v1` is working in these areas:

- market data refresh with `akshare` and a sample fallback provider
- lightweight prior scoring for candidate selection
- real text ingestion for A-shares:
  - Eastmoney stock news via `stock_news_em`
  - CNINFO disclosure lists via `stock_zh_a_disclosure_report_cninfo`
  - announcement PDF body extraction and local cache
- `ResearchAgent` and `DecisionAgent` with structured JSON outputs
- deterministic risk review before any order enters simulation
- daily simulation and portfolio state persistence in SQLite
- local cockpit for reviewing signals, portfolio, and run history

Still intentionally out of scope for `v1`:

- real brokerage execution
- full announcement body extraction
- intraday or high-frequency trading
- portfolio optimization beyond the current rule-based risk layer

## Architecture

```text
service/server      FastAPI API
service/frontend    React + Vite cockpit
engine/market       A-share market data and prior scores
engine/text         Real text providers and derived events
engine/agents       ResearchAgent / DecisionAgent / LLM providers
engine/risk         Deterministic risk shell
engine/sim          Daily A-share simulation
engine/execution    Future vn.py / QMT adapters
engine/storage      SQLite persistence
storage/            Local artifacts, parquet, jsonl, state.db
tests/              Backend tests
```

## Decision Flow

1. Refresh market data and compute prior signals.
2. Select a focus universe from long candidates, avoid candidates, and current positions.
3. Pull real text events for focus symbols and merge them with derived risk/price events.
4. Generate per-symbol `ResearchCard` outputs.
5. Aggregate them into portfolio-level `TradeIntent` proposals.
6. Run deterministic risk checks that can approve, clip, delay, or reject each intent.
7. Send approved intents into the daily A-share simulator.
8. Persist runs, cards, decisions, risk reviews, positions, fills, and artifacts.

## Data Sources

`v1` currently uses:

- `akshare` market data for A-share symbols and daily bars
- `akshare.stock_news_em(symbol)` for stock news
- `akshare.stock_zh_a_disclosure_report_cninfo(...)` for disclosure metadata

Announcement handling now downloads the disclosure PDF when available, extracts body text locally,
and caches three artifacts:

- `storage/text/disclosures/<announcement_id>.json`
- `storage/text/disclosures/pdf/<announcement_id>.pdf`
- `storage/text/disclosures/text/<announcement_id>.txt`

Current limitation:

- very long filings are truncated before being passed into the event stream; full extracted text remains in the local cache

## Environment

An isolated Conda environment is provided:

```powershell
conda env create -f environment.yml
conda activate ashare-agent
python -m pip install -r requirements.txt
```

Verified backend runtime:

- Python `3.11`

## Configuration

Copy `.env.example` into your shell environment or set variables directly:

```powershell
$env:DEEPSEEK_API_KEY="your-key"
$env:QWEN_API_KEY="your-key"
$env:ASHARE_MARKET_PROVIDER="akshare"
$env:ASHARE_TEXT_PROVIDER="akshare"
$env:ASHARE_TEXT_LOOKBACK_DAYS="30"
$env:ASHARE_MAX_NEWS_PER_SYMBOL="6"
$env:ASHARE_MAX_ANNOUNCEMENTS_PER_SYMBOL="6"
```

Useful options:

- `DEEPSEEK_MODEL`
- `QWEN_MODEL`
- `ASHARE_STORAGE_ROOT`
- `ASHARE_DEFAULT_WATCHLIST`
- `ASHARE_BLACKLIST_SYMBOLS`

If both LLM providers are unavailable, the system automatically degrades into risk-only mode and blocks new entries.

## Running

Backend:

```powershell
python -m service.server.main
```

Or use the PowerShell helper:

```powershell
.\scripts\run_backend.ps1
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

## API

- `GET /api/health`
- `POST /api/data/refresh-market`
- `POST /api/data/refresh-text`
- `POST /api/research/run`
- `POST /api/decision/run`
- `POST /api/risk/run`
- `POST /api/sim/run-daily`
- `GET /api/dashboard/summary`
- `GET /api/signals/today`
- `GET /api/portfolio/current`
- `GET /api/runs/{run_id}`
- `GET /api/backtest/summary`

## Testing

Backend tests:

```powershell
D:\miniconda\envs\ashare-agent\python.exe -m pytest -q
```

The latest verification passed with `6` tests.

## Roadmap

High-value next steps:

- announcement body extraction and caching
- better A-share text coverage and deduplication
- regime-aware risk throttling
- vn.py / QMT execution adapter
- more explicit run auditing in the frontend

## License

No standalone license file is currently included in this rebuilt repository. Add one before distributing the project externally.
