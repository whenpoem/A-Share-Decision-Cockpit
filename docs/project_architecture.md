# Project Architecture

```mermaid
flowchart LR
    A["CLI Entrypoints<br/>run_cli.py / astock_prob.cli"] --> B["Config Loader<br/>configs/*.json"]
    B --> C["Data Ingestion<br/>AkShare / CSV Provider"]
    C --> D["Raw Cache<br/>price / valuation / financial / benchmark CSV"]
    D --> E["Feature Engineering<br/>returns / vol / ATR / MACD / RSI / valuation / fundamentals"]
    D --> F["Label Generation<br/>60-day terminal bucket + touch events"]
    E --> G["Model Dataset"]
    F --> G
    G --> H["GBM Baseline<br/>EWMA drift + volatility + Monte Carlo"]
    G --> I["Bootstrap Baseline<br/>historical analog paths"]
    G --> J["ML Layer<br/>LightGBM or sklearn fallback"]
    H --> K["Validation Blend"]
    I --> K
    J --> K
    K --> L["Calibration<br/>temperature scaling / isotonic"]
    L --> M["Constraints<br/>monotonic touch probabilities"]
    M --> N["Artifacts<br/>latest.pkl / metrics.json"]
    N --> O["Predict"]
    G --> P["Walk-forward Backtest"]
    O --> Q["Reports<br/>CSV / Markdown / HTML / PNG"]
    P --> Q
```

## Layer Notes

- CLI layer: exposes `fetch-data`, `train`, `backtest`, `predict`, and `run-live`.
- Data layer: fetches raw market and financial data, normalizes fields, and preserves existing cache on transient failures.
- Feature layer: converts time series into model-ready daily factors.
- Label layer: creates future 60-trading-day terminal-return buckets and up/down touch events.
- Modeling layer: combines parametric baseline, non-parametric baseline, and ML probabilities.
- Calibration layer: improves probabilistic reliability rather than only ranking ability.
- Reporting layer: emits analyst-facing outputs for both latest prediction and walk-forward quality review.

## Maintenance Rules

- If a data interface becomes unstable, patch the provider first and keep the downstream schema unchanged.
- If you add factors, keep them point-in-time aligned and add tests around future leakage.
- If you add a new model, plug it into the ensemble after validation and calibration rather than replacing all baselines.
- If you extend coverage beyond two stocks, keep symbol configuration externalized in `configs/*.json`.
