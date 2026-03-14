import { useEffect, useState } from "react";

type DashboardSummary = {
  run: null | { run_id: string; as_of_date: string; status: string; summary: any };
  portfolio: { positions: any[]; gross_exposure: number; nav: number };
  signals: any[];
  risk: { traffic_light: string; flags: string[] };
};

type SignalsPayload = {
  run: null | { run_id: string; as_of_date: string; status: string };
  cards: any[];
  decision: any;
  risk: any;
};

type BacktestSummary = {
  runs: number;
  total_return: number;
  max_drawdown: number;
  equity_curve: Array<{ as_of_date: string; nav: number }>;
};

const API = "http://127.0.0.1:8000";

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export default function App() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [signals, setSignals] = useState<SignalsPayload | null>(null);
  const [backtest, setBacktest] = useState<BacktestSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      setError(null);
      const [dashboard, todaySignals, simSummary] = await Promise.all([
        getJson<DashboardSummary>("/api/dashboard/summary"),
        getJson<SignalsPayload>("/api/signals/today"),
        getJson<BacktestSummary>("/api/backtest/summary"),
      ]);
      setSummary(dashboard);
      setSignals(todaySignals);
      setBacktest(simSummary);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const runDaily = async () => {
    try {
      setLoading(true);
      setError(null);
      await getJson("/api/sim/run-daily", { method: "POST", body: JSON.stringify({}) });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "运行失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div>
          <p className="eyebrow">AI-Trader Inspired</p>
          <h1>A股决策驾驶舱</h1>
          <p className="lede">
            LLM 原始建议、硬风控裁决与日频模拟盘在一个本地工作台里串起来。
          </p>
        </div>
        <div className="sidebar-card">
          <button className="primary" onClick={runDaily} disabled={loading}>
            {loading ? "正在运行..." : "运行今日流程"}
          </button>
          <button className="secondary" onClick={refresh}>
            刷新面板
          </button>
        </div>
        <div className="sidebar-card">
          <span className={`lamp lamp-${summary?.risk.traffic_light ?? "amber"}`} />
          <div>
            <strong>风控灯</strong>
            <p>{summary?.risk.flags?.join(" / ") || "尚未运行"}</p>
          </div>
        </div>
      </aside>

      <main className="main-grid">
        {error ? <div className="banner error">{error}</div> : null}
        <section className="hero-card">
          <div>
            <p className="eyebrow">Decision Cockpit</p>
            <h2>
              {summary?.run ? `最新运行 ${summary.run.run_id}` : "还没有运行记录"}
            </h2>
            <p>
              {summary?.run
                ? `观察日 ${summary.run.as_of_date}，状态 ${summary.run.status}`
                : "点击左侧按钮生成第一批研究卡、决策与模拟结果。"}
            </p>
          </div>
          <div className="hero-stats">
            <div>
              <span>组合净值</span>
              <strong>{summary ? summary.portfolio.nav.toLocaleString() : "--"}</strong>
            </div>
            <div>
              <span>总仓位</span>
              <strong>{summary ? pct(summary.portfolio.gross_exposure) : "--"}</strong>
            </div>
            <div>
              <span>模拟运行数</span>
              <strong>{backtest?.runs ?? 0}</strong>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h3>今日候选与研究卡</h3>
            <span>{signals?.cards.length ?? 0} 只</span>
          </div>
          <div className="signal-list">
            {(signals?.cards ?? []).slice(0, 8).map((card) => (
              <article className="signal-card" key={card.symbol}>
                <div className="signal-topline">
                  <strong>{card.symbol}</strong>
                  <span className={`tag tag-${card.stance}`}>{card.stance}</span>
                  <span className="tag quality">{card.event_quality}</span>
                </div>
                <p>{card.summary}</p>
                <div className="chip-row">
                  {(card.drivers ?? []).slice(0, 3).map((item: string) => (
                    <span className="chip" key={item}>
                      {item}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h3>LLM 原始建议 vs 风控后建议</h3>
          </div>
          <div className="decision-grid">
            <div className="subpanel">
              <h4>原始决策</h4>
              <p className="subpanel-note">{signals?.decision?.rationale ?? "暂无"}</p>
              {(signals?.decision?.trade_intents ?? []).slice(0, 8).map((intent: any) => (
                <div className="intent-row" key={`${intent.symbol}-${intent.action}`}>
                  <span>{intent.symbol}</span>
                  <strong>{intent.action}</strong>
                  <span>{pct(intent.target_weight)}</span>
                </div>
              ))}
            </div>
            <div className="subpanel">
              <h4>风控裁决</h4>
              {["approved", "clipped", "delayed", "rejected"].map((bucket) => (
                <div key={bucket} className="risk-bucket">
                  <div className="risk-bucket-title">{bucket}</div>
                  {((signals?.risk?.[bucket] ?? []) as any[]).slice(0, 6).map((item) => (
                    <div className="intent-row" key={`${bucket}-${item.symbol}`}>
                      <span>{item.symbol}</span>
                      <strong>{pct(item.approved_weight)}</strong>
                      <span>{(item.risk_flags ?? []).join(", ") || "ok"}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h3>组合与风险</h3>
            <span>{summary?.portfolio.positions.length ?? 0} 个持仓</span>
          </div>
          <div className="table-like">
            <div className="table-head">
              <span>标的</span>
              <span>市值</span>
              <span>成本</span>
              <span>最新价</span>
            </div>
            {(summary?.portfolio.positions ?? []).map((position) => (
              <div className="table-row" key={position.symbol}>
                <span>{position.symbol}</span>
                <span>{Number(position.market_value).toFixed(0)}</span>
                <span>{Number(position.avg_cost).toFixed(2)}</span>
                <span>{Number(position.last_price).toFixed(2)}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h3>Simulation</h3>
            <span>
              收益 {backtest ? pct(backtest.total_return) : "--"} / 回撤{" "}
              {backtest ? pct(backtest.max_drawdown) : "--"}
            </span>
          </div>
          <div className="equity-list">
            {(backtest?.equity_curve ?? []).slice(-8).map((point) => (
              <div className="intent-row" key={point.as_of_date}>
                <span>{point.as_of_date}</span>
                <strong>{point.nav.toFixed(0)}</strong>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

