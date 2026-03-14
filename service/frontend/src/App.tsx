import { useEffect, useMemo, useState } from "react";

type Mode = "backtest" | "paper" | "live";
type Tab = "control" | "backtest" | "paper" | "live" | "signals" | "memory" | "diagnostics";

type ChartPoint = {
  label: string;
  value: number;
};

const API = "http://127.0.0.1:8000";

const MODE_LABELS: Record<Mode, string> = {
  backtest: "回测",
  paper: "模拟盘",
  live: "真实盘",
};

const TAB_LABELS: Record<Tab, string> = {
  control: "控制台",
  backtest: "回测",
  paper: "模拟盘",
  live: "真实盘",
  signals: "信号与决策",
  memory: "记忆",
  diagnostics: "诊断",
};

const TASK_STATUS_LABELS: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

const TASK_TYPE_LABELS: Record<string, string> = {
  refresh_market: "刷新行情",
  refresh_text: "刷新文本",
  run_backtest: "运行回测",
  run_paper_cycle: "运行模拟盘周期",
  run_live_cycle: "运行真实盘周期",
  start_live_session: "启动真实盘会话",
  stop_live_session: "停止真实盘会话",
  rebuild_memory: "重建记忆",
  sync_broker_state: "同步券商状态",
};

const MODE_STATUS_LABELS: Record<string, string> = {
  idle: "空闲",
  running: "运行中",
  waiting_approval: "等待审批",
  connected: "已连接",
  disabled: "已禁用",
  error: "异常",
};

const RUN_STATUS_LABELS: Record<string, string> = {
  ok: "正常",
  degraded: "降级",
  error: "错误",
};

const STANCE_LABELS: Record<string, string> = {
  bullish: "偏多",
  bearish: "偏空",
  neutral: "中性",
};

const EVENT_QUALITY_LABELS: Record<string, string> = {
  verified: "已验证",
  mixed: "混合",
  weak: "较弱",
};

const ACTION_LABELS: Record<string, string> = {
  buy: "买入",
  sell: "卖出",
  reduce: "减仓",
  hold: "持有",
  open_long: "开仓偏多",
  trim: "缩仓",
  avoid: "回避",
};

const METRIC_LABELS: Record<string, string> = {
  total_return: "累计收益",
  annual_return: "年化收益",
  annual_volatility: "年化波动",
  max_drawdown: "最大回撤",
  sortino: "Sortino",
  win_rate: "胜率",
  profit_factor: "盈亏比",
  avg_holding_days: "平均持有天数",
  avg_turnover: "日均换手",
  fee_ratio: "费用占比",
  excess_return: "超额收益",
};

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

function labelOf(table: Record<string, string>, value: string | undefined, fallback = "--"): string {
  if (!value) {
    return fallback;
  }
  return table[value] ?? value;
}

function fmtPct(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function fmtNum(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return value.toLocaleString("zh-CN", { maximumFractionDigits: 2 });
}

function fmtDate(value: string | undefined): string {
  if (!value) {
    return "--";
  }
  return value.slice(0, 10);
}

function cumulativeReturn(points: ChartPoint[]): number | undefined {
  if (points.length < 2 || points[0].value === 0) {
    return undefined;
  }
  return points[points.length - 1].value / points[0].value - 1;
}

function toBacktestNavPoints(summary: any): ChartPoint[] {
  return (summary?.equity_curve ?? []).map((item: any) => ({
    label: item.as_of_date,
    value: Number(item.nav ?? 0),
  }));
}

function toDrawdownPoints(summary: any): ChartPoint[] {
  return (summary?.drawdown_curve ?? []).map((item: any) => ({
    label: item.as_of_date,
    value: Number(item.drawdown ?? 0),
  }));
}

function toPortfolioNavPoints(memory: any): ChartPoint[] {
  return (memory?.portfolio_history ?? [])
    .map((item: any) => ({
      label: item.as_of_date,
      value: Number(item.payload?.current_nav ?? 0),
    }))
    .filter((item: ChartPoint) => item.value > 0);
}

function LineChart({
  title,
  points,
  percent = false,
  tone = "accent",
}: {
  title: string;
  points: ChartPoint[];
  percent?: boolean;
  tone?: "accent" | "danger";
}) {
  if (points.length === 0) {
    return (
      <div className="chart-card">
        <div className="chart-header">
          <h4>{title}</h4>
        </div>
        <div className="chart-empty">暂无可视化数据</div>
      </div>
    );
  }

  const width = 760;
  const height = 220;
  const padding = 24;
  const values = points.map((item) => item.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || Math.max(Math.abs(max), 1);
  const path = points
    .map((item, index) => {
      const x = padding + (index / Math.max(points.length - 1, 1)) * (width - padding * 2);
      const y = height - padding - ((item.value - min) / span) * (height - padding * 2);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  const stroke = tone === "danger" ? "var(--red)" : "var(--accent)";
  const latest = points[points.length - 1];
  const first = points[0];

  return (
    <div className="chart-card">
      <div className="chart-header">
        <div>
          <h4>{title}</h4>
          <p>
            起点 {fmtDate(first.label)} · 终点 {fmtDate(latest.label)}
          </p>
        </div>
        <strong>{percent ? fmtPct(latest.value) : fmtNum(latest.value)}</strong>
      </div>
      <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <path className="chart-grid" d={`M ${padding} ${padding} H ${width - padding}`} />
        <path
          className="chart-grid"
          d={`M ${padding} ${height / 2} H ${width - padding}`}
        />
        <path
          className="chart-grid"
          d={`M ${padding} ${height - padding} H ${width - padding}`}
        />
        <path d={path} fill="none" stroke={stroke} strokeWidth="3.5" strokeLinecap="round" />
      </svg>
      <div className="chart-footer">
        <span>{percent ? fmtPct(min) : fmtNum(min)}</span>
        <span>{percent ? fmtPct(max) : fmtNum(max)}</span>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function JsonPanel({ title, payload }: { title: string; payload: any }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>{title}</h3>
      </div>
      <pre className="code-block">{payload ? JSON.stringify(payload, null, 2) : "暂无数据"}</pre>
    </section>
  );
}

function App() {
  const [mode, setMode] = useState<Mode>("paper");
  const [tab, setTab] = useState<Tab>("control");
  const [system, setSystem] = useState<any>(null);
  const [modeStatus, setModeStatus] = useState<any>(null);
  const [dashboard, setDashboard] = useState<any>(null);
  const [signals, setSignals] = useState<any>(null);
  const [memory, setMemory] = useState<any>(null);
  const [diagnostics, setDiagnostics] = useState<any>(null);
  const [backtestRuns, setBacktestRuns] = useState<any[]>([]);
  const [selectedBacktestRunId, setSelectedBacktestRunId] = useState<string>("");
  const [paperQueue, setPaperQueue] = useState<any[]>([]);
  const [liveQueue, setLiveQueue] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [backtestForm, setBacktestForm] = useState({
    start_date: "2025-01-01",
    end_date: "2025-12-31",
    initial_capital: "1000000",
    watchlist: "",
  });

  const taskList = system?.tasks ?? [];
  const latestTask = taskList[0];
  const selectedBacktest = useMemo(
    () => backtestRuns.find((item) => item.run_id === selectedBacktestRunId)?.summary ?? null,
    [backtestRuns, selectedBacktestRunId],
  );

  const refresh = async (currentMode: Mode = mode) => {
    try {
      setError(null);
      const queueEndpoint =
        currentMode === "live" ? "/api/live/approval-queue" : "/api/paper/approval-queue";
      const [
        systemPayload,
        modePayload,
        dashboardPayload,
        signalsPayload,
        memoryPayload,
        diagnosticsPayload,
        backtestPayload,
        queuePayload,
      ] = await Promise.all([
        getJson<any>("/api/status/system"),
        getJson<any>(`/api/status/mode/${currentMode}`),
        getJson<any>(`/api/dashboard/summary?mode=${currentMode}`),
        getJson<any>(`/api/signals/today?mode=${currentMode}`),
        getJson<any>(`/api/memory/${currentMode}`),
        getJson<any>("/api/diagnostics"),
        getJson<any>("/api/backtest/runs"),
        getJson<any>(queueEndpoint),
      ]);
      setSystem(systemPayload);
      setModeStatus(modePayload);
      setDashboard(dashboardPayload);
      setSignals(signalsPayload);
      setMemory(memoryPayload);
      setDiagnostics(diagnosticsPayload);
      setBacktestRuns(backtestPayload.runs ?? []);
      if (!selectedBacktestRunId && (backtestPayload.runs?.length ?? 0) > 0) {
        setSelectedBacktestRunId(backtestPayload.runs[0].run_id);
      }
      if (currentMode === "paper") {
        setPaperQueue(queuePayload.tickets ?? []);
      }
      if (currentMode === "live") {
        setLiveQueue(queuePayload.tickets ?? []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载控制台失败");
    }
  };

  useEffect(() => {
    refresh(mode);
  }, [mode]);

  useEffect(() => {
    const timer = window.setInterval(() => refresh(mode), 5000);
    return () => window.clearInterval(timer);
  }, [mode]);

  useEffect(() => {
    const socket = new WebSocket("ws://127.0.0.1:8000/ws/status");
    socket.onmessage = (event) => {
      try {
        setSystem(JSON.parse(event.data));
      } catch {
        return;
      }
    };
    return () => socket.close();
  }, []);

  const submitAction = async (label: string, path: string, init?: RequestInit) => {
    try {
      setLoading(true);
      setError(null);
      await getJson(path, init);
      await refresh(mode);
    } catch (err) {
      setError(err instanceof Error ? `${label}失败: ${err.message}` : `${label}失败`);
    } finally {
      setLoading(false);
    }
  };

  const submitBacktest = async () => {
    await submitAction("运行回测", "/api/control/run-backtest", {
      method: "POST",
      body: JSON.stringify({
        start_date: backtestForm.start_date,
        end_date: backtestForm.end_date,
        initial_capital: Number(backtestForm.initial_capital),
        watchlist: backtestForm.watchlist
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean),
      }),
    });
    setTab("backtest");
  };

  const approvalActions = useMemo(
    () => ({
      approve: async (ticketId: string) =>
        submitAction(
          "批准订单",
          `${mode === "live" ? "/api/live/approve" : "/api/paper/approve"}/${ticketId}`,
          { method: "POST" },
        ),
      reject: async (ticketId: string) =>
        submitAction(
          "拒绝订单",
          `${mode === "live" ? "/api/live/reject" : "/api/paper/reject"}/${ticketId}`,
          { method: "POST" },
        ),
    }),
    [mode],
  );

  const currentNavSeries = useMemo(() => toPortfolioNavPoints(memory), [memory]);
  const backtestNavSeries = useMemo(() => toBacktestNavPoints(selectedBacktest), [selectedBacktest]);
  const backtestDrawdownSeries = useMemo(
    () => toDrawdownPoints(selectedBacktest),
    [selectedBacktest],
  );

  const renderControl = () => (
    <div className="grid two">
      <section className="panel">
        <div className="panel-head">
          <h3>运行控制</h3>
        </div>
        <div className="action-grid">
          <button
            onClick={() =>
              submitAction("运行模拟盘周期", "/api/control/run-paper-cycle", {
                method: "POST",
                body: JSON.stringify({}),
              })
            }
            disabled={loading}
          >
            运行模拟盘周期
          </button>
          <button
            onClick={() =>
              submitAction("运行真实盘周期", "/api/control/run-live-cycle", {
                method: "POST",
                body: JSON.stringify({}),
              })
            }
            disabled={loading}
          >
            运行真实盘周期
          </button>
          <button
            onClick={() =>
              submitAction("启动真实盘会话", "/api/control/start-live-session", {
                method: "POST",
              })
            }
            disabled={loading}
          >
            启动真实盘会话
          </button>
          <button
            onClick={() =>
              submitAction("停止真实盘会话", "/api/control/stop-live-session", {
                method: "POST",
              })
            }
            disabled={loading}
          >
            停止真实盘会话
          </button>
          <button
            onClick={() =>
              submitAction("重建记忆", "/api/control/rebuild-memory", {
                method: "POST",
                body: JSON.stringify({ mode }),
              })
            }
            disabled={loading}
          >
            重建记忆
          </button>
          <button onClick={() => refresh(mode)} disabled={loading}>
            刷新状态
          </button>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h3>任务列表</h3>
        </div>
        <div className="list">
          {taskList.length === 0 ? <p className="muted">暂无任务记录。</p> : null}
          {taskList.map((task: any) => (
            <div key={task.task_id} className="list-row">
              <div>
                <strong>{labelOf(TASK_TYPE_LABELS, task.task_type)}</strong>
                <p>{labelOf(MODE_LABELS, task.mode as Mode, "系统")}</p>
              </div>
              <div className="list-meta">
                <strong>{labelOf(TASK_STATUS_LABELS, task.status)}</strong>
                <span>{fmtPct(task.progress)}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel span-two">
        <LineChart
          title={`${MODE_LABELS[mode]}净值曲线`}
          points={currentNavSeries}
        />
      </section>
    </div>
  );

  const renderBacktest = () => (
    <div className="grid two">
      <section className="panel">
        <div className="panel-head">
          <h3>运行回测</h3>
        </div>
        <div className="form-grid">
          <label>
            开始日期
            <input
              value={backtestForm.start_date}
              onChange={(event) =>
                setBacktestForm((prev) => ({ ...prev, start_date: event.target.value }))
              }
            />
          </label>
          <label>
            结束日期
            <input
              value={backtestForm.end_date}
              onChange={(event) =>
                setBacktestForm((prev) => ({ ...prev, end_date: event.target.value }))
              }
            />
          </label>
          <label>
            初始资金
            <input
              value={backtestForm.initial_capital}
              onChange={(event) =>
                setBacktestForm((prev) => ({ ...prev, initial_capital: event.target.value }))
              }
            />
          </label>
          <label className="full">
            股票池
            <input
              value={backtestForm.watchlist}
              placeholder="例如: 000001,600519,300750"
              onChange={(event) =>
                setBacktestForm((prev) => ({ ...prev, watchlist: event.target.value }))
              }
            />
          </label>
        </div>
        <button onClick={submitBacktest} disabled={loading}>
          启动回测
        </button>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h3>回测记录</h3>
        </div>
        <div className="list">
          {backtestRuns.length === 0 ? <p className="muted">暂无回测记录。</p> : null}
          {backtestRuns.map((item) => (
            <button
              key={item.run_id}
              className={item.run_id === selectedBacktestRunId ? "list-row selectable active" : "list-row selectable"}
              onClick={() => setSelectedBacktestRunId(item.run_id)}
            >
              <div>
                <strong>{item.run_id}</strong>
                <p>
                  {item.start_date} 到 {item.end_date}
                </p>
              </div>
              <div className="list-meta">
                <strong>{labelOf(TASK_STATUS_LABELS, item.status, item.status)}</strong>
                <span>{fmtPct(item.summary?.metrics?.total_return)}</span>
              </div>
            </button>
          ))}
        </div>
      </section>

      <section className="panel span-two">
        <div className="metric-grid">
          {Object.entries(selectedBacktest?.metrics ?? {}).map(([key, value]) => (
            <MetricCard
              key={key}
              label={METRIC_LABELS[key] ?? key}
              value={key.includes("drawdown") || key.includes("return") || key.includes("rate") || key.includes("turnover") || key.includes("fee")
                ? fmtPct(Number(value))
                : fmtNum(Number(value))}
            />
          ))}
        </div>
      </section>

      <section className="panel span-two">
        <LineChart title="回测净值曲线" points={backtestNavSeries} />
      </section>

      <section className="panel span-two">
        <LineChart title="回测回撤曲线" points={backtestDrawdownSeries} percent tone="danger" />
      </section>
    </div>
  );

  const renderQueue = (tickets: any[], queueMode: "paper" | "live") => (
    <section className="panel">
      <div className="panel-head">
        <h3>{queueMode === "live" ? "真实盘审批队列" : "模拟盘审批队列"}</h3>
      </div>
      <div className="list">
        {tickets.length === 0 ? <p className="muted">当前没有待审批订单。</p> : null}
        {tickets.map((ticket) => (
          <div key={ticket.ticket_id} className="approval-card">
            <div>
              <strong>
                {ticket.symbol} · {labelOf(ACTION_LABELS, ticket.side, ticket.side)}
              </strong>
              <p>
                目标仓位 {fmtPct(ticket.target_weight)} · 计划数量 {fmtNum(ticket.planned_quantity)}
              </p>
              <p>{ticket.reason || "暂无理由"}</p>
              {ticket.risk_flags?.length ? (
                <div className="chip-row">
                  {ticket.risk_flags.map((flag: string) => (
                    <span key={flag} className="chip">
                      {flag}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>
            <div className="approval-actions">
              <button onClick={() => approvalActions.approve(ticket.ticket_id)} disabled={loading}>
                批准
              </button>
              <button
                className="secondary"
                onClick={() => approvalActions.reject(ticket.ticket_id)}
                disabled={loading}
              >
                拒绝
              </button>
            </div>
          </div>
        ))}
      </div>
    </section>
  );

  const renderPaper = () => (
    <div className="grid two">
      <section className="panel">
        <div className="panel-head">
          <h3>模拟盘表现</h3>
        </div>
        <div className="mini-grid">
          <MetricCard label="净值收益" value={fmtPct(cumulativeReturn(currentNavSeries))} />
          <MetricCard label="当前净值" value={fmtNum(dashboard?.portfolio?.nav)} />
          <MetricCard label="当前仓位" value={fmtPct(dashboard?.portfolio?.gross_exposure)} />
        </div>
      </section>
      {renderQueue(paperQueue, "paper")}
      <section className="panel span-two">
        <LineChart title="模拟盘净值曲线" points={currentNavSeries} />
      </section>
    </div>
  );

  const renderLive = () => (
    <div className="grid two">
      <section className="panel">
        <div className="panel-head">
          <h3>真实盘状态</h3>
        </div>
        <div className="mini-grid">
          <MetricCard label="账户净值" value={fmtNum(system?.live_account?.equity)} />
          <MetricCard label="可用现金" value={fmtNum(system?.live_account?.cash)} />
          <MetricCard label="券商连接" value={system?.live_account?.connected ? "已连接" : "未连接"} />
        </div>
        <p className="muted">
          {system?.live_account?.message || "当前以 live-ready 模式展示，真实下单仍受券商连接与审批流控制。"}
        </p>
      </section>
      {renderQueue(liveQueue, "live")}
      <section className="panel span-two">
        <LineChart title="真实盘净值曲线" points={currentNavSeries} />
      </section>
    </div>
  );

  const renderSignals = () => (
    <div className="grid two">
      <section className="panel">
        <div className="panel-head">
          <h3>研究卡</h3>
        </div>
        <div className="list">
          {(signals?.cards ?? []).length === 0 ? <p className="muted">暂无研究卡。</p> : null}
          {(signals?.cards ?? []).map((card: any) => (
            <article key={card.symbol} className="signal-card">
              <div className="signal-top">
                <strong>{card.symbol}</strong>
                <span className={`tag ${card.stance}`}>{labelOf(STANCE_LABELS, card.stance)}</span>
                <span className="tag muted-tag">{labelOf(EVENT_QUALITY_LABELS, card.event_quality)}</span>
                <span className="tag muted-tag">{card.provider_name}</span>
              </div>
              <p>{card.summary}</p>
              {card.drivers?.length ? (
                <div className="chip-row">
                  {card.drivers.map((item: string) => (
                    <span key={item} className="chip">
                      {item}
                    </span>
                  ))}
                </div>
              ) : null}
              {card.risks?.length ? (
                <div className="note-list">
                  {card.risks.map((risk: string) => (
                    <p key={risk} className="muted">
                      风险: {risk}
                    </p>
                  ))}
                </div>
              ) : null}
            </article>
          ))}
        </div>
      </section>
      <div className="section-stack">
        <JsonPanel title="组合决策" payload={signals?.decision} />
        <JsonPanel title="风控结果" payload={signals?.risk} />
      </div>
    </div>
  );

  const renderMemory = () => (
    <div className="grid two">
      <section className="panel">
        <div className="panel-head">
          <h3>持仓记忆</h3>
        </div>
        <div className="list">
          {(memory?.positions ?? []).length === 0 ? <p className="muted">暂无持仓记忆。</p> : null}
          {(memory?.positions ?? []).map((item: any) => (
            <div key={item.symbol} className="list-row">
              <div>
                <strong>{item.symbol}</strong>
                <p>{item.current_thesis || item.initial_thesis || "暂无 thesis"}</p>
              </div>
              <div className="list-meta">
                <strong>{item.is_open ? "持仓中" : "已平仓"}</strong>
                <span>{item.holding_days ?? 0} 天</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h3>组合记忆</h3>
        </div>
        <div className="kv-grid">
          <div>
            <span>市场状态</span>
            <strong>{memory?.portfolio?.market_regime ?? "--"}</strong>
          </div>
          <div>
            <span>当前净值</span>
            <strong>{fmtNum(memory?.portfolio?.current_nav)}</strong>
          </div>
          <div>
            <span>历史峰值</span>
            <strong>{fmtNum(memory?.portfolio?.peak_nav)}</strong>
          </div>
          <div>
            <span>连续失败数</span>
            <strong>{fmtNum(memory?.portfolio?.consecutive_failures)}</strong>
          </div>
        </div>
        {memory?.portfolio?.risk_flags?.length ? (
          <div className="chip-row">
            {memory.portfolio.risk_flags.map((flag: string) => (
              <span key={flag} className="chip">
                {flag}
              </span>
            ))}
          </div>
        ) : null}
      </section>

      <section className="panel span-two">
        <LineChart title={`${MODE_LABELS[mode]}记忆净值曲线`} points={currentNavSeries} />
      </section>

      <section className="panel span-two">
        <div className="panel-head">
          <h3>决策日志</h3>
        </div>
        <div className="timeline">
          {(memory?.journal ?? []).length === 0 ? <p className="muted">暂无决策日志。</p> : null}
          {(memory?.journal ?? []).map((item: any) => (
            <div key={item.id} className="timeline-row">
              <strong>{item.symbol}</strong>
              <span>{item.stage}</span>
              <span>{fmtDate(item.as_of_date)}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );

  const renderDiagnostics = () => (
    <div className="grid two">
      <JsonPanel title="系统诊断" payload={diagnostics?.system} />
      <JsonPanel title="当前模式状态" payload={modeStatus} />
      <JsonPanel title="模拟盘诊断" payload={diagnostics?.paper} />
      <JsonPanel title="真实盘诊断" payload={diagnostics?.live} />
      <JsonPanel title="回测诊断" payload={diagnostics?.backtest} />
    </div>
  );

  return (
    <div className="cockpit-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">A-SHARE AI WORKSTATION</p>
          <h1>A 股本地决策工作台</h1>
          <p className="lede">降低文本噪音、强化技术先验、统一管理回测、模拟盘、真实盘和收益可视化。</p>
        </div>
        <div className="mode-switch">
          {(["backtest", "paper", "live"] as Mode[]).map((item) => (
            <button
              key={item}
              className={item === mode ? "mode-button active" : "mode-button"}
              onClick={() => setMode(item)}
            >
              {MODE_LABELS[item]}
            </button>
          ))}
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          {(["control", "backtest", "paper", "live", "signals", "memory", "diagnostics"] as Tab[]).map(
            (item) => (
              <button
                key={item}
                className={item === tab ? "nav-button active" : "nav-button"}
                onClick={() => setTab(item)}
              >
                {TAB_LABELS[item]}
              </button>
            ),
          )}
          <div className="sidebar-card">
            <strong>模式</strong>
            <span>{MODE_LABELS[mode]}</span>
            <strong>状态</strong>
            <span>{labelOf(MODE_STATUS_LABELS, modeStatus?.state?.status)}</span>
            <strong>待审批</strong>
            <span>{modeStatus?.state?.approval_count ?? 0}</span>
          </div>
          <div className="sidebar-card">
            <strong>最新任务</strong>
            <span>{labelOf(TASK_TYPE_LABELS, latestTask?.task_type, "暂无")}</span>
            <span>{labelOf(TASK_STATUS_LABELS, latestTask?.status)}</span>
            {latestTask?.error ? <span className="muted">{latestTask.error}</span> : null}
          </div>
        </aside>

        <main className="content">
          <section className="hero-card">
            <div>
              <p className="eyebrow">{TAB_LABELS[tab]}</p>
              <h2>{dashboard?.run?.run_id ?? "尚无运行记录"}</h2>
              <p>
                {dashboard?.run
                  ? `观察日 ${fmtDate(dashboard.run.as_of_date)} · 状态 ${labelOf(RUN_STATUS_LABELS, dashboard.run.status)}`
                  : "先运行一次回测或模拟盘周期，再查看信号、审批、记忆和收益曲线。"}
              </p>
            </div>
            <div className="hero-stats">
              <div>
                <span>净值</span>
                <strong>{fmtNum(dashboard?.portfolio?.nav)}</strong>
              </div>
              <div>
                <span>现金</span>
                <strong>{fmtNum(dashboard?.portfolio?.cash)}</strong>
              </div>
              <div>
                <span>仓位</span>
                <strong>{fmtPct(dashboard?.portfolio?.gross_exposure)}</strong>
              </div>
            </div>
          </section>

          {error ? <div className="banner error">{error}</div> : null}

          {tab === "control" ? renderControl() : null}
          {tab === "backtest" ? renderBacktest() : null}
          {tab === "paper" ? renderPaper() : null}
          {tab === "live" ? renderLive() : null}
          {tab === "signals" ? renderSignals() : null}
          {tab === "memory" ? renderMemory() : null}
          {tab === "diagnostics" ? renderDiagnostics() : null}
        </main>
      </div>
    </div>
  );
}

export default App;
