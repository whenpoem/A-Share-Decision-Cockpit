import { useEffect, useMemo, useState } from "react";

type Mode = "backtest" | "paper" | "live";
type Tab = "control" | "backtest" | "paper" | "live" | "signals" | "memory" | "diagnostics";

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
  bullish: "看多",
  bearish: "看空",
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

const STAGE_LABELS: Record<string, string> = {
  research: "研究",
  decision: "决策",
  risk: "风控",
  execution: "执行",
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

function labelOf(table: Record<string, string>, value: string | undefined, fallback = "--"): string {
  if (!value) {
    return fallback;
  }
  return table[value] ?? value;
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
  const [selectedBacktest, setSelectedBacktest] = useState<any>(null);
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
      if (currentMode === "paper") {
        setPaperQueue(queuePayload.tickets ?? []);
      }
      if (currentMode === "live") {
        setLiveQueue(queuePayload.tickets ?? []);
      }
      if (!selectedBacktest && (backtestPayload.runs?.length ?? 0) > 0) {
        setSelectedBacktest(backtestPayload.runs[0].summary);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载控制台失败");
    }
  };

  useEffect(() => {
    refresh(mode);
  }, [mode]);

  useEffect(() => {
    const timer = window.setInterval(() => refresh(mode), 4000);
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
      setError(err instanceof Error ? `${label}失败：${err.message}` : `${label}失败`);
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

  return (
    <div className="cockpit-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">A股 AI 工作台</p>
          <h1>A 股本地决策工作台</h1>
          <p className="lede">一个本地控制台里完成回测、模拟盘、审批、记忆和执行状态查看。</p>
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
                  ? `观察日 ${dashboard.run.as_of_date} · 状态 ${labelOf(RUN_STATUS_LABELS, dashboard.run.status)}`
                  : "先运行一次回测或模拟盘周期，再查看信号、审批和记忆。"}
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

          {tab === "control" ? (
            <div className="grid two">
              <section className="panel">
                <div className="panel-head">
                  <h3>运行控制</h3>
                </div>
                <div className="action-grid">
                  <button
                    onClick={() =>
                      submitAction("运行模拟盘", "/api/control/run-paper-cycle", {
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
                      submitAction("运行真实盘", "/api/control/run-live-cycle", {
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
                      submitAction("启动真实盘", "/api/control/start-live-session", {
                        method: "POST",
                      })
                    }
                    disabled={loading}
                  >
                    启动真实盘会话
                  </button>
                  <button
                    onClick={() =>
                      submitAction("停止真实盘", "/api/control/stop-live-session", {
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
                  {taskList.map((task: any) => (
                    <div key={task.task_id} className="list-row">
                      <div>
                        <strong>{labelOf(TASK_TYPE_LABELS, task.task_type)}</strong>
                        <p>{labelOf(MODE_LABELS, task.mode, "系统")}</p>
                        {task.error ? <p>{task.error}</p> : null}
                      </div>
                      <div>
                        <span>{labelOf(TASK_STATUS_LABELS, task.status)}</span>
                        <p>{fmtPct(task.progress)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          ) : null}

          {tab === "backtest" ? (
            <div className="grid two">
              <section className="panel">
                <div className="panel-head">
                  <h3>回测设置</h3>
                </div>
                <div className="form-grid">
                  <label>
                    开始日期
                    <input
                      type="date"
                      value={backtestForm.start_date}
                      onChange={(event) =>
                        setBacktestForm({ ...backtestForm, start_date: event.target.value })
                      }
                    />
                  </label>
                  <label>
                    结束日期
                    <input
                      type="date"
                      value={backtestForm.end_date}
                      onChange={(event) =>
                        setBacktestForm({ ...backtestForm, end_date: event.target.value })
                      }
                    />
                  </label>
                  <label>
                    初始资金
                    <input
                      value={backtestForm.initial_capital}
                      onChange={(event) =>
                        setBacktestForm({
                          ...backtestForm,
                          initial_capital: event.target.value,
                        })
                      }
                    />
                  </label>
                  <label className="full">
                    股票池
                    <input
                      value={backtestForm.watchlist}
                      onChange={(event) =>
                        setBacktestForm({ ...backtestForm, watchlist: event.target.value })
                      }
                      placeholder="000001,600519,300750"
                    />
                  </label>
                </div>
                <button onClick={submitBacktest} disabled={loading}>
                  运行日频滚动回测
                </button>
              </section>

              <section className="panel">
                <div className="panel-head">
                  <h3>回测记录</h3>
                </div>
                <div className="list">
                  {backtestRuns.map((run: any) => (
                    <button
                      key={run.run_id}
                      className="list-row selectable"
                      onClick={() => setSelectedBacktest(run.summary)}
                    >
                      <div>
                        <strong>{run.run_id}</strong>
                        <p>
                          {run.start_date} 到 {run.end_date}
                        </p>
                      </div>
                      <div>
                        <span>{labelOf(TASK_STATUS_LABELS, run.status, run.status ?? "--")}</span>
                        <p>{fmtPct(run.summary?.metrics?.total_return)}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </section>

              <section className="panel span-two">
                <div className="panel-head">
                  <h3>回测结果</h3>
                </div>
                {selectedBacktest ? (
                  <>
                    <div className="metric-grid">
                      {Object.entries(selectedBacktest.metrics ?? {}).map(([key, value]) => (
                        <div key={key} className="metric-card">
                          <span>{METRIC_LABELS[key] ?? key}</span>
                          <strong>
                            {typeof value === "number" &&
                            key !== "profit_factor" &&
                            key !== "avg_holding_days"
                              ? fmtPct(value)
                              : String(value)}
                          </strong>
                        </div>
                      ))}
                    </div>
                    <div className="timeline">
                      {(selectedBacktest.equity_curve ?? []).slice(-20).map((point: any) => (
                        <div key={point.as_of_date} className="timeline-row">
                          <span>{point.as_of_date}</span>
                          <strong>{fmtNum(point.nav)}</strong>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="muted">还没有选中的回测结果。</p>
                )}
              </section>
            </div>
          ) : null}

          {tab === "paper" ? (
            <section className="panel">
              <div className="panel-head">
                <h3>模拟盘审批队列</h3>
              </div>
              <ApprovalQueue
                tickets={paperQueue}
                onApprove={approvalActions.approve}
                onReject={approvalActions.reject}
              />
            </section>
          ) : null}

          {tab === "live" ? (
            <div className="grid two">
              <section className="panel">
                <div className="panel-head">
                  <h3>券商账户</h3>
                </div>
                <div className="metric-grid">
                  <div className="metric-card">
                    <span>券商适配器</span>
                    <strong>{system?.live_account?.provider ?? "--"}</strong>
                  </div>
                  <div className="metric-card">
                    <span>连接状态</span>
                    <strong>{system?.live_account?.connected ? "已连接" : "未连接"}</strong>
                  </div>
                  <div className="metric-card">
                    <span>现金</span>
                    <strong>{fmtNum(system?.live_account?.cash)}</strong>
                  </div>
                  <div className="metric-card">
                    <span>权益</span>
                    <strong>{fmtNum(system?.live_account?.equity)}</strong>
                  </div>
                </div>
              </section>
              <section className="panel">
                <div className="panel-head">
                  <h3>真实盘审批队列</h3>
                </div>
                <ApprovalQueue
                  tickets={liveQueue}
                  onApprove={approvalActions.approve}
                  onReject={approvalActions.reject}
                />
              </section>
            </div>
          ) : null}

          {tab === "signals" ? (
            <div className="grid two">
              <section className="panel">
                <div className="panel-head">
                  <h3>研究卡</h3>
                </div>
                <div className="list">
                  {(signals?.cards ?? []).map((card: any) => (
                    <div key={card.symbol} className="signal-card">
                      <div className="signal-top">
                        <strong>{card.symbol}</strong>
                        <span className={`tag ${card.stance}`}>{labelOf(STANCE_LABELS, card.stance)}</span>
                        <span className="tag muted-tag">
                          {labelOf(EVENT_QUALITY_LABELS, card.event_quality)}
                        </span>
                      </div>
                      <p>{card.summary}</p>
                      <div className="chip-row">
                        {(card.drivers ?? []).slice(0, 4).map((item: string) => (
                          <span key={item} className="chip">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
              <section className="panel">
                <div className="panel-head">
                  <h3>决策与风控</h3>
                </div>
                <pre className="code-block">{JSON.stringify(signals?.decision ?? {}, null, 2)}</pre>
                <pre className="code-block">{JSON.stringify(signals?.risk ?? {}, null, 2)}</pre>
              </section>
            </div>
          ) : null}

          {tab === "memory" ? (
            <div className="grid two">
              <section className="panel">
                <div className="panel-head">
                  <h3>持仓记忆</h3>
                </div>
                <div className="list">
                  {(memory?.positions ?? []).map((item: any) => (
                    <div key={item.symbol} className="list-row">
                      <div>
                        <strong>{item.symbol}</strong>
                        <p>{item.current_thesis || item.last_research_summary || "还没有投资逻辑摘要"}</p>
                      </div>
                      <div>
                        <span>{labelOf(ACTION_LABELS, item.last_decision_action, "持有")}</span>
                        <p>{item.holding_days ?? 0} 天</p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
              <section className="panel">
                <div className="panel-head">
                  <h3>决策日志</h3>
                </div>
                <div className="timeline">
                  {(memory?.journal ?? []).slice(0, 30).map((entry: any) => (
                    <div key={entry.id} className="timeline-row">
                      <span>{entry.as_of_date}</span>
                      <strong>{entry.symbol}</strong>
                      <span>{labelOf(STAGE_LABELS, entry.stage)}</span>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          ) : null}

          {tab === "diagnostics" ? (
            <div className="grid two">
              <section className="panel">
                <div className="panel-head">
                  <h3>模式状态</h3>
                </div>
                <pre className="code-block">{JSON.stringify(system?.modes ?? [], null, 2)}</pre>
              </section>
              <section className="panel">
                <div className="panel-head">
                  <h3>诊断信息</h3>
                </div>
                <pre className="code-block">{JSON.stringify(diagnostics ?? {}, null, 2)}</pre>
              </section>
            </div>
          ) : null}
        </main>
      </div>
    </div>
  );
}

function ApprovalQueue({
  tickets,
  onApprove,
  onReject,
}: {
  tickets: any[];
  onApprove: (ticketId: string) => Promise<void>;
  onReject: (ticketId: string) => Promise<void>;
}) {
  if (!tickets.length) {
    return <p className="muted">当前没有待审批订单。</p>;
  }
  return (
    <div className="list">
      {tickets.map((ticket) => (
        <div key={ticket.ticket_id} className="approval-card">
          <div>
            <strong>{ticket.symbol}</strong>
            <p>
              {labelOf(ACTION_LABELS, ticket.side)} · 目标仓位 {fmtPct(ticket.target_weight)}
            </p>
            <p>{ticket.reason || "没有记录理由。"}</p>
          </div>
          <div className="approval-actions">
            <button onClick={() => onApprove(ticket.ticket_id)}>批准</button>
            <button className="secondary" onClick={() => onReject(ticket.ticket_id)}>
              拒绝
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

export default App;
