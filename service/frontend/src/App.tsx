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
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
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

  const currentQueue = mode === "live" ? liveQueue : paperQueue;
  const taskList = system?.tasks ?? [];
  const latestTask = taskList[0];

  const refresh = async (currentMode: Mode = mode) => {
    try {
      setError(null);
      const queueEndpoint = currentMode === "live" ? "/api/live/approval-queue" : "/api/paper/approval-queue";
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
      setPaperQueue(currentMode === "paper" ? queuePayload.tickets ?? [] : paperQueue);
      setLiveQueue(currentMode === "live" ? queuePayload.tickets ?? [] : liveQueue);
      if (!selectedBacktest && (backtestPayload.runs?.length ?? 0) > 0) {
        setSelectedBacktest(backtestPayload.runs[0].summary);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load cockpit.");
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
      setError(err instanceof Error ? `${label}: ${err.message}` : `${label} failed`);
    } finally {
      setLoading(false);
    }
  };

  const submitBacktest = async () => {
    await submitAction("Backtest", "/api/control/run-backtest", {
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
          "Approve",
          `${mode === "live" ? "/api/live/approve" : "/api/paper/approve"}/${ticketId}`,
          { method: "POST" },
        ),
      reject: async (ticketId: string) =>
        submitAction(
          "Reject",
          `${mode === "live" ? "/api/live/reject" : "/api/paper/reject"}/${ticketId}`,
          { method: "POST" },
        ),
    }),
    [mode],
  );

  const pageTitle = {
    control: "Control Center",
    backtest: "Backtest",
    paper: "Paper Trading",
    live: "Live Trading",
    signals: "Signals & Decisions",
    memory: "Memory",
    diagnostics: "Diagnostics",
  }[tab];

  return (
    <div className="cockpit-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">A-share AI workstation</p>
          <h1>A-Share Decision Cockpit</h1>
          <p className="lede">
            Three-mode research, decision, approval, memory, and execution control from one local console.
          </p>
        </div>
        <div className="mode-switch">
          {(["backtest", "paper", "live"] as Mode[]).map((item) => (
            <button
              key={item}
              className={item === mode ? "mode-button active" : "mode-button"}
              onClick={() => setMode(item)}
            >
              {item.toUpperCase()}
            </button>
          ))}
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          {(["control", "backtest", "paper", "live", "signals", "memory", "diagnostics"] as Tab[]).map((item) => (
            <button
              key={item}
              className={item === tab ? "nav-button active" : "nav-button"}
              onClick={() => setTab(item)}
            >
              {item}
            </button>
          ))}
          <div className="sidebar-card">
            <strong>Mode</strong>
            <span>{mode.toUpperCase()}</span>
            <strong>Status</strong>
            <span>{modeStatus?.state?.status ?? "--"}</span>
            <strong>Approvals</strong>
            <span>{modeStatus?.state?.approval_count ?? 0}</span>
          </div>
          <div className="sidebar-card">
            <strong>Latest Task</strong>
            <span>{latestTask?.task_type ?? "none"}</span>
            <span>{latestTask?.status ?? "--"}</span>
          </div>
        </aside>

        <main className="content">
          <section className="hero-card">
            <div>
              <p className="eyebrow">{pageTitle}</p>
              <h2>{dashboard?.run?.run_id ?? "No run yet"}</h2>
              <p>
                {dashboard?.run
                  ? `As of ${dashboard.run.as_of_date} · ${dashboard.run.status}`
                  : "Start a paper cycle, connect live mode, or launch a walk-forward backtest."}
              </p>
            </div>
            <div className="hero-stats">
              <div>
                <span>NAV</span>
                <strong>{fmtNum(dashboard?.portfolio?.nav)}</strong>
              </div>
              <div>
                <span>Cash</span>
                <strong>{fmtNum(dashboard?.portfolio?.cash)}</strong>
              </div>
              <div>
                <span>Exposure</span>
                <strong>{fmtPct(dashboard?.portfolio?.gross_exposure)}</strong>
              </div>
            </div>
          </section>

          {error ? <div className="banner error">{error}</div> : null}

          {tab === "control" ? (
            <div className="grid two">
              <section className="panel">
                <div className="panel-head">
                  <h3>Run Controls</h3>
                </div>
                <div className="action-grid">
                  <button onClick={() => submitAction("Paper cycle", "/api/control/run-paper-cycle", { method: "POST", body: JSON.stringify({}) })} disabled={loading}>
                    Run Paper Cycle
                  </button>
                  <button onClick={() => submitAction("Live cycle", "/api/control/run-live-cycle", { method: "POST", body: JSON.stringify({}) })} disabled={loading}>
                    Run Live Cycle
                  </button>
                  <button onClick={() => submitAction("Start live", "/api/control/start-live-session", { method: "POST" })} disabled={loading}>
                    Start Live Session
                  </button>
                  <button onClick={() => submitAction("Stop live", "/api/control/stop-live-session", { method: "POST" })} disabled={loading}>
                    Stop Live Session
                  </button>
                  <button onClick={() => submitAction("Rebuild memory", "/api/control/rebuild-memory", { method: "POST", body: JSON.stringify({ mode }) })} disabled={loading}>
                    Rebuild Memory
                  </button>
                  <button onClick={() => refresh(mode)} disabled={loading}>
                    Refresh State
                  </button>
                </div>
              </section>

              <section className="panel">
                <div className="panel-head">
                  <h3>Task Feed</h3>
                </div>
                <div className="list">
                  {taskList.map((task: any) => (
                    <div key={task.task_id} className="list-row">
                      <div>
                        <strong>{task.task_type}</strong>
                        <p>{task.mode ?? "system"}</p>
                      </div>
                      <div>
                        <span>{task.status}</span>
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
                  <h3>Backtest Setup</h3>
                </div>
                <div className="form-grid">
                  <label>
                    Start
                    <input type="date" value={backtestForm.start_date} onChange={(event) => setBacktestForm({ ...backtestForm, start_date: event.target.value })} />
                  </label>
                  <label>
                    End
                    <input type="date" value={backtestForm.end_date} onChange={(event) => setBacktestForm({ ...backtestForm, end_date: event.target.value })} />
                  </label>
                  <label>
                    Initial Capital
                    <input value={backtestForm.initial_capital} onChange={(event) => setBacktestForm({ ...backtestForm, initial_capital: event.target.value })} />
                  </label>
                  <label className="full">
                    Watchlist
                    <input value={backtestForm.watchlist} onChange={(event) => setBacktestForm({ ...backtestForm, watchlist: event.target.value })} placeholder="000001,600519,300750" />
                  </label>
                </div>
                <button onClick={submitBacktest} disabled={loading}>
                  Run Walk-forward Backtest
                </button>
              </section>

              <section className="panel">
                <div className="panel-head">
                  <h3>Backtest Runs</h3>
                </div>
                <div className="list">
                  {backtestRuns.map((run: any) => (
                    <button key={run.run_id} className="list-row selectable" onClick={() => setSelectedBacktest(run.summary)}>
                      <div>
                        <strong>{run.run_id}</strong>
                        <p>{run.start_date} → {run.end_date}</p>
                      </div>
                      <div>
                        <span>{run.status}</span>
                        <p>{fmtPct(run.summary?.metrics?.total_return)}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </section>

              <section className="panel span-two">
                <div className="panel-head">
                  <h3>Backtest Result</h3>
                </div>
                {selectedBacktest ? (
                  <>
                    <div className="metric-grid">
                      {Object.entries(selectedBacktest.metrics ?? {}).map(([key, value]) => (
                        <div key={key} className="metric-card">
                          <span>{key}</span>
                          <strong>{typeof value === "number" ? fmtPct(value) : String(value)}</strong>
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
                  <p className="muted">No backtest selected.</p>
                )}
              </section>
            </div>
          ) : null}

          {tab === "paper" ? (
            <section className="panel">
              <div className="panel-head">
                <h3>Paper Trading Queue</h3>
              </div>
              <ApprovalQueue tickets={paperQueue} onApprove={approvalActions.approve} onReject={approvalActions.reject} />
            </section>
          ) : null}

          {tab === "live" ? (
            <div className="grid two">
              <section className="panel">
                <div className="panel-head">
                  <h3>Broker Account</h3>
                </div>
                <div className="metric-grid">
                  <div className="metric-card">
                    <span>Provider</span>
                    <strong>{system?.live_account?.provider ?? "--"}</strong>
                  </div>
                  <div className="metric-card">
                    <span>Connected</span>
                    <strong>{String(system?.live_account?.connected ?? false)}</strong>
                  </div>
                  <div className="metric-card">
                    <span>Cash</span>
                    <strong>{fmtNum(system?.live_account?.cash)}</strong>
                  </div>
                  <div className="metric-card">
                    <span>Equity</span>
                    <strong>{fmtNum(system?.live_account?.equity)}</strong>
                  </div>
                </div>
              </section>
              <section className="panel">
                <div className="panel-head">
                  <h3>Live Approval Queue</h3>
                </div>
                <ApprovalQueue tickets={liveQueue} onApprove={approvalActions.approve} onReject={approvalActions.reject} />
              </section>
            </div>
          ) : null}

          {tab === "signals" ? (
            <div className="grid two">
              <section className="panel">
                <div className="panel-head">
                  <h3>Research Cards</h3>
                </div>
                <div className="list">
                  {(signals?.cards ?? []).map((card: any) => (
                    <div key={card.symbol} className="signal-card">
                      <div className="signal-top">
                        <strong>{card.symbol}</strong>
                        <span className={`tag ${card.stance}`}>{card.stance}</span>
                        <span className="tag muted-tag">{card.event_quality}</span>
                      </div>
                      <p>{card.summary}</p>
                      <div className="chip-row">
                        {(card.drivers ?? []).slice(0, 4).map((item: string) => (
                          <span key={item} className="chip">{item}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
              <section className="panel">
                <div className="panel-head">
                  <h3>Decision & Risk</h3>
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
                  <h3>Position Memory</h3>
                </div>
                <div className="list">
                  {(memory?.positions ?? []).map((item: any) => (
                    <div key={item.symbol} className="list-row">
                      <div>
                        <strong>{item.symbol}</strong>
                        <p>{item.current_thesis || item.last_research_summary || "No thesis yet"}</p>
                      </div>
                      <div>
                        <span>{item.last_decision_action ?? "hold"}</span>
                        <p>{item.holding_days ?? 0} days</p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
              <section className="panel">
                <div className="panel-head">
                  <h3>Decision Journal</h3>
                </div>
                <div className="timeline">
                  {(memory?.journal ?? []).slice(0, 30).map((entry: any) => (
                    <div key={entry.id} className="timeline-row">
                      <span>{entry.as_of_date}</span>
                      <strong>{entry.symbol}</strong>
                      <span>{entry.stage}</span>
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
                  <h3>Mode State</h3>
                </div>
                <pre className="code-block">{JSON.stringify(system?.modes ?? [], null, 2)}</pre>
              </section>
              <section className="panel">
                <div className="panel-head">
                  <h3>Diagnostics Feed</h3>
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
    return <p className="muted">No pending approvals.</p>;
  }
  return (
    <div className="list">
      {tickets.map((ticket) => (
        <div key={ticket.ticket_id} className="approval-card">
          <div>
            <strong>{ticket.symbol}</strong>
            <p>{ticket.side} · target {fmtPct(ticket.target_weight)}</p>
            <p>{ticket.reason || "No rationale recorded."}</p>
          </div>
          <div className="approval-actions">
            <button onClick={() => onApprove(ticket.ticket_id)}>Approve</button>
            <button className="secondary" onClick={() => onReject(ticket.ticket_id)}>
              Reject
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

export default App;

