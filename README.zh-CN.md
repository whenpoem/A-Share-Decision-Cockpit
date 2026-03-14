# A 股本地决策工作台

[English](README.md)

这是一个面向 A 股的本地 AI 决策系统，支持三种运行模式：

- `Backtest`：日频 walk-forward 回测
- `Paper`：人工审批后的模拟盘
- `Live-ready`：带审批流的真实盘准备态，内置 mock broker 和 QMT/Ptrade 适配接口

系统把数值先验、真实文本、LLM 研究与决策、确定性风控、连续记忆、前端驾驶舱整合到一套单机工作流里。

## 当前包含的能力

- `FastAPI` 后端控制面
- `React + Vite` 前端控制台
- `DeepSeek` 主模型
- LLM 交易意图之外的确定性风控壳
- A 股日频规则：
  - `T+1`
  - 100 股一手
  - 涨跌停阻塞
  - 手续费与印花税
- 持仓记忆、thesis 记忆、决策日志、组合记忆
- 任务状态、模式状态、审批队列、诊断面板

## V2 完成度

仓库当前已经是可运行的 `v2` 基线。

已完成：

- 三模式主流程：
  - `Backtest`
  - `Paper`
  - `Live-ready`
- 日频 walk-forward 回测与结果持久化
- 审批式模拟盘
- live session 控制、审批流、broker 状态显示
- 本地 mock live broker 全链路
- `QMT/Ptrade` 适配接口和 feature gate
- 基于 `baostock` 的行情刷新、`akshare` 财务补充，以及 sample fallback
- A 股真实文本接入：
  - 东方财富个股新闻
  - 巨潮资讯公告元数据
  - 公告 PDF 下载、正文提取和本地缓存
- `ResearchAgent` 与 `DecisionAgent` 的结构化 JSON 输出
- `approved / clipped / delayed / rejected` 风控裁决
- 持仓记忆、thesis 演化、组合记忆持久化
- 前端控制台可查看控制、审批、回测、信号、记忆、诊断

当前边界：

- 真实 `QMT/Ptrade` 发单还需要你本机完成券商 SDK 接线
- `Live-ready` 这一层已经包含 API、状态机、审批队列、mock 执行和 adapter contract
- 公开文本源目前偏日频研究，盘中事件基础设施还不完整

## 架构目录

```text
service/server      FastAPI API 与 WebSocket 状态推送
service/frontend    React + Vite 控制台
engine/market       A 股行情与 prior signal
engine/text         文本抓取与公告缓存
engine/agents       ResearchAgent / DecisionAgent / LLM provider
engine/risk         确定性风控
engine/sim          A 股日频模拟与估值更新
engine/memory       持仓记忆 / thesis 记忆 / journal
engine/execution    Mock broker 与 live-ready adapter
engine/runtime      任务管理
engine/storage      模式状态与系统状态的 SQLite 存储
storage/backtest    回测模式产物与状态
storage/paper       模拟盘模式产物与状态
storage/live        真实盘模式产物与状态
tests/              后端测试
```

## 决策流程

1. 刷新行情并计算 prior signal。
2. 从多头候选、风险候选和当前持仓里选出 focus universe。
3. 拉取 focus symbols 的新闻和公告事件。
4. 生成逐标的 `ResearchCard`。
5. 聚合成组合级 `TradeIntentSet`。
6. 进入确定性风控审查。
7. 按模式进入不同执行路径：
   - `Backtest`：直接进入日频模拟撮合
   - `Paper`：进入模拟盘审批队列
   - `Live-ready`：进入真实盘审批队列
8. 持久化 run、journal、持仓、审批、订单、成交和 artifact。

## 数据源

当前使用：

- `baostock` A 股股票列表和日线行情
- `akshare` 财务快照补充
- `akshare.stock_news_em(symbol)` 东方财富新闻
- `akshare.stock_zh_a_disclosure_report_cninfo(...)` 巨潮公告元数据

公告相关缓存会写入：

- `storage/<mode>/text/disclosures/<announcement_id>.json`
- `storage/<mode>/text/disclosures/pdf/<announcement_id>.pdf`
- `storage/<mode>/text/disclosures/text/<announcement_id>.txt`

超长公告不会整篇进入 agent 事件流，进入事件流的是裁剪后的正文片段，完整提取文本保存在本地缓存。

## 环境准备

创建并激活独立 Conda 环境：

```powershell
conda env create -f environment.yml
conda activate ashare-agent
python -m pip install -r requirements.txt
```

当前后端验证环境：

- Python `3.11`

## 环境变量

可以直接在 PowerShell 设置，也可以参考 `.env.example`。

常用变量：

```powershell
$env:DEEPSEEK_API_KEY="your-key"
$env:ASHARE_MARKET_PROVIDER="baostock"
$env:ASHARE_TEXT_PROVIDER="akshare"
$env:ASHARE_STORAGE_ROOT="D:\bug\storage"
```

真实盘相关变量：

```powershell
$env:ASHARE_LIVE_ENABLED="true"
$env:ASHARE_LIVE_BROKER="mock"
$env:ASHARE_LIVE_ACCOUNT_ID="demo-account"
```

常用可选项：

- `DEEPSEEK_MODEL`
- `ASHARE_DEFAULT_WATCHLIST`
- `ASHARE_BLACKLIST_SYMBOLS`
- `ASHARE_QMT_TERMINAL_PATH`
- `ASHARE_QMT_USER`
- `ASHARE_QMT_PASSWORD`

如果 DeepSeek 不可用，或者返回的结构化结果不可用，系统会降级到 risk-only 行为，并阻止新开仓。

## 启动方式

后端：

```powershell
python -m service.server.main
```

前端：

```powershell
cd service/frontend
npm install
npm run dev
```

默认地址：

- API：`http://127.0.0.1:8000`
- 控制台：`http://127.0.0.1:5173`
- 状态 WebSocket：`ws://127.0.0.1:8000/ws/status`

## 使用方式

### 1. 回测

- 打开控制台
- 切换到 `BACKTEST`
- 进入 `Backtest` 页面
- 设置：
  - `start date`
  - `end date`
  - `initial capital`
  - 可选 `watchlist`
- 启动回测
- 查看：
  - 任务状态
  - 收益指标
  - 净值曲线
  - 每日决策回放

### 2. 模拟盘

- 切换到 `PAPER`
- 在 `Control Center` 启动 paper cycle
- 进入 `Paper Trading`
- 查看待审批订单
- 手工 `Approve / Reject`
- 再到 `Signals`、`Memory`、`Diagnostics` 查看更新结果

### 3. 真实盘准备态

- 设置 live 相关环境变量
- 在 `Control Center` 启动 live session
- 进入 `Live Trading`
- 查看 broker 状态和审批队列
- 手工批准订单

当 `ASHARE_LIVE_BROKER=mock` 时，本地可以跑完整的审批到成交闭环。

当使用真实券商路径时，adapter contract 已经到位，本机 SDK 接线仍需要你本地完成。

## API 总览

核心接口：

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

兼容旧脚本的接口仍然保留：

- `POST /api/sim/run-daily`

## 测试

后端测试：

```powershell
D:\miniconda\envs\ashare-agent\python.exe -m pytest -q
```

前端构建：

```powershell
cd service/frontend
npm run build
```

最新验证结果：

- 后端：`9 passed`
- 前端构建：通过

## 后续建议

下一步最值得继续推进的方向：

- 本机 `QMT/Ptrade` SDK 接线
- 更丰富的 A 股文本源和来源质量排序
- 更细的市场 regime 联动风控
- 控制台里的回测可视化和单日 drill-down
- 更强的 live execution 审计视图

## License

项目采用 MIT License，详见 [LICENSE](LICENSE)。
