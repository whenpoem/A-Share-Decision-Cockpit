# A 股本地决策工作台

[English](README.md)

这个仓库已经从旧的“概率研究项目”重建为一套本地 A 股决策工作台，核心流程参考 `AI-Trader`，定位是单人本地研究和模拟盘：

- `market -> text -> research -> decision -> risk -> simulation`
- `FastAPI` 后端
- `React + Vite` 前端驾驶舱
- `DeepSeek` 主模型，`Qwen` 回退
- `LLM 提交易意图 + 确定性风控壳`
- A 股日频模拟盘，内置 `T+1`、100 股一手、涨跌停、手续费、印花税

## 当前完成度

目前 `v1` 已经完成：

- 基于 `akshare` 的 A 股行情刷新，并带有 sample fallback
- 轻量级 prior model，用于候选池筛选
- 真实 A 股文本接入：
  - 东方财富个股新闻 `stock_news_em`
  - 巨潮资讯公告列表 `stock_zh_a_disclosure_report_cninfo`
  - 公告 PDF 正文提取与本地缓存
- `ResearchAgent` 和 `DecisionAgent`，输出结构化 JSON
- 独立风控审批层，能够 `approved / clipped / delayed / rejected`
- A 股日频模拟盘和 SQLite 状态持久化
- 本地前端驾驶舱，可查看信号、组合、运行结果

当前还没有做的内容：

- 真实券商执行
- 公告正文全文抓取
- 高频 / 日内交易
- 更复杂的组合优化器

## 架构目录

```text
service/server      FastAPI API
service/frontend    React + Vite 驾驶舱
engine/market       A 股行情与 prior score
engine/text         真实文本源与派生事件
engine/agents       ResearchAgent / DecisionAgent / LLM provider
engine/risk         确定性风控层
engine/sim          A 股日频模拟盘
engine/execution    未来 vn.py / QMT 执行适配层
engine/storage      SQLite 持久化
storage/            本地 artifacts、parquet、jsonl、state.db
tests/              后端测试
```

## 决策流程

1. 刷新行情并计算 prior signal。
2. 从多头候选、风险候选和当前持仓中选出 focus universe。
3. 为 focus symbols 拉取真实新闻/公告，并和派生事件合并。
4. 生成逐股票的 `ResearchCard`。
5. 聚合成组合级别的 `TradeIntent`。
6. 进入确定性风控，决定通过、裁剪、延后还是拒绝。
7. 通过后的交易意图进入 A 股日频模拟盘。
8. 持久化 run、研究卡、决策、风控、持仓、成交和 artifacts。

## 当前文本数据源

`v1` 现在接的是：

- `akshare` A 股行情
- `akshare.stock_news_em(symbol)` 个股新闻
- `akshare.stock_zh_a_disclosure_report_cninfo(...)` 公告元数据

公告侧现在会在可用时下载 PDF、提取正文并缓存到本地：

- `storage/text/disclosures/<announcement_id>.json`
- `storage/text/disclosures/pdf/<announcement_id>.pdf`
- `storage/text/disclosures/text/<announcement_id>.txt`

现阶段的限制：

- 很长的公告不会整篇直接塞进事件流；传给 agent 的是裁剪后的正文片段，完整提取文本保留在本地缓存

## 环境配置

已经提供独立的 Conda 环境：

```powershell
conda env create -f environment.yml
conda activate ashare-agent
python -m pip install -r requirements.txt
```

当前后端验证使用的是：

- Python `3.11`

## 环境变量

可以参考 `.env.example`，或者直接在 PowerShell 里设置：

```powershell
$env:DEEPSEEK_API_KEY="your-key"
$env:QWEN_API_KEY="your-key"
$env:ASHARE_MARKET_PROVIDER="akshare"
$env:ASHARE_TEXT_PROVIDER="akshare"
$env:ASHARE_TEXT_LOOKBACK_DAYS="30"
$env:ASHARE_MAX_NEWS_PER_SYMBOL="6"
$env:ASHARE_MAX_ANNOUNCEMENTS_PER_SYMBOL="6"
```

常用可选项：

- `DEEPSEEK_MODEL`
- `QWEN_MODEL`
- `ASHARE_STORAGE_ROOT`
- `ASHARE_DEFAULT_WATCHLIST`
- `ASHARE_BLACKLIST_SYMBOLS`

如果两个 LLM 都不可用，系统会自动降级到 `risk-only mode`，禁止新开仓。

## 启动方式

后端：

```powershell
python -m service.server.main
```

或者直接用脚本：

```powershell
.\scripts\run_backend.ps1
```

前端：

```powershell
cd service/frontend
npm install
npm run dev
```

默认地址：

- API: `http://127.0.0.1:8000`
- Cockpit: `http://127.0.0.1:5173`

## API 列表

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

## 测试

后端测试命令：

```powershell
D:\miniconda\envs\ashare-agent\python.exe -m pytest -q
```

当前已经验证通过 `6` 个测试。

## 下一步建议

最值得继续推进的方向：

- 公告正文抓取与缓存
- 更完整的 A 股文本覆盖和去重
- 与市场 regime 联动的动态风控
- `vn.py / QMT` 执行适配层
- 前端增加更明确的 run 审计与回溯

## License

本项目采用 MIT License，详见 [LICENSE](LICENSE)。
