# A 股概率研究系统

这是一个面向 A 股个股研究的轻量级概率预测项目，用于在日频数据基础上，持续更新未来 `60` 个交易日的：

- 期末收益区间概率分布
- 关键涨跌幅目标的区间触达概率
- 每日模型置信度与中文研究报告

项目当前以 `601727` 和 `002202` 为目标输出股票，但训练时不只使用这两只股票，而是使用更大的相关股票池做横截面建模，以提升样本量、稳定性和概率校准质量。

## 项目亮点

- 我希望通过这个项目去建模未来收益的概率分布
- 将 `期末分布` 与 `触达某个百分比的概率` 明确拆成两个不同任务
- 使用 `GBM + bootstrap + LightGBM` 做概率融合
- 输出 CSV、Markdown、HTML 三种结果，兼容了程序和人工阅读
- 强调概率校准，使用 `log loss`、`RPS`、`Brier`、`ECE` 等指标评估模型

## 这个项目在做什么

大多数常见的股票预测项目会给出一个单点目标，或者只做涨跌二分类。这个项目的核心思路不同：

- `terminal distribution`：预测 60 个交易日后，股票更可能落在哪个收益区间
- `touch probability`：预测未来 60 个交易日内，股票是否会在任意一天触达某个目标价位
- `calibration`：判断模型给出的 70% 概率，历史上是否真的接近 70% 发生率


## 架构概览

整个项目按六层组织：

1. `data ingestion`
   - AkShare / Eastmoney 数据抓取
   - 本地缓存
   - 股票池解析和失败回退
2. `feature engineering`
   - 收益率、波动率、ATR、量价、相对强弱、估值、财务特征
3. `label generation`
   - 60 日前瞻终值收益分桶
   - `+/-5%` 到 `+/-30%` 的触达事件标签
4. `modeling`
   - `GBM` 基线
   - 相似状态 `bootstrap` 基线
   - 全局横截面 `LightGBM`
5. `calibration and constraints`
   - 温度缩放
   - isotonic 校准
   - 触达概率单调约束
6. `backtest and reporting`
   - walk-forward 回测
   - 指标统计
   - Markdown / HTML / 图表输出

## 当前范围

- 目标输出股票：`601727`、`002202`
- 更新频率：每日收盘后
- 数据频率：日线
- 训练方式：更大相关股票池的全局横截面模型
- 第一阶段不包含：
  - 盘中分钟级刷新
  - NLP/新闻因子
  - 自动交易执行

## 快速开始

1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

2. 刷新市场数据

```bash
python run_cli.py refresh-data --config configs/default.json
```

3. 训练全局模型

```bash
python run_cli.py train-global --config configs/default.json
```

4. 运行 walk-forward 回测

```bash
python run_cli.py backtest --config configs/default.json
```

5. 生成每日预测 CSV

```bash
python run_cli.py score-daily --config configs/default.json
```

6. 生成日报

```bash
python run_cli.py report-daily --config configs/default.json
```

7. 运行整套日更流水线

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_live.ps1
```

## 仓库结构

```text
src/astock_prob/
  data/         数据源、缓存、股票池解析
  features/     特征工程
  labels/       标签生成
  modeling/     基线模型、ML 模型、校准、约束
  backtest/     walk-forward 回测
  reporting/    Markdown/HTML/图表输出
configs/        配置文件
docs/           架构文档
scripts/        辅助脚本
tests/          单元测试与集成测试
```

## 主要输出

主要产物会写到 `artifacts/reports/`，包括：

- `daily_terminal_predictions.csv`
- `daily_touch_predictions.csv`
- `daily_model_health.json`
- `daily_report.md`
- `daily_report.html`

这些输出通常包含：

- 终值收益分布表
- 关键阈值触达概率表
- 模型健康度与置信度标记
- 收盘后的可视化日报

## 使用的评估指标

项目使用概率预测指标，而不是只看方向命中率：

- `terminal_log_loss`
- `terminal_rps`
- `touch_brier_mean`
- `touch_ece_mean`

其中：

- `log loss / RPS` 用于衡量期末分布质量
- `Brier` 用于衡量触达事件概率误差
- `ECE` 用于衡量概率校准质量

## 设计取舍

- 只做日线：优先保证稳定性和复现性
- 用横截面训练：避免只用两只股票导致样本过少
- 先做概率校准：让输出的 70% 更接近“真实 70%”
- 兼容免费数据源不稳定：依赖缓存和失败回退逻辑

## 路线图

- 将实时训练股票池扩展到配置目标的 `80-120+` 只
- 在更大股票池上重跑完整 walk-forward 回测
- 增强市场状态和行业相对强弱特征
- 优化报告页面与模型分项对比图
- 增加 CSV 数据适配器，降低对公共 API 稳定性的依赖

## 说明

- 免费数据接口存在延迟、失败和字段不稳定的问题
- 本仓库属于研究项目，不构成任何投资建议
- 输出概率是模型结果，不代表对未来市场结果的保证
