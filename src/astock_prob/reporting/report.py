from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from astock_prob.config import AppConfig
from astock_prob.labels.generator import touch_column_name
from astock_prob.utils import format_probability


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def generate_prediction_report(
    terminal_df: pd.DataFrame,
    touch_df: pd.DataFrame,
    config: AppConfig,
    output_stem: str = "latest_prediction",
    health: Dict[str, object] | None = None,
    recent_quality: Dict[str, float] | None = None,
) -> Dict[str, Path]:
    report_dir = config.paths.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    terminal_heatmap_path = report_dir / f"{output_stem}_terminal_heatmap.png"
    touch_heatmap_path = report_dir / f"{output_stem}_touch_heatmap.png"
    touch_ladder_path = report_dir / f"{output_stem}_touch_ladder.png"
    _plot_terminal_heatmap(terminal_df, config, terminal_heatmap_path)
    _plot_touch_heatmap(touch_df, touch_heatmap_path)
    _plot_touch_ladder(touch_df, config, touch_ladder_path)
    summary = _build_prediction_summary(terminal_df, touch_df, config, health or {})
    markdown = _prediction_markdown(summary, terminal_df, touch_df, terminal_heatmap_path.name, touch_heatmap_path.name, touch_ladder_path.name, recent_quality or {}, health or {})
    markdown_path = report_dir / f"{output_stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path = report_dir / f"{output_stem}.html"
    html_path.write_text(_prediction_html(summary, terminal_df, touch_df, terminal_heatmap_path.name, touch_heatmap_path.name, touch_ladder_path.name, recent_quality or {}, health or {}), encoding="utf-8")
    return {"markdown": markdown_path, "html": html_path, "terminal_heatmap": terminal_heatmap_path, "touch_heatmap": touch_heatmap_path, "touch_ladder": touch_ladder_path}


def generate_backtest_report(
    terminal_df: pd.DataFrame,
    touch_df: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    touch_task_metrics: pd.DataFrame,
    summary_metrics: Dict[str, float],
    config: AppConfig,
    output_stem: str = "backtest_report",
) -> Dict[str, Path]:
    report_dir = config.paths.report_dir
    terminal_heatmap_path = report_dir / f"{output_stem}_terminal_heatmap.png"
    touch_calibration_path = report_dir / f"{output_stem}_touch_calibration.png"
    metric_path = report_dir / f"{output_stem}_metrics.png"
    task_path = report_dir / f"{output_stem}_touch_task_metrics.png"
    if not terminal_df.empty:
        latest_terminal = terminal_df.sort_values("as_of_date").groupby("symbol").tail(len(config.bucket_labels))
        _plot_terminal_heatmap(latest_terminal, config, terminal_heatmap_path)
    if not touch_df.empty:
        _plot_touch_calibration_panels(touch_df, config, touch_calibration_path)
    if not monthly_metrics.empty:
        _plot_metric_timeseries(monthly_metrics, metric_path)
    if not touch_task_metrics.empty:
        _plot_touch_task_metrics(touch_task_metrics, task_path)
    markdown = (
        "# Walk-forward Backtest Report\n\n"
        f"{_markdown_table(pd.DataFrame([summary_metrics]).round(6) if summary_metrics else pd.DataFrame())}\n\n"
        f"![Terminal heatmap]({terminal_heatmap_path.name})\n\n"
        f"![Touch calibration]({touch_calibration_path.name})\n\n"
        f"![Metric time series]({metric_path.name})\n\n"
        f"![Touch task metrics]({task_path.name})\n\n"
        f"{_markdown_table(monthly_metrics.round(6) if not monthly_metrics.empty else pd.DataFrame())}\n"
    )
    markdown_path = report_dir / f"{output_stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path = report_dir / f"{output_stem}.html"
    html_path.write_text(
        _basic_html(
            "Walk-forward Backtest Report",
            [
                _panel_html("Summary Metrics", _html_table(pd.DataFrame([summary_metrics]).round(6) if summary_metrics else pd.DataFrame())),
                _panel_html("Visual Review", f'<div class="visuals"><img src="{escape(terminal_heatmap_path.name)}"/><img src="{escape(touch_calibration_path.name)}"/><img src="{escape(metric_path.name)}"/><img src="{escape(task_path.name)}"/></div>'),
                _panel_html("Monthly Metrics", _html_table(monthly_metrics.round(6) if not monthly_metrics.empty else pd.DataFrame())),
            ],
        ),
        encoding="utf-8",
    )
    return {"markdown": markdown_path, "html": html_path, "terminal_heatmap": terminal_heatmap_path, "touch_calibration": touch_calibration_path, "metrics": metric_path, "touch_task_metrics": task_path}


def _plot_terminal_heatmap(terminal_df: pd.DataFrame, config: AppConfig, path: Path) -> None:
    if terminal_df.empty:
        return
    pivot = terminal_df.pivot_table(index="symbol", columns="return_bucket", values="ensemble_prob", aggfunc="last").reindex(columns=config.bucket_labels).fillna(0.0)
    fig, ax = plt.subplots(figsize=(14, 3.8))
    image = ax.imshow(pivot.to_numpy(), cmap=_terminal_cmap(), aspect="auto", vmin=0.0, vmax=max(0.2, float(pivot.to_numpy().max())))
    ax.set_title("期末收益概率热力图", loc="left", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(image, ax=ax, pad=0.02)
    _finalize_plot(fig, path)


def _plot_touch_heatmap(touch_df: pd.DataFrame, path: Path) -> None:
    if touch_df.empty:
        return
    ordered = [f"+{pct}%" for pct in (5, 10, 15, 20, 25, 30)] + [f"-{pct}%" for pct in (5, 10, 15, 20, 25, 30)]
    pivot = touch_df.pivot_table(index="symbol", columns="return_bucket", values="ensemble_prob", aggfunc="last").reindex(columns=ordered).fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    image = ax.imshow(pivot.to_numpy(), cmap=_touch_cmap(), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("触达概率热力图", loc="left", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(image, ax=ax, pad=0.02)
    _finalize_plot(fig, path)


def _plot_touch_ladder(touch_df: pd.DataFrame, config: AppConfig, path: Path) -> None:
    if touch_df.empty:
        return
    thresholds = [int(threshold * 100) for threshold in config.touch_thresholds]
    symbols = list(dict.fromkeys(touch_df["symbol"].astype(str).tolist()))
    fig, axes = plt.subplots(1, len(symbols), figsize=(6.2 * len(symbols), 4.2), sharey=True)
    if len(symbols) == 1:
        axes = [axes]
    for ax, symbol in zip(axes, symbols):
        subset = touch_df[touch_df["symbol"].astype(str) == symbol]
        up = [_touch_prob(subset, f"+{threshold}%") for threshold in thresholds]
        down = [_touch_prob(subset, f"-{threshold}%") for threshold in thresholds]
        ax.plot(thresholds, up, marker="o", linewidth=2.4, color="#c2410c", label="上行")
        ax.plot(thresholds, down, marker="o", linewidth=2.4, color="#2563eb", label="下行")
        ax.fill_between(thresholds, up, 0, color="#fdba74", alpha=0.18)
        ax.fill_between(thresholds, down, 0, color="#93c5fd", alpha=0.18)
        ax.set_title(symbol, fontsize=12, fontweight="bold")
        ax.set_xticks(thresholds)
        ax.set_xticklabels([f"{value}%" for value in thresholds])
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend(frameon=False)
    _finalize_plot(fig, path)


def _plot_touch_calibration_panels(touch_df: pd.DataFrame, config: AppConfig, path: Path) -> None:
    tasks = [(0.1, "up"), (0.1, "down"), (0.2, "up"), (0.2, "down"), (0.3, "up"), (0.3, "down")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (threshold, direction) in zip(axes, tasks):
        key = touch_column_name(threshold, direction)
        subset = touch_df[touch_df["touch_key"] == key].copy()
        ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.1)
        if not subset.empty:
            subset["prob_bin"] = pd.cut(subset["ensemble_prob"], bins=np.linspace(0, 1, config.model.calibration_bins + 1), include_lowest=True)
            grouped = subset.groupby("prob_bin", observed=False).agg(mean_prob=("ensemble_prob", "mean"), actual=("actual_touch", "mean"), samples=("actual_touch", "count")).dropna()
            if not grouped.empty:
                color = "#c2410c" if direction == "up" else "#2563eb"
                ax.plot(grouped["mean_prob"], grouped["actual"], marker="o", linewidth=2.0, color=color)
                for _, row in grouped.iterrows():
                    ax.text(float(row["mean_prob"]), float(row["actual"]) + 0.03, f"n={int(row['samples'])}", fontsize=7, ha="center")
        ax.set_title(f"{'+' if direction == 'up' else '-'}{int(threshold * 100)}%", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2)
    _finalize_plot(fig, path)


def _plot_metric_timeseries(monthly_metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    month_values = pd.to_datetime(monthly_metrics["month_start"])
    ax.plot(month_values, monthly_metrics["terminal_log_loss"], color="#b45309", linewidth=2.2, label="Terminal log loss")
    ax.plot(month_values, monthly_metrics["touch_brier_mean"], color="#0f766e", linewidth=2.2, label="Touch Brier")
    if "touch_ece_mean" in monthly_metrics.columns:
        ax.plot(month_values, monthly_metrics["touch_ece_mean"], color="#2563eb", linewidth=2.2, label="Touch ECE")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    _finalize_plot(fig, path)


def _plot_touch_task_metrics(task_metrics: pd.DataFrame, path: Path) -> None:
    grouped = task_metrics.groupby("touch_key", observed=False).agg(ece=("ece", "mean"), brier=("brier", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].bar(grouped["touch_key"], grouped["ece"], color="#2563eb")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Touch task ECE", loc="left")
    axes[1].bar(grouped["touch_key"], grouped["brier"], color="#0f766e")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Touch task Brier", loc="left")
    _finalize_plot(fig, path)


def _build_prediction_summary(terminal_df: pd.DataFrame, touch_df: pd.DataFrame, config: AppConfig, health: Dict[str, object]) -> list[dict[str, object]]:
    if terminal_df.empty:
        return []
    bucket_order = {label: idx for idx, label in enumerate(config.bucket_labels)}
    rows = []
    for symbol in dict.fromkeys(terminal_df["symbol"].astype(str).tolist()):
        terminal_slice = terminal_df[terminal_df["symbol"].astype(str) == symbol].copy()
        terminal_slice["bucket_order"] = terminal_slice["return_bucket"].map(bucket_order)
        terminal_slice = terminal_slice.sort_values("bucket_order")
        touch_slice = touch_df[touch_df["symbol"].astype(str) == symbol]
        most_likely = terminal_slice.loc[terminal_slice["ensemble_prob"].idxmax()]
        right_tail = float(terminal_slice.loc[terminal_slice["return_bucket"].isin(["[15%,20%)", "[20%,25%)", "[25%,30%)", ">=30%"]), "ensemble_prob"].sum())
        left_tail = float(terminal_slice.loc[terminal_slice["return_bucket"].isin(["<-30%", "[-30%,-25%)", "[-25%,-20%)", "[-20%,-15%)"]), "ensemble_prob"].sum())
        up20 = _touch_prob(touch_slice, "+20%")
        down20 = _touch_prob(touch_slice, "-20%")
        confidence = health.get("symbols", {}).get(symbol, {}).get("confidence_flag", "unknown")
        bias = "上行弹性占优" if up20 > down20 + 0.15 and right_tail >= left_tail else "高波动震荡"
        rows.append({"symbol": symbol, "confidence_flag": confidence, "most_likely_bucket": str(most_likely["return_bucket"]), "most_likely_prob": float(most_likely["ensemble_prob"]), "right_tail_prob": right_tail, "left_tail_prob": left_tail, "touch_up_20": up20, "touch_down_20": down20, "bias": bias, "view": f"最可能区间为 {most_likely['return_bucket']}，+20%/-20% 触达概率为 {format_probability(up20)} / {format_probability(down20)}，当前置信度 {confidence}。"})
    return rows


def _prediction_markdown(summary: list[dict[str, object]], terminal_df: pd.DataFrame, touch_df: pd.DataFrame, terminal_heatmap_name: str, touch_heatmap_name: str, touch_ladder_name: str, recent_quality: Dict[str, float], health: Dict[str, object]) -> str:
    highlights = "\n".join(f"- {row['symbol']}: {row['view']}" for row in summary)
    return (
        "# 三个月概率预测日报\n\n"
        f"观察日: {terminal_df['as_of_date'].iloc[0] if not terminal_df.empty else ''}\n\n"
        f"整体置信度: {health.get('overall_confidence_flag', 'unknown')}\n\n"
        "## 核心结论\n\n"
        f"{highlights}\n\n"
        "## 最近 60 日模型质量\n\n"
        f"{_markdown_table(_quality_frame(recent_quality))}\n\n"
        "## 模型健康度\n\n"
        f"{_markdown_table(_health_frame(health))}\n\n"
        f"![终值概率热力图]({terminal_heatmap_name})\n\n"
        f"![触达概率热力图]({touch_heatmap_name})\n\n"
        f"![触达阶梯图]({touch_ladder_name})\n\n"
        f"{_markdown_table(_summary_frame(summary))}\n\n"
        f"{_markdown_table(_terminal_frame(terminal_df))}\n\n"
        f"{_markdown_table(_touch_frame(touch_df))}\n"
    )


def _prediction_html(summary: list[dict[str, object]], terminal_df: pd.DataFrame, touch_df: pd.DataFrame, terminal_heatmap_name: str, touch_heatmap_name: str, touch_ladder_name: str, recent_quality: Dict[str, float], health: Dict[str, object]) -> str:
    insights = "".join(f"<li>{escape(str(row['symbol']))}: {escape(str(row['view']))}</li>" for row in summary)
    panels = [
        _panel_html("执行摘要", f"<ul>{insights}</ul>"),
        _panel_html("最近 60 日模型质量", _html_table(_quality_frame(recent_quality))),
        _panel_html("模型健康度", _html_table(_health_frame(health))),
        _panel_html("标的摘要", _html_table(_summary_frame(summary))),
        _panel_html("图形总览", f'<div class="visuals"><img src="{escape(terminal_heatmap_name)}"/><img src="{escape(touch_heatmap_name)}"/><img src="{escape(touch_ladder_name)}"/></div>'),
        _panel_html("期末收益分布", _html_table(_terminal_frame(terminal_df))),
        _panel_html("触达概率明细", _html_table(_touch_frame(touch_df))),
    ]
    return _basic_html("三个月概率预测日报", panels, subtitle=f"观察日: {escape(str(terminal_df['as_of_date'].iloc[0] if not terminal_df.empty else ''))} | 整体置信度: {escape(str(health.get('overall_confidence_flag', 'unknown')))}")


def _quality_frame(recent_quality: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([recent_quality]).round(6) if recent_quality else pd.DataFrame()


def _health_frame(health: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for symbol, payload in health.get("symbols", {}).items():
        rows.append({"symbol": symbol, "confidence_flag": payload.get("confidence_flag", "unknown"), "drift_score": round(float(payload.get("drift_score", 0.0) or 0.0), 4), "feature_missing_ratio": round(float(payload.get("feature_missing_ratio", 0.0) or 0.0), 4), "reasons": ",".join(payload.get("reasons", []))})
    return pd.DataFrame(rows)


def _summary_frame(summary: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(summary)
    if frame.empty:
        return frame
    display = frame[["symbol", "confidence_flag", "most_likely_bucket", "most_likely_prob", "right_tail_prob", "left_tail_prob", "touch_up_20", "touch_down_20", "bias"]].copy()
    for col in ["most_likely_prob", "right_tail_prob", "left_tail_prob", "touch_up_20", "touch_down_20"]:
        display[col] = display[col].map(format_probability)
    return display


def _terminal_frame(terminal_df: pd.DataFrame) -> pd.DataFrame:
    if terminal_df.empty:
        return pd.DataFrame()
    frame = terminal_df[["symbol", "as_of_date", "return_bucket", "confidence_flag", "ensemble_prob", "gbm_prob", "bootstrap_prob", "ml_prob"]].copy()
    for col in ["ensemble_prob", "gbm_prob", "bootstrap_prob", "ml_prob"]:
        frame[col] = frame[col].map(format_probability)
    return frame


def _touch_frame(touch_df: pd.DataFrame) -> pd.DataFrame:
    if touch_df.empty:
        return pd.DataFrame()
    frame = touch_df[["symbol", "as_of_date", "return_bucket", "direction", "confidence_flag", "ensemble_prob", "gbm_prob", "bootstrap_prob", "ml_prob"]].copy()
    for col in ["ensemble_prob", "gbm_prob", "bootstrap_prob", "ml_prob"]:
        frame[col] = frame[col].map(format_probability)
    return frame


def _touch_prob(touch_df: pd.DataFrame, bucket: str) -> float:
    subset = touch_df.loc[touch_df["return_bucket"] == bucket, "ensemble_prob"]
    return float(subset.iloc[0]) if not subset.empty else 0.0


def _basic_html(title: str, panels: list[str], subtitle: str = "") -> str:
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/><title>{escape(title)}</title><style>body{{margin:0;font-family:"Segoe UI Variable","Microsoft YaHei",sans-serif;color:#16202a;background:linear-gradient(180deg,#f7f1e6,#efe5d5)}}.shell{{max-width:1280px;margin:0 auto;padding:32px 22px 60px}}.hero{{background:linear-gradient(135deg,#b45309,#7c2d12);color:#fff8ef;border-radius:28px;padding:28px 32px;box-shadow:0 20px 50px rgba(71,43,17,.16)}}.hero h1{{margin:0 0 8px;font-size:34px}}.panel{{margin-top:18px;padding:22px;background:rgba(255,253,249,.96);border:1px solid #dacdbd;border-radius:24px;box-shadow:0 18px 46px rgba(71,43,17,.10)}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:10px 12px;border-bottom:1px solid #ece3d4;text-align:left;vertical-align:top}}th{{background:#f6ecdf;color:#6b3f1c}}.visuals{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}.visuals img{{width:100%;border:1px solid #e1d6c8;border-radius:18px;background:#fff9f1}}@media (max-width:980px){{.visuals{{grid-template-columns:1fr}}}}</style></head><body><div class="shell"><section class="hero"><h1>{escape(title)}</h1><p>{subtitle}</p></section>{''.join(panels)}</div></body></html>"""


def _panel_html(title: str, body: str) -> str:
    return f'<section class="panel"><h2>{escape(title)}</h2>{body}</section>'


def _terminal_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("terminal_prob", ["#fff7ed", "#fdba74", "#f97316", "#c2410c"])


def _touch_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("touch_prob", ["#eff6ff", "#93c5fd", "#60a5fa", "#1d4ed8"])


def _finalize_plot(fig: plt.Figure, path: Path) -> None:
    fig.patch.set_facecolor("#fffaf2")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{escape(str(col))}</th>" for col in frame.columns)
    body = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{escape(str(row[col]))}</td>" for col in frame.columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    cols = list(frame.columns)
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join([header, separator] + rows)
