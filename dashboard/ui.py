from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime
from html import escape
from io import StringIO
import math
from pathlib import Path
import re
from typing import Iterator

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


PAGE_ORDER = [
    "Overview",
    "Backtest",
    "Confluence Analysis",
    "ML Analysis",
]

PAGE_ICONS = {
    "Overview": "📊",
    "Backtest": "🧪",
    "Confluence Analysis": "🧩",
    "ML Analysis": "🤖",
}


def render_metric_row(metrics: list[tuple[str, object]]) -> None:
    #Rendering a simple one-line metric row
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        display = _format_metric_display(label, value)
        value_tone = _metric_value_tone_class(label, value)
        value_class = "dashboard-stat-value dashboard-stat-value-compact" if len(display) > 20 else "dashboard-stat-value"
        value_class = f"{value_class} {value_tone}".strip()
        with column:
            st.markdown(
                f"""
                <div class="dashboard-stat-card">
                  <div class="dashboard-stat-label">{escape(str(label))}</div>
                  <div class="{value_class}">{escape(display)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metric_grid(metrics: list[tuple[str, object]], *, columns: int = 2) -> None:
    #Rendering a compact metric-card grid for dense summary panels
    if columns <= 0:
        columns = 1

    for start in range(0, len(metrics), columns):
        row = metrics[start : start + columns]
        grid_columns = st.columns(columns)
        for column, (label, value) in zip(grid_columns, row):
            display = _format_metric_display(label, value)
            value_tone = _metric_value_tone_class(label, value)
            value_class = "dashboard-stat-value dashboard-stat-value-compact" if len(display) > 18 else "dashboard-stat-value"
            value_class = f"{value_class} {value_tone}".strip()
            with column:
                st.markdown(
                    f"""
                    <div class="dashboard-stat-card dashboard-stat-card-compact">
                      <div class="dashboard-stat-label">{escape(str(label))}</div>
                      <div class="{value_class}">{escape(display)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_page_header(
    *,
    current_page: str,
    title: str,
    subtitle: str,
) -> None:
    #Rendering a simplified page title block
    icon_text = escape(PAGE_ICONS.get(current_page, ""))
    title_text = escape(title)
    subtitle_text = escape(subtitle)

    st.markdown(
        f"""
        <div class="dashboard-page-heading">
          <div class="dashboard-page-heading-row">
            <div class="dashboard-page-icon">{icon_text}</div>
            <div class="dashboard-hero-title">{title_text}</div>
          </div>
          <div class="dashboard-hero-subtitle">{subtitle_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def dashboard_panel(
    title: str,
    *,
    subtitle: str | None = None,
    eyebrow: str | None = None,
    action: str | None = None,
) -> Iterator[st.delta_generator.DeltaGenerator]:
    #Grouping content inside a styled bordered container panel
    with st.container(border=True) as container:
        header_bits = []
        if eyebrow:
            header_bits.append(f'<div class="dashboard-kicker">{escape(eyebrow)}</div>')
        header_bits.append(f'<div class="dashboard-panel-title">{escape(title)}</div>')
        if subtitle:
            header_bits.append(f'<div class="dashboard-help">{escape(subtitle)}</div>')

        action_html = f'<div class="dashboard-panel-action">{escape(action)}</div>' if action else ""
        st.markdown(
            f"""
            <div class="dashboard-panel-header">
              <div class="dashboard-panel-copy">
                {''.join(header_bits)}
              </div>
              {action_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
        yield container
def render_section_label(text: str) -> None:
    #Rendering a muted uppercase section label
    st.markdown(f'<div class="dashboard-kicker">{escape(text)}</div>', unsafe_allow_html=True)


def _to_streamlit_cell(value: object) -> object:
    #Converting mixed Python objects into Arrow-friendly display values
    if isinstance(value, pd.Timestamp):
        return _format_timestamp_for_display(value)
    if isinstance(value, datetime):
        return _format_timestamp_for_display(pd.Timestamp(value))
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return _format_display_text(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return value


def _format_metric_number(label: str, value: float) -> str:
    #Format numeric metric values using label-aware rules
    label_text = label.lower()
    if not math.isfinite(value):
        if "pf" in label_text or "profit factor" in label_text:
            return "No Losses"
        return "-"
    if any(token in label_text for token in ["total trades", "wins", "losses", "breakevens", "count", "trades"]):
        return f"{int(round(value)):,}"

    if any(token in label_text for token in ["expectancy", "avg r", "delta r", "pnl r"]):
        return f"{value:,.4f}"

    if any(token in label_text for token in ["rate", "%", "drawdown", "return", "factor"]):
        return f"{value:,.2f}"

    return f"{value:,.2f}"


def _format_metric_display(label: object, value: object) -> str:
    #Format metric values so overview cards stay readable
    safe_value = _to_streamlit_cell(value)
    if safe_value in (None, "", {}):
        return "-"
    if isinstance(safe_value, bool):
        return "Yes" if safe_value else "No"
    if isinstance(safe_value, (int, np.integer)):
        return f"{int(safe_value):,}"
    if isinstance(safe_value, (float, np.floating)):
        return _format_metric_number(str(label), float(safe_value))
    return str(safe_value)


def _metric_value_tone_class(label: object, value: object) -> str:
    #Applying positive/negative color cues to directional metrics
    safe_value = _to_streamlit_cell(value)
    if isinstance(safe_value, bool) or not isinstance(safe_value, (int, float, np.integer, np.floating)):
        return ""

    label_text = str(label).lower()
    numeric = float(safe_value)

    if "profit factor" in label_text:
        if numeric > 1.0:
            return "dashboard-stat-value-positive"
        if numeric < 1.0:
            return "dashboard-stat-value-negative"
        return ""

    if any(token in label_text for token in ("loss", "drawdown", "dd")):
        if numeric > 0:
            return "dashboard-stat-value-negative"
        if numeric < 0:
            return "dashboard-stat-value-positive"
        return ""

    if any(token in label_text for token in ("pnl", "profit", "return", "expectancy")):
        if numeric > 0:
            return "dashboard-stat-value-positive"
        if numeric < 0:
            return "dashboard-stat-value-negative"
        return ""

    return ""


def _format_timestamp_for_display(value: pd.Timestamp) -> str:
    #Format timestamps so cards and tables stay compact
    if pd.isna(value):
        return ""

    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)

    if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0:
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m-%d %H:%M")


def _format_datetime_text(text: str) -> str:
    #Format ISO-like datetime text into a cleaner display string
    candidate = text.strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}", candidate):
        return text

    ts = pd.to_datetime(candidate, errors="coerce")
    if pd.isna(ts):
        return text
    return _format_timestamp_for_display(pd.Timestamp(ts))


def _format_display_text(text: str) -> str:
    #Cleaning long text values used inside metric cards
    candidate = text.strip()
    if not candidate:
        return text

    if "->" in candidate:
        left, right = (part.strip() for part in candidate.split("->", 1))
        return f"{_format_datetime_text(left)} to {_format_datetime_text(right)}"

    return _format_datetime_text(candidate)


def render_key_value_table(items: list[tuple[str, object]], *, hide_index: bool = True) -> None:
    #Rendering key/value content as a compact table
    frame = pd.DataFrame(items, columns=["Setting", "Value"])
    frame["Setting"] = frame["Setting"].map(lambda value: str(_to_streamlit_cell(value)))
    frame["Value"] = frame["Value"].map(lambda value: str(_to_streamlit_cell(value)))
    st.dataframe(frame, width="stretch", hide_index=hide_index)


def render_matplotlib_equity_chart(curve: pd.DataFrame, *, line_color: str = "#97D6FF") -> None:
    #Rendering the equity curve with Matplotlib for a more traditional backtest view
    if curve.empty or "equity" not in curve.columns:
        st.info("No chart data available.")
        return

    series = curve.copy()
    equity = pd.to_numeric(series["equity"], errors="coerce").ffill()
    time_values = pd.to_datetime(series.get("time"), errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=150)
    fig.patch.set_alpha(0)
    ax.set_facecolor("#111216")

    if isinstance(time_values, pd.Series) and time_values.notna().sum() >= 2:
        if getattr(time_values.dt, "tz", None) is not None:
            time_values = time_values.dt.tz_convert(None)
        x_values = time_values
        ax.plot(x_values, equity, color=line_color, linewidth=2.35)
        ax.fill_between(x_values, equity.to_numpy(dtype=float), equity.min(), color=line_color, alpha=0.12)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        x_values = pd.to_numeric(series.get("step"), errors="coerce").fillna(0)
        ax.plot(x_values, equity, color=line_color, linewidth=2.35)
        ax.fill_between(x_values, equity.to_numpy(dtype=float), equity.min(), color=line_color, alpha=0.12)
        ax.set_xlabel("Trade Number", color="#B8AEA4", fontsize=10)

    ax.tick_params(axis="x", colors="#9D9388", labelsize=9)
    ax.tick_params(axis="y", colors="#9D9388", labelsize=9)
    ax.grid(axis="y", color="white", alpha=0.06, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("Equity", color="#B8AEA4", fontsize=10)
    ax.margins(x=0.02)
    fig.tight_layout()

    svg_buffer = StringIO()
    fig.savefig(svg_buffer, format="svg", transparent=True, bbox_inches="tight")
    svg_markup = svg_buffer.getvalue()
    st.markdown(
        f'<div class="dashboard-mpl-chart">{svg_markup}</div>',
        unsafe_allow_html=True,
    )
    plt.close(fig)

