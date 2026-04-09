from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.backtest_views import (
    render_backtest_page,
    render_confluence_page,
    render_overview,
)
from dashboard.data import latest_run_or_none, list_run_directories
from dashboard.ml_views import render_ml_page
from dashboard.theme import apply_theme
from dashboard.ui import PAGE_ICONS, PAGE_ORDER, render_section_label
from src.mt5.config import BACKTEST_OUTPUT_DIR, OUTPUTS_DIR


st.set_page_config(page_title="TBot Dashboard", page_icon="T", layout="wide")
apply_theme()


MENU_ITEMS = [(item, PAGE_ICONS[item]) for item in PAGE_ORDER]


def _resolve_selected_run(runs: list[Path], session_key: str, default_root: Path) -> Path | None:
    if not runs:
        st.session_state.pop(session_key, None)
        return None

    current = st.session_state.get(session_key)
    current_path = Path(current) if current else None
    if current_path in runs:
        selected = current_path
    else:
        selected = latest_run_or_none(default_root) or runs[0]
        st.session_state[session_key] = str(selected)

    return selected


def main() -> None:
    #Routing between the dashboard pages and current selected artifacts
    backtest_runs = list_run_directories(BACKTEST_OUTPUT_DIR)
    ml_root = OUTPUTS_DIR / "ml_models"
    ml_runs = list_run_directories(ml_root)

    st.session_state.setdefault("dashboard_page", "Overview")
    valid_pages = {item for item, _ in MENU_ITEMS}
    if st.session_state["dashboard_page"] not in valid_pages:
        st.session_state["dashboard_page"] = "Overview"

    selected_backtest_run = _resolve_selected_run(backtest_runs, "dashboard_selected_backtest_run", BACKTEST_OUTPUT_DIR)
    selected_ml_run = _resolve_selected_run(ml_runs, "dashboard_selected_ml_run", ml_root)

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-mini-chart">
              <div class="sidebar-mini-chart-bars">
                <span style="height: 34%"></span>
                <span style="height: 56%"></span>
                <span style="height: 42%"></span>
                <span style="height: 70%"></span>
                <span style="height: 61%"></span>
                <span style="height: 88%"></span>
                <span style="height: 74%"></span>
                <span style="height: 52%"></span>
                <span style="height: 64%"></span>
                <span style="height: 82%"></span>
                <span style="height: 58%"></span>
                <span style="height: 72%"></span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_section_label("Menu")
        for item, icon in MENU_ITEMS:
            is_active = st.session_state["dashboard_page"] == item
            if st.button(
                f"{icon}  {item}",
                key=f"nav_{item}",
                width="stretch",
                type="primary" if is_active else "secondary",
            ):
                st.session_state["dashboard_page"] = item
                st.rerun()

    page = st.session_state["dashboard_page"]
    if page == "Overview":
        render_overview(selected_backtest_run, backtest_runs)
    elif page == "Backtest":
        render_backtest_page(PROJECT_ROOT)
    elif page == "Confluence Analysis":
        render_confluence_page(selected_backtest_run, backtest_runs)
    else:
        render_ml_page(selected_ml_run, PROJECT_ROOT)


if __name__ == "__main__":
    main()
