from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dashboard.data import (
    build_backtest_run_labels,
    build_equity_curve,
    delete_backtest_run_and_dataset,
    extract_optional_confluence_columns,
    load_backtest_bundle,
    summarize_confluence_impact,
    summarize_confluence_sets,
    summarize_confluence_summary,
)
from dashboard.settings_execution import (
    get_backtest_settings,
    initialize_backtest_state,
    render_engine_control_inputs,
    render_run_configuration_inputs,
    render_strategy_component_inputs,
    run_backtest_with_settings,
    validate_backtest_settings,
)
from dashboard.ui import dashboard_panel, render_matplotlib_equity_chart, render_metric_grid, render_metric_row, render_page_header
from src.mt5.config import INITIAL_CASH


@st.cache_data(show_spinner=False)
def load_backtest_bundle_cached(run_dir_str: str) -> dict[str, object]:
    return load_backtest_bundle(Path(run_dir_str))


def _render_scroll_anchor(anchor_id: str) -> None:
    #page jump to a target section after a dashboard action
    st.markdown(f"<div id=\"{anchor_id}\"></div>", unsafe_allow_html=True)
    if st.session_state.get("dashboard_scroll_target") != anchor_id:
        return

    components.html(
        f"""
        <script>
        const anchor = window.parent.document.getElementById("{anchor_id}");
        if (anchor) {{
          anchor.scrollIntoView({{ behavior: "smooth", block: "start" }});
        }}
        </script>
        """,
        height=0,
    )
    st.session_state.pop("dashboard_scroll_target", None)


def _render_saved_backtest_selector(
    selected_backtest_run: Path | None,
    backtest_runs: list[Path],
    *,
    key_prefix: str,
) -> Path | None:
    #Rendering the shared saved-backtest selector used by analysis pages
    if selected_backtest_run is None:
        st.info("No backtest runs are available yet. Run a backtest from the Backtest page to populate this workspace.")
        return None

    run_options = [str(run_dir) for run_dir in backtest_runs]
    if not run_options:
        st.info("No saved backtest runs are available.")
        return None

    run_label_map = build_backtest_run_labels(tuple(run_options))
    selected_run_str = str(selected_backtest_run)
    if selected_run_str not in run_options:
        selected_run_str = run_options[0]

    st.caption("Saved Backtest Runs")
    selector_left, selector_right = st.columns((12, 1))
    with selector_left:
        selected_run_str = st.selectbox(
            "Saved Backtest Runs",
            options=run_options,
            index=run_options.index(selected_run_str),
            format_func=lambda value: run_label_map.get(value, Path(value).name),
            label_visibility="collapsed",
            key=f"{key_prefix}_saved_backtests",
        )
    with selector_right:
        if st.button(
            "x",
            key=f"{key_prefix}_delete_selected_run",
            width="stretch",
            help=f"Delete {run_label_map.get(selected_run_str, Path(selected_run_str).name)}",
        ):
            try:
                load_backtest_bundle_cached.clear()
                st.cache_data.clear()
                delete_backtest_run_and_dataset(Path(selected_run_str))
                remaining_runs = [item for item in run_options if item != selected_run_str]
                st.session_state["dashboard_selected_backtest_run"] = remaining_runs[0] if remaining_runs else None
                st.rerun()
            except PermissionError:
                st.session_state["dashboard_delete_error"] = (
                    "That run could not be deleted because Windows or OneDrive is still holding the folder. "
                    "Close anything using that run and try again."
                )
                st.rerun()
            except OSError as exc:
                st.session_state["dashboard_delete_error"] = f"Unable to delete the selected run: {exc}"
                st.rerun()

    st.session_state["dashboard_selected_backtest_run"] = selected_run_str
    return Path(selected_run_str)


def render_overview(selected_backtest_run: Path | None, backtest_runs: list[Path]) -> None:
    #Showing the executive strategy summary for the selected run
    render_page_header(
        current_page="Overview",
        title="Overview",
        subtitle="Displays a high-level summary of the backtest performance, including account balance, net profit/loss, total trades, win rate, and drawdown.",
    )
    notice = st.session_state.pop("dashboard_backtest_notice", None)
    if notice:
        st.success(str(notice))
    delete_error = st.session_state.pop("dashboard_delete_error", None)
    if delete_error:
        st.error(str(delete_error))

    selected_backtest_run = _render_saved_backtest_selector(
        selected_backtest_run,
        backtest_runs,
        key_prefix="overview",
    )
    if selected_backtest_run is None:
        return

    run_options = [str(run_dir) for run_dir in backtest_runs]
    run_label_map = build_backtest_run_labels(tuple(run_options))
    selected_run_str = str(selected_backtest_run)
    bundle = load_backtest_bundle_cached(str(selected_backtest_run))
    account = bundle["account_info"]
    stats = bundle["trade_stats"]
    trades = bundle["trades"]
    curve = build_equity_curve(trades, float(account.get("initial_balance", INITIAL_CASH)))
    initial_balance = float(account.get("initial_balance", INITIAL_CASH) or INITIAL_CASH)
    ending_balance = float(stats.get("ending_balance", account.get("ending_balance", initial_balance)) or initial_balance)
    return_pct = ((ending_balance - initial_balance) / initial_balance * 100.0) if initial_balance else 0.0
    trades_count = int(stats.get("trades", account.get("trades", 0)) or 0)
    net_pnl = float(stats.get("net_pnl", account.get("net_pnl", 0.0)) or 0.0)
    avg_trade_pnl = (net_pnl / trades_count) if trades_count else 0.0

    trade_left, trade_right = st.columns(2)
    with trade_left:
        with dashboard_panel(
            "Trade Metrics",
            subtitle="Executed-trade outcomes for the selected backtest run.",
            eyebrow="Trade Performance",
        ):
            render_metric_grid(
                [
                    ("Total Trades", trades_count),
                    ("Wins", stats.get("wins")),
                    ("Losses", stats.get("losses")),
                    ("Breakevens", stats.get("breakevens")),
                    ("Win Rate", stats.get("win_rate")),
                    ("Expectancy R", stats.get("expectancy_r")),
                    ("Profit Factor", stats.get("profit_factor")),
                    ("Avg Trade PnL", avg_trade_pnl),
                ],
                columns=2,
            )

    with trade_right:
        with dashboard_panel(
            "Account Metrics",
            subtitle="Account-level results after spread, slippage, and commission are applied.",
            eyebrow="Account Performance",
        ):
            render_metric_grid(
                [
                    ("Initial Balance", initial_balance),
                    ("Ending Balance", ending_balance),
                    ("Net PnL", net_pnl),
                    ("Return %", return_pct),
                    ("Max Drawdown", stats.get("max_drawdown")),
                    ("Max DD %", stats.get("max_drawdown_pct")),
                    ("Initial Max Drawdown", stats.get("initial_balance_max_drawdown")),
                    ("Initial DD %", stats.get("initial_balance_max_drawdown_pct")),
                    ("Gross Profit", stats.get("gross_profit")),
                    ("Gross Loss", stats.get("gross_loss")),
                ],
                columns=2,
            )

    with dashboard_panel(
        "Equity Curve",
        subtitle="Realized balance progression across the selected backtest run.",
        eyebrow="Performance",
        action=run_label_map.get(selected_run_str, selected_backtest_run.name),
    ):
        render_matplotlib_equity_chart(curve, line_color="#97D6FF")


def render_backtest_page(project_root: Path) -> None:
    #Showing configuration, controls, and the selected run output summary
    initialize_backtest_state()
    render_page_header(
        current_page="Backtest",
        title="Backtest",
        subtitle="Run and evaluate the trading strategy on historical data. You can select specific symbols, timeframe, and date ranges. Outputs performance metrics, trade distribution, and equity progression based on cost-adjusted execution (spread, slippage, commission).",
    )

    with dashboard_panel(
        "Run Configuration and Strategy Toggles",
        subtitle="Set the market, date range, costs, and optional confluences before launching a run.",
        eyebrow="Setup",
    ):
        config_left, config_right = st.columns((1.15, 1))
        with config_left:
            st.markdown("##### Run Configuration")
            render_run_configuration_inputs()
        with config_right:
            st.markdown("##### Strategy Toggles")
            render_strategy_component_inputs()

        action_left, action_right = st.columns((1, 1))
        with action_left:
            st.empty()
        with action_right:
            if st.button("Run Backtest", key="dashboard_run_backtest", width="stretch"):
                settings = get_backtest_settings()
                validation_error = validate_backtest_settings(settings)
                if validation_error:
                    st.error(validation_error)
                else:
                    st.session_state["dashboard_backtest_request"] = settings
                    st.session_state["dashboard_scroll_target"] = "dashboard-run-status-anchor"
                    st.rerun()

    status = st.session_state.get("dashboard_backtest_status")
    with dashboard_panel(
        "Run Status",
        subtitle="Backtest progress and logs will appear here after a dashboard-triggered run starts.",
        eyebrow="Status",
    ):
        _render_scroll_anchor("dashboard-run-status-anchor")
        pending_request = st.session_state.get("dashboard_backtest_request")
        if pending_request is not None:
            st.info("Backtest launched from the dashboard. Running now...")
            with st.spinner("Running backtest..."):
                result = run_backtest_with_settings(project_root, pending_request)

            st.session_state["dashboard_backtest_status"] = result
            st.session_state.pop("dashboard_backtest_request", None)
            st.cache_data.clear()

            if int(result.get("returncode", 1)) == 0 and result.get("latest_run"):
                st.session_state["dashboard_selected_backtest_run"] = result["latest_run"]
                st.session_state["dashboard_page"] = "Overview"
                st.session_state["dashboard_backtest_notice"] = "Backtest completed successfully. Overview updated with the latest run."
                st.session_state.pop("dashboard_scroll_target", None)
            else:
                st.session_state["dashboard_scroll_target"] = "dashboard-run-status-anchor"
            st.rerun()

        if status:
            latest_run = Path(status["latest_run"]).name if status.get("latest_run") else "No run produced"
            stdout_text = str(status.get("stdout", "")).strip()
            stderr_text = str(status.get("stderr", "")).strip()
            log_sections: list[str] = [
                f"Return code: {status.get('returncode')}",
                f"Latest run: {latest_run}",
            ]
            if stdout_text:
                log_sections.append(stdout_text)
            if stderr_text:
                log_sections.append("[stderr]\n" + stderr_text)

            if int(status.get("returncode", 1)) == 0:
                st.success("The latest dashboard backtest completed successfully.")
            else:
                st.error("The latest dashboard backtest finished with errors.")
            st.code("\n\n".join(section for section in log_sections if section), language="text")

    with dashboard_panel(
        "Engine Controls",
        subtitle="Control caching behaviour and reset the editable dashboard inputs when needed.",
        eyebrow="Execution",
    ):
        render_engine_control_inputs()
        _, button_right = st.columns((1.3, 1))
        with button_right:
            if st.button("Reset Inputs", width="stretch"):
                for key in list(st.session_state.keys()):
                    if key.startswith("dashboard_bt_"):
                        del st.session_state[key]
                initialize_backtest_state()
                st.rerun()
def render_confluence_page(selected_backtest_run: Path | None, backtest_runs: list[Path]) -> None:
    #Showing trade-level confluence summary, impact, and best-set analysis
    render_page_header(
        current_page="Confluence Analysis",
        title="Confluence Analysis",
        subtitle="Breaks down the contribution of each confluence to trade outcomes. Shows how often each confluence is present and its impact on performance (e.g., win rate, expectancy). Used to identify which set of confluences have the strongest edge.",
    )

    delete_error = st.session_state.pop("dashboard_delete_error", None)
    if delete_error:
        st.error(str(delete_error))

    selected_backtest_run = _render_saved_backtest_selector(
        selected_backtest_run,
        backtest_runs,
        key_prefix="confluence",
    )
    if selected_backtest_run is None:
        return

    bundle = load_backtest_bundle_cached(str(selected_backtest_run))
    trades = bundle["trades"]
    if trades.empty:
        st.info("This run does not contain any trades.")
        return

    summary = summarize_confluence_summary(trades)
    impact = summarize_confluence_impact(trades)
    confluence_sets = summarize_confluence_sets(trades)
    if summary.empty:
        st.info("No optional confluences were found in this trade export.")
        return

    total_trades = len(trades)
    optional_columns = extract_optional_confluence_columns(trades)
    optional_active_count = (
        trades[optional_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).mean()
        if optional_columns
        else 0.0
    )
    most_common = summary.iloc[0]
    best_lift = impact.iloc[0] if not impact.empty else None

    with dashboard_panel(
        "Confluence Summary",
        subtitle="Trade-level count and coverage for each confluence in the selected backtest.",
        eyebrow="Summary",
    ):
        render_metric_row(
            [
                ("Trades Analysed", total_trades),
                ("Avg Active Confluences", optional_active_count),
                ("Most Frequent Confluence", most_common.get("confluence")),
                ("Top Active Rate %", most_common.get("active_rate_pct")),
            ]
        )
        st.dataframe(summary, width="stretch", hide_index=True)

    with dashboard_panel(
        "Confluence Impact (With vs Without)",
        subtitle="Does each confluence actually improve performance when present on executed trades?",
        eyebrow="Impact",
    ):
        if best_lift is not None:
            render_metric_row(
                [
                    ("Best Improving Confluence", best_lift.get("confluence")),
                    ("Win Rate Delta pp", best_lift.get("win_rate_delta_pp")),
                    ("Expectancy Delta R", best_lift.get("expectancy_delta_r")),
                    ("Trades With", best_lift.get("trades_with")),
                ]
            )
        st.dataframe(impact, width="stretch", hide_index=True)

    set_left, set_right = st.columns((1.05, 1))
    best_set_candidates = confluence_sets[confluence_sets["trade_count"] >= 2] if not confluence_sets.empty else confluence_sets
    if best_set_candidates.empty and not confluence_sets.empty:
        best_set_candidates = confluence_sets
    default_set = best_set_candidates.iloc[0]["confluence_set"] if not best_set_candidates.empty else None

    with set_left:
        with dashboard_panel(
            "Best Set Of Confluences",
            subtitle="Ranked confluence sets discovered from winning trades in this backtest, excluding the fixed core confluences.",
            eyebrow="Set",
        ):
            if confluence_sets.empty:
                st.info("No confluence sets could be derived from this backtest.")
            else:
                st.dataframe(confluence_sets.head(12), width="stretch", hide_index=True)

    with set_right:
        with dashboard_panel(
            "Set Performance",
            subtitle="Performance for the selected confluence set across non-breakeven trades within this specific backtest.",
            eyebrow="Performance",
        ):
            if confluence_sets.empty:
                st.info("No set-level performance is available for this run.")
            else:
                set_options = confluence_sets["confluence_set"].tolist()
                selected_set = st.selectbox(
                    "Confluence Set",
                    options=set_options,
                    index=set_options.index(default_set) if default_set in set_options else 0,
                )
                selected_row = confluence_sets.loc[confluence_sets["confluence_set"] == selected_set].iloc[0]
                render_metric_grid(
                    [
                        ("Confluence Set", selected_row.get("confluence_set")),
                        ("Set Size", selected_row.get("set_size")),
                        ("Trade Count", selected_row.get("trade_count")),
                        ("Win Rate", selected_row.get("win_rate")),
                        ("Expectancy R", selected_row.get("expectancy_r")),
                        ("PnL Cash", selected_row.get("pnl_cash")),
                    ],
                    columns=2,
                )
