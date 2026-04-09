from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import date, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.data import normalize_symbol_name
from src.mt5.connector import MT5Connector
from src.mt5.config import (
    BACKTEST_OUTPUT_DIR,
    BASE_SPREAD_PIPS,
    COMMISSION_PER_LOT,
    END_DATE,
    FORCE_REBUILD_FEATURES,
    INITIAL_CASH,
    PIPELINE_TOGGLES,
    PRINT_STAGE_TIMINGS,
    RISK_PCT,
    SLIPPAGE_PIPS,
    START_DATE,
    SYMBOLS,
    TARGET_RR,
    TIMEFRAMES,
    USE_FEATURE_CACHE,
    USE_RAW_CACHE,
)

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


AVAILABLE_TIMEFRAMES = ["M5", "M10", "M15", "M30", "H1"]
PREFERRED_SYMBOLS = [
    "AUDUSD",
    "EURUSD",
    "EURJPY",
    "GBPUSD",
    "USDJPY",
    "USDCAD",
    "USDCHF",
    "AUDJPY",
    "XAUUSD",
    "XAGUSD",
]
FX_CODES = {
    "AUD",
    "CAD",
    "CHF",
    "CNH",
    "CNY",
    "EUR",
    "GBP",
    "HKD",
    "JPY",
    "MXN",
    "NOK",
    "NZD",
    "SEK",
    "SGD",
    "TRY",
    "USD",
    "ZAR",
}
STATE_PREFIX = "dashboard_bt_"


def _state_key(name: str) -> str:
    return f"{STATE_PREFIX}{name}"


def _fallback_symbols() -> list[str]:
    #Keeping a stable fallback symbol list if MT5 is unavailable
    merged = [normalize_symbol_name(symbol) for symbol in [*SYMBOLS, *PREFERRED_SYMBOLS]]
    deduped = [symbol for symbol in dict.fromkeys(merged) if symbol]
    return deduped or ["GBPUSD"]


def _is_precious_metal_symbol(symbol: str) -> bool:
    #Keeping only gold and silver broker symbol variants
    upper_symbol = symbol.upper()
    return any(token in upper_symbol for token in ("XAUUSD", "XAGUSD", "GOLD", "SILVER"))


def _is_forex_symbol(symbol: str) -> bool:
    #Detecting FX pairs from the broker symbol prefix
    match = re.match(r"^([A-Z]{6})", symbol.upper())
    if not match:
        return False

    pair = match.group(1)
    base, quote = pair[:3], pair[3:6]
    return base in FX_CODES and quote in FX_CODES and base != quote


def _build_dashboard_connector() -> MT5Connector:
    #building a fast-fail MT5 connector for dashboard metadata lookups
    return MT5Connector(
        MT5Connector.creds_from_env(),
        timeout_sec=2.0,
        retries=1,
        retry_backoff_sec=0.25,
    )


@st.cache_data(show_spinner=False, ttl=60)
def load_mt5_symbol_options() -> list[str]:
    #Load only forex and gold/silver symbols and showing them with normal names
    fallback = _fallback_symbols()
    if mt5 is None:
        return fallback

    try:
        with _build_dashboard_connector() as connector:
            symbols = connector.list_symbols()
    except Exception:
        return fallback

    unique = list(dict.fromkeys(symbols))
    filtered = [symbol for symbol in unique if _is_forex_symbol(symbol) or _is_precious_metal_symbol(symbol)]
    if not filtered:
        return fallback

    canonical_symbols = [normalize_symbol_name(symbol) for symbol in filtered]
    unique_symbols = [symbol for symbol in dict.fromkeys(canonical_symbols) if symbol]
    preferred_matches = [symbol for symbol in PREFERRED_SYMBOLS if symbol in unique_symbols]
    remaining = sorted(symbol for symbol in unique_symbols if symbol not in preferred_matches)
    ordered = preferred_matches + remaining
    return ordered or fallback


def _mt5_timeframe_constant(timeframe: str):
    #Mapping the dashboard timeframe strings to MT5 constants
    if mt5 is None:
        return None

    return {
        "M5": mt5.TIMEFRAME_M5,
        "M10": mt5.TIMEFRAME_M10,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
    }.get(timeframe)


def _read_bar_timestamp(symbol: str, timeframe, position: int) -> pd.Timestamp | None:
    #Fetching a single bar timestamp from MT5 by position
    if mt5 is None:
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, position, 1)
    if rates is None or len(rates) == 0:
        return None

    return pd.to_datetime(int(rates[0]["time"]), unit="s", utc=True, errors="coerce")


@st.cache_data(show_spinner=False, ttl=60)
def load_mt5_date_bounds(symbol: str, timeframe: str) -> tuple[date, date] | None:
    #Resolving the full available MT5 history range for one symbol/timeframe
    tf_constant = _mt5_timeframe_constant(timeframe)
    if tf_constant is None:
        return None

    try:
        with _build_dashboard_connector() as connector:
            resolved_symbol = connector.resolve_symbol(symbol)

            latest_bar = _read_bar_timestamp(resolved_symbol, tf_constant, 0)
            if latest_bar is None or pd.isna(latest_bar):
                return None

            probe = 1
            last_valid = 0
            while probe <= 10_000_000:
                probe_bar = _read_bar_timestamp(resolved_symbol, tf_constant, probe)
                if probe_bar is None or pd.isna(probe_bar):
                    break
                last_valid = probe
                probe *= 2

            low = last_valid
            high = probe
            while low + 1 < high:
                mid = (low + high) // 2
                mid_bar = _read_bar_timestamp(resolved_symbol, tf_constant, mid)
                if mid_bar is None or pd.isna(mid_bar):
                    high = mid
                else:
                    low = mid

            earliest_bar = _read_bar_timestamp(resolved_symbol, tf_constant, low)
            if earliest_bar is None or pd.isna(earliest_bar):
                earliest_bar = latest_bar

            earliest_date = pd.Timestamp(earliest_bar).tz_convert(timezone.utc).date()
            latest_date = pd.Timestamp(latest_bar).tz_convert(timezone.utc).date()
            return earliest_date, latest_date
    except Exception:
        return None


def _default_symbol() -> str:
    #Choosing the best initial dashboard symbol
    options = load_mt5_symbol_options()
    configured = normalize_symbol_name(SYMBOLS[0]) if SYMBOLS else None
    if configured in options:
        return configured
    return options[0] if options else "GBPUSD"


def _default_timeframe() -> str:
    #Choosing the best initial dashboard timeframe
    configured = TIMEFRAMES[0] if TIMEFRAMES else None
    if configured in AVAILABLE_TIMEFRAMES:
        return configured
    return "M30"


def _ensure_selection_state() -> None:
    #Keeping the selected symbol and timeframe valid against current options
    symbol_options = load_mt5_symbol_options()
    if st.session_state[_state_key("symbol")] not in symbol_options:
        st.session_state[_state_key("symbol")] = symbol_options[0] if symbol_options else "GBPUSD"

    if st.session_state[_state_key("timeframe")] not in AVAILABLE_TIMEFRAMES:
        st.session_state[_state_key("timeframe")] = _default_timeframe()


def _sync_date_state(symbol: str, timeframe: str) -> tuple[date, date] | None:
    #Clamping the dashboard date inputs to the selected MT5 history range
    available_range = load_mt5_date_bounds(symbol, timeframe)
    if available_range is None:
        return None

    range_start, range_end = available_range
    state_start = st.session_state[_state_key("start_date")]
    state_end = st.session_state[_state_key("end_date")]
    clamped_start = max(range_start, min(state_start, range_end))
    clamped_end = max(range_start, min(state_end, range_end))

    if clamped_start >= clamped_end:
        if range_start < range_end:
            one_day = timedelta(days=1)
            if clamped_end <= range_start:
                clamped_start = range_start
                clamped_end = min(range_end, range_start + one_day)
            else:
                clamped_start = max(range_start, clamped_end - one_day)
            if clamped_start >= clamped_end:
                clamped_start = range_start
                clamped_end = range_end
        else:
            clamped_start = range_start
            clamped_end = range_end

    st.session_state[_state_key("start_date")] = clamped_start
    st.session_state[_state_key("end_date")] = clamped_end

    st.session_state[_state_key("range_symbol")] = symbol
    st.session_state[_state_key("range_timeframe")] = timeframe
    return available_range


def initialize_backtest_state() -> None:
    #Seeding the editable dashboard state from the current config defaults
    defaults = {
        "symbol": _default_symbol(),
        "timeframe": _default_timeframe(),
        "start_date": START_DATE.date(),
        "end_date": END_DATE.date(),
        "initial_cash": float(INITIAL_CASH),
        "risk_pct": float(RISK_PCT) * 100.0,
        "target_rr": float(TARGET_RR),
        "base_spread": float(BASE_SPREAD_PIPS),
        "base_slippage": float(SLIPPAGE_PIPS),
        "commission_per_lot": float(COMMISSION_PER_LOT),
        "use_cache": bool(USE_RAW_CACHE and USE_FEATURE_CACHE),
        "rebuild_features": bool(FORCE_REBUILD_FEATURES),
        "print_stage_timings": bool(PRINT_STAGE_TIMINGS),
        "use_session_liquidity": bool(PIPELINE_TOGGLES.use_session_liquidity),
        "use_volume_profile": bool(PIPELINE_TOGGLES.use_volume_profile),
        "use_imbalance": bool(PIPELINE_TOGGLES.use_imbalance),
        "use_fibonacci": bool(PIPELINE_TOGGLES.use_fibonacci),
        "use_liquidity": bool(PIPELINE_TOGGLES.use_liquidity),
        "use_support_resistance": bool(PIPELINE_TOGGLES.use_support_resistance),
        "range_symbol": None,
        "range_timeframe": None,
    }

    for name, value in defaults.items():
        st.session_state.setdefault(_state_key(name), value)

    _ensure_selection_state()


def get_backtest_settings() -> dict[str, Any]:
    #Reading the current editable dashboard settings
    initialize_backtest_state()
    return {
        "symbol": str(st.session_state[_state_key("symbol")]),
        "timeframe": str(st.session_state[_state_key("timeframe")]),
        "start_date": st.session_state[_state_key("start_date")],
        "end_date": st.session_state[_state_key("end_date")],
        "initial_cash": float(st.session_state[_state_key("initial_cash")]),
        "risk_pct": float(st.session_state[_state_key("risk_pct")]),
        "target_rr": float(st.session_state[_state_key("target_rr")]),
        "base_spread": float(st.session_state[_state_key("base_spread")]),
        "base_slippage": float(st.session_state[_state_key("base_slippage")]),
        "commission_per_lot": float(st.session_state[_state_key("commission_per_lot")]),
        "use_cache": bool(st.session_state[_state_key("use_cache")]),
        "rebuild_features": bool(st.session_state[_state_key("rebuild_features")]),
        "print_stage_timings": bool(st.session_state[_state_key("print_stage_timings")]),
        "use_session_liquidity": bool(st.session_state[_state_key("use_session_liquidity")]),
        "use_volume_profile": bool(st.session_state[_state_key("use_volume_profile")]),
        "use_imbalance": bool(st.session_state[_state_key("use_imbalance")]),
        "use_fibonacci": bool(st.session_state[_state_key("use_fibonacci")]),
        "use_liquidity": bool(st.session_state[_state_key("use_liquidity")]),
        "use_support_resistance": bool(st.session_state[_state_key("use_support_resistance")]),
    }


def reset_backtest_state() -> None:
    #Resetting the dashboard inputs back to config defaults
    for key in list(st.session_state.keys()):
        if key.startswith(STATE_PREFIX):
            del st.session_state[key]
    initialize_backtest_state()


def render_run_configuration_inputs() -> None:
    #Rendering the editable run-configuration block
    initialize_backtest_state()
    left, right = st.columns(2)
    symbol_options = load_mt5_symbol_options()
    available_range = _sync_date_state(
        str(st.session_state[_state_key("symbol")]),
        str(st.session_state[_state_key("timeframe")]),
    )

    with left:
        st.selectbox(
            "Symbol",
            options=symbol_options,
            key=_state_key("symbol"),
            help="Showing broker symbols from MT5. Type to search the full list.",
        )
        st.selectbox("Timeframe", options=AVAILABLE_TIMEFRAMES, key=_state_key("timeframe"))

        selected_symbol = str(st.session_state[_state_key("symbol")])
        selected_timeframe = str(st.session_state[_state_key("timeframe")])
        available_range = _sync_date_state(selected_symbol, selected_timeframe)
        date_kwargs: dict[str, Any] = {}
        if available_range is not None:
            range_start, range_end = available_range
            date_kwargs = {"min_value": range_start, "max_value": range_end}
            st.caption(f"Available MT5 range: {range_start.isoformat()} to {range_end.isoformat()}")
        else:
            st.caption("Available MT5 range could not be loaded. Falling back to the current dashboard dates.")

        st.date_input("Start Date", key=_state_key("start_date"), **date_kwargs)
        st.date_input("End Date", key=_state_key("end_date"), **date_kwargs)

    with right:
        st.number_input("Risk %", min_value=0.1, max_value=10.0, step=0.1, key=_state_key("risk_pct"))
        st.number_input("Target RR", min_value=1.0, max_value=10.0, step=0.1, key=_state_key("target_rr"))
        st.number_input("Base Spread", min_value=0.0, max_value=20.0, step=0.1, key=_state_key("base_spread"))
        st.number_input("Base Slippage", min_value=0.0, max_value=10.0, step=0.1, key=_state_key("base_slippage"))
        st.number_input("Commission Per Lot", min_value=0.0, max_value=50.0, step=0.5, key=_state_key("commission_per_lot"))


def render_strategy_component_inputs() -> None:
    #Rendering core and optional feature controls
    initialize_backtest_state()
    left, right = st.columns(2)

    with left:
        st.markdown("##### Core Confluences")
        st.caption("Always enabled for every backtest run.")
        st.checkbox("ATR", value=True, disabled=True)
        st.checkbox("Market Structure", value=True, disabled=True)
        st.checkbox("Supply & Demand", value=True, disabled=True)

    with right:
        st.markdown("##### Extra Confluences")
        st.caption("Optional layers that can be toggled on or off.")
        st.checkbox("Session Liquidity", key=_state_key("use_session_liquidity"))
        st.checkbox("Volume Profile", key=_state_key("use_volume_profile"))
        st.checkbox("Imbalance (FVG)", key=_state_key("use_imbalance"))
        st.checkbox("Fibonacci", key=_state_key("use_fibonacci"))
        st.checkbox("Liquidity", key=_state_key("use_liquidity"))
        st.checkbox("Support and Resistance", key=_state_key("use_support_resistance"))


def render_engine_control_inputs() -> None:
    #Rendering cache and execution toggles
    initialize_backtest_state()
    left, right, third = st.columns(3)
    with left:
        st.checkbox("Use Cache", key=_state_key("use_cache"))
    with right:
        st.checkbox("Rebuild Features", key=_state_key("rebuild_features"))
    with third:
        st.checkbox("Print Stage Timings", key=_state_key("print_stage_timings"))


def _bool_env(value: bool) -> str:
    return "true" if value else "false"


def build_backtest_env(settings: dict[str, Any]) -> dict[str, str]:
    #Translating the dashboard settings into runtime env overrides
    use_cache = bool(settings["use_cache"])
    env = {
        "TBOT_SYMBOLS": settings["symbol"],
        "TBOT_TIMEFRAMES": settings["timeframe"],
        "TBOT_START_DATE": settings["start_date"].isoformat(),
        "TBOT_END_DATE": settings["end_date"].isoformat(),
        "TBOT_INITIAL_CASH": str(settings["initial_cash"]),
        "TBOT_RISK_PCT": str(settings["risk_pct"] / 100.0),
        "TBOT_TARGET_RR": str(settings["target_rr"]),
        "TBOT_BASE_SPREAD_PIPS": str(settings["base_spread"]),
        "TBOT_SLIPPAGE_PIPS": str(settings["base_slippage"]),
        "TBOT_COMMISSION_PER_LOT": str(settings["commission_per_lot"]),
        "TBOT_USE_RAW_CACHE": _bool_env(use_cache),
        "TBOT_USE_FEATURE_CACHE": _bool_env(use_cache),
        "TBOT_FORCE_REBUILD_FEATURES": _bool_env(settings["rebuild_features"]),
        "TBOT_PRINT_STAGE_TIMINGS": _bool_env(settings["print_stage_timings"]),
        "TBOT_USE_SESSION_LIQUIDITY": _bool_env(settings["use_session_liquidity"]),
        "TBOT_USE_VOLUME_PROFILE": _bool_env(settings["use_volume_profile"]),
        "TBOT_USE_IMBALANCE": _bool_env(settings["use_imbalance"]),
        "TBOT_USE_FIBONACCI": _bool_env(settings["use_fibonacci"]),
        "TBOT_USE_LIQUIDITY": _bool_env(settings["use_liquidity"]),
        "TBOT_USE_SUPPORT_RESISTANCE": _bool_env(settings["use_support_resistance"]),
    }
    return env


def validate_backtest_settings(settings: dict[str, Any]) -> str | None:
    #Validating the editable backtest settings before execution
    if not settings["symbol"]:
        return "Select one symbol."
    if not settings["timeframe"]:
        return "Select one timeframe."
    if settings["start_date"] >= settings["end_date"]:
        return "End date must be after start date."
    return None


def run_backtest_with_settings(project_root: Path, settings: dict[str, Any]) -> dict[str, Any]:
    #Running the configured backtest entrypoint with dashboard overrides
    env = os.environ.copy()
    env.update(build_backtest_env(settings))

    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_backtest"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        env=env,
    )

    latest_run = None
    runs = sorted(
        [path for path in BACKTEST_OUTPUT_DIR.iterdir() if path.is_dir()] if BACKTEST_OUTPUT_DIR.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if runs:
        latest_run = runs[0]

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "latest_run": str(latest_run) if latest_run is not None else None,
    }
