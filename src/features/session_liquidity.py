from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SessionLiquidityConfig:
    """
    Session windows are interpreted in the timezone of df['time'] if timezone-aware.
    If naive, they are treated as-is.

    Times are half-open intervals: [start_hour, end_hour)
    """
    asia_start_hour: int = 0
    asia_end_hour: int = 8

    london_start_hour: int = 8
    london_end_hour: int = 13

    new_york_start_hour: int = 13
    new_york_end_hour: int = 22

    sweep_valid_bars: int = 24
    require_close_back_through: bool = True
    use_wick_sweeps: bool = True


def _require_columns(df: pd.DataFrame) -> None:
    needed = {"time", "open", "high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")


def _validate_config(cfg: SessionLiquidityConfig) -> None:
    hour_fields = [
        cfg.asia_start_hour,
        cfg.asia_end_hour,
        cfg.london_start_hour,
        cfg.london_end_hour,
        cfg.new_york_start_hour,
        cfg.new_york_end_hour,
    ]
    if any((h < 0 or h > 24) for h in hour_fields):
        raise ValueError("Session hours must be between 0 and 24.")

    if cfg.sweep_valid_bars < 1:
        raise ValueError("sweep_valid_bars must be >= 1.")


def _prepare_time(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["time"], errors="coerce")
    if ts.isna().any():
        raise ValueError("Column 'time' contains non-datetime values.")
    return ts


def _first_existing_column(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _session_name_for_hour(hour: int, cfg: SessionLiquidityConfig) -> Optional[str]:
    if cfg.asia_start_hour <= hour < cfg.asia_end_hour:
        return "Asia"
    if cfg.london_start_hour <= hour < cfg.london_end_hour:
        return "London"
    if cfg.new_york_start_hour <= hour < cfg.new_york_end_hour:
        return "NewYork"
    return None


def _is_session_level_sweep(
    sweep_price: float,
    close_price: float,
    reference_level: float,
    cfg: SessionLiquidityConfig,
    *,
    swept_below: bool,
) -> bool:
    wick_swept = sweep_price < reference_level if swept_below else sweep_price > reference_level
    close_back_through = close_price >= reference_level if swept_below else close_price <= reference_level

    if not wick_swept:
        return False

    if cfg.require_close_back_through:
        return close_back_through

    return cfg.use_wick_sweeps


def compute_session_liquidity_features(
    df: pd.DataFrame,
    config: Optional[SessionLiquidityConfig] = None,
) -> pd.DataFrame:
    _require_columns(df)
    cfg = config or SessionLiquidityConfig()
    _validate_config(cfg)

    out = df.copy()
    ts = _prepare_time(out)

    if not ts.is_monotonic_increasing:
        raise ValueError("Column 'time' must be sorted in ascending order.")

    hours = ts.dt.hour.to_numpy()
    dates = ts.dt.floor("D")

    n = len(out)

    session_name = np.empty(n, dtype=object)
    for i in range(n):
        session_name[i] = _session_name_for_hour(int(hours[i]), cfg)

    out["session_name"] = session_name

    prev_asia_high = np.full(n, np.nan)
    prev_asia_low = np.full(n, np.nan)
    prev_london_high = np.full(n, np.nan)
    prev_london_low = np.full(n, np.nan)
    prev_newyork_high = np.full(n, np.nan)
    prev_newyork_low = np.full(n, np.nan)

    session_low_swept = np.zeros(n, dtype=bool)
    session_high_swept = np.zeros(n, dtype=bool)

    session_liquidity_long_valid = np.zeros(n, dtype=bool)
    session_liquidity_short_valid = np.zeros(n, dtype=bool)

    demand_valid_col = _first_existing_column(out, ["sd_demand_valid", "demand_zone_active"])
    supply_valid_col = _first_existing_column(out, ["sd_supply_valid", "supply_zone_active"])

    last_session_sweep_bar = -10_000

    active_session_name: Optional[str] = None
    active_session_day = None
    active_session_high: Optional[float] = None
    active_session_low: Optional[float] = None

    stored_session_levels: dict[str, tuple[float, float] | None] = {
        "Asia": None,
        "London": None,
        "NewYork": None,
    }

    for i in range(n):
        #Rolling the current bar into the active session block
        cur_session = session_name[i]
        cur_day = dates.iat[i]

        new_session_block = (
            i == 0
            or cur_session != active_session_name
            or cur_day != active_session_day
        )

        if new_session_block:
            #Saving the completed session range before starting a new block
            if (
                active_session_name is not None
                and active_session_high is not None
                and active_session_low is not None
            ):
                stored_session_levels[active_session_name] = (
                    float(active_session_high),
                    float(active_session_low),
                )

            active_session_name = cur_session
            active_session_day = cur_day

            if cur_session is not None:
                active_session_high = float(out["high"].iat[i])
                active_session_low = float(out["low"].iat[i])
            else:
                active_session_high = None
                active_session_low = None
        else:
            if active_session_name is not None:
                active_session_high = max(float(active_session_high), float(out["high"].iat[i]))
                active_session_low = min(float(active_session_low), float(out["low"].iat[i]))

        asia_levels = stored_session_levels["Asia"]
        london_levels = stored_session_levels["London"]
        newyork_levels = stored_session_levels["NewYork"]

        if asia_levels is not None:
            prev_asia_high[i] = asia_levels[0]
            prev_asia_low[i] = asia_levels[1]

        if london_levels is not None:
            prev_london_high[i] = london_levels[0]
            prev_london_low[i] = london_levels[1]

        if newyork_levels is not None:
            prev_newyork_high[i] = newyork_levels[0]
            prev_newyork_low[i] = newyork_levels[1]

        high_i = float(out["high"].iat[i])
        low_i = float(out["low"].iat[i])
        close_i = float(out["close"].iat[i])

        downside_swept_any = False
        upside_swept_any = False

        downside_candidates = []
        upside_candidates = []

        if asia_levels is not None:
            downside_candidates.append(asia_levels[1])
            upside_candidates.append(asia_levels[0])

        if london_levels is not None:
            downside_candidates.append(london_levels[1])
            upside_candidates.append(london_levels[0])

        if newyork_levels is not None:
            downside_candidates.append(newyork_levels[1])
            upside_candidates.append(newyork_levels[0])

        #Checking whether any stored session high or low was just swept
        downside_swept_any = any(
            _is_session_level_sweep(low_i, close_i, float(reference_low), cfg, swept_below=True)
            for reference_low in downside_candidates
        )
        upside_swept_any = any(
            _is_session_level_sweep(high_i, close_i, float(reference_high), cfg, swept_below=False)
            for reference_high in upside_candidates
        )

        if downside_swept_any:
            session_low_swept[i] = True

        if upside_swept_any:
            session_high_swept[i] = True
        if downside_swept_any or upside_swept_any:
            last_session_sweep_bar = i

        #Treating recent session sweeps as context for either demand or supply
        recent_session_sweep = (i - last_session_sweep_bar) <= cfg.sweep_valid_bars
        demand_valid = (
            bool(out[demand_valid_col].iat[i])
            if demand_valid_col is not None and pd.notna(out[demand_valid_col].iat[i])
            else False
        )
        supply_valid = (
            bool(out[supply_valid_col].iat[i])
            if supply_valid_col is not None and pd.notna(out[supply_valid_col].iat[i])
            else False
        )

        session_liquidity_long_valid[i] = recent_session_sweep and demand_valid
        session_liquidity_short_valid[i] = recent_session_sweep and supply_valid

    out["prev_asia_high"] = prev_asia_high
    out["prev_asia_low"] = prev_asia_low
    out["prev_london_high"] = prev_london_high
    out["prev_london_low"] = prev_london_low
    out["prev_newyork_high"] = prev_newyork_high
    out["prev_newyork_low"] = prev_newyork_low

    out["session_liquidity_long_valid"] = session_liquidity_long_valid
    out["session_liquidity_short_valid"] = session_liquidity_short_valid

    #Keeping the combined session-liquidity flag for reporting and ML
    out["Session_Liquidity_Valid"] = (
        out["session_liquidity_long_valid"] | out["session_liquidity_short_valid"]
    )

    #Keeping session-state flags for engine snapshots and spread logic
    out["is_asia_session"] = out["session_name"].eq("Asia")
    out["is_london_session"] = out["session_name"].eq("London")
    out["is_newyork_session"] = out["session_name"].eq("NewYork")

    #Current session windows are non-overlapping, but the engine expects the flag
    out["is_london_newyork_overlap"] = False

    #Keeping sweep diagnostics available for analysis
    out["session_low_swept"] = session_low_swept
    out["session_high_swept"] = session_high_swept

    return out


def validate_session_liquidity_output(df: pd.DataFrame) -> None:
    required = {
        "session_liquidity_long_valid",
        "session_liquidity_short_valid",
        "Session_Liquidity_Valid",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Session liquidity output missing columns: {sorted(missing)}")

    for col in required:
        if df[col].dtype != bool:
            raise ValueError(f"Column '{col}' must be boolean.")
