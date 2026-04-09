from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FibonacciConfig:
    """
    Directional half-leg Fibonacci confluence.

    Uses latest confirmed numeric market-structure swing legs.

    Bullish impulse:
        prior swing low -> later swing high

    Bearish impulse:
        prior swing high -> later swing low

    Validation:
    - Long valid when long entry / demand zone lies in bullish 0 -> 0.5 leg band
    - Short valid when short entry / supply zone lies in bearish 0 -> 0.5 leg band

    Directional mapping:
    - Bullish leg: 0 = swing low, 0.5 = midpoint, 1 = swing high
    - Bearish leg: 0 = swing high, 0.5 = midpoint, 1 = swing low
    """
    min_leg_size: float = 0.0
    zone_overlap_tolerance: float = 0.0


def _require_columns(df: pd.DataFrame) -> None:
    needed = {"high", "low"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")


def _validate_config(cfg: FibonacciConfig) -> None:
    if cfg.min_leg_size < 0:
        raise ValueError("min_leg_size must be >= 0.")
    if cfg.zone_overlap_tolerance < 0:
        raise ValueError("zone_overlap_tolerance must be >= 0.")


def _first_existing_column(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _price_inside_zone(price: float, zone_low: float, zone_high: float, tolerance: float = 0.0) -> bool:
    return (zone_low - tolerance) <= price <= (zone_high + tolerance)


def _zones_overlap(
    a_low: float,
    a_high: float,
    b_low: float,
    b_high: float,
    tolerance: float = 0.0,
) -> bool:
    return (a_high + tolerance) >= b_low and (a_low - tolerance) <= b_high


def compute_fibonacci_features(
    df: pd.DataFrame,
    config: Optional[FibonacciConfig] = None,
) -> pd.DataFrame:
    _require_columns(df)
    cfg = config or FibonacciConfig()
    _validate_config(cfg)

    out = df.copy()
    n = len(out)

    #Using confirmed swing-price columns instead of raw pivot markers
    swing_low_col = _first_existing_column(
        out,
        [
            "ext_swing_low",
            "int_swing_low",
        ],
    )
    swing_high_col = _first_existing_column(
        out,
        [
            "ext_swing_high",
            "int_swing_high",
        ],
    )

    if swing_low_col is None or swing_high_col is None:
        #Returning empty validity if market-structure swings are unavailable
        out["fibonacci_long_valid"] = False
        out["fibonacci_short_valid"] = False
        out["Fibonacci_Valid"] = False
        out["fib_bull_impulse_low"] = np.nan
        out["fib_bull_impulse_high"] = np.nan
        out["fib_bear_impulse_low"] = np.nan
        out["fib_bear_impulse_high"] = np.nan
        out["fib_long_band_low"] = np.nan
        out["fib_long_band_high"] = np.nan
        out["fib_short_band_low"] = np.nan
        out["fib_short_band_high"] = np.nan
        return out

    demand_zone_low_col = _first_existing_column(
        out,
        ["demand_zone_bottom", "demand_zone_low", "active_demand_zone_low"],
    )
    demand_zone_high_col = _first_existing_column(
        out,
        ["demand_zone_top", "demand_zone_high", "active_demand_zone_high"],
    )
    supply_zone_low_col = _first_existing_column(
        out,
        ["supply_zone_bottom", "supply_zone_low", "active_supply_zone_low"],
    )
    supply_zone_high_col = _first_existing_column(
        out,
        ["supply_zone_top", "supply_zone_high", "active_supply_zone_high"],
    )

    long_entry_col = _first_existing_column(
        out,
        ["demand_zone_entry", "long_entry_price", "long_entry", "entry_price_long"],
    )
    short_entry_col = _first_existing_column(
        out,
        ["supply_zone_entry", "short_entry_price", "short_entry", "entry_price_short"],
    )

    bullish_impulse_low = np.full(n, np.nan)
    bullish_impulse_high = np.full(n, np.nan)
    bearish_impulse_low = np.full(n, np.nan)
    bearish_impulse_high = np.full(n, np.nan)

    fib_long_band_low = np.full(n, np.nan)
    fib_long_band_high = np.full(n, np.nan)
    fib_short_band_low = np.full(n, np.nan)
    fib_short_band_high = np.full(n, np.nan)

    fibonacci_long_valid = np.zeros(n, dtype=bool)
    fibonacci_short_valid = np.zeros(n, dtype=bool)

    last_swing_low_price: Optional[float] = None
    last_swing_low_idx: Optional[int] = None
    last_swing_high_price: Optional[float] = None
    last_swing_high_idx: Optional[int] = None

    active_bull_low: Optional[float] = None
    active_bull_high: Optional[float] = None
    active_bear_low: Optional[float] = None
    active_bear_high: Optional[float] = None

    swing_low_series = out[swing_low_col]
    swing_high_series = out[swing_high_col]

    for i in range(n):
        #Tracking the latest confirmed swing points
        sl = swing_low_series.iat[i]
        sh = swing_high_series.iat[i]

        if pd.notna(sl):
            last_swing_low_price = float(sl)
            last_swing_low_idx = i

            if last_swing_high_price is not None and last_swing_high_idx is not None:
                if last_swing_high_idx < i:
                    leg_low = float(sl)
                    leg_high = float(last_swing_high_price)
                    leg_size = leg_high - leg_low
                    if leg_size >= cfg.min_leg_size:
                        active_bear_low = leg_low
                        active_bear_high = leg_high

        if pd.notna(sh):
            last_swing_high_price = float(sh)
            last_swing_high_idx = i

            if last_swing_low_price is not None and last_swing_low_idx is not None:
                if last_swing_low_idx < i:
                    leg_low = float(last_swing_low_price)
                    leg_high = float(sh)
                    leg_size = leg_high - leg_low
                    if leg_size >= cfg.min_leg_size:
                        active_bull_low = leg_low
                        active_bull_high = leg_high

        #Projecting the bullish 0 -> 0.5 band for long-side validation
        if active_bull_low is not None and active_bull_high is not None:
            zero_level = active_bull_low
            one_level = active_bull_high
            half_level = zero_level + 0.5 * (one_level - zero_level)

            bullish_impulse_low[i] = zero_level
            bullish_impulse_high[i] = one_level
            fib_long_band_low[i] = zero_level
            fib_long_band_high[i] = half_level

            long_valid = False

            if long_entry_col is not None:
                entry_val = out[long_entry_col].iat[i]
                if pd.notna(entry_val):
                    long_valid = _price_inside_zone(
                        float(entry_val),
                        zero_level,
                        half_level,
                        cfg.zone_overlap_tolerance,
                    )

            if (
                not long_valid
                and demand_zone_low_col is not None
                and demand_zone_high_col is not None
            ):
                dz_low = out[demand_zone_low_col].iat[i]
                dz_high = out[demand_zone_high_col].iat[i]
                if pd.notna(dz_low) and pd.notna(dz_high):
                    long_valid = _zones_overlap(
                        float(dz_low),
                        float(dz_high),
                        zero_level,
                        half_level,
                        cfg.zone_overlap_tolerance,
                    )

            fibonacci_long_valid[i] = long_valid

        #Projecting the bearish 0 -> 0.5 band for short-side validation
        if active_bear_low is not None and active_bear_high is not None:
            zero_level = active_bear_high
            one_level = active_bear_low
            half_level = zero_level - 0.5 * (zero_level - one_level)

            bearish_impulse_low[i] = one_level
            bearish_impulse_high[i] = zero_level
            fib_short_band_low[i] = half_level
            fib_short_band_high[i] = zero_level

            short_valid = False

            if short_entry_col is not None:
                entry_val = out[short_entry_col].iat[i]
                if pd.notna(entry_val):
                    short_valid = _price_inside_zone(
                        float(entry_val),
                        half_level,
                        zero_level,
                        cfg.zone_overlap_tolerance,
                    )

            if (
                not short_valid
                and supply_zone_low_col is not None
                and supply_zone_high_col is not None
            ):
                sz_low = out[supply_zone_low_col].iat[i]
                sz_high = out[supply_zone_high_col].iat[i]
                if pd.notna(sz_low) and pd.notna(sz_high):
                    short_valid = _zones_overlap(
                        float(sz_low),
                        float(sz_high),
                        half_level,
                        zero_level,
                        cfg.zone_overlap_tolerance,
                    )

            fibonacci_short_valid[i] = short_valid

    out["fib_bull_impulse_low"] = bullish_impulse_low
    out["fib_bull_impulse_high"] = bullish_impulse_high
    out["fib_bear_impulse_low"] = bearish_impulse_low
    out["fib_bear_impulse_high"] = bearish_impulse_high

    out["fib_long_band_low"] = fib_long_band_low
    out["fib_long_band_high"] = fib_long_band_high
    out["fib_short_band_low"] = fib_short_band_low
    out["fib_short_band_high"] = fib_short_band_high

    out["fibonacci_long_valid"] = fibonacci_long_valid
    out["fibonacci_short_valid"] = fibonacci_short_valid

    out["Fibonacci_Valid"] = out["fibonacci_long_valid"] | out["fibonacci_short_valid"]

    return out


def validate_fibonacci_output(df: pd.DataFrame) -> None:
    required = {
        "fibonacci_long_valid",
        "fibonacci_short_valid",
        "Fibonacci_Valid",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fibonacci output missing columns: {sorted(missing)}")

    for col in required:
        if df[col].dtype != bool:
            raise ValueError(f"Column '{col}' must be boolean.")
