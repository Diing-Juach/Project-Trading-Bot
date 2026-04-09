from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class LiquidityConfig:
    """
    Entry-sweep liquidity confluence anchored to a valid supply/demand zone.

    Long idea:
    - demand zone is valid
    - a sell-side liquidity pool sits into / just below that demand area
    - price sweeps that pool while retracing into the zone
    - the sweep remains valid for a short window into the entry

    Short idea mirrors the above using buy-side liquidity around supply.
    """

    pool_tolerance: float = 0.00015
    sweep_valid_bars: int = 20
    require_two_touch_pool: bool = False
    zone_proximity_atr: float = 0.50


@dataclass(slots=True)
class LiquidityPool:
    price: float
    created_idx: int
    last_touch_idx: int
    touch_count: int = 1


def _require_columns(df: pd.DataFrame) -> None:
    needed = {"high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")


def _validate_config(cfg: LiquidityConfig) -> None:
    if cfg.pool_tolerance < 0:
        raise ValueError("pool_tolerance must be >= 0.")
    if cfg.sweep_valid_bars < 1:
        raise ValueError("sweep_valid_bars must be >= 1.")
    if cfg.zone_proximity_atr < 0:
        raise ValueError("zone_proximity_atr must be >= 0.")


def _first_existing_column(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _coerce_optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_optional_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(round(float(value)))
    except Exception:
        return None


def _zone_created_idx(i: int, age_value) -> int | None:
    age = _coerce_optional_int(age_value)
    if age is None:
        return None
    created_idx = i - age
    if created_idx < 0:
        return None
    return created_idx


def _zone_key(zone_low: float, zone_high: float, created_idx: int) -> tuple[int, float, float]:
    return (
        int(created_idx),
        round(float(zone_low), 10),
        round(float(zone_high), 10),
    )


def _bar_overlaps_zone(
    bar_low: float,
    bar_high: float,
    zone_low: float,
    zone_high: float,
    tolerance: float = 0.0,
) -> bool:
    return (bar_high + tolerance) >= zone_low and (bar_low - tolerance) <= zone_high


def _effective_zone_tolerance(cfg: LiquidityConfig, atr_value: float | None) -> float:
    atr_tol = 0.0
    if atr_value is not None and np.isfinite(atr_value) and atr_value > 0:
        atr_tol = float(atr_value) * cfg.zone_proximity_atr
    return max(float(cfg.pool_tolerance), atr_tol)


def _update_pools(
    pools: list[LiquidityPool],
    *,
    price: float,
    i: int,
    cfg: LiquidityConfig,
) -> list[LiquidityPool]:
    if not pools:
        return [LiquidityPool(price=float(price), created_idx=i, last_touch_idx=i, touch_count=1)]

    best_idx = -1
    best_distance = float("inf")
    for idx, pool in enumerate(pools):
        dist = abs(float(price) - pool.price)
        if dist <= cfg.pool_tolerance and dist < best_distance:
            best_idx = idx
            best_distance = dist

    updated = list(pools)
    if best_idx >= 0:
        pool = updated[best_idx]
        new_touch_count = pool.touch_count + 1
        pool.price = ((pool.price * pool.touch_count) + float(price)) / new_touch_count
        pool.last_touch_idx = i
        pool.touch_count = new_touch_count
        updated[best_idx] = pool
        return updated

    updated.append(LiquidityPool(price=float(price), created_idx=i, last_touch_idx=i, touch_count=1))
    return updated


def _pool_confirmed(pool: LiquidityPool, cfg: LiquidityConfig) -> bool:
    required_touches = 2 if cfg.require_two_touch_pool else 1
    return pool.touch_count >= required_touches


def _pick_relevant_pool(
    pools: list[LiquidityPool],
    *,
    zone_low: float,
    zone_high: float,
    tolerance: float,
    cfg: LiquidityConfig,
) -> Optional[LiquidityPool]:
    candidates = [
        pool
        for pool in pools
        if _pool_confirmed(pool, cfg)
        and (zone_low - tolerance) <= pool.price <= (zone_high + tolerance)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda pool: (pool.last_touch_idx, pool.created_idx, pool.price))


def compute_liquidity_features(
    df: pd.DataFrame,
    config: Optional[LiquidityConfig] = None,
) -> pd.DataFrame:
    _require_columns(df)
    cfg = config or LiquidityConfig()
    _validate_config(cfg)

    out = df.copy()
    n = len(out)

    swing_high_col = _first_existing_column(out, ["ext_swing_high", "int_swing_high"])
    swing_low_col = _first_existing_column(out, ["ext_swing_low", "int_swing_low"])
    atr_col = _first_existing_column(out, ["atr_feature", "atr"])

    if atr_col is not None:
        atr_vals = pd.to_numeric(out[atr_col], errors="coerce").to_numpy(dtype=float)
    else:
        atr_vals = np.full(n, np.nan)

    if swing_high_col is None or swing_low_col is None:
        out["liquidity_long_valid"] = False
        out["liquidity_short_valid"] = False
        out["Liquidity_Valid"] = False
        out["liquidity_pool_high"] = np.nan
        out["liquidity_pool_low"] = np.nan
        return out

    high = out["high"].astype(float).to_numpy()
    low = out["low"].astype(float).to_numpy()
    close = out["close"].astype(float).to_numpy()

    swing_high_series = out[swing_high_col]
    swing_low_series = out[swing_low_col]

    liquidity_pool_high = np.full(n, np.nan)
    liquidity_pool_low = np.full(n, np.nan)

    liquidity_long_valid = np.zeros(n, dtype=bool)
    liquidity_short_valid = np.zeros(n, dtype=bool)

    low_pools: list[LiquidityPool] = []
    high_pools: list[LiquidityPool] = []

    demand_valid_col = _first_existing_column(out, ["sd_demand_valid", "demand_zone_active"])
    demand_zone_low_col = _first_existing_column(out, ["demand_zone_bottom", "demand_zone_low"])
    demand_zone_high_col = _first_existing_column(out, ["demand_zone_top", "demand_zone_high"])
    demand_zone_age_col = _first_existing_column(out, ["demand_zone_age"])

    supply_valid_col = _first_existing_column(out, ["sd_supply_valid", "supply_zone_active"])
    supply_zone_low_col = _first_existing_column(out, ["supply_zone_bottom", "supply_zone_low"])
    supply_zone_high_col = _first_existing_column(out, ["supply_zone_top", "supply_zone_high"])
    supply_zone_age_col = _first_existing_column(out, ["supply_zone_age"])

    current_long_zone_key: tuple[int, float, float] | None = None
    current_short_zone_key: tuple[int, float, float] | None = None
    last_long_sweep_bar = -10_000
    last_short_sweep_bar = -10_000
    last_long_sweep_zone_key: tuple[int, float, float] | None = None
    last_short_sweep_zone_key: tuple[int, float, float] | None = None

    for i in range(n):
        #Updating visible liquidity pools from newly confirmed swings
        sh = swing_high_series.iat[i]
        sl = swing_low_series.iat[i]

        if pd.notna(sh):
            high_pools = _update_pools(high_pools, price=float(sh), i=i, cfg=cfg)
        if pd.notna(sl):
            low_pools = _update_pools(low_pools, price=float(sl), i=i, cfg=cfg)

        tol = _effective_zone_tolerance(cfg, _coerce_optional_float(atr_vals[i]))

        #Checking for sell-side liquidity sweeps into the active demand zone
        demand_valid = bool(out[demand_valid_col].iat[i]) if demand_valid_col is not None and pd.notna(out[demand_valid_col].iat[i]) else False
        demand_zone_low = _coerce_optional_float(out[demand_zone_low_col].iat[i]) if demand_zone_low_col is not None else None
        demand_zone_high = _coerce_optional_float(out[demand_zone_high_col].iat[i]) if demand_zone_high_col is not None else None
        demand_created_idx = _zone_created_idx(i, out[demand_zone_age_col].iat[i]) if demand_zone_age_col is not None else None

        if demand_valid and demand_zone_low is not None and demand_zone_high is not None and demand_created_idx is not None:
            long_zone_key = _zone_key(demand_zone_low, demand_zone_high, demand_created_idx)
        else:
            long_zone_key = None

        if long_zone_key != current_long_zone_key:
            #Resetting long-side sweep state when the demand zone changes
            current_long_zone_key = long_zone_key
            last_long_sweep_bar = -10_000
            last_long_sweep_zone_key = None

        if current_long_zone_key is not None and demand_zone_low is not None and demand_zone_high is not None:
            relevant_low_pool = _pick_relevant_pool(
                low_pools,
                zone_low=demand_zone_low,
                zone_high=demand_zone_high,
                tolerance=tol,
                cfg=cfg,
            )
            if relevant_low_pool is not None:
                liquidity_pool_low[i] = relevant_low_pool.price

                if (
                    low[i] < relevant_low_pool.price
                    and close[i] >= relevant_low_pool.price
                    and _bar_overlaps_zone(low[i], high[i], demand_zone_low, demand_zone_high, tol)
                ):
                    #Removing the pool once its liquidity has been swept
                    last_long_sweep_bar = i
                    last_long_sweep_zone_key = current_long_zone_key
                    low_pools = [pool for pool in low_pools if pool is not relevant_low_pool]

        #Checking for buy-side liquidity sweeps into the active supply zone
        supply_valid = bool(out[supply_valid_col].iat[i]) if supply_valid_col is not None and pd.notna(out[supply_valid_col].iat[i]) else False
        supply_zone_low = _coerce_optional_float(out[supply_zone_low_col].iat[i]) if supply_zone_low_col is not None else None
        supply_zone_high = _coerce_optional_float(out[supply_zone_high_col].iat[i]) if supply_zone_high_col is not None else None
        supply_created_idx = _zone_created_idx(i, out[supply_zone_age_col].iat[i]) if supply_zone_age_col is not None else None

        if supply_valid and supply_zone_low is not None and supply_zone_high is not None and supply_created_idx is not None:
            short_zone_key = _zone_key(supply_zone_low, supply_zone_high, supply_created_idx)
        else:
            short_zone_key = None

        if short_zone_key != current_short_zone_key:
            #Resetting short-side sweep state when the supply zone changes
            current_short_zone_key = short_zone_key
            last_short_sweep_bar = -10_000
            last_short_sweep_zone_key = None

        if current_short_zone_key is not None and supply_zone_low is not None and supply_zone_high is not None:
            relevant_high_pool = _pick_relevant_pool(
                high_pools,
                zone_low=supply_zone_low,
                zone_high=supply_zone_high,
                tolerance=tol,
                cfg=cfg,
            )
            if relevant_high_pool is not None:
                liquidity_pool_high[i] = relevant_high_pool.price

                if (
                    high[i] > relevant_high_pool.price
                    and close[i] <= relevant_high_pool.price
                    and _bar_overlaps_zone(low[i], high[i], supply_zone_low, supply_zone_high, tol)
                ):
                    #Removing the pool once its liquidity has been swept
                    last_short_sweep_bar = i
                    last_short_sweep_zone_key = current_short_zone_key
                    high_pools = [pool for pool in high_pools if pool is not relevant_high_pool]

        #Keeping the sweep valid for a short entry window after it happens
        recent_downside_sweep = (
            current_long_zone_key is not None
            and last_long_sweep_zone_key == current_long_zone_key
            and (i - last_long_sweep_bar) <= cfg.sweep_valid_bars
        )
        recent_upside_sweep = (
            current_short_zone_key is not None
            and last_short_sweep_zone_key == current_short_zone_key
            and (i - last_short_sweep_bar) <= cfg.sweep_valid_bars
        )

        liquidity_long_valid[i] = recent_downside_sweep
        liquidity_short_valid[i] = recent_upside_sweep

    out["liquidity_pool_high"] = liquidity_pool_high
    out["liquidity_pool_low"] = liquidity_pool_low

    out["liquidity_long_valid"] = liquidity_long_valid
    out["liquidity_short_valid"] = liquidity_short_valid
    #Keeping the combined flag for reporting and ML exports
    out["Liquidity_Valid"] = out["liquidity_long_valid"] | out["liquidity_short_valid"]

    return out


def validate_liquidity_output(df: pd.DataFrame) -> None:
    required = {
        "liquidity_long_valid",
        "liquidity_short_valid",
        "Liquidity_Valid",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Liquidity output missing columns: {sorted(missing)}")

    for col in required:
        if df[col].dtype != bool:
            raise ValueError(f"Column '{col}' must be boolean.")
