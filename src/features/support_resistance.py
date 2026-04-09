from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SupportResistanceConfig:
    pivot_lookback: int = 3
    cluster_tolerance_atr: float = 0.35
    recency_bars: int = 250
    min_touches: int = 1

    min_runner_rr: float = 2.3
    max_runner_rr: float = 4.0
    target_buffer_atr: float = 0.05


def _require_ohlc(df: pd.DataFrame) -> None:
    needed = {"high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns for support/resistance: {sorted(missing)}")


def _pick_atr_column(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for col in ("atr_feature", "atr"):
        if col in df.columns:
            return col
    for col in df.columns:
        if "atr" in col.lower():
            return col
    return None


def _pivot_high_confirmed(high: pd.Series, left: int, right: int) -> pd.Series:
    window = left + right + 1
    center_max = high.rolling(window, center=True, min_periods=window).max()
    raw = (high == center_max).fillna(False)
    return raw.shift(right, fill_value=False).astype(bool)


def _pivot_low_confirmed(low: pd.Series, left: int, right: int) -> pd.Series:
    window = left + right + 1
    center_min = low.rolling(window, center=True, min_periods=window).min()
    raw = (low == center_min).fillna(False)
    return raw.shift(right, fill_value=False).astype(bool)


def _cluster_levels(levels: list[float], tolerance: float) -> list[tuple[float, int]]:
    if not levels:
        return []

    lvls = sorted(float(x) for x in levels if np.isfinite(x))
    if not lvls:
        return []

    clusters: list[list[float]] = [[lvls[0]]]
    for x in lvls[1:]:
        cluster_mean = float(np.mean(clusters[-1]))
        if abs(x - cluster_mean) <= tolerance:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    return [(float(np.mean(c)), len(c)) for c in clusters]


def compute_support_resistance_features(
    df: pd.DataFrame,
    config: Optional[SupportResistanceConfig] = None,
    atr_column: Optional[str] = None,
) -> pd.DataFrame:
    _require_ohlc(df)
    cfg = config or SupportResistanceConfig()

    out = df.copy()
    n = len(out)

    atr_col = _pick_atr_column(out, atr_column)
    if atr_col is None:
        raise ValueError("Support/resistance requires an ATR column, but none was found.")

    left = max(1, int(cfg.pivot_lookback))
    right = left

    out["sr_pivot_high"] = _pivot_high_confirmed(out["high"], left, right)
    out["sr_pivot_low"] = _pivot_low_confirmed(out["low"], left, right)

    sr_nearest_resistance = np.full(n, np.nan, dtype=float)
    sr_nearest_support = np.full(n, np.nan, dtype=float)
    sr_long_reference_level = np.full(n, np.nan, dtype=float)
    sr_short_reference_level = np.full(n, np.nan, dtype=float)
    sr_long_target = np.full(n, np.nan, dtype=float)
    sr_short_target = np.full(n, np.nan, dtype=float)
    sr_long_target_rr = np.full(n, np.nan, dtype=float)
    sr_short_target_rr = np.full(n, np.nan, dtype=float)
    sr_long_valid = np.zeros(n, dtype=bool)
    sr_short_valid = np.zeros(n, dtype=bool)

    pivot_high_flags = out["sr_pivot_high"].to_numpy(dtype=bool)
    pivot_low_flags = out["sr_pivot_low"].to_numpy(dtype=bool)
    pivot_high_prices = out["high"].to_numpy(dtype=float)
    pivot_low_prices = out["low"].to_numpy(dtype=float)
    atr_vals = out[atr_col].to_numpy(dtype=float)
    close_vals = out["close"].to_numpy(dtype=float)

    demand_entry_vals = out["demand_zone_entry"].to_numpy(dtype=float) if "demand_zone_entry" in out.columns else None
    demand_stop_vals = out["demand_zone_stop"].to_numpy(dtype=float) if "demand_zone_stop" in out.columns else None
    supply_entry_vals = out["supply_zone_entry"].to_numpy(dtype=float) if "supply_zone_entry" in out.columns else None
    supply_stop_vals = out["supply_zone_stop"].to_numpy(dtype=float) if "supply_zone_stop" in out.columns else None

    recent_highs: deque[tuple[int, float]] = deque()
    recent_lows: deque[tuple[int, float]] = deque()

    for i in range(n):
        #Maintaining recent pivot highs and lows inside the recency window
        if pivot_high_flags[i] and np.isfinite(pivot_high_prices[i]):
            recent_highs.append((i, float(pivot_high_prices[i])))
        if pivot_low_flags[i] and np.isfinite(pivot_low_prices[i]):
            recent_lows.append((i, float(pivot_low_prices[i])))

        oldest_allowed = i - cfg.recency_bars
        while recent_highs and recent_highs[0][0] < oldest_allowed:
            recent_highs.popleft()
        while recent_lows and recent_lows[0][0] < oldest_allowed:
            recent_lows.popleft()

        atr_i = atr_vals[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        #Clustering nearby pivots into support and resistance levels
        tolerance = max(atr_i * cfg.cluster_tolerance_atr, 1e-8)
        high_clusters = _cluster_levels([price for _, price in recent_highs], tolerance)
        low_clusters = _cluster_levels([price for _, price in recent_lows], tolerance)

        high_levels = [lvl for lvl, touches in high_clusters if touches >= cfg.min_touches]
        low_levels = [lvl for lvl, touches in low_clusters if touches >= cfg.min_touches]

        close_i = close_vals[i]
        higher_res = [lvl for lvl in high_levels if lvl > close_i]
        lower_sup = [lvl for lvl in low_levels if lvl < close_i]

        if higher_res:
            sr_nearest_resistance[i] = min(higher_res)
        if lower_sup:
            sr_nearest_support[i] = max(lower_sup)

        #Checking whether a long runner has enough room to the next resistance
        if demand_entry_vals is not None and demand_stop_vals is not None:
            entry = demand_entry_vals[i]
            stop = demand_stop_vals[i]
            if np.isfinite(entry) and np.isfinite(stop):
                risk = float(entry - stop)
                if risk > 0:
                    two_r_price = float(entry) + (2.0 * risk)
                    valid_res_levels = [lvl for lvl in high_levels if lvl > two_r_price]
                    if valid_res_levels:
                        raw_res = min(valid_res_levels)
                        final_tp = raw_res - (atr_i * cfg.target_buffer_atr)
                        rr = (final_tp - float(entry)) / risk
                        sr_long_reference_level[i] = raw_res
                        sr_long_target[i] = final_tp
                        sr_long_target_rr[i] = rr
                        if cfg.min_runner_rr <= rr <= cfg.max_runner_rr:
                            sr_long_valid[i] = True

        #Checking whether a short runner has enough room to the next support
        if supply_entry_vals is not None and supply_stop_vals is not None:
            entry = supply_entry_vals[i]
            stop = supply_stop_vals[i]
            if np.isfinite(entry) and np.isfinite(stop):
                risk = float(stop - entry)
                if risk > 0:
                    two_r_price = float(entry) - (2.0 * risk)
                    valid_sup_levels = [lvl for lvl in low_levels if lvl < two_r_price]
                    if valid_sup_levels:
                        raw_sup = max(valid_sup_levels)
                        final_tp = raw_sup + (atr_i * cfg.target_buffer_atr)
                        rr = (float(entry) - final_tp) / risk
                        sr_short_reference_level[i] = raw_sup
                        sr_short_target[i] = final_tp
                        sr_short_target_rr[i] = rr
                        if cfg.min_runner_rr <= rr <= cfg.max_runner_rr:
                            sr_short_valid[i] = True

    out["sr_nearest_resistance"] = sr_nearest_resistance
    out["sr_nearest_support"] = sr_nearest_support
    out["sr_long_reference_level"] = sr_long_reference_level
    out["sr_short_reference_level"] = sr_short_reference_level
    out["sr_long_target"] = sr_long_target
    out["sr_short_target"] = sr_short_target
    out["sr_long_target_rr"] = sr_long_target_rr
    out["sr_short_target_rr"] = sr_short_target_rr
    out["support_resistance_long_valid"] = sr_long_valid
    out["support_resistance_short_valid"] = sr_short_valid
    #Keeping the combined flag for reporting and ML exports
    out["Support_Resistance_Valid"] = sr_long_valid | sr_short_valid

    return out


def validate_support_resistance_output(df: pd.DataFrame) -> None:
    required = {
        "sr_nearest_resistance",
        "sr_nearest_support",
        "sr_long_reference_level",
        "sr_short_reference_level",
        "sr_long_target",
        "sr_short_target",
        "sr_long_target_rr",
        "sr_short_target_rr",
        "support_resistance_long_valid",
        "support_resistance_short_valid",
        "Support_Resistance_Valid",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Support/resistance output missing required columns: {sorted(missing)}")
