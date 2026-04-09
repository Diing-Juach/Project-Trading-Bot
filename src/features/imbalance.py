from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ImbalanceConfig:
    """
    Fair value gap / imbalance confluence anchored to a valid supply/demand zone.

    Intended logic:
    - a valid demand/supply zone already exists
    - price departs from that zone
    - one or more FVGs are created during that departure leg
    - on the retrace back toward the zone, price trades back into one of those
      departure imbalances
    - from that point until the zone changes, the imbalance confluence is valid

    Bullish FVG:
        high[i-2] < low[i]

    Bearish FVG:
        low[i-2] > high[i]
    """

    min_gap_size: float = 0.0
    fill_tolerance: float = 0.0
    max_age_bars: int = 120
    keep_only_most_recent: bool = False
    zone_overlap_tolerance: float = 0.0
    relevance_tolerance_atr: float = 0.10


@dataclass(slots=True)
class DepartureImbalance:
    low: float
    high: float
    created_idx: int


def _require_columns(df: pd.DataFrame) -> None:
    needed = {"high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")


def _validate_config(cfg: ImbalanceConfig) -> None:
    if cfg.min_gap_size < 0:
        raise ValueError("min_gap_size must be >= 0.")
    if cfg.fill_tolerance < 0:
        raise ValueError("fill_tolerance must be >= 0.")
    if cfg.max_age_bars < 1:
        raise ValueError("max_age_bars must be >= 1.")
    if cfg.zone_overlap_tolerance < 0:
        raise ValueError("zone_overlap_tolerance must be >= 0.")
    if cfg.relevance_tolerance_atr < 0:
        raise ValueError("relevance_tolerance_atr must be >= 0.")


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


def _bar_overlaps_zone(
    bar_low: float,
    bar_high: float,
    zone_low: float,
    zone_high: float,
    tolerance: float = 0.0,
) -> bool:
    return (bar_high + tolerance) >= zone_low and (bar_low - tolerance) <= zone_high


def _effective_tolerance(cfg: ImbalanceConfig, atr_value: float | None) -> float:
    atr_tol = 0.0
    if atr_value is not None and np.isfinite(atr_value) and atr_value > 0:
        atr_tol = float(atr_value) * cfg.relevance_tolerance_atr
    return max(float(cfg.zone_overlap_tolerance), atr_tol)


def _zone_key(zone_low: float, zone_high: float, created_idx: int) -> tuple[int, float, float]:
    return (
        int(created_idx),
        round(float(zone_low), 10),
        round(float(zone_high), 10),
    )


def _zone_created_idx(i: int, age_value) -> int | None:
    age = _coerce_optional_int(age_value)
    if age is None:
        return None
    created_idx = i - age
    if created_idx < 0:
        return None
    return created_idx


def _select_display_gap(
    tracked_gaps: list[DepartureImbalance],
    filled_gap: DepartureImbalance | None,
) -> DepartureImbalance | None:
    if filled_gap is not None:
        return filled_gap
    if not tracked_gaps:
        return None
    return tracked_gaps[-1]


def _trim_gaps_for_age(
    gaps: list[DepartureImbalance],
    *,
    i: int,
    cfg: ImbalanceConfig,
) -> list[DepartureImbalance]:
    return [gap for gap in gaps if (i - gap.created_idx) <= cfg.max_age_bars]


def compute_imbalance_features(
    df: pd.DataFrame,
    config: Optional[ImbalanceConfig] = None,
) -> pd.DataFrame:
    _require_columns(df)
    cfg = config or ImbalanceConfig()
    _validate_config(cfg)

    out = df.copy()
    n = len(out)

    high = out["high"].astype(float).to_numpy()
    low = out["low"].astype(float).to_numpy()

    atr_col = _first_existing_column(out, ["atr_feature", "atr"])
    if atr_col is not None:
        atr_vals = pd.to_numeric(out[atr_col], errors="coerce").to_numpy(dtype=float)
    else:
        atr_vals = np.full(n, np.nan)

    bullish_created = np.zeros(n, dtype=bool)
    bearish_created = np.zeros(n, dtype=bool)
    bullish_filled = np.zeros(n, dtype=bool)
    bearish_filled = np.zeros(n, dtype=bool)

    bullish_zone_low = np.full(n, np.nan)
    bullish_zone_high = np.full(n, np.nan)
    bearish_zone_low = np.full(n, np.nan)
    bearish_zone_high = np.full(n, np.nan)

    active_bullish_zone_low = np.full(n, np.nan)
    active_bullish_zone_high = np.full(n, np.nan)
    active_bearish_zone_low = np.full(n, np.nan)
    active_bearish_zone_high = np.full(n, np.nan)

    imbalance_long_valid = np.zeros(n, dtype=bool)
    imbalance_short_valid = np.zeros(n, dtype=bool)

    demand_valid_col = _first_existing_column(out, ["sd_demand_valid", "demand_zone_active"])
    demand_touched_col = _first_existing_column(out, ["sd_demand_touched", "demand_touched"])
    demand_zone_low_col = _first_existing_column(out, ["demand_zone_bottom", "demand_zone_low"])
    demand_zone_high_col = _first_existing_column(out, ["demand_zone_top", "demand_zone_high"])
    demand_zone_entry_col = _first_existing_column(out, ["demand_zone_entry", "long_entry_price", "long_entry"])
    demand_zone_age_col = _first_existing_column(out, ["demand_zone_age"])

    supply_valid_col = _first_existing_column(out, ["sd_supply_valid", "supply_zone_active"])
    supply_touched_col = _first_existing_column(out, ["sd_supply_touched", "supply_touched"])
    supply_zone_low_col = _first_existing_column(out, ["supply_zone_bottom", "supply_zone_low"])
    supply_zone_high_col = _first_existing_column(out, ["supply_zone_top", "supply_zone_high"])
    supply_zone_entry_col = _first_existing_column(out, ["supply_zone_entry", "short_entry_price", "short_entry"])
    supply_zone_age_col = _first_existing_column(out, ["supply_zone_age"])

    current_long_zone_key: tuple[int, float, float] | None = None
    current_short_zone_key: tuple[int, float, float] | None = None

    tracked_long_gaps: list[DepartureImbalance] = []
    tracked_short_gaps: list[DepartureImbalance] = []

    long_filled_gap: DepartureImbalance | None = None
    short_filled_gap: DepartureImbalance | None = None

    long_fill_seen = False
    short_fill_seen = False

    for i in range(n):
        #Detecting fresh bullish and bearish three-candle gaps
        bull_gap_low = np.nan
        bull_gap_high = np.nan
        bear_gap_low = np.nan
        bear_gap_high = np.nan

        if i >= 2:
            bull_gap_low = float(high[i - 2])
            bull_gap_high = float(low[i])
            bull_gap_size = bull_gap_high - bull_gap_low
            if bull_gap_size > cfg.min_gap_size:
                bullish_created[i] = True
                bullish_zone_low[i] = bull_gap_low
                bullish_zone_high[i] = bull_gap_high

            bear_gap_low = float(high[i])
            bear_gap_high = float(low[i - 2])
            bear_gap_size = bear_gap_high - bear_gap_low
            if bear_gap_size > cfg.min_gap_size:
                bearish_created[i] = True
                bearish_zone_low[i] = bear_gap_low
                bearish_zone_high[i] = bear_gap_high

        tol = _effective_tolerance(cfg, _coerce_optional_float(atr_vals[i]))

        #Binding bullish departure gaps to the currently active demand zone
        demand_valid = bool(out[demand_valid_col].iat[i]) if demand_valid_col is not None and pd.notna(out[demand_valid_col].iat[i]) else False
        demand_touched = bool(out[demand_touched_col].iat[i]) if demand_touched_col is not None and pd.notna(out[demand_touched_col].iat[i]) else False
        demand_zone_low = _coerce_optional_float(out[demand_zone_low_col].iat[i]) if demand_zone_low_col is not None else None
        demand_zone_high = _coerce_optional_float(out[demand_zone_high_col].iat[i]) if demand_zone_high_col is not None else None
        demand_zone_entry = _coerce_optional_float(out[demand_zone_entry_col].iat[i]) if demand_zone_entry_col is not None else None
        demand_created_idx = _zone_created_idx(i, out[demand_zone_age_col].iat[i]) if demand_zone_age_col is not None else None

        if demand_valid and demand_zone_low is not None and demand_zone_high is not None and demand_created_idx is not None:
            long_zone_key = _zone_key(demand_zone_low, demand_zone_high, demand_created_idx)
        else:
            long_zone_key = None

        if long_zone_key != current_long_zone_key:
            #Resetting tracked bullish gaps when the demand zone changes
            current_long_zone_key = long_zone_key
            tracked_long_gaps = []
            long_filled_gap = None
            long_fill_seen = False

        if current_long_zone_key is not None:
            tracked_long_gaps = _trim_gaps_for_age(tracked_long_gaps, i=i, cfg=cfg)

            if bullish_created[i] and not demand_touched and not long_fill_seen:
                #Recording bullish gaps created during the move away from demand
                if (
                    demand_created_idx is not None
                    and i >= demand_created_idx
                    and demand_zone_entry is not None
                    and bull_gap_high >= (demand_zone_entry - tol)
                ):
                    gap = DepartureImbalance(low=float(bull_gap_low), high=float(bull_gap_high), created_idx=i)
                    tracked_long_gaps = [gap] if cfg.keep_only_most_recent else [*tracked_long_gaps, gap]

            if not long_fill_seen:
                #Marking the first retrace back into a tracked bullish gap
                for gap in reversed(tracked_long_gaps):
                    if i <= gap.created_idx:
                        continue
                    if _bar_overlaps_zone(float(low[i]), float(high[i]), gap.low, gap.high, cfg.fill_tolerance):
                        long_fill_seen = True
                        long_filled_gap = gap
                        bullish_filled[i] = True
                        break

            display_gap = _select_display_gap(tracked_long_gaps, long_filled_gap)
            if display_gap is not None:
                active_bullish_zone_low[i] = display_gap.low
                active_bullish_zone_high[i] = display_gap.high

            imbalance_long_valid[i] = long_fill_seen

        #Binding bearish departure gaps to the currently active supply zone
        supply_valid = bool(out[supply_valid_col].iat[i]) if supply_valid_col is not None and pd.notna(out[supply_valid_col].iat[i]) else False
        supply_touched = bool(out[supply_touched_col].iat[i]) if supply_touched_col is not None and pd.notna(out[supply_touched_col].iat[i]) else False
        supply_zone_low = _coerce_optional_float(out[supply_zone_low_col].iat[i]) if supply_zone_low_col is not None else None
        supply_zone_high = _coerce_optional_float(out[supply_zone_high_col].iat[i]) if supply_zone_high_col is not None else None
        supply_zone_entry = _coerce_optional_float(out[supply_zone_entry_col].iat[i]) if supply_zone_entry_col is not None else None
        supply_created_idx = _zone_created_idx(i, out[supply_zone_age_col].iat[i]) if supply_zone_age_col is not None else None

        if supply_valid and supply_zone_low is not None and supply_zone_high is not None and supply_created_idx is not None:
            short_zone_key = _zone_key(supply_zone_low, supply_zone_high, supply_created_idx)
        else:
            short_zone_key = None

        if short_zone_key != current_short_zone_key:
            #Resetting tracked bearish gaps when the supply zone changes
            current_short_zone_key = short_zone_key
            tracked_short_gaps = []
            short_filled_gap = None
            short_fill_seen = False

        if current_short_zone_key is not None:
            tracked_short_gaps = _trim_gaps_for_age(tracked_short_gaps, i=i, cfg=cfg)

            if bearish_created[i] and not supply_touched and not short_fill_seen:
                #Recording bearish gaps created during the move away from supply
                if (
                    supply_created_idx is not None
                    and i >= supply_created_idx
                    and supply_zone_entry is not None
                    and bear_gap_low <= (supply_zone_entry + tol)
                ):
                    gap = DepartureImbalance(low=float(bear_gap_low), high=float(bear_gap_high), created_idx=i)
                    tracked_short_gaps = [gap] if cfg.keep_only_most_recent else [*tracked_short_gaps, gap]

            if not short_fill_seen:
                #Marking the first retrace back into a tracked bearish gap
                for gap in reversed(tracked_short_gaps):
                    if i <= gap.created_idx:
                        continue
                    if _bar_overlaps_zone(float(low[i]), float(high[i]), gap.low, gap.high, cfg.fill_tolerance):
                        short_fill_seen = True
                        short_filled_gap = gap
                        bearish_filled[i] = True
                        break

            display_gap = _select_display_gap(tracked_short_gaps, short_filled_gap)
            if display_gap is not None:
                active_bearish_zone_low[i] = display_gap.low
                active_bearish_zone_high[i] = display_gap.high

            imbalance_short_valid[i] = short_fill_seen

    out["bullish_imbalance_created"] = bullish_created
    out["bearish_imbalance_created"] = bearish_created
    out["bullish_imbalance_filled"] = bullish_filled
    out["bearish_imbalance_filled"] = bearish_filled

    out["bullish_imbalance_zone_low"] = bullish_zone_low
    out["bullish_imbalance_zone_high"] = bullish_zone_high
    out["bearish_imbalance_zone_low"] = bearish_zone_low
    out["bearish_imbalance_zone_high"] = bearish_zone_high

    out["active_bullish_imbalance_low"] = active_bullish_zone_low
    out["active_bullish_imbalance_high"] = active_bullish_zone_high
    out["active_bearish_imbalance_low"] = active_bearish_zone_low
    out["active_bearish_imbalance_high"] = active_bearish_zone_high

    out["imbalance_long_valid"] = imbalance_long_valid
    out["imbalance_short_valid"] = imbalance_short_valid
    #Keeping the combined validity flag for reporting and ML exports
    out["Imbalance_Valid"] = out["imbalance_long_valid"] | out["imbalance_short_valid"]

    return out


def validate_imbalance_output(df: pd.DataFrame) -> None:
    required = {
        "imbalance_long_valid",
        "imbalance_short_valid",
        "Imbalance_Valid",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Imbalance output missing columns: {sorted(missing)}")

    for col in required:
        if df[col].dtype != bool:
            raise ValueError(f"Column '{col}' must be boolean.")
