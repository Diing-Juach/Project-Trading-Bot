from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Dict

import numpy as np
import pandas as pd

from src.mt5.symbol_spec import infer_pip_size


ZoneType = Literal["demand", "supply"]
InvalidationMethod = Literal["close", "wick"]


@dataclass(slots=True)
class SupplyDemandConfig:
    swing_len: int = 3
    base_lookback_left: int = 2
    base_lookforward_right: int = 1
    departure_lookahead: int = 5

    atr_period: int = 14
    displacement_body_atr: float = 0.45
    displacement_range_atr: float = 0.75
    departure_close_frac: float = 0.60
    base_max_range_atr: float = 1.20

    bos_lookback: int = 40

    zone_from_last_opposing_candle: bool = True
    min_zone_width_atr: float = 0.03
    max_zone_width_atr: float = 1.50

    max_zone_width_pips: float = 10.0
    stop_buffer_atr: float = 0.03
    stop_buffer_pips: float = 0.5
    max_stop_pips: float = 10.0

    max_retests: int = 1
    invalidation_method: InvalidationMethod = "close"
    penetration_tolerance_atr: float = 0.05

    active_zone_max_age: int = 60
    min_bars_between_same_type_zones: int = 8
    prefer_nearest_zone: bool = True


@dataclass(slots=True)
class Zone:
    zone_type: ZoneType
    created_bar: int
    pivot_bar: int
    base_start: int
    base_end: int
    top: float
    bottom: float
    entry_price: float
    stop_price: float
    width_pips: float
    stop_pips: float
    departure_bar: int
    retests: int = 0
    broken: bool = False
    touched: bool = False
    active: bool = True
    was_inside_prev_bar: bool = False


def _require_ohlc(df: pd.DataFrame) -> None:
    needed = {"open", "high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    return _true_range(df).rolling(period, min_periods=period).mean()


def _pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    window = left + right + 1
    center_max = high.rolling(window, center=True, min_periods=window).max()
    raw = (high == center_max).fillna(False)
    return raw.shift(right, fill_value=False).astype(bool)


def _pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    window = left + right + 1
    center_min = low.rolling(window, center=True, min_periods=window).min()
    raw = (low == center_min).fillna(False)
    return raw.shift(right, fill_value=False).astype(bool)


def _candle_body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _candle_range(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df["low"]


def _upper_close_frac(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["low"]) / rng


def _lower_close_frac(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["high"] - df["close"]) / rng


def _prepare_last_opposing_indices(opens: np.ndarray, closes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(opens)
    last_bearish = np.full(n, -1, dtype=int)
    last_bullish = np.full(n, -1, dtype=int)
    lbear = -1
    lbull = -1
    for i in range(n):
        if closes[i] < opens[i]:
            lbear = i
        if closes[i] > opens[i]:
            lbull = i
        last_bearish[i] = lbear
        last_bullish[i] = lbull
    return last_bearish, last_bullish


def _prepare_next_true_index(flags: np.ndarray) -> np.ndarray:
    n = len(flags)
    out = np.full(n, -1, dtype=int)
    next_idx = -1
    for i in range(n - 1, -1, -1):
        if flags[i]:
            next_idx = i
        out[i] = next_idx
    return out


def _range_sum(prefix: np.ndarray, start: int, end: int) -> float:
    if start > end:
        return 0.0
    return float(prefix[end + 1] - prefix[start])


def _build_zone_from_base(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    last_bearish_idx: np.ndarray,
    last_bullish_idx: np.ndarray,
    base_start: int,
    base_end: int,
    zone_type: ZoneType,
    atr_value: float,
    pip_size: float,
    cfg: SupplyDemandConfig,
) -> Optional[Dict[str, float]]:
    if base_start < 0 or base_end >= len(highs) or base_start > base_end:
        return None

    idx = None
    if cfg.zone_from_last_opposing_candle:
        if zone_type == "demand":
            idx = int(last_bearish_idx[base_end])
        else:
            idx = int(last_bullish_idx[base_end])
        if idx < base_start:
            idx = None

    if idx is not None:
        o = opens[idx]
        h = highs[idx]
        l = lows[idx]
        c = closes[idx]
        if zone_type == "demand":
            top = max(o, c)
            bottom = l
            entry_price = top
        else:
            top = h
            bottom = min(o, c)
            entry_price = bottom
    else:
        base_high = float(np.max(highs[base_start:base_end + 1]))
        base_low = float(np.min(lows[base_start:base_end + 1]))
        last_open = float(opens[base_end])
        last_close = float(closes[base_end])
        if zone_type == "demand":
            bottom = base_low
            top = max(last_open, last_close)
            entry_price = top
        else:
            top = base_high
            bottom = min(last_open, last_close)
            entry_price = bottom

    width = top - bottom
    if not np.isfinite(width) or width <= 0:
        return None

    width_pips = width / pip_size
    if width_pips > cfg.max_zone_width_pips:
        return None
    if width < atr_value * cfg.min_zone_width_atr:
        return None
    if width > atr_value * cfg.max_zone_width_atr:
        return None

    stop_buffer = max(cfg.stop_buffer_atr * atr_value, cfg.stop_buffer_pips * pip_size)
    if zone_type == "demand":
        stop_price = bottom - stop_buffer
        stop_pips = (entry_price - stop_price) / pip_size
    else:
        stop_price = top + stop_buffer
        stop_pips = (stop_price - entry_price) / pip_size

    if stop_pips > cfg.max_stop_pips:
        return None

    return {
        "top": float(top),
        "bottom": float(bottom),
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "width_pips": float(width_pips),
        "stop_pips": float(stop_pips),
    }


def compute_supply_demand_features(
    df: pd.DataFrame,
    config: Optional[SupplyDemandConfig] = None,
    symbol: Optional[str] = None,
    pip_size: Optional[float] = None,
    atr_column: Optional[str] = None,
) -> pd.DataFrame:
    _require_ohlc(df)
    cfg = config or SupplyDemandConfig()
    pip = pip_size or infer_pip_size(symbol)

    out = df.copy()
    if atr_column is not None and atr_column in out.columns:
        out["atr"] = out[atr_column]
    else:
        out["atr"] = _atr(out, cfg.atr_period)

    out["pivot_high"] = _pivot_high(out["high"], cfg.swing_len, cfg.swing_len)
    out["pivot_low"] = _pivot_low(out["low"], cfg.swing_len, cfg.swing_len)

    #Identifying strong departure candles that can validate new zones
    body = _candle_body(out)
    rng = _candle_range(out)
    upper_close = _upper_close_frac(out)
    lower_close = _lower_close_frac(out)

    bull_departure = (
        (out["close"] > out["open"])
        & (body >= cfg.displacement_body_atr * out["atr"])
        & (rng >= cfg.displacement_range_atr * out["atr"])
        & (upper_close >= cfg.departure_close_frac)
    ).fillna(False)

    bear_departure = (
        (out["close"] < out["open"])
        & (body >= cfg.displacement_body_atr * out["atr"])
        & (rng >= cfg.displacement_range_atr * out["atr"])
        & (lower_close >= cfg.departure_close_frac)
    ).fillna(False)

    n = len(out)
    opens = out["open"].to_numpy(dtype=float)
    highs = out["high"].to_numpy(dtype=float)
    lows = out["low"].to_numpy(dtype=float)
    closes = out["close"].to_numpy(dtype=float)
    atr_vals = out["atr"].to_numpy(dtype=float)
    rng_vals = rng.to_numpy(dtype=float)
    pivot_high = out["pivot_high"].to_numpy(dtype=bool)
    pivot_low = out["pivot_low"].to_numpy(dtype=bool)
    bull_departure_vals = bull_departure.to_numpy(dtype=bool)
    bear_departure_vals = bear_departure.to_numpy(dtype=bool)

    range_prefix = np.zeros(n + 1, dtype=float)
    range_prefix[1:] = np.cumsum(np.nan_to_num(rng_vals, nan=0.0))
    next_bull_departure = _prepare_next_true_index(bull_departure_vals)
    next_bear_departure = _prepare_next_true_index(bear_departure_vals)
    last_bearish_idx, last_bullish_idx = _prepare_last_opposing_indices(opens, closes)

    zones: List[Zone] = []
    last_supply_created = -10_000
    last_demand_created = -10_000

    for confirm_bar in range(n):
            #Building demand zones from confirmed lows plus bullish departure
            if pivot_low[confirm_bar]:
                pivot_bar = confirm_bar - cfg.swing_len
                if pivot_bar >= 0:
                    atr_value = atr_vals[pivot_bar]
                    if np.isfinite(atr_value) and atr_value > 0:
                        base_start = max(0, pivot_bar - cfg.base_lookback_left)
                        base_end = min(n - 1, pivot_bar + cfg.base_lookforward_right)
                        base_count = (base_end - base_start + 1)
                        base_mean_range = _range_sum(range_prefix, base_start, base_end) / float(base_count)

                        if base_mean_range <= (cfg.base_max_range_atr * atr_value):
                            dep_search_start = min(n - 1, base_end + 1)
                            dep_limit = min(n - 1, base_end + cfg.departure_lookahead)
                            dep_idx = next_bull_departure[dep_search_start] if dep_search_start < n else -1

                            if dep_idx != -1 and dep_idx <= dep_limit:
                                created_bar = max(confirm_bar, int(dep_idx))
                                if created_bar - last_demand_created >= cfg.min_bars_between_same_type_zones:
                                    zone_bits = _build_zone_from_base(
                                        opens,
                                        highs,
                                        lows,
                                        closes,
                                        last_bearish_idx,
                                        last_bullish_idx,
                                        base_start,
                                        base_end,
                                        "demand",
                                        atr_value,
                                        pip,
                                        cfg,
                                    )
                                    if zone_bits is not None:
                                        zones.append(
                                            Zone(
                                                zone_type="demand",
                                                created_bar=created_bar,
                                                pivot_bar=pivot_bar,
                                                base_start=base_start,
                                                base_end=base_end,
                                                top=zone_bits["top"],
                                                bottom=zone_bits["bottom"],
                                                entry_price=zone_bits["entry_price"],
                                                stop_price=zone_bits["stop_price"],
                                                width_pips=zone_bits["width_pips"],
                                                stop_pips=zone_bits["stop_pips"],
                                                departure_bar=int(dep_idx),
                                            )
                                        )
                                        last_demand_created = created_bar

            #Building supply zones from confirmed highs plus bearish departure
            if pivot_high[confirm_bar]:
                pivot_bar = confirm_bar - cfg.swing_len
                if pivot_bar >= 0:
                    atr_value = atr_vals[pivot_bar]
                    if np.isfinite(atr_value) and atr_value > 0:
                        base_start = max(0, pivot_bar - cfg.base_lookback_left)
                        base_end = min(n - 1, pivot_bar + cfg.base_lookforward_right)
                        base_count = (base_end - base_start + 1)
                        base_mean_range = _range_sum(range_prefix, base_start, base_end) / float(base_count)

                        if base_mean_range <= (cfg.base_max_range_atr * atr_value):
                            dep_search_start = min(n - 1, base_end + 1)
                            dep_limit = min(n - 1, base_end + cfg.departure_lookahead)
                            dep_idx = next_bear_departure[dep_search_start] if dep_search_start < n else -1

                            if dep_idx != -1 and dep_idx <= dep_limit:
                                created_bar = max(confirm_bar, int(dep_idx))
                                if created_bar - last_supply_created >= cfg.min_bars_between_same_type_zones:
                                    zone_bits = _build_zone_from_base(
                                        opens,
                                        highs,
                                        lows,
                                        closes,
                                        last_bearish_idx,
                                        last_bullish_idx,
                                        base_start,
                                        base_end,
                                        "supply",
                                        atr_value,
                                        pip,
                                        cfg,
                                    )
                                    if zone_bits is not None:
                                        zones.append(
                                            Zone(
                                                zone_type="supply",
                                                created_bar=created_bar,
                                                pivot_bar=pivot_bar,
                                                base_start=base_start,
                                                base_end=base_end,
                                                top=zone_bits["top"],
                                                bottom=zone_bits["bottom"],
                                                entry_price=zone_bits["entry_price"],
                                                stop_price=zone_bits["stop_price"],
                                                width_pips=zone_bits["width_pips"],
                                                stop_pips=zone_bits["stop_pips"],
                                                departure_bar=int(dep_idx),
                                            )
                                        )
                                        last_supply_created = created_bar


    demand_cols = {
        "demand_zone_active": np.zeros(n, dtype=bool),
        "demand_zone_top": np.full(n, np.nan),
        "demand_zone_bottom": np.full(n, np.nan),
        "demand_zone_entry": np.full(n, np.nan),
        "demand_zone_stop": np.full(n, np.nan),
        "demand_zone_width_pips": np.full(n, np.nan),
        "demand_zone_stop_pips": np.full(n, np.nan),
        "demand_zone_retests": np.full(n, np.nan),
        "demand_zone_age": np.full(n, np.nan),
        "demand_touched": np.zeros(n, dtype=bool),
    }

    supply_cols = {
        "supply_zone_active": np.zeros(n, dtype=bool),
        "supply_zone_top": np.full(n, np.nan),
        "supply_zone_bottom": np.full(n, np.nan),
        "supply_zone_entry": np.full(n, np.nan),
        "supply_zone_stop": np.full(n, np.nan),
        "supply_zone_width_pips": np.full(n, np.nan),
        "supply_zone_stop_pips": np.full(n, np.nan),
        "supply_zone_retests": np.full(n, np.nan),
        "supply_zone_age": np.full(n, np.nan),
        "supply_touched": np.zeros(n, dtype=bool),
    }

    active_demand: List[Zone] = []
    active_supply: List[Zone] = []

    #Grouping zones by the bar where they become active in the engine
    zones_by_creation: Dict[int, List[Zone]] = {}
    for z in zones:
        zones_by_creation.setdefault(z.created_bar, []).append(z)

    for i in range(n):
        #Activating any new zones that become available on this bar
        for z in zones_by_creation.get(i, []):
            if z.zone_type == "demand":
                active_demand.append(z)
            else:
                active_supply.append(z)

        high_i = highs[i]
        low_i = lows[i]
        close_i = closes[i]
        atr_i = atr_vals[i]

        #Updating zone touch, retest, age, and invalidation state
        for pool in (active_demand, active_supply):
            for z in pool:
                if not z.active:
                    continue

                age = i - z.created_bar
                if age < 0:
                    continue
                if age > cfg.active_zone_max_age:
                    z.active = False
                    continue
                if not np.isfinite(atr_i) or atr_i <= 0:
                    continue

                inside_now = high_i >= z.bottom and low_i <= z.top

                if i > z.created_bar and inside_now and not z.was_inside_prev_bar:
                    z.touched = True
                    z.retests += 1
                    if z.retests > cfg.max_retests:
                        z.active = False
                        continue

                tol = cfg.penetration_tolerance_atr * atr_i
                if z.zone_type == "demand":
                    break_value = close_i if cfg.invalidation_method == "close" else low_i
                    if break_value < (z.bottom - tol):
                        z.broken = True
                        z.active = False
                        continue
                else:
                    break_value = close_i if cfg.invalidation_method == "close" else high_i
                    if break_value > (z.top + tol):
                        z.broken = True
                        z.active = False
                        continue

                z.was_inside_prev_bar = inside_now

        #Selecting the nearest still-valid demand and supply zone for this bar
        valid_demand = [z for z in active_demand if z.active and not z.broken]
        valid_supply = [z for z in active_supply if z.active and not z.broken]

        if cfg.prefer_nearest_zone and valid_demand:
            valid_demand.sort(key=lambda z: abs(close_i - z.entry_price))
        if cfg.prefer_nearest_zone and valid_supply:
            valid_supply.sort(key=lambda z: abs(close_i - z.entry_price))

        if valid_demand:
            z = valid_demand[0]
            demand_cols["demand_zone_active"][i] = True
            demand_cols["demand_zone_top"][i] = z.top
            demand_cols["demand_zone_bottom"][i] = z.bottom
            demand_cols["demand_zone_entry"][i] = z.entry_price
            demand_cols["demand_zone_stop"][i] = z.stop_price
            demand_cols["demand_zone_width_pips"][i] = z.width_pips
            demand_cols["demand_zone_stop_pips"][i] = z.stop_pips
            demand_cols["demand_zone_retests"][i] = z.retests
            demand_cols["demand_zone_age"][i] = i - z.created_bar
            demand_cols["demand_touched"][i] = i > z.created_bar and (low_i <= z.top and high_i >= z.bottom)

        if valid_supply:
            z = valid_supply[0]
            supply_cols["supply_zone_active"][i] = True
            supply_cols["supply_zone_top"][i] = z.top
            supply_cols["supply_zone_bottom"][i] = z.bottom
            supply_cols["supply_zone_entry"][i] = z.entry_price
            supply_cols["supply_zone_stop"][i] = z.stop_price
            supply_cols["supply_zone_width_pips"][i] = z.width_pips
            supply_cols["supply_zone_stop_pips"][i] = z.stop_pips
            supply_cols["supply_zone_retests"][i] = z.retests
            supply_cols["supply_zone_age"][i] = i - z.created_bar
            supply_cols["supply_touched"][i] = i > z.created_bar and (high_i >= z.bottom and low_i <= z.top)

    for k, v in {**demand_cols, **supply_cols}.items():
        out[k] = v

    out["sd_demand_valid"] = (
        out["demand_zone_active"]
        & (out["demand_zone_stop_pips"] <= cfg.max_stop_pips)
        & (out["demand_zone_retests"].fillna(0) <= cfg.max_retests)
        & (out["demand_zone_age"].fillna(-1) >= 1)
    )

    out["sd_supply_valid"] = (
        out["supply_zone_active"]
        & (out["supply_zone_stop_pips"] <= cfg.max_stop_pips)
        & (out["supply_zone_retests"].fillna(0) <= cfg.max_retests)
        & (out["supply_zone_age"].fillna(-1) >= 1)
    )

    out["sd_demand_touched"] = out["demand_touched"]
    out["sd_supply_touched"] = out["supply_touched"]

    #Keeping basic directional entry flags available for downstream rules
    out["long_entry_signal"] = out["sd_demand_valid"] & out["sd_demand_touched"]
    out["short_entry_signal"] = out["sd_supply_valid"] & out["sd_supply_touched"]

    return out
