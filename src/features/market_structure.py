from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd


BreakMethod = Literal["close"]


@dataclass(slots=True)
class MarketStructureConfig:
    internal_swing_len: int = 3
    external_swing_len: int = 8
    atr_period: int = 14

    bos_lookback: int = 50
    displacement_body_atr: float = 1.0
    require_displacement_for_mss: bool = True

    equal_tolerance_atr: float = 0.10
    break_method: BreakMethod = "close"


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


def _body_size(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _classify_swing_high(
    current_high: float,
    prev_high: Optional[float],
    atr_value: float,
    tol_atr: float,
) -> Optional[str]:
    if prev_high is None or not np.isfinite(prev_high) or not np.isfinite(atr_value):
        return None
    tol = atr_value * tol_atr
    if abs(current_high - prev_high) <= tol:
        return "EQH"
    if current_high > prev_high:
        return "HH"
    return "LH"


def _classify_swing_low(
    current_low: float,
    prev_low: Optional[float],
    atr_value: float,
    tol_atr: float,
) -> Optional[str]:
    if prev_low is None or not np.isfinite(prev_low) or not np.isfinite(atr_value):
        return None
    tol = atr_value * tol_atr
    if abs(current_low - prev_low) <= tol:
        return "EQL"
    if current_low > prev_low:
        return "HL"
    return "LL"


def _infer_trend_from_external_labels(last_high_label: Optional[str], last_low_label: Optional[str]) -> int:
    if last_high_label == "HH" and last_low_label == "HL":
        return 1
    if last_high_label == "LH" and last_low_label == "LL":
        return -1
    return 0


def compute_market_structure_features(
    df: pd.DataFrame,
    config: Optional[MarketStructureConfig] = None,
    atr_column: Optional[str] = None,
) -> pd.DataFrame:
    _require_ohlc(df)
    cfg = config or MarketStructureConfig()

    out = df.copy()
    n = len(out)

    if atr_column is not None and atr_column in out.columns:
        out["atr"] = out[atr_column]
    else:
        out["atr"] = _atr(out, cfg.atr_period)
    out["body"] = _body_size(out)

    out["int_pivot_high"] = _pivot_high(out["high"], cfg.internal_swing_len, cfg.internal_swing_len)
    out["int_pivot_low"] = _pivot_low(out["low"], cfg.internal_swing_len, cfg.internal_swing_len)
    out["ext_pivot_high"] = _pivot_high(out["high"], cfg.external_swing_len, cfg.external_swing_len)
    out["ext_pivot_low"] = _pivot_low(out["low"], cfg.external_swing_len, cfg.external_swing_len)

    cols = {
        "int_swing_high": np.full(n, np.nan),
        "int_swing_low": np.full(n, np.nan),
        "ext_swing_high": np.full(n, np.nan),
        "ext_swing_low": np.full(n, np.nan),
        "int_high_label": np.full(n, None, dtype=object),
        "int_low_label": np.full(n, None, dtype=object),
        "ext_high_label": np.full(n, None, dtype=object),
        "ext_low_label": np.full(n, None, dtype=object),
        "ext_structure_high": np.full(n, np.nan),
        "ext_structure_low": np.full(n, np.nan),
        "bullish_bias": np.zeros(n, dtype=bool),
        "bearish_bias": np.zeros(n, dtype=bool),
        "trend_state": np.zeros(n, dtype=int),
        "bos_up": np.zeros(n, dtype=bool),
        "bos_down": np.zeros(n, dtype=bool),
        "choch_up": np.zeros(n, dtype=bool),
        "choch_down": np.zeros(n, dtype=bool),
        "mss_up": np.zeros(n, dtype=bool),
        "mss_down": np.zeros(n, dtype=bool),
        "sweep_high": np.zeros(n, dtype=bool),
        "sweep_low": np.zeros(n, dtype=bool),
        "last_broken_high": np.full(n, np.nan),
        "last_broken_low": np.full(n, np.nan),
    }

    int_pivot_high = out["int_pivot_high"].to_numpy(dtype=bool)
    int_pivot_low = out["int_pivot_low"].to_numpy(dtype=bool)
    ext_pivot_high = out["ext_pivot_high"].to_numpy(dtype=bool)
    ext_pivot_low = out["ext_pivot_low"].to_numpy(dtype=bool)

    highs = out["high"].to_numpy(dtype=float)
    lows = out["low"].to_numpy(dtype=float)
    closes = out["close"].to_numpy(dtype=float)
    atr_vals = out["atr"].to_numpy(dtype=float)
    body_vals = out["body"].to_numpy(dtype=float)

    last_int_high: Optional[float] = None
    last_int_low: Optional[float] = None

    last_ext_high: Optional[float] = None
    last_ext_low: Optional[float] = None
    last_ext_high_label: Optional[str] = None
    last_ext_low_label: Optional[str] = None
    ext_high_broken = False
    ext_low_broken = False
    trend = 0

    for i in range(n):
        #Updating internal swing labels as pivots become confirmed
        high_i = highs[i]
        low_i = lows[i]
        close_i = closes[i]
        atr_i = atr_vals[i]

        if int_pivot_high[i]:
            pivot_idx = i - cfg.internal_swing_len
            if pivot_idx >= 0:
                price = float(highs[pivot_idx])
                atr_src = atr_vals[pivot_idx]
                cols["int_swing_high"][i] = price
                cols["int_high_label"][i] = _classify_swing_high(price, last_int_high, atr_src, cfg.equal_tolerance_atr)
                last_int_high = price

        if int_pivot_low[i]:
            pivot_idx = i - cfg.internal_swing_len
            if pivot_idx >= 0:
                price = float(lows[pivot_idx])
                atr_src = atr_vals[pivot_idx]
                cols["int_swing_low"][i] = price
                cols["int_low_label"][i] = _classify_swing_low(price, last_int_low, atr_src, cfg.equal_tolerance_atr)
                last_int_low = price

        if ext_pivot_high[i]:
            pivot_idx = i - cfg.external_swing_len
            if pivot_idx >= 0:
                price = float(highs[pivot_idx])
                atr_src = atr_vals[pivot_idx]
                cols["ext_swing_high"][i] = price
                label = _classify_swing_high(price, last_ext_high, atr_src, cfg.equal_tolerance_atr)
                cols["ext_high_label"][i] = label
                last_ext_high = price
                last_ext_high_label = label
                ext_high_broken = False

        if ext_pivot_low[i]:
            pivot_idx = i - cfg.external_swing_len
            if pivot_idx >= 0:
                price = float(lows[pivot_idx])
                atr_src = atr_vals[pivot_idx]
                cols["ext_swing_low"][i] = price
                label = _classify_swing_low(price, last_ext_low, atr_src, cfg.equal_tolerance_atr)
                cols["ext_low_label"][i] = label
                last_ext_low = price
                last_ext_low_label = label
                ext_low_broken = False

        #Projecting the latest external structure into directional bias
        cols["ext_structure_high"][i] = last_ext_high if last_ext_high is not None else np.nan
        cols["ext_structure_low"][i] = last_ext_low if last_ext_low is not None else np.nan

        inferred = _infer_trend_from_external_labels(last_ext_high_label, last_ext_low_label)
        if inferred != 0:
            trend = inferred

        displaced_up = np.isfinite(atr_i) and body_vals[i] >= (cfg.displacement_body_atr * atr_i)
        displaced_down = displaced_up

        #Marking BOS, CHoCH, MSS, and liquidity sweeps from external structure breaks
        if last_ext_high is not None and not ext_high_broken:
            if close_i > last_ext_high:
                cols["last_broken_high"][i] = last_ext_high
                cols["bos_up"][i] = trend >= 0
                cols["choch_up"][i] = trend < 0
                cols["mss_up"][i] = cols["choch_up"][i] and (
                    displaced_up if cfg.require_displacement_for_mss else True
                )
                ext_high_broken = True
                trend = 1
            elif high_i > last_ext_high and close_i <= last_ext_high:
                cols["sweep_high"][i] = True

        if last_ext_low is not None and not ext_low_broken:
            if close_i < last_ext_low:
                cols["last_broken_low"][i] = last_ext_low
                cols["bos_down"][i] = trend <= 0
                cols["choch_down"][i] = trend > 0
                cols["mss_down"][i] = cols["choch_down"][i] and (
                    displaced_down if cfg.require_displacement_for_mss else True
                )
                ext_low_broken = True
                trend = -1
            elif low_i < last_ext_low and close_i >= last_ext_low:
                cols["sweep_low"][i] = True

        cols["trend_state"][i] = trend
        cols["bullish_bias"][i] = trend == 1
        cols["bearish_bias"][i] = trend == -1

    #Attaching the full structure state back onto the feature frame
    for k, v in cols.items():
        out[k] = v

    return out
