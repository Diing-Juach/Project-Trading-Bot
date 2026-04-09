from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Literal


ATRMethod = Literal["sma", "ema", "rma"]


def _require_ohlc(df: pd.DataFrame) -> None:
    needed = {"high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def true_range(df: pd.DataFrame) -> pd.Series:
    #Comparing the current bar range against the prior close gap
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(
    df: pd.DataFrame,
    period: int = 14,
    method: ATRMethod = "rma",
) -> pd.Series:
    _require_ohlc(df)

    #Building ATR from the standard true range series
    tr = true_range(df)

    if method == "sma":
        return tr.rolling(period, min_periods=period).mean()

    if method == "ema":
        return tr.ewm(span=period, adjust=False).mean()

    if method == "rma":
        #Using Wilder's smoothing for the trading-oriented default
        alpha = 1.0 / period
        return tr.ewm(alpha=alpha, adjust=False).mean()

    raise ValueError(f"Unknown ATR method: {method}")


def add_atr(
    df: pd.DataFrame,
    period: int = 14,
    method: ATRMethod = "rma",
    column_name: str = "atr",
) -> pd.DataFrame:
    df = df.copy()
    df[column_name] = atr(df, period=period, method=method)
    return df
