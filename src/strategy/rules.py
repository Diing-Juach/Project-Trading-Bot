from __future__ import annotations

import pandas as pd


def _bool_col(df: pd.DataFrame, *names: str) -> pd.Series:
    """
    Return the first existing column from `names` as a bool Series.
    If none exist, return all-False.
    """
    for name in names:
        if name in df.columns:
            return df[name].fillna(False).astype(bool)
    return pd.Series(False, index=df.index, dtype=bool)


def generate_signals(
    df: pd.DataFrame,
    structure_lookback: int = 20,
) -> pd.DataFrame:
    df = df.copy()

    #Resolving structure and break columns across old and new schemas
    structure_bullish = _bool_col(df, "structure_bullish", "bullish_bias")
    structure_bearish = _bool_col(df, "structure_bearish", "bearish_bias")

    choch_up = _bool_col(df, "choch_up", "bullish_choch")
    choch_down = _bool_col(df, "choch_down", "bearish_choch")
    bos_up = _bool_col(df, "bos_up", "bullish_bos")
    bos_down = _bool_col(df, "bos_down", "bearish_bos")
    mss_up = _bool_col(df, "mss_up", "bullish_mss")
    mss_down = _bool_col(df, "mss_down", "bearish_mss")

    #Using explicit recent-break flags when available, otherwise deriving them
    if "recent_bull_break" in df.columns:
        recent_bull_break = df["recent_bull_break"].fillna(False).astype(bool)
    else:
        recent_bull_break = (
            (choch_up | bos_up | mss_up)
            .rolling(structure_lookback, min_periods=1)
            .max()
            .astype(bool)
        )

    if "recent_bear_break" in df.columns:
        recent_bear_break = df["recent_bear_break"].fillna(False).astype(bool)
    else:
        recent_bear_break = (
            (choch_down | bos_down | mss_down)
            .rolling(structure_lookback, min_periods=1)
            .max()
            .astype(bool)
        )

    sd_demand_valid = _bool_col(df, "sd_demand_valid")
    sd_supply_valid = _bool_col(df, "sd_supply_valid")
    sd_demand_touched = _bool_col(df, "sd_demand_touched", "demand_touched")
    sd_supply_touched = _bool_col(df, "sd_supply_touched", "supply_touched")

    demand_zone_entry = df["demand_zone_entry"] if "demand_zone_entry" in df.columns else pd.Series(float("nan"), index=df.index)
    supply_zone_entry = df["supply_zone_entry"] if "supply_zone_entry" in df.columns else pd.Series(float("nan"), index=df.index)

    #Requiring the signal bar to reject back out of the active zone
    long_rejection = (
        sd_demand_touched
        & (df["close"] >= demand_zone_entry)
        & (df["close"] > df["open"])
    )

    short_rejection = (
        sd_supply_touched
        & (df["close"] <= supply_zone_entry)
        & (df["close"] < df["open"])
    )

    #Keeping directional validity flags available for reporting and ML
    df["atr_long_valid"] = df["atr_feature"].notna() if "atr_feature" in df.columns else True
    df["atr_short_valid"] = df["atr_feature"].notna() if "atr_feature" in df.columns else True

    df["supply_demand_long_valid"] = sd_demand_valid
    df["supply_demand_short_valid"] = sd_supply_valid

    df["market_structure_long_valid"] = structure_bullish & recent_bull_break
    df["market_structure_short_valid"] = structure_bearish & recent_bear_break

    #Combining zone validity, structure bias, recent breaks, and rejection into entries
    df["long_signal"] = (
        sd_demand_valid
        & structure_bullish
        & recent_bull_break
        & long_rejection
    )

    df["short_signal"] = (
        sd_supply_valid
        & structure_bearish
        & recent_bear_break
        & short_rejection
    )

    return df
