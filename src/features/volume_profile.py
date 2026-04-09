from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class VolumeProfileConfig:
    """
    Lightweight session-based volume profile.

    - Uses available bar volume (tick_volume preferred, then volume, then real_volume)
    - Builds one session profile from typical-price buckets
    - Stores only the completed session POC
    - Validates directional support by checking whether the trade location
      (entry or active supply/demand zone) is near the latest completed POC

    This is intentionally lightweight and dissertation-friendly.
    """
    asia_start_hour: int = 0
    asia_end_hour: int = 8

    london_start_hour: int = 8
    london_end_hour: int = 13

    new_york_start_hour: int = 13
    new_york_end_hour: int = 22

    rounding_decimals: int = 5
    poc_tolerance: float = 0.00020


def _require_columns(df: pd.DataFrame) -> None:
    needed = {"time", "high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")


def _validate_config(cfg: VolumeProfileConfig) -> None:
    hours = [
        cfg.asia_start_hour,
        cfg.asia_end_hour,
        cfg.london_start_hour,
        cfg.london_end_hour,
        cfg.new_york_start_hour,
        cfg.new_york_end_hour,
    ]
    if any((h < 0 or h > 24) for h in hours):
        raise ValueError("Session hours must be between 0 and 24.")
    if cfg.rounding_decimals < 0:
        raise ValueError("rounding_decimals must be >= 0.")
    if cfg.poc_tolerance < 0:
        raise ValueError("poc_tolerance must be >= 0.")


def _prepare_time(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["time"], errors="coerce")
    if ts.isna().any():
        raise ValueError("Column 'time' contains non-datetime values.")
    return ts


def _session_name_for_hour(hour: int, cfg: VolumeProfileConfig) -> Optional[str]:
    if cfg.asia_start_hour <= hour < cfg.asia_end_hour:
        return "Asia"
    if cfg.london_start_hour <= hour < cfg.london_end_hour:
        return "London"
    if cfg.new_york_start_hour <= hour < cfg.new_york_end_hour:
        return "NewYork"
    return None


def _resolve_volume_series(df: pd.DataFrame) -> pd.Series:
    for col in ("tick_volume", "volume", "real_volume"):
        if col in df.columns:
            vol = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            return vol.astype(float)
    return pd.Series(np.ones(len(df), dtype=float), index=df.index, name="synthetic_volume")


def _compute_poc_from_profile(profile: Dict[float, float]) -> Optional[float]:
    if not profile:
        return None
    best_price = max(profile.items(), key=lambda x: (x[1], x[0]))[0]
    return float(best_price)


def _first_existing_column(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _price_near_level(price: float, level: float, tolerance: float) -> bool:
    return abs(price - level) <= tolerance


def _zone_overlaps_level(zone_low: float, zone_high: float, level: float, tolerance: float) -> bool:
    return (zone_high + tolerance) >= level and (zone_low - tolerance) <= level


def compute_volume_profile_features(
    df: pd.DataFrame,
    config: Optional[VolumeProfileConfig] = None,
) -> pd.DataFrame:
    _require_columns(df)
    cfg = config or VolumeProfileConfig()
    _validate_config(cfg)

    out = df.copy()
    ts = _prepare_time(out)

    if not ts.is_monotonic_increasing:
        raise ValueError("Column 'time' must be sorted in ascending order.")

    volume = _resolve_volume_series(out)
    typical_price = ((out["high"] + out["low"] + out["close"]) / 3.0).astype(float)

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

    hours = ts.dt.hour.to_numpy()
    dates = ts.dt.floor("D")

    n = len(out)
    session_name = np.empty(n, dtype=object)
    for i in range(n):
        session_name[i] = _session_name_for_hour(int(hours[i]), cfg)

    out["vp_session_name"] = session_name

    reference_poc = np.full(n, np.nan)
    reference_session_name = np.full(n, None, dtype=object)

    long_valid = np.zeros(n, dtype=bool)
    short_valid = np.zeros(n, dtype=bool)

    active_session_name: Optional[str] = None
    active_session_day = None
    active_profile: Dict[float, float] = {}

    last_completed_session_name: Optional[str] = None
    last_completed_poc: Optional[float] = None

    for i in range(n):
        #Rolling the current bar into the active session profile
        cur_session = session_name[i]
        cur_day = dates.iat[i]

        new_session_block = (
            i == 0
            or cur_session != active_session_name
            or cur_day != active_session_day
        )

        if new_session_block:
            #Freezing the previous session profile into a single reference POC
            if active_session_name is not None and active_profile:
                last_completed_session_name = active_session_name
                last_completed_poc = _compute_poc_from_profile(active_profile)

            active_session_name = cur_session
            active_session_day = cur_day
            active_profile = {}

        if cur_session is not None:
            #Accumulating bar volume into rounded typical-price buckets
            px = round(float(typical_price.iat[i]), cfg.rounding_decimals)
            vol = max(0.0, float(volume.iat[i]))
            active_profile[px] = active_profile.get(px, 0.0) + vol

        if last_completed_poc is not None:
            #Validating the current trade location against the latest completed POC
            reference_poc[i] = float(last_completed_poc)
            reference_session_name[i] = last_completed_session_name

            poc = float(last_completed_poc)
            tol = float(cfg.poc_tolerance)

            lv = False
            sv = False

            if long_entry_col is not None:
                entry_val = out[long_entry_col].iat[i]
                if pd.notna(entry_val):
                    lv = _price_near_level(float(entry_val), poc, tol)

            if (
                not lv
                and demand_zone_low_col is not None
                and demand_zone_high_col is not None
            ):
                dz_low = out[demand_zone_low_col].iat[i]
                dz_high = out[demand_zone_high_col].iat[i]
                if pd.notna(dz_low) and pd.notna(dz_high):
                    lv = _zone_overlaps_level(float(dz_low), float(dz_high), poc, tol)

            if short_entry_col is not None:
                entry_val = out[short_entry_col].iat[i]
                if pd.notna(entry_val):
                    sv = _price_near_level(float(entry_val), poc, tol)

            if (
                not sv
                and supply_zone_low_col is not None
                and supply_zone_high_col is not None
            ):
                sz_low = out[supply_zone_low_col].iat[i]
                sz_high = out[supply_zone_high_col].iat[i]
                if pd.notna(sz_low) and pd.notna(sz_high):
                    sv = _zone_overlaps_level(float(sz_low), float(sz_high), poc, tol)

            long_valid[i] = lv
            short_valid[i] = sv

    out["vp_reference_session_name"] = reference_session_name
    out["vp_reference_poc"] = reference_poc

    out["volume_profile_long_valid"] = long_valid
    out["volume_profile_short_valid"] = short_valid
    #Keeping the combined flag for reporting and ML exports
    out["Volume_Profile_Valid"] = out["volume_profile_long_valid"] | out["volume_profile_short_valid"]

    return out


def validate_volume_profile_output(df: pd.DataFrame) -> None:
    required = {
        "volume_profile_long_valid",
        "volume_profile_short_valid",
        "Volume_Profile_Valid",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Volume profile output missing columns: {sorted(missing)}")

    for col in required:
        if df[col].dtype != bool:
            raise ValueError(f"Column '{col}' must be boolean.")
