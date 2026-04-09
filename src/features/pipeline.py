from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import md5
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd

from .atr import ATRMethod, add_atr
from .market_structure import MarketStructureConfig, compute_market_structure_features
from .supply_demand import SupplyDemandConfig, compute_supply_demand_features
from .session_liquidity import (
    SessionLiquidityConfig,
    compute_session_liquidity_features,
    validate_session_liquidity_output,
)
from .volume_profile import (
    VolumeProfileConfig,
    compute_volume_profile_features,
    validate_volume_profile_output,
)
from .imbalance import (
    ImbalanceConfig,
    compute_imbalance_features,
    validate_imbalance_output,
)
from .fibonacci import (
    FibonacciConfig,
    compute_fibonacci_features,
    validate_fibonacci_output,
)
from .liquidity import (
    LiquidityConfig,
    compute_liquidity_features,
    validate_liquidity_output,
)
from .support_resistance import (
    SupportResistanceConfig,
    compute_support_resistance_features,
    validate_support_resistance_output,
)


@dataclass(slots=True)
class FeaturePipelineConfig:
    atr_period: int = 14
    atr_method: ATRMethod = "rma"
    atr_column_name: str = "atr_feature"
    pip_size: Optional[float] = None

    market_structure: Optional[MarketStructureConfig] = None
    supply_demand: Optional[SupplyDemandConfig] = None
    session_liquidity: Optional[SessionLiquidityConfig] = None
    volume_profile: Optional[VolumeProfileConfig] = None
    imbalance: Optional[ImbalanceConfig] = None
    fibonacci: Optional[FibonacciConfig] = None
    liquidity: Optional[LiquidityConfig] = None
    support_resistance: Optional[SupportResistanceConfig] = None

    use_session_liquidity: bool = True
    use_volume_profile: bool = True
    use_imbalance: bool = True
    use_fibonacci: bool = True
    use_liquidity: bool = True
    use_support_resistance: bool = True

    def signature(self) -> str:
        payload = {
            "atr_period": self.atr_period,
            "atr_method": self.atr_method,
            "atr_column_name": self.atr_column_name,
            "pip_size": self.pip_size,
            "use_session_liquidity": self.use_session_liquidity,
            "use_volume_profile": self.use_volume_profile,
            "use_imbalance": self.use_imbalance,
            "use_fibonacci": self.use_fibonacci,
            "use_liquidity": self.use_liquidity,
            "use_support_resistance": self.use_support_resistance,
            "market_structure": _dataclass_to_dict(self.market_structure),
            "supply_demand": _dataclass_to_dict(self.supply_demand),
            "session_liquidity": _dataclass_to_dict(self.session_liquidity),
            "volume_profile": _dataclass_to_dict(self.volume_profile),
            "imbalance": _dataclass_to_dict(self.imbalance),
            "fibonacci": _dataclass_to_dict(self.fibonacci),
            "liquidity": _dataclass_to_dict(self.liquidity),
            "support_resistance": _dataclass_to_dict(self.support_resistance),
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return md5(raw).hexdigest()[:16]


def _dataclass_to_dict(value):
    if value is None:
        return None
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)


def _ensure_false_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = False
        else:
            out[col] = out[col].fillna(False).astype(bool)
    return out


def _ensure_nan_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _add_session_liquidity_defaults(df: pd.DataFrame) -> pd.DataFrame:
    return _ensure_false_columns(
        df,
        ["session_liquidity_long_valid", "session_liquidity_short_valid", "Session_Liquidity_Valid"],
    )


def _add_volume_profile_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_false_columns(
        df,
        ["volume_profile_long_valid", "volume_profile_short_valid", "Volume_Profile_Valid"],
    )
    out = _ensure_nan_columns(out, ["vp_reference_poc", "vp_distance_to_poc"])
    return out


def _add_imbalance_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_false_columns(
        df,
        [
            "bullish_imbalance_created",
            "bearish_imbalance_created",
            "bullish_imbalance_filled",
            "bearish_imbalance_filled",
            "imbalance_long_valid",
            "imbalance_short_valid",
            "Imbalance_Valid",
        ],
    )
    out = _ensure_nan_columns(
        out,
        [
            "bullish_imbalance_zone_low",
            "bullish_imbalance_zone_high",
            "bearish_imbalance_zone_low",
            "bearish_imbalance_zone_high",
            "active_bullish_imbalance_low",
            "active_bullish_imbalance_high",
            "active_bearish_imbalance_low",
            "active_bearish_imbalance_high",
        ],
    )
    return out


def _add_fibonacci_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_false_columns(
        df,
        ["fibonacci_long_valid", "fibonacci_short_valid", "Fibonacci_Valid"],
    )
    out = _ensure_nan_columns(
        out,
        [
            "fib_bull_impulse_low",
            "fib_bull_impulse_high",
            "fib_bear_impulse_low",
            "fib_bear_impulse_high",
            "fib_long_band_low",
            "fib_long_band_high",
            "fib_short_band_low",
            "fib_short_band_high",
        ],
    )
    return out


def _add_liquidity_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_false_columns(
        df,
        [
            "liquidity_long_valid",
            "liquidity_short_valid",
            "Liquidity_Valid",
        ],
    )
    out = _ensure_nan_columns(out, ["liquidity_pool_high", "liquidity_pool_low"])
    return out


def _add_support_resistance_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_false_columns(
        df,
        ["support_resistance_long_valid", "support_resistance_short_valid", "Support_Resistance_Valid"],
    )
    out = _ensure_nan_columns(
        out,
        [
            "sr_nearest_resistance",
            "sr_nearest_support",
            "sr_long_reference_level",
            "sr_short_reference_level",
            "sr_long_target",
            "sr_short_target",
            "sr_long_target_rr",
            "sr_short_target_rr",
        ],
    )
    return out


def build_feature_frame(
    df: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    config: Optional[FeaturePipelineConfig] = None,
    cache_dir: Optional[Path] = None,
    cache_key: Optional[str] = None,
    use_cache: bool = False,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    cfg = config or FeaturePipelineConfig()

    cache_path: Optional[Path] = None
    if cache_dir is not None and cache_key:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}_{cfg.signature()}.pkl"

    if use_cache and cache_path is not None and cache_path.exists() and not force_rebuild:
        return pd.read_pickle(cache_path)

    out = add_atr(
        df,
        period=cfg.atr_period,
        method=cfg.atr_method,
        column_name=cfg.atr_column_name,
    )

    out = compute_market_structure_features(
        out,
        config=cfg.market_structure,
        atr_column=cfg.atr_column_name,
    )

    out = compute_supply_demand_features(
        out,
        config=cfg.supply_demand,
        symbol=symbol,
        pip_size=cfg.pip_size,
        atr_column=cfg.atr_column_name,
    )

    if cfg.use_session_liquidity:
        out = compute_session_liquidity_features(out, config=cfg.session_liquidity)
        validate_session_liquidity_output(out)
    else:
        out = _add_session_liquidity_defaults(out)

    if cfg.use_volume_profile:
        out = compute_volume_profile_features(out, config=cfg.volume_profile)
        validate_volume_profile_output(out)
    else:
        out = _add_volume_profile_defaults(out)

    if cfg.use_imbalance:
        out = compute_imbalance_features(out, config=cfg.imbalance)
        validate_imbalance_output(out)
    else:
        out = _add_imbalance_defaults(out)

    if cfg.use_fibonacci:
        out = compute_fibonacci_features(out, config=cfg.fibonacci)
        validate_fibonacci_output(out)
    else:
        out = _add_fibonacci_defaults(out)

    if cfg.use_liquidity:
        out = compute_liquidity_features(out, config=cfg.liquidity)
        validate_liquidity_output(out)
    else:
        out = _add_liquidity_defaults(out)

    if cfg.use_support_resistance:
        out = compute_support_resistance_features(
            out,
            config=cfg.support_resistance,
            atr_column=cfg.atr_column_name,
        )
        validate_support_resistance_output(out)
    else:
        out = _add_support_resistance_defaults(out)

    if cache_path is not None:
        out.to_pickle(cache_path)

    return out
