from __future__ import annotations

from dataclasses import asdict, dataclass
import logging

from .connector import MT5Connector
from .config import PIP_SIZE_OVERRIDES


log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    requested_symbol: str | None
    broker_symbol: str | None
    pip_size: float
    digits: int | None = None
    point: float | None = None
    tick_size: float | None = None
    tick_value: float | None = None
    contract_size: float | None = None
    volume_min: float | None = None
    volume_max: float | None = None
    volume_step: float | None = None
    spread_points: float | None = None
    currency_base: str | None = None
    currency_profit: str | None = None
    source: str = "fallback"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _safe_float(value) -> float | None:
    #Returning a positive float or None for invalid values
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if value > 0 else None


def _safe_int(value) -> int | None:
    #Returning an int or None for invalid values
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def fallback_pip_size(symbol: str | None) -> float:
    #Using heuristic pip sizes when broker metadata is unavailable
    if symbol:
        s = symbol.upper()
        if "JPY" in s:
            return 0.01
        if s in {"XAUUSD", "GOLD"} or "XAU" in s:
            return 0.10
    return 0.0001


def _override_pip_size(symbol: str | None) -> float | None:
    #Checking explicit pip-size overrides before broker inference
    if not symbol:
        return None

    s = symbol.upper()
    if s in PIP_SIZE_OVERRIDES:
        return float(PIP_SIZE_OVERRIDES[s])

    for key, value in PIP_SIZE_OVERRIDES.items():
        key_u = str(key).upper()
        if s.startswith(key_u):
            return float(value)

    return None


def infer_pip_size(
    symbol: str | None,
    *,
    digits: int | None = None,
    point: float | None = None,
    tick_size: float | None = None,
) -> float:
    #Deriving pip size from broker point/digits when available
    point = _safe_float(point)
    tick_size = _safe_float(tick_size)
    digits = _safe_int(digits)

    if point is not None:
        # Most broker feeds use fractional pricing:
        # 5-digit quotes -> 0.00001 point, 0.0001 pip
        # 3-digit quotes -> 0.001 point, 0.01 pip
        # The same normalization works well for metals quoted with 3 decimals.
        if digits in (3, 5):
            return point * 10.0
        if digits is not None:
            return point

        if tick_size is not None:
            return tick_size
        return point

    #Falling back to symbol heuristics if broker precision is missing
    return fallback_pip_size(symbol)


def build_symbol_spec(symbol: str | None, *, broker_symbol: str | None = None, info: dict | None = None) -> SymbolSpec:
    #Normalizing broker symbol metadata into one reusable spec object
    info = info or {}
    digits = _safe_int(info.get("digits"))
    point = _safe_float(info.get("point"))
    tick_size = _safe_float(info.get("trade_tick_size"))
    tick_value = _safe_float(info.get("trade_tick_value"))
    contract_size = _safe_float(info.get("trade_contract_size"))
    volume_min = _safe_float(info.get("volume_min"))
    volume_max = _safe_float(info.get("volume_max"))
    volume_step = _safe_float(info.get("volume_step"))
    spread_points = _safe_float(info.get("spread"))
    currency_base = info.get("currency_base") or None
    currency_profit = info.get("currency_profit") or None

    #Applying explicit overrides before broker-derived pip inference
    override = _override_pip_size(symbol) or _override_pip_size(broker_symbol)
    if override is not None:
        pip_size = float(override)
        source = "override"
    else:
        pip_size = infer_pip_size(
            broker_symbol or symbol,
            digits=digits,
            point=point,
            tick_size=tick_size,
        )
        source = "broker" if info else "fallback"

    return SymbolSpec(
        requested_symbol=symbol,
        broker_symbol=broker_symbol or symbol,
        pip_size=float(pip_size),
        digits=digits,
        point=point,
        tick_size=tick_size,
        tick_value=tick_value,
        contract_size=contract_size,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
        spread_points=spread_points,
        currency_base=currency_base,
        currency_profit=currency_profit,
        source=source,
    )


def load_symbol_spec(symbol: str | None) -> SymbolSpec:
    if not symbol:
        return build_symbol_spec(symbol)

    try:
        #Fetching live broker metadata for the requested symbol
        creds = MT5Connector.creds_from_env()
        with MT5Connector(creds=creds) as conn:
            broker_symbol = conn.resolve_symbol(symbol)
            info = conn.symbol_info(broker_symbol)
            return build_symbol_spec(symbol, broker_symbol=broker_symbol, info=info)
    except Exception as exc:
        #Falling back to static heuristics if broker lookup fails
        log.warning("Falling back to heuristic pip size for %s: %s", symbol, exc)
        return build_symbol_spec(symbol)
