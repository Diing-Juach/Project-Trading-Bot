from __future__ import annotations

import math


def value_per_price_unit_per_lot(
    *,
    tick_size: float | None = None,
    tick_value: float | None = None,
    contract_size: float | None = None,
) -> float:
    #Preferring broker tick economics and falling back to contract size
    if tick_size is not None and tick_size > 0 and tick_value is not None and tick_value > 0:
        return tick_value / tick_size
    if contract_size is not None and contract_size > 0:
        return contract_size
    return 1.0


def _step_precision(step: float) -> int:
    text = f"{step:.10f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def _round_lots_down(
    lots: float,
    *,
    volume_step: float | None = None,
    volume_min: float | None = None,
    volume_max: float | None = None,
) -> float:
    if lots <= 0:
        return 0.0

    #Rounding down so sizing never risks more than intended
    rounded = float(lots)
    if volume_step is not None and volume_step > 0:
        rounded = math.floor(rounded / volume_step) * volume_step

    if volume_max is not None and volume_max > 0:
        rounded = min(rounded, volume_max)

    if volume_min is not None and volume_min > 0 and rounded < volume_min:
        return 0.0

    precision = _step_precision(volume_step) if volume_step is not None and volume_step > 0 else 6
    return round(max(rounded, 0.0), precision)


def calculate_size(
    balance: float,
    risk_pct: float,
    entry: float,
    stop: float,
    *,
    tick_size: float | None = None,
    tick_value: float | None = None,
    contract_size: float | None = None,
    volume_step: float | None = None,
    volume_min: float | None = None,
    volume_max: float | None = None,
) -> float:
    #Sizing lots from cash risk, stop distance, and broker symbol economics
    risk_amount = balance * risk_pct
    dist = abs(entry - stop)
    if dist <= 0:
        return 0.0

    cash_per_price = value_per_price_unit_per_lot(
        tick_size=tick_size,
        tick_value=tick_value,
        contract_size=contract_size,
    )
    if cash_per_price <= 0:
        return 0.0

    raw_lots = risk_amount / (dist * cash_per_price)
    return _round_lots_down(
        raw_lots,
        volume_step=volume_step,
        volume_min=volume_min,
        volume_max=volume_max,
    )


def should_move_be(trade, high: float, low: float) -> bool:
    #Checking whether price has reached at least 1R from the original entry
    initial_risk = getattr(trade, "effective_initial_risk", None)

    if initial_risk is None or initial_risk <= 0:
        initial_stop = getattr(trade, "initial_stop", trade.stop)
        if trade.direction == "long":
            initial_risk = trade.entry - initial_stop
        else:
            initial_risk = initial_stop - trade.entry

    if initial_risk <= 0:
        return False

    if trade.direction == "long":
        rr = (high - trade.entry) / initial_risk
    else:
        rr = (trade.entry - low) / initial_risk

    return rr >= 1.0
