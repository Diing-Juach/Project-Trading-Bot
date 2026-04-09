from __future__ import annotations

from typing import Tuple


def get_target_from_risk(entry: float, risk: float, direction: str, rr_multiple: float = 2.0) -> float:
    if direction == "long":
        return entry + (risk * rr_multiple)

    return entry - (risk * rr_multiple)


def breakeven_price_to_cover_costs(
    entry: float,
    direction: str,
    remaining_commission: float,
    remaining_lots: float,
    cash_per_price_unit_per_lot: float,
) -> float:
    denominator = remaining_lots * cash_per_price_unit_per_lot
    if denominator <= 0:
        return entry

    price_offset = remaining_commission / denominator
    if direction == "long":
        return entry + price_offset
    return entry - price_offset


def runner_trailing_stop_candidate(direction: str, close_prev: float, atr_prev: float, atr_multiple: float) -> float:
    if direction == "long":
        return close_prev - (atr_multiple * atr_prev)
    return close_prev + (atr_multiple * atr_prev)


def runner_hit_status(
    direction: str,
    high: float,
    low: float,
    stop_price: float,
    target_price: float | None,
) -> Tuple[bool, bool]:
    if direction == "long":
        stop_hit = low <= stop_price
        target_hit = target_price is not None and high >= target_price
    else:
        stop_hit = high >= stop_price
        target_hit = target_price is not None and low <= target_price
    return stop_hit, target_hit
