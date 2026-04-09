from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trade:
    #Core trade direction, pricing, and risk state
    direction: str
    entry: float
    stop: float
    target: float
    risk: float
    size: float
    open_bar: int

    #Close-state and realized trade outcome
    close_bar: int | None = None
    breakeven_moved: bool = False
    result: float = 0.0

    entry_time: object | None = None
    exit_time: object | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    balance_after: float | None = None

    #Commission and breakeven tracking
    initial_stop: float | None = None
    effective_initial_risk: float | None = None
    commission_paid: float = 0.0
    is_breakeven_exit: bool = False

    #Snapshot of all entry-time reporting fields
    report: dict[str, Any] = field(default_factory=dict)

    #Live size and realized PnL after partial exits
    remaining_size: float = 0.0
    realized_pnl: float = 0.0

    #Partial-take-profit state
    partial_taken: bool = False
    partial_price: float | None = None
    partial_bar: int | None = None
    partial_fraction: float = 0.30

    #Entry-time runner plan
    runner_qualified: bool = False
    runner_active: bool = False
    runner_activation_bar: int | None = None
    runner_reference_level: float | None = None
    runner_target_price: float | None = None
    runner_target_rr: float | None = None

    #Runner live state
    runner_stop_price: float | None = None
    runner_exit_reason: str | None = None
    final_exit_type: str | None = None
