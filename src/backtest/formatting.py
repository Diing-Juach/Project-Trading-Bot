from __future__ import annotations

from datetime import date, datetime, timezone
import math
from pathlib import Path
from typing import Any

import pandas as pd


_TWO_DP_KEYWORDS = (
    "balance",
    "pnl",
    "profit",
    "loss",
    "drawdown",
    "roi_pct",
    "avg_pnl",
    "largest_",
    "commission",
    "price",
    "pips",
    "bars",
)

_FOUR_DP_KEYWORDS = (
    "ratio",
    "sharpe",
    "expectancy",
    "profit_factor",
    "avg_rr",
    "cutoff",
    "threshold",
)

_PERCENT_KEYS = {
    "win_rate",
    "baseline_freq",
    "accepted_freq",
    "long_signal_hit_rate",
    "short_signal_hit_rate",
    "overall_signal_hit_rate",
}


def _format_dateish(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")
    return str(value)


def make_run_name(symbol: str, timeframe: str, start: Any, end: Any) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{symbol}_{timeframe}_{_format_dateish(start)}_{_format_dateish(end)}_{stamp}"


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_stat_value(key: str | None, value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        key_l = (key or "").lower()

        if key_l.endswith("_pct") or key_l in _PERCENT_KEYS:
            return round(value, 2)
        if any(token in key_l for token in _TWO_DP_KEYWORDS):
            return round(value, 2)
        if any(token in key_l for token in _FOUR_DP_KEYWORDS):
            return round(value, 4)
        return round(value, 4)

    return value


def format_stats_payload(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {str(k): format_stats_payload(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [format_stats_payload(v, key=key) for v in value]
    return format_stat_value(key, value)


def to_serializable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if pd.isna(value):
        return None
    return value
