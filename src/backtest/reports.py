from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable

import pandas as pd

from src.backtest.formatting import ensure_directory, format_stats_payload, make_run_name, to_serializable
from src.backtest.models import Trade


_STD_NORMAL = NormalDist()
_EULER_MASCHERONI = 0.5772156649


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _trade_pnl_returns(trades: Iterable[Trade], initial_balance: float) -> pd.Series:
    if initial_balance <= 0:
        return pd.Series(dtype="float64")

    #Normalizing trade PnL into simple return units for Sharpe-style stats
    return pd.Series(
        [_safe_float(t.pnl) / initial_balance for t in trades],
        dtype="float64",
    ).replace([math.inf, -math.inf], pd.NA).dropna()


def _compute_sharpe_from_returns(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0

    std = float(returns.std(ddof=1))
    if std <= 0.0:
        return 0.0

    return float((returns.mean() / std) * math.sqrt(len(returns)))


def _resolve_trial_sharpe_variance(summary: Dict[str, Any]) -> float:
    explicit = summary.get("sharpe_trial_variance")
    if explicit is not None:
        value = _safe_float(explicit, default=0.0)
        return value if value > 0.0 else 0.0

    trial_sharpes = summary.get("trial_sharpes")
    if not isinstance(trial_sharpes, (list, tuple)):
        return 0.0

    series = pd.Series(
        [_safe_float(value, default=float("nan")) for value in trial_sharpes],
        dtype="float64",
    ).replace([math.inf, -math.inf], pd.NA).dropna()

    if len(series) < 2:
        return 0.0

    variance = float(series.var(ddof=1))
    return variance if math.isfinite(variance) and variance > 0.0 else 0.0


def _expected_max_sharpe(num_trials: int, sharpe_variance: float) -> float:
    if num_trials <= 1 or sharpe_variance <= 0.0:
        return 0.0

    sigma = math.sqrt(sharpe_variance)
    max_z = (
        ((1.0 - _EULER_MASCHERONI) * _STD_NORMAL.inv_cdf(1.0 - (1.0 / num_trials)))
        + (_EULER_MASCHERONI * _STD_NORMAL.inv_cdf(1.0 - (1.0 / (num_trials * math.e))))
    )
    return float(sigma * max_z)


def _compute_deflated_sharpe_stats(
    summary: Dict[str, Any],
    trades: Iterable[Trade],
    initial_balance: float,
) -> Dict[str, Any]:
    #Turning trade returns into Sharpe and Deflated Sharpe style diagnostics
    returns = _trade_pnl_returns(trades, initial_balance)
    observed_sharpe = _compute_sharpe_from_returns(returns)

    if len(returns) < 2:
        return {
            "sharpe": observed_sharpe,
            "deflated_sharpe_ratio": 0.0,
            "deflated_sharpe_threshold": 0.0,
            "deflated_sharpe_trials": 1,
            "deflated_sharpe_mode": "insufficient_observations",
        }

    raw_trials = summary.get("independent_trials", 1)
    try:
        num_trials = max(1, int(raw_trials))
    except Exception:
        num_trials = 1

    sharpe_variance = _resolve_trial_sharpe_variance(summary)
    threshold_sharpe = _expected_max_sharpe(num_trials, sharpe_variance)

    skew = float(returns.skew()) if len(returns) >= 3 else 0.0
    excess_kurtosis = float(returns.kurt()) if len(returns) >= 4 else 0.0
    kurtosis = excess_kurtosis + 3.0

    denominator_term = (
        1.0
        - (skew * observed_sharpe)
        + (((kurtosis - 1.0) / 4.0) * observed_sharpe * observed_sharpe)
    )

    if denominator_term <= 0.0 or not math.isfinite(denominator_term):
        deflated = 0.0
    else:
        z_score = (
            (observed_sharpe - threshold_sharpe)
            * math.sqrt(len(returns) - 1.0)
            / math.sqrt(denominator_term)
        )
        deflated = float(_STD_NORMAL.cdf(z_score))

    if num_trials <= 1 or sharpe_variance <= 0.0:
        mode = "single_trial_psr_equivalent"
    else:
        mode = "multiple_testing_adjusted"

    return {
        "sharpe": observed_sharpe,
        "deflated_sharpe_ratio": deflated,
        "deflated_sharpe_threshold": threshold_sharpe,
        "deflated_sharpe_trials": num_trials,
        "deflated_sharpe_mode": mode,
    }


def _compute_avg_bars_in_trade(trades: list[Trade]) -> float:
    durations = [
        float(t.close_bar - t.open_bar)
        for t in trades
        if t.close_bar is not None and t.open_bar is not None
    ]
    if not durations:
        return 0.0
    return float(sum(durations) / len(durations))


def _compute_trade_stats(
    summary: Dict[str, Any],
    trades: Iterable[Trade],
) -> Dict[str, Any]:
    trades = list(trades)

    #Breaking the finished trade list into wins, losses, and breakevens
    wins = [t for t in trades if t.pnl > 0 and not t.is_breakeven_exit]
    losses = [t for t in trades if t.pnl < 0 and not t.is_breakeven_exit]
    breakevens = [t for t in trades if t.is_breakeven_exit]

    gross_profit = float(sum(t.pnl for t in wins))
    gross_loss = abs(float(sum(t.pnl for t in losses)))

    avg_win = float(gross_profit / len(wins)) if wins else 0.0
    avg_loss = float(sum(t.pnl for t in losses) / len(losses)) if losses else 0.0
    avg_pnl = float(sum(t.pnl for t in trades) / len(trades)) if trades else 0.0
    avg_rr = float(sum(t.result for t in trades) / len(trades)) if trades else 0.0

    #Rebuilding account-level stats from the engine summary and trade list
    initial_balance = _safe_float(summary.get("initial_balance", 0.0))
    ending_balance = _safe_float(summary.get("balance", initial_balance))
    net_pnl = ending_balance - initial_balance
    roi_pct = float((net_pnl / initial_balance) * 100.0) if initial_balance > 0 else 0.0
    sharpe_stats = _compute_deflated_sharpe_stats(summary, trades, initial_balance)

    largest_win = float(max((t.pnl for t in trades), default=0.0))
    largest_loss = float(min((t.pnl for t in trades), default=0.0))

    return {
        "initial_balance": initial_balance,
        "ending_balance": ending_balance,
        "net_pnl": float(net_pnl),
        "roi_pct": roi_pct,
        "trades": int(len(trades)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "breakevens": int(len(breakevens)),
        "win_rate": _safe_float(summary.get("win_rate", 0.0)),
        "expectancy_r": _safe_float(summary.get("expectancy_r", 0.0)),
        "avg_rr_per_trade": avg_rr,
        "profit_factor": _safe_float(summary.get("profit_factor", 0.0)),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_pnl_per_trade": avg_pnl,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "initial_balance_max_drawdown": _safe_float(summary.get("initial_balance_max_drawdown", 0.0)),
        "initial_balance_max_drawdown_pct": _safe_float(summary.get("initial_balance_max_drawdown_pct", 0.0)),
        "max_drawdown": _safe_float(summary.get("max_drawdown", 0.0)),
        "max_drawdown_pct": _safe_float(summary.get("max_drawdown_pct", 0.0)),
        "sharpe": sharpe_stats["sharpe"],
        "deflated_sharpe_ratio": sharpe_stats["deflated_sharpe_ratio"],
        "deflated_sharpe_threshold": sharpe_stats["deflated_sharpe_threshold"],
        "deflated_sharpe_trials": sharpe_stats["deflated_sharpe_trials"],
        "deflated_sharpe_mode": sharpe_stats["deflated_sharpe_mode"],
        "avg_bars_in_trade": _compute_avg_bars_in_trade(trades),
    }


def _compute_account_info(
    summary: Dict[str, Any],
    *,
    symbol: str,
    timeframe: str,
    start: Any,
    end: Any,
) -> Dict[str, Any]:
    #Building the compact account-level summary saved beside trade stats
    initial_balance = _safe_float(summary.get("initial_balance", 0.0))
    ending_balance = _safe_float(summary.get("balance", initial_balance))
    net_pnl = ending_balance - initial_balance
    roi_pct = float((net_pnl / initial_balance) * 100.0) if initial_balance > 0 else 0.0

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "initial_balance": initial_balance,
        "ending_balance": ending_balance,
        "net_pnl": float(net_pnl),
        "roi_pct": roi_pct,
        "initial_balance_max_drawdown": _safe_float(summary.get("initial_balance_max_drawdown", 0.0)),
        "initial_balance_max_drawdown_pct": _safe_float(summary.get("initial_balance_max_drawdown_pct", 0.0)),
        "max_drawdown": _safe_float(summary.get("max_drawdown", 0.0)),
        "max_drawdown_pct": _safe_float(summary.get("max_drawdown_pct", 0.0)),
        "trades": int(summary.get("trades", 0)),
        "wins": int(summary.get("wins", 0)),
        "losses": int(summary.get("losses", 0)),
        "breakevens": int(summary.get("breakevens", 0)),
    }


def save_backtest_artifacts(
    *,
    summary: Dict[str, Any],
    trades: Iterable[Trade],
    engine: Any,
    output_dir: str | Path,
    symbol: str,
    timeframe: str,
    start: Any,
    end: Any,
) -> Dict[str, str]:
    trades = list(trades)

    #Creating a fresh artifact directory for this backtest run
    root = ensure_directory(output_dir)
    run_dir = ensure_directory(root / make_run_name(symbol, timeframe, start, end))

    account_info_path = run_dir / "account_info.json"
    trade_stats_path = run_dir / "trade_stats.json"
    trades_csv_path = run_dir / "trades.csv"

    #Formatting the report payloads before writing them to disk
    account_info = _compute_account_info(
        summary,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    trade_stats = _compute_trade_stats(summary, trades)
    account_info = format_stats_payload(account_info)
    trade_stats = format_stats_payload(trade_stats)

    with account_info_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(account_info), handle, indent=2)

    with trade_stats_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(trade_stats), handle, indent=2)

    #Saving the engine's trade report frame with the dissertation-friendly schema
    engine.get_trade_report_frame().to_csv(trades_csv_path, index=False)

    return {
        "run_dir": str(run_dir),
        "account_info_json": str(account_info_path),
        "trade_stats_json": str(trade_stats_path),
        "trades_csv": str(trades_csv_path),
    }
