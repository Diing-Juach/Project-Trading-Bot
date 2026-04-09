from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PNL_TOLERANCE = 1e-6


@dataclass(slots=True)
class RankingConfig:
    keep_ratio_min: float = 0.35
    keep_ratio_max: float = 1.00
    keep_ratio_step: float = 0.05
    min_trades: int = 12
    min_pnl_retention: float = 0.95
    max_profit_factor: float = 5.0


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    equity = series.cumsum()
    drawdown = equity.cummax() - equity
    return float(drawdown.max()) if len(drawdown) else 0.0


def trading_metrics(df: pd.DataFrame) -> dict[str, float]:
    #Summarizing selection quality using trading-oriented metrics
    if len(df) == 0:
        return {
            "trades": 0.0,
            "pnl": 0.0,
            "pnl_r": 0.0,
            "expectancy": 0.0,
            "expectancy_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_r": 0.0,
        }

    pnl = pd.to_numeric(df["PnL_Cash"], errors="coerce").fillna(0.0)
    pnl_r = pd.to_numeric(df["PnL_R"], errors="coerce").fillna(0.0)

    wins = pnl[pnl > PNL_TOLERANCE]
    losses = pnl[pnl < -PNL_TOLERANCE]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    if gross_loss > PNL_TOLERANCE:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > PNL_TOLERANCE:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return {
        "trades": float(len(df)),
        "pnl": float(pnl.sum()),
        "pnl_r": float(pnl_r.sum()),
        "expectancy": float(pnl.mean()),
        "expectancy_r": float(pnl_r.mean()),
        "win_rate": float((pnl > PNL_TOLERANCE).mean() * 100.0),
        "profit_factor": profit_factor,
        "max_drawdown": _max_drawdown(pnl),
        "max_drawdown_r": _max_drawdown(pnl_r),
    }


def confluence_lift(base_df: pd.DataFrame, accepted_df: pd.DataFrame) -> pd.DataFrame:
    #Measuring which confluences appear more often in the selected trades
    cols = [col for col in base_df.columns if "Valid" in col]
    rows: list[dict[str, float | str]] = []
    tol = 1e-12

    for col in cols:
        base = pd.to_numeric(base_df[col], errors="coerce").fillna(0.0).mean()
        accepted = (
            pd.to_numeric(accepted_df[col], errors="coerce").fillna(0.0).mean()
            if len(accepted_df)
            else 0.0
        )
        rows.append(
            {
                "feature": col,
                "baseline_freq": float(base * 100.0),
                "accepted_freq": float(accepted * 100.0),
                "lift": float(accepted / base) if base > 0 else 0.0,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["feature", "baseline_freq", "accepted_freq", "lift"])

    changed = (out["accepted_freq"] - out["baseline_freq"]).abs() > tol
    out = out.loc[changed].copy()
    if out.empty:
        return pd.DataFrame(columns=["feature", "baseline_freq", "accepted_freq", "lift"])

    return out.sort_values(["lift", "accepted_freq"], ascending=[False, False], kind="stable").reset_index(drop=True)


def _select_top_by_ratio(df: pd.DataFrame, scores: np.ndarray, keep_ratio: float) -> tuple[pd.DataFrame, float]:
    #Keeping the top-scored fraction of rows and returning the cutoff score
    if len(df) == 0:
        return df.copy(), float("nan")

    keep_count = max(1, int(np.ceil(len(df) * keep_ratio)))
    ranked = df.copy()
    ranked["_score"] = scores
    ranked = ranked.sort_values("_score", ascending=False, kind="stable")
    selected = ranked.iloc[:keep_count].sort_index(kind="stable")
    cutoff = float(ranked.iloc[keep_count - 1]["_score"])
    return selected.drop(columns="_score"), cutoff


def _select_top_by_ratio_grouped(
    df: pd.DataFrame,
    *,
    keep_ratio: float,
    group_col: str,
    score_col: str,
) -> tuple[pd.DataFrame, float]:
    #Applying the top-fraction selection separately inside each group
    if len(df) == 0:
        return df.copy(), float("nan")
    if group_col not in df.columns:
        raise KeyError(f"Grouped keep-ratio selection requires column '{group_col}'.")
    if score_col not in df.columns:
        raise KeyError(f"Grouped keep-ratio selection requires column '{score_col}'.")

    parts: list[pd.DataFrame] = []
    cutoffs: list[float] = []

    for _, group_df in df.groupby(group_col, sort=True):
        scores = pd.to_numeric(group_df[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        selected, cutoff = _select_top_by_ratio(group_df, scores, keep_ratio)
        parts.append(selected)
        if np.isfinite(cutoff):
            cutoffs.append(float(cutoff))

    selected = pd.concat(parts, axis=0).sort_index(kind="stable") if parts else df.iloc[0:0].copy()
    grouped_cutoff = float(np.median(cutoffs)) if cutoffs else float("nan")
    return selected, grouped_cutoff


def _objective_score(
    metrics: dict[str, float],
    baseline: dict[str, float],
    keep_ratio: float,
    cfg: RankingConfig,
) -> float:
    #Scoring a keep-ratio candidate by PnL retention, expectancy, and drawdown
    pnl_retention = metrics["pnl"] / baseline["pnl"] if baseline["pnl"] > 0 else 1.0
    dd_improvement = (
        (baseline["max_drawdown_r"] - metrics["max_drawdown_r"]) / baseline["max_drawdown_r"]
        if baseline["max_drawdown_r"] > 0
        else 0.0
    )
    profit_factor = (
        min(metrics["profit_factor"], cfg.max_profit_factor)
        if np.isfinite(metrics["profit_factor"])
        else cfg.max_profit_factor
    )

    return float(
        (4.0 * pnl_retention)
        + (2.0 * metrics["expectancy_r"])
        + (1.0 * dd_improvement)
        + (0.5 * profit_factor)
        - (0.05 * keep_ratio)
    )


def find_keep_ratio_by_group(
    df: pd.DataFrame,
    cfg: RankingConfig,
    *,
    group_col: str = "fold",
    score_col: str = "score",
) -> tuple[float, float, pd.DataFrame]:
    #Sweeping keep ratios on grouped validation scores to find the best selection rate
    baseline = trading_metrics(df)
    rows: list[dict[str, float]] = []

    for keep_ratio in np.arange(cfg.keep_ratio_min, cfg.keep_ratio_max + 1e-9, cfg.keep_ratio_step):
        selected, cutoff = _select_top_by_ratio_grouped(
            df,
            keep_ratio=float(keep_ratio),
            group_col=group_col,
            score_col=score_col,
        )
        metrics = trading_metrics(selected)
        pnl_retention = metrics["pnl"] / baseline["pnl"] if baseline["pnl"] > 0 else np.nan
        dd_ratio = metrics["max_drawdown_r"] / baseline["max_drawdown_r"] if baseline["max_drawdown_r"] > 0 else np.nan
        objective = _objective_score(metrics, baseline, float(keep_ratio), cfg)
        rows.append(
            {
                "keep_ratio": float(keep_ratio),
                "cutoff_score": cutoff,
                **metrics,
                "pnl_retention": float(pnl_retention) if np.isfinite(pnl_retention) else np.nan,
                "drawdown_ratio": float(dd_ratio) if np.isfinite(dd_ratio) else np.nan,
                "objective_score": objective,
            }
        )

    sweep = pd.DataFrame(rows)
    sweep = sweep.assign(
        profit_factor_capped=sweep["profit_factor"].replace(np.inf, cfg.max_profit_factor)
    )

    #Filtering to candidates that preserve enough trades and PnL
    eligible = sweep[
        (sweep["trades"] >= cfg.min_trades)
        & (
            (sweep["pnl_retention"] >= cfg.min_pnl_retention)
            if baseline["pnl"] > 0
            else True
        )
    ].copy()

    #Falling back to the best objective score if nothing meets the retention constraint
    if eligible.empty:
        print(
            f"[ML] No keep ratio retained at least {cfg.min_pnl_retention:.0%} of validation PnL. "
            "Falling back to best objective score."
        )
        best = sweep.sort_values(
            ["objective_score", "pnl", "expectancy_r", "profit_factor_capped", "max_drawdown_r", "keep_ratio"],
            ascending=[False, False, False, False, True, True],
            kind="stable",
        ).iloc[0]
        return float(best["keep_ratio"]), float(best["cutoff_score"]), sweep.drop(columns="profit_factor_capped")

    best = eligible.sort_values(
        ["pnl", "expectancy_r", "profit_factor_capped", "max_drawdown_r", "keep_ratio"],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).iloc[0]
    return float(best["keep_ratio"]), float(best["cutoff_score"]), sweep.drop(columns="profit_factor_capped")


def select_by_keep_ratio(
    df: pd.DataFrame,
    keep_ratio: float,
    *,
    group_col: str = "fold",
    score_col: str = "score",
) -> pd.DataFrame:
    #Applying the chosen keep ratio to grouped scored rows
    selected, _ = _select_top_by_ratio_grouped(
        df,
        keep_ratio=keep_ratio,
        group_col=group_col,
        score_col=score_col,
    )
    return selected
