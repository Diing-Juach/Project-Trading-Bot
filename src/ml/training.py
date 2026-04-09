from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.backtest.formatting import format_stats_payload
from src.mt5.config import BACKTEST_OUTPUT_DIR, ML_DATASET_OUTPUT_DIR, OUTPUTS_DIR

from .dataset import JOIN_KEY_COLUMNS, load_training_data
from .features import add_trade_quality_features, apply_imputer, build_sample_weights, fit_imputer, vectorize_splits
from .ranking import (
    RankingConfig,
    find_keep_ratio_by_group,
    confluence_lift,
    select_by_keep_ratio,
    trading_metrics,
)


@dataclass(slots=True)
class TrainConfig:
    data_dir: str
    pattern: str = "*.csv"
    backtest_dir: str = str(BACKTEST_OUTPUT_DIR)
    target: str = "Label_PositivePnL"
    output_dir: str = str(OUTPUTS_DIR / "ml_models" / "xgb_ranker_v1")

    train_ratio: float = 0.70
    walk_forward_folds: int = 3

    keep_ratio_min: float = 0.35
    keep_ratio_max: float = 1.00
    keep_ratio_step: float = 0.05
    min_trades_threshold: int = 12
    min_pnl_retention: float = 0.95

    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.04
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    min_child_weight: float = 2.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    early_stopping_rounds: int = 40
    random_state: int = 42


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    #Summarizing the loaded dataset before training
    grouped = (
        df.groupby(["Symbol", "Timeframe"]).size().reset_index(name="trades").sort_values(["Symbol", "Timeframe"])
        if {"Symbol", "Timeframe"}.issubset(df.columns)
        else pd.DataFrame()
    )
    return {
        "rows": int(len(df)),
        "date_start": df["Trade_Date"].min(),
        "date_end": df["Trade_Date"].max(),
        "target_distribution": (
            df["Label_PositivePnL"].value_counts(dropna=False).to_dict()
            if "Label_PositivePnL" in df.columns
            else {}
        ),
        "by_symbol_timeframe": grouped.to_dict(orient="records"),
    }


def build_walk_forward_folds(df: pd.DataFrame, cfg: TrainConfig) -> list[dict[str, Any]]:
    #Building expanding walk-forward train/validation/test folds
    unique_dates = pd.Index(df["Trade_Date"].sort_values().unique())
    if len(unique_dates) < 6:
        raise ValueError("Need at least 6 unique trade timestamps to create walk-forward folds.")

    initial_train_size = max(1, int(len(unique_dates) * cfg.train_ratio))
    initial_train_size = min(initial_train_size, len(unique_dates) - 2)

    remaining = len(unique_dates) - initial_train_size
    max_supported_folds = max(1, remaining // 2)
    fold_count = min(cfg.walk_forward_folds, max_supported_folds)
    if fold_count < 1:
        raise ValueError("Not enough trade timestamps to build walk-forward folds.")
    if fold_count < cfg.walk_forward_folds:
        print(
            f"[ML] Using {fold_count} walk-forward folds instead of requested "
            f"{cfg.walk_forward_folds} based on available trade timestamps."
        )

    window_size = max(1, remaining // (fold_count * 2))
    folds: list[dict[str, Any]] = []

    #Stepping through each anchored fold in chronological order
    for fold_idx in range(fold_count):
        train_end_idx = initial_train_size + (2 * fold_idx * window_size)
        valid_end_idx = train_end_idx + window_size
        test_end_idx = valid_end_idx + window_size

        if train_end_idx >= len(unique_dates) or valid_end_idx >= len(unique_dates):
            break
        if fold_idx == fold_count - 1 or test_end_idx >= len(unique_dates):
            test_end_idx = len(unique_dates)
        if test_end_idx <= valid_end_idx:
            break

        train_cut = unique_dates[train_end_idx - 1]
        valid_cut = unique_dates[valid_end_idx - 1]
        test_cut = unique_dates[test_end_idx - 1]

        train_df = df[df["Trade_Date"] <= train_cut].copy()
        valid_df = df[(df["Trade_Date"] > train_cut) & (df["Trade_Date"] <= valid_cut)].copy()
        test_df = df[(df["Trade_Date"] > valid_cut) & (df["Trade_Date"] <= test_cut)].copy()

        if train_df.empty or valid_df.empty or test_df.empty:
            continue

        folds.append(
            {
                "fold": fold_idx + 1,
                "train_df": train_df,
                "valid_df": valid_df,
                "test_df": test_df,
                "train_start": train_df["Trade_Date"].min(),
                "train_end": train_df["Trade_Date"].max(),
                "valid_start": valid_df["Trade_Date"].min(),
                "valid_end": valid_df["Trade_Date"].max(),
                "test_start": test_df["Trade_Date"].min(),
                "test_end": test_df["Trade_Date"].max(),
            }
        )

    if not folds:
        raise ValueError("Walk-forward fold generation produced no usable folds.")

    return folds


def _train_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray,
    cfg: TrainConfig,
    *,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
) -> XGBClassifier:
    #Training the XGBoost classifier with optional early stopping
    params = {
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "min_child_weight": cfg.min_child_weight,
        "reg_lambda": cfg.reg_lambda,
        "reg_alpha": cfg.reg_alpha,
        "random_state": cfg.random_state,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }

    fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["verbose"] = False

    try:
        if X_valid is not None and y_valid is not None:
            model = XGBClassifier(**params, early_stopping_rounds=cfg.early_stopping_rounds)
        else:
            model = XGBClassifier(**params)
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight)

    return model


def _build_feature_importance_table(model: XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    #Exporting non-zero model feature importance values
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return pd.DataFrame(columns=["feature", "importance"])
    out = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    out = out[out["importance"] > 0].reset_index(drop=True)
    return out


def _attach_scores(
    df: pd.DataFrame,
    scores: np.ndarray,
    *,
    fold: int,
) -> pd.DataFrame:
    #Attaching model scores and within-fold rank percentiles to a split
    scored = df.copy()
    scored["fold"] = int(fold)
    scored["score"] = scores
    scored["rank_pct"] = scored["score"].rank(method="first", pct=True, ascending=False)
    return scored


def _build_prediction_table(df: pd.DataFrame, target: str, selected_index: pd.Index) -> pd.DataFrame:
    #Building the compact prediction export used for analysis
    cols = JOIN_KEY_COLUMNS + ["fold", "PnL_Cash", "PnL_R", target, "rank_pct"]
    out = df[cols].copy()
    out["selected"] = 0
    out.loc[out.index.isin(selected_index), "selected"] = 1
    return out


def train(cfg: TrainConfig) -> dict[str, Any]:
    #Loading datasets and enriching them with trade-quality features
    raw_df = load_training_data(
        data_dir=cfg.data_dir,
        pattern=cfg.pattern,
        backtest_dir=cfg.backtest_dir,
        target=cfg.target,
    )
    raw_df = add_trade_quality_features(raw_df)

    #Preparing walk-forward folds and ranking settings
    folds = build_walk_forward_folds(raw_df, cfg)
    ranking_cfg = RankingConfig(
        keep_ratio_min=cfg.keep_ratio_min,
        keep_ratio_max=cfg.keep_ratio_max,
        keep_ratio_step=cfg.keep_ratio_step,
        min_trades=cfg.min_trades_threshold,
        min_pnl_retention=cfg.min_pnl_retention,
    )

    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    #Training one model per fold and scoring validation/test slices
    for fold in folds:
        train_df = fold["train_df"]
        valid_df = fold["valid_df"]
        test_df = fold["test_df"]

        X_train, X_valid, X_test = vectorize_splits(train_df, valid_df, test_df, target=cfg.target)
        y_train = pd.to_numeric(train_df[cfg.target], errors="coerce").astype(int)
        y_valid = pd.to_numeric(valid_df[cfg.target], errors="coerce").astype(int)

        imputer = fit_imputer(X_train)
        X_train_imp = apply_imputer(imputer, X_train)
        X_valid_imp = apply_imputer(imputer, X_valid)
        X_test_imp = apply_imputer(imputer, X_test)

        model = _train_xgb_model(
            X_train_imp,
            y_train,
            build_sample_weights(train_df),
            cfg,
            X_valid=X_valid_imp,
            y_valid=y_valid,
        )

        validation_parts.append(
            _attach_scores(
                valid_df,
                model.predict_proba(X_valid_imp)[:, 1],
                fold=fold["fold"],
            )
        )
        test_parts.append(
            _attach_scores(
                test_df,
                model.predict_proba(X_test_imp)[:, 1],
                fold=fold["fold"],
            )
        )

    #Combining fold predictions for global validation/test analysis
    validation_scored = pd.concat(validation_parts, axis=0).sort_values("Trade_Date", kind="stable")
    test_scored = pd.concat(test_parts, axis=0).sort_values("Trade_Date", kind="stable")

    #Selecting the keep ratio on validation and applying it to test
    keep_ratio, _cutoff_score, keep_ratio_sweep = find_keep_ratio_by_group(
        validation_scored,
        ranking_cfg,
        group_col="fold",
        score_col="score",
    )
    accepted_valid = select_by_keep_ratio(validation_scored, keep_ratio, group_col="fold", score_col="score")
    accepted_test = select_by_keep_ratio(test_scored, keep_ratio, group_col="fold", score_col="score")

    valid_baseline = trading_metrics(validation_scored)
    valid_filtered = trading_metrics(accepted_valid)
    test_baseline = trading_metrics(test_scored)
    test_filtered = trading_metrics(accepted_test)

    valid_predictions = _build_prediction_table(validation_scored, cfg.target, accepted_valid.index)
    test_predictions = _build_prediction_table(test_scored, cfg.target, accepted_test.index)

    #Summarizing each walk-forward fold separately
    fold_summaries: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = fold["fold"]
        valid_fold = validation_scored[validation_scored["fold"] == fold_id]
        test_fold = test_scored[test_scored["fold"] == fold_id]
        accepted_valid_fold = accepted_valid[accepted_valid["fold"] == fold_id]
        accepted_test_fold = accepted_test[accepted_test["fold"] == fold_id]
        fold_summaries.append(
            {
                "fold": int(fold_id),
                "train_rows": int(len(fold["train_df"])),
                "valid_rows": int(len(valid_fold)),
                "test_rows": int(len(test_fold)),
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "valid_start": fold["valid_start"],
                "valid_end": fold["valid_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "validation": {
                    "baseline": trading_metrics(valid_fold),
                    "filtered": trading_metrics(accepted_valid_fold),
                },
                "test": {
                    "baseline": trading_metrics(test_fold),
                    "filtered": trading_metrics(accepted_test_fold),
                },
            }
        )

    fold_summary_rows = [
        {
            "fold": fold["fold"],
            "train_rows": fold["train_rows"],
            "valid_rows": fold["valid_rows"],
            "test_rows": fold["test_rows"],
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "valid_start": fold["valid_start"],
            "valid_end": fold["valid_end"],
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "validation_baseline_pnl": fold["validation"]["baseline"]["pnl"],
            "validation_filtered_pnl": fold["validation"]["filtered"]["pnl"],
            "test_baseline_pnl": fold["test"]["baseline"]["pnl"],
            "test_filtered_pnl": fold["test"]["filtered"]["pnl"],
            "test_baseline_expectancy_r": fold["test"]["baseline"]["expectancy_r"],
            "test_filtered_expectancy_r": fold["test"]["filtered"]["expectancy_r"],
            "test_baseline_profit_factor": fold["test"]["baseline"]["profit_factor"],
            "test_filtered_profit_factor": fold["test"]["filtered"]["profit_factor"],
            "test_baseline_max_drawdown_r": fold["test"]["baseline"]["max_drawdown_r"],
            "test_filtered_max_drawdown_r": fold["test"]["filtered"]["max_drawdown_r"],
        }
        for fold in fold_summaries
    ]

    #Retraining one final model on the full dataset for saved artifacts
    empty_df = raw_df.iloc[0:0].copy()
    X_full, _, _ = vectorize_splits(raw_df, empty_df, empty_df, target=cfg.target)
    imputer_final = fit_imputer(X_full)
    X_full_imp = apply_imputer(imputer_final, X_full)
    y_full = pd.to_numeric(raw_df[cfg.target], errors="coerce").astype(int)
    final_model = _train_xgb_model(
        X_full_imp,
        y_full,
        build_sample_weights(raw_df),
        cfg,
    )

    #Writing the trained bundle and analysis exports
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": final_model,
        "imputer": imputer_final,
        "feature_columns": list(X_full.columns),
        "target": cfg.target,
        "keep_ratio": float(keep_ratio),
        "selection_method": "walk_forward_top_x_percent",
        "walk_forward_folds": int(len(folds)),
        "config": asdict(cfg),
    }
    with (output_dir / "model_bundle.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)

    keep_ratio_sweep = keep_ratio_sweep.drop(columns=["cutoff_score"], errors="ignore")
    keep_ratio_sweep.to_csv(output_dir / "validation_keep_ratio_sweep.csv", index=False)
    valid_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(fold_summary_rows).to_csv(output_dir / "walk_forward_fold_summary.csv", index=False)
    confluence_lift(validation_scored, accepted_valid).to_csv(output_dir / "validation_confluence_lift.csv", index=False)
    confluence_lift(test_scored, accepted_test).to_csv(output_dir / "test_confluence_lift.csv", index=False)
    _build_feature_importance_table(final_model, list(X_full.columns)).to_csv(
        output_dir / "feature_importance.csv",
        index=False,
    )

    #Saving the top-level training summary used by the report/dashboard
    summary = {
        "dataset_overview": _dataset_overview(raw_df),
        "evaluation_mode": "walk_forward",
        "walk_forward_folds": int(len(folds)),
        "initial_train_ratio": float(cfg.train_ratio),
        "selected_keep_ratio": float(keep_ratio),
        "feature_count": int(len(X_full.columns)),
        "validation": {"baseline": valid_baseline, "filtered": valid_filtered},
        "test": {"baseline": test_baseline, "filtered": test_filtered},
        "folds": fold_summaries,
    }
    summary = format_stats_payload(summary)
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2)

    print(f"Walk-forward folds: {len(folds)}")
    print(f"Selected keep ratio: {keep_ratio:.2f}")
    print(f"Validation trades kept: {len(accepted_valid)}/{len(validation_scored)}")
    print(f"Test trades kept: {len(accepted_test)}/{len(test_scored)}")
    print(
        f"Validation baseline pnl: {valid_baseline['pnl']:.2f} | "
        f"filtered pnl: {valid_filtered['pnl']:.2f}"
    )
    print(
        f"Test baseline pnl: {test_baseline['pnl']:.2f} | "
        f"filtered pnl: {test_filtered['pnl']:.2f}"
    )

    return {"output_dir": str(output_dir), **summary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an XGBoost trade-ranking filter.")
    parser.add_argument("--data-dir", default=str(ML_DATASET_OUTPUT_DIR))
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--backtest-dir", default=str(BACKTEST_OUTPUT_DIR))
    parser.add_argument("--target", default="Label_PositivePnL")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "ml_models" / "xgb_ranker_v1"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--walk-forward-folds", type=int, default=3)
    parser.add_argument("--keep-ratio-min", type=float, default=0.35)
    parser.add_argument("--keep-ratio-max", type=float, default=1.00)
    parser.add_argument("--keep-ratio-step", type=float, default=0.05)
    parser.add_argument("--min-trades-threshold", type=int, default=12)
    parser.add_argument("--min-pnl-retention", type=float, default=0.95)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        pattern=args.pattern,
        backtest_dir=args.backtest_dir,
        target=args.target,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        walk_forward_folds=args.walk_forward_folds,
        keep_ratio_min=args.keep_ratio_min,
        keep_ratio_max=args.keep_ratio_max,
        keep_ratio_step=args.keep_ratio_step,
        min_trades_threshold=args.min_trades_threshold,
        min_pnl_retention=args.min_pnl_retention,
    )
    train(cfg)


if __name__ == "__main__":
    main()
