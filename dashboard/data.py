from __future__ import annotations

import json
import os
import pickle
import re
import stat
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.mt5.config import ML_DATASET_OUTPUT_DIR


BACKTEST_CONFLUENCE_COLUMNS = [
    "SupplyAndDemand_Valid",
    "Atr_Valid",
    "Market_Structure_Valid",
    "Session_Liquidity_Valid",
    "Volume_Profile_Valid",
    "Imbalance_Valid",
    "Fibonacci_Valid",
    "Liquidity_Valid",
    "Support_Resistance_Valid",
]

CORE_CONFLUENCE_COLUMNS = [
    "SupplyAndDemand_Valid",
    "Atr_Valid",
    "Market_Structure_Valid",
]

CONFLUENCE_LABELS = {
    "SupplyAndDemand_Valid": "Supply & Demand",
    "Atr_Valid": "ATR",
    "Market_Structure_Valid": "Market Structure",
    "Session_Liquidity_Valid": "Session Liquidity",
    "Volume_Profile_Valid": "Volume Profile",
    "Imbalance_Valid": "Imbalance (FVG)",
    "Fibonacci_Valid": "Fibonacci",
    "Liquidity_Valid": "Liquidity",
    "Support_Resistance_Valid": "Support & Resistance",
}

BACKTEST_RUN_NAME_PATTERN = re.compile(
    r"^(?P<symbol>.+?)_(?P<timeframe>[A-Z0-9]+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})_(?P<stamp>\d{8}T\d{6}Z)$"
)
ML_DATASET_NAME_PATTERN = re.compile(
    r"^(?P<symbol>.+?)_(?P<timeframe>[A-Z0-9]+)_ml_dataset$"
)
ML_RUN_NAME_PATTERN = re.compile(
    r"^(?P<symbol>.+?)_(?P<timeframe>[A-Z0-9]+)_(?P<mode>[A-Za-z0-9_]+)_(?P<stamp>\d{8}T\d{6}Z)$"
)


def normalize_symbol_name(symbol: str | None) -> str:
    # Cleaning broker symbol variants into a normal display symbol
    if symbol is None:
        return ""

    raw = str(symbol).strip()
    if not raw:
        return raw

    upper = raw.upper()
    if "XAUUSD" in upper or "GOLD" in upper:
        return "XAUUSD"
    if "XAGUSD" in upper or "SILVER" in upper:
        return "XAGUSD"

    pair_match = re.match(r"^([A-Z]{6})", upper)
    if pair_match:
        return pair_match.group(1)

    return raw


@st.cache_data(show_spinner=False, ttl=5)
def list_run_directories(root: Path) -> list[Path]:
    #Returning run folders newest first
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


@st.cache_data(show_spinner=False, ttl=5)
def list_csv_files(root: Path) -> list[Path]:
    #Listing flat CSV artifacts in a directory
    if not root.exists():
        return []
    return sorted(root.glob("*.csv"), key=lambda path: path.name.lower())


@st.cache_data(show_spinner=False, ttl=5)
def read_json_file(path: Path) -> dict[str, Any]:
    #Reading a JSON artifact if it exists
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(show_spinner=False, ttl=5)
def read_csv_file(path: Path) -> pd.DataFrame:
    #Loading a full CSV artifact if it exists
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    #Flattening nested metric dictionaries for quick display
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def find_backtest_artifacts(run_dir: Path) -> dict[str, Path]:
    #Resolve the standard files saved by a backtest run
    return {
        "account_info": run_dir / "account_info.json",
        "trade_stats": run_dir / "trade_stats.json",
        "trades": run_dir / "trades.csv",
    }


def find_ml_artifacts(run_dir: Path) -> dict[str, Path]:
    #Resolve the standard files saved by an ML training run
    return {
        "summary": run_dir / "training_summary.json",
        "folds": run_dir / "walk_forward_fold_summary.csv",
        "importance": run_dir / "feature_importance.csv",
        "keep_ratio_sweep": run_dir / "validation_keep_ratio_sweep.csv",
        "validation_predictions": run_dir / "validation_predictions.csv",
        "test_predictions": run_dir / "test_predictions.csv",
        "validation_confluence_lift": run_dir / "validation_confluence_lift.csv",
        "test_confluence_lift": run_dir / "test_confluence_lift.csv",
        "model_bundle": run_dir / "model_bundle.pkl",
    }


@st.cache_data(show_spinner=False, ttl=5)
def load_model_bundle_metadata(path: Path) -> dict[str, Any]:
    #Extracting light metadata fields from the saved model bundle
    if not path.exists():
        return {}

    try:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
    except Exception:
        return {}

    if not isinstance(bundle, dict):
        return {}

    return {
        "target": bundle.get("target"),
        "keep_ratio": bundle.get("keep_ratio"),
        "selection_method": bundle.get("selection_method"),
        "walk_forward_folds": bundle.get("walk_forward_folds"),
        "feature_columns": bundle.get("feature_columns", []),
        "config": bundle.get("config", {}),
    }


@st.cache_data(show_spinner=False, ttl=5)
def latest_run_or_none(root: Path) -> Path | None:
    #Picking the newest run folder when available
    runs = list_run_directories(root)
    return runs[0] if runs else None


@st.cache_data(show_spinner=False, ttl=5)
def build_backtest_run_labels(run_dir_strings: tuple[str, ...]) -> dict[str, str]:
    #Building clean display labels for saved backtest runs
    grouped: dict[tuple[str, str, str, str], list[tuple[str, str]]] = {}
    labels: dict[str, str] = {}

    for run_dir_str in run_dir_strings:
        run_name = Path(run_dir_str).name
        match = BACKTEST_RUN_NAME_PATTERN.match(run_name)
        if not match:
            labels[run_dir_str] = run_name
            continue

        key = (
            normalize_symbol_name(match.group("symbol")),
            match.group("timeframe"),
            match.group("start"),
            match.group("end"),
        )
        grouped.setdefault(key, []).append((run_dir_str, match.group("stamp")))

    for (symbol, timeframe, start, end), entries in grouped.items():
        for version, (run_dir_str, _) in enumerate(sorted(entries, key=lambda item: item[1]), start=1):
            labels[run_dir_str] = f"{symbol}_{timeframe}_{start}_to_{end}_v{version}"

    return labels


@st.cache_data(show_spinner=False, ttl=5)
def build_ml_dataset_labels(dataset_strings: tuple[str, ...]) -> dict[str, str]:
    #Building clean display labels for saved ML dataset exports
    labels: dict[str, str] = {}
    for dataset_str in dataset_strings:
        dataset_name = Path(dataset_str).stem
        match = ML_DATASET_NAME_PATTERN.match(dataset_name)
        if not match:
            labels[dataset_str] = dataset_name.removesuffix("_ml_dataset")
            continue

        symbol = normalize_symbol_name(match.group("symbol"))
        timeframe = match.group("timeframe")
        labels[dataset_str] = f"{symbol}_{timeframe}"
    return labels


@st.cache_data(show_spinner=False, ttl=5)
def build_ml_run_labels(run_dir_strings: tuple[str, ...]) -> dict[str, str]:
    #Building clean display labels for saved ML training runs
    grouped: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
    labels: dict[str, str] = {}

    for run_dir_str in run_dir_strings:
        run_name = Path(run_dir_str).name
        match = ML_RUN_NAME_PATTERN.match(run_name)
        if not match:
            labels[run_dir_str] = run_name
            continue

        key = (
            normalize_symbol_name(match.group("symbol")),
            match.group("timeframe"),
            match.group("mode").lower(),
        )
        grouped.setdefault(key, []).append((run_dir_str, match.group("stamp")))

    for (symbol, timeframe, mode), entries in grouped.items():
        for version, (run_dir_str, _) in enumerate(sorted(entries, key=lambda item: item[1]), start=1):
            labels[run_dir_str] = f"{symbol}_{timeframe}_{mode}_v{version}"

    return labels


def _format_date_text(value: pd.Timestamp) -> str:
    #Format timestamps into a compact date string for dataset summaries
    if pd.isna(value):
        return ""

    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False, ttl=5)
def load_ml_dataset_summary(path: Path) -> dict[str, Any]:
    #Load a lightweight summary for one saved ML dataset
    frame = read_csv_file(path)
    if frame.empty:
        return {
            "rows": 0,
            "symbol": None,
            "timeframe": None,
            "date_range": None,
        }

    trade_date = pd.to_datetime(frame.get("Trade_Date"), errors="coerce")
    date_range = None
    if trade_date.notna().any():
        start = pd.Timestamp(trade_date.min())
        end = pd.Timestamp(trade_date.max())
        date_range = f"{_format_date_text(start)} to {_format_date_text(end)}"

    symbol = None
    if "Symbol" in frame.columns and frame["Symbol"].dropna().any():
        symbol = normalize_symbol_name(str(frame["Symbol"].dropna().iloc[0]))

    timeframe = None
    if "Timeframe" in frame.columns and frame["Timeframe"].dropna().any():
        timeframe = str(frame["Timeframe"].dropna().iloc[0])

    return {
        "rows": int(len(frame)),
        "symbol": symbol,
        "timeframe": timeframe,
        "date_range": date_range,
    }


def delete_run_directory(run_dir: Path) -> None:
    #delete a saved artifact folder and all files under it
    if not run_dir.exists() or not run_dir.is_dir():
        return

    def _make_writable(path: Path) -> None:
        #Clearing read-only flags before deletion attempts
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except FileNotFoundError:
            return
        except OSError:
            return

    def _delete_tree(path: Path) -> None:
        #Deleting children first so Windows can remove the parent folder cleanly
        for child in sorted(path.iterdir(), key=lambda item: len(item.parts), reverse=True):
            if child.is_dir():
                _delete_tree(child)
                _make_writable(child)
                child.rmdir()
            else:
                _make_writable(child)
                child.unlink()

        _make_writable(path)
        path.rmdir()

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            _delete_tree(run_dir)
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.35 * (attempt + 1))

    if last_error is not None:
        raise last_error


def delete_dataset_file(path: Path) -> None:
    #Removing one saved ML dataset export
    if not path.exists() or not path.is_file():
        return

    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    path.unlink()


def resolve_backtest_ml_dataset_path(run_dir: Path) -> Path | None:
    #Mapping one saved backtest run back to its exported ML dataset file
    match = BACKTEST_RUN_NAME_PATTERN.match(run_dir.name)
    if not match:
        return None

    symbol = match.group("symbol")
    timeframe = match.group("timeframe")
    return ML_DATASET_OUTPUT_DIR / f"{symbol}_{timeframe}_ml_dataset.csv"


def _has_other_backtest_runs_for_dataset(run_dir: Path, dataset_path: Path | None) -> bool:
    #Checking whether another saved run still depends on the same dataset export
    if dataset_path is None:
        return False

    backtest_root = run_dir.parent
    if not backtest_root.exists():
        return False

    for sibling in backtest_root.iterdir():
        if sibling == run_dir or not sibling.is_dir():
            continue
        if resolve_backtest_ml_dataset_path(sibling) == dataset_path:
            return True

    return False


def delete_backtest_run_and_dataset(run_dir: Path) -> dict[str, Path | None]:
    #Removing a backtest run and deleting the shared dataset only when no sibling run still uses it
    dataset_path = resolve_backtest_ml_dataset_path(run_dir)
    should_delete_dataset = dataset_path is not None and not _has_other_backtest_runs_for_dataset(run_dir, dataset_path)
    delete_run_directory(run_dir)

    if should_delete_dataset and dataset_path is not None and dataset_path.exists():
        delete_dataset_file(dataset_path)

    return {
        "run_dir": run_dir,
        "dataset_path": dataset_path,
    }


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    #Normalizing mixed boolean-like data into a clean bool series
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "none": False,
            "nan": False,
            "": False,
        }
    )
    return mapped.fillna(False).astype(bool)


def normalize_trade_frame(df: pd.DataFrame) -> pd.DataFrame:
    #Preparing the trade export for filtering and charting
    if df.empty:
        return df

    out = df.copy()

    for col in ("Trade_Date", "Exit_Time", "Signal_Time", "Last_Partial_Time"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    for col in (
        "PnL_Cash",
        "PnL_R",
        "Entry_Price",
        "Exit_Price",
        "Initial_Stop",
        "Final_Stop",
        "Take_Profit",
        "Risk_Price",
        "Risk_Pips",
        "Target_Pips",
        "Lots",
        "Balance_After",
        "Confluence_Count",
        "Commission_Cash",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in extract_confluence_columns(out):
        out[col] = _coerce_bool_series(out[col])

    for col in ("Label_Hit2R", "Label_RunnerUsed", "Runner_Qualified", "Runner_Active"):
        if col in out.columns:
            out[col] = _coerce_bool_series(out[col])

    out["Outcome_Group"] = classify_trade_outcomes(out)
    return out


@st.cache_data(show_spinner=False, ttl=5)
def load_backtest_bundle(run_dir: Path) -> dict[str, Any]:
    #Loading the saved backtest files for a single run
    artifacts = find_backtest_artifacts(run_dir)
    account_info = read_json_file(artifacts["account_info"])
    trade_stats = read_json_file(artifacts["trade_stats"])
    trades = normalize_trade_frame(read_csv_file(artifacts["trades"]))
    return {
        "run_dir": run_dir,
        "artifacts": artifacts,
        "account_info": account_info,
        "trade_stats": trade_stats,
        "trades": trades,
    }


@st.cache_data(show_spinner=False, ttl=5)
def load_ml_bundle(run_dir: Path) -> dict[str, Any]:
    #Loading the saved ML files for a single training run
    artifacts = find_ml_artifacts(run_dir)
    return {
        "run_dir": run_dir,
        "artifacts": artifacts,
        "summary": read_json_file(artifacts["summary"]),
        "folds": read_csv_file(artifacts["folds"]),
        "importance": read_csv_file(artifacts["importance"]),
        "keep_ratio_sweep": read_csv_file(artifacts["keep_ratio_sweep"]),
        "validation_predictions": read_csv_file(artifacts["validation_predictions"]),
        "test_predictions": read_csv_file(artifacts["test_predictions"]),
        "validation_confluence_lift": read_csv_file(artifacts["validation_confluence_lift"]),
        "test_confluence_lift": read_csv_file(artifacts["test_confluence_lift"]),
        "bundle_metadata": load_model_bundle_metadata(artifacts["model_bundle"]),
    }


def build_equity_curve(trades: pd.DataFrame, initial_balance: float) -> pd.DataFrame:
    #Reconstructing a simple realized equity curve from the trade report
    if trades.empty:
        return pd.DataFrame(columns=["step", "time", "equity"])

    out = trades.sort_values("Trade_Date", kind="stable").copy()
    if "Balance_After" in out.columns and out["Balance_After"].notna().any():
        equity = out["Balance_After"].ffill()
    else:
        pnl = pd.to_numeric(out.get("PnL_Cash", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
        equity = float(initial_balance) + pnl.cumsum()

    curve = pd.DataFrame(
        {
            "step": range(1, len(out) + 1),
            "time": out.get("Trade_Date", pd.Series(pd.NaT, index=out.index)),
            "equity": equity,
        }
    )

    start_row = pd.DataFrame(
        [{"step": 0, "time": curve["time"].min() if not curve["time"].dropna().empty else pd.NaT, "equity": float(initial_balance)}]
    )
    return pd.concat([start_row, curve], ignore_index=True)


def classify_trade_outcomes(trades: pd.DataFrame) -> pd.Series:
    #Bucketing each trade into win, loss, or breakeven
    if trades.empty:
        return pd.Series(dtype="object")

    if "Outcome" in trades.columns:
        text = trades["Outcome"].astype(str).str.strip().str.lower()
        mapped = text.map({"win": "Win", "loss": "Loss", "breakeven": "Breakeven"})
        if mapped.notna().all():
            return mapped

    pnl_r = pd.to_numeric(trades.get("PnL_R", pd.Series(0.0, index=trades.index)), errors="coerce").fillna(0.0)
    pnl_cash = pd.to_numeric(trades.get("PnL_Cash", pd.Series(0.0, index=trades.index)), errors="coerce").fillna(0.0)
    pnl = pnl_r.where(pnl_r.ne(0.0), pnl_cash)

    outcome = np.where(pnl > 0, "Win", np.where(pnl < 0, "Loss", "Breakeven"))
    return pd.Series(outcome, index=trades.index, dtype="object")


def extract_confluence_columns(trades: pd.DataFrame) -> list[str]:
    #Keeping the known confluence columns in a stable order
    if trades.empty:
        return []

    ordered = [col for col in BACKTEST_CONFLUENCE_COLUMNS if col in trades.columns]
    extras = [
        col
        for col in trades.columns
        if col.endswith("_Valid")
        and col not in ordered
        and not col.startswith("Label_")
    ]
    return ordered + sorted(extras)


def extract_optional_confluence_columns(trades: pd.DataFrame) -> list[str]:
    #Keeping only non-core confluences for analysis views
    return [col for col in extract_confluence_columns(trades) if col not in CORE_CONFLUENCE_COLUMNS]


def format_confluence_name(name: str) -> str:
    #Converting saved column names into cleaner dashboard labels
    if name in CONFLUENCE_LABELS:
        return CONFLUENCE_LABELS[name]

    return name.replace("_Valid", "").replace("_", " ").strip()


def _trade_metric_series(trades: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    #Returning aligned trade outcome, R, and cash series
    outcome = trades["Outcome_Group"] if "Outcome_Group" in trades.columns else classify_trade_outcomes(trades)
    pnl_r = pd.to_numeric(trades.get("PnL_R", pd.Series(0.0, index=trades.index)), errors="coerce").fillna(0.0)
    pnl_cash = pd.to_numeric(trades.get("PnL_Cash", pd.Series(0.0, index=trades.index)), errors="coerce").fillna(0.0)
    return outcome, pnl_r, pnl_cash


def _subset_performance(mask: pd.Series, outcome: pd.Series, pnl_r: pd.Series, pnl_cash: pd.Series) -> dict[str, float]:
    #omputing trade metrics for one filtered subset
    trade_count = int(mask.sum())
    if trade_count == 0:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "expectancy_r": 0.0,
            "pnl_cash": 0.0,
        }

    subset_outcome = outcome.loc[mask]
    subset_pnl_r = pnl_r.loc[mask]
    subset_pnl_cash = pnl_cash.loc[mask]
    return {
        "trade_count": trade_count,
        "win_rate": float((subset_outcome == "Win").mean() * 100.0),
        "expectancy_r": float(subset_pnl_r.mean()),
        "pnl_cash": float(subset_pnl_cash.sum()),
    }


def summarize_confluence_summary(trades: pd.DataFrame) -> pd.DataFrame:
    #Summarizing per-confluence count and active-trade performance
    if trades.empty:
        return pd.DataFrame()

    outcome, pnl_r, pnl_cash = _trade_metric_series(trades)
    total_trades = len(trades)
    rows: list[dict[str, Any]] = []

    for col in extract_optional_confluence_columns(trades):
        mask = _coerce_bool_series(trades[col])
        with_metrics = _subset_performance(mask, outcome, pnl_r, pnl_cash)
        rows.append(
            {
                "confluence": format_confluence_name(col),
                "active_trade_count": with_metrics["trade_count"],
                "inactive_trade_count": int(total_trades - with_metrics["trade_count"]),
                "active_rate_pct": float((with_metrics["trade_count"] / total_trades) * 100.0) if total_trades else 0.0,
                "win_rate_when_active": with_metrics["win_rate"],
                "avg_r_when_active": with_metrics["expectancy_r"],
                "pnl_when_active": with_metrics["pnl_cash"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["active_trade_count", "win_rate_when_active", "avg_r_when_active"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def summarize_confluence_impact(trades: pd.DataFrame) -> pd.DataFrame:
    #Comparing each confluence with-vs-without on executed trades
    if trades.empty:
        return pd.DataFrame()

    outcome, pnl_r, pnl_cash = _trade_metric_series(trades)
    rows: list[dict[str, Any]] = []

    for col in extract_optional_confluence_columns(trades):
        with_mask = _coerce_bool_series(trades[col])
        without_mask = ~with_mask
        with_metrics = _subset_performance(with_mask, outcome, pnl_r, pnl_cash)
        without_metrics = _subset_performance(without_mask, outcome, pnl_r, pnl_cash)

        rows.append(
            {
                "confluence": format_confluence_name(col),
                "trades_with": with_metrics["trade_count"],
                "win_rate_with": with_metrics["win_rate"],
                "expectancy_with": with_metrics["expectancy_r"],
                "pnl_with": with_metrics["pnl_cash"],
                "trades_without": without_metrics["trade_count"],
                "win_rate_without": without_metrics["win_rate"],
                "expectancy_without": without_metrics["expectancy_r"],
                "pnl_without": without_metrics["pnl_cash"],
                "win_rate_delta_pp": with_metrics["win_rate"] - without_metrics["win_rate"],
                "expectancy_delta_r": with_metrics["expectancy_r"] - without_metrics["expectancy_r"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["expectancy_delta_r", "win_rate_delta_pp", "trades_with"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def summarize_confluence_sets(trades: pd.DataFrame) -> pd.DataFrame:
    #Ranking optional confluence sets discovered from winning trades and scoring them on non-breakeven outcomes
    if trades.empty:
        return pd.DataFrame()

    optional_columns = extract_optional_confluence_columns(trades)
    if not optional_columns:
        return pd.DataFrame()

    trade_signature = trades[optional_columns].apply(
        lambda row: tuple(col for col in optional_columns if bool(row[col])),
        axis=1,
    )
    outcome, pnl_r, pnl_cash = _trade_metric_series(trades)
    positive_mask = ((outcome == "Win") | (pnl_r > 0) | (pnl_cash > 0)) & trade_signature.map(bool)
    if not positive_mask.any():
        return pd.DataFrame()

    evaluation_mask = outcome != "Breakeven"
    evaluation_signature = trade_signature.loc[evaluation_mask]
    evaluation_outcome = outcome.loc[evaluation_mask]
    evaluation_pnl_r = pnl_r.loc[evaluation_mask]
    evaluation_pnl_cash = pnl_cash.loc[evaluation_mask]
    rows: list[dict[str, Any]] = []

    for active_set in trade_signature.loc[positive_mask].drop_duplicates():
        mask = evaluation_signature == active_set
        metrics = _subset_performance(mask, evaluation_outcome, evaluation_pnl_r, evaluation_pnl_cash)
        readable = [format_confluence_name(col) for col in active_set]
        rows.append(
            {
                "confluence_set": ", ".join(readable),
                "set_size": len(active_set),
                "trade_count": metrics["trade_count"],
                "win_rate": metrics["win_rate"],
                "expectancy_r": metrics["expectancy_r"],
                "pnl_cash": metrics["pnl_cash"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["expectancy_r", "win_rate", "trade_count", "set_size"],
        ascending=[False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def _align_timestamp_to_series(series: pd.Series, value: pd.Timestamp | None) -> pd.Timestamp | None:
    #Align a filter timestamp to the timezone of trade series
    if value is None:
        return None

    ts = pd.Timestamp(value)
    series_tz = getattr(series.dt, "tz", None)
    value_tz = ts.tzinfo

    if series_tz is not None:
        if value_tz is None:
            return ts.tz_localize(series_tz)
        return ts.tz_convert(series_tz)

    if value_tz is not None:
        return ts.tz_localize(None)

    return ts


def _expand_date_filter_end(value: pd.Timestamp | None) -> pd.Timestamp | None:
    #Expanding the date only end filters so the full selected day stays included
    if value is None:
        return None

    ts = pd.Timestamp(value)
    if ts == ts.normalize():
        return ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def filter_trades(
    trades: pd.DataFrame,
    *,
    symbol: str = "All",
    timeframe: str = "All",
    side: str = "All",
    outcome: str = "All",
    date_from: pd.Timestamp | None = None,
    date_to: pd.Timestamp | None = None,
    confluence_name: str = "Any",
    confluence_state: str = "Any",
    runner_state: str = "Any",
    min_r: float | None = None,
    max_r: float | None = None,
) -> pd.DataFrame:
    #Apply dashboard trade-level filters
    if trades.empty:
        return trades

    filtered = trades.copy()

    if symbol != "All" and "Symbol" in filtered.columns:
        filtered = filtered[filtered["Symbol"] == symbol]
    if timeframe != "All" and "Timeframe" in filtered.columns:
        filtered = filtered[filtered["Timeframe"] == timeframe]
    if side != "All" and "Position" in filtered.columns:
        filtered = filtered[filtered["Position"].astype(str).str.lower() == side.lower()]
    if outcome != "All" and "Outcome_Group" in filtered.columns:
        filtered = filtered[filtered["Outcome_Group"] == outcome]

    if date_from is not None and "Trade_Date" in filtered.columns:
        start_ts = _align_timestamp_to_series(filtered["Trade_Date"], pd.Timestamp(date_from))
        filtered = filtered[filtered["Trade_Date"] >= start_ts]
    if date_to is not None and "Trade_Date" in filtered.columns:
        inclusive_end = _expand_date_filter_end(date_to)
        end_ts = _align_timestamp_to_series(filtered["Trade_Date"], inclusive_end)
        filtered = filtered[filtered["Trade_Date"] <= end_ts]

    if confluence_name != "Any" and confluence_name in filtered.columns and confluence_state != "Any":
        series = _coerce_bool_series(filtered[confluence_name])
        filtered = filtered[series] if confluence_state == "Valid" else filtered[~series]

    if runner_state != "Any":
        runner_series = pd.Series(False, index=filtered.index)
        for col in ("Label_RunnerUsed", "Runner_Qualified", "Runner_Active"):
            if col in filtered.columns:
                runner_series = runner_series | _coerce_bool_series(filtered[col])
        filtered = filtered[runner_series] if runner_state == "Used" else filtered[~runner_series]

    pnl_r = pd.to_numeric(filtered.get("PnL_R", pd.Series(0.0, index=filtered.index)), errors="coerce")
    if min_r is not None:
        filtered = filtered[pnl_r >= float(min_r)]
    if max_r is not None:
        filtered = filtered[pnl_r <= float(max_r)]

    return filtered.sort_values("Trade_Date", kind="stable") if "Trade_Date" in filtered.columns else filtered


def trade_detail_sections(row: pd.Series) -> dict[str, Any]:
    #Grouping one trade into the panels shown in the details view
    active_confluences = [
        col
        for col in BACKTEST_CONFLUENCE_COLUMNS
        if col in row.index and bool(_coerce_bool_series(pd.Series([row.get(col)])).iat[0])
    ]

    structure_keys = [
        "bos_up",
        "bos_down",
        "choch_up",
        "choch_down",
        "mss_up",
        "mss_down",
        "trend_state",
    ]
    zone_keys = [
        "demand_zone_entry",
        "demand_zone_stop",
        "demand_zone_bottom",
        "demand_zone_top",
        "demand_zone_width_pips",
        "demand_zone_stop_pips",
        "supply_zone_entry",
        "supply_zone_stop",
        "supply_zone_bottom",
        "supply_zone_top",
        "supply_zone_width_pips",
        "supply_zone_stop_pips",
    ]

    return {
        "trade": {
            "entry_time": row.get("Trade_Date"),
            "exit_time": row.get("Exit_Time"),
            "side": row.get("Position"),
            "entry_price": row.get("Entry_Price"),
            "stop": row.get("Initial_Stop"),
            "target": row.get("Take_Profit"),
            "final_pnl_cash": row.get("PnL_Cash"),
            "final_pnl_r": row.get("PnL_R"),
            "hit_2r": row.get("Label_Hit2R"),
            "runner_used": row.get("Label_RunnerUsed"),
        },
        "confluences": active_confluences,
        "structure": {key: row.get(key) for key in structure_keys if key in row.index},
        "zone": {key: row.get(key) for key in zone_keys if key in row.index},
    }
