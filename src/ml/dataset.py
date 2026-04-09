from __future__ import annotations

from pathlib import Path

import pandas as pd


LABEL_COLUMNS = [
    "Label_PositivePnL",
    "Label_Hit2R",
    "Label_RunnerUsed",
]

OUTCOME_COLUMNS = [
    "PnL_Cash",
    "PnL_R",
    "Outcome",
    "Duration_Bars",
    "Commission_Cash",
]

JOIN_KEY_COLUMNS = [
    "Symbol",
    "Timeframe",
    "Trade_Date",
    "Entry_Bar",
    "Position",
]


def normalize_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    #Normalizing timestamps and identifier columns for stable joins
    if "Trade_Date" in out.columns:
        trade_date = pd.to_datetime(out["Trade_Date"], errors="coerce", utc=True)
        out["Trade_Date"] = trade_date.dt.tz_convert(None)

    for col in ("Symbol", "Timeframe", "Position"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.upper()

    if "Entry_Bar" in out.columns:
        out["Entry_Bar"] = pd.to_numeric(out["Entry_Bar"], errors="coerce").astype("Int64")

    return out


def _read_trade_report(path: Path) -> pd.DataFrame:
    #Loading only the trade-report columns needed for ML enrichment
    columns = JOIN_KEY_COLUMNS + [col for col in OUTCOME_COLUMNS if col not in JOIN_KEY_COLUMNS]
    report = pd.read_csv(path, usecols=lambda col: col in columns)
    report = normalize_join_keys(report)
    report["Backtest_Trade_Source"] = path.as_posix()
    return report.drop_duplicates(subset=JOIN_KEY_COLUMNS, keep="last")


def _find_matching_trade_report(
    dataset_path: Path,
    df: pd.DataFrame,
    backtest_dir: Path,
) -> pd.DataFrame:
    #Inferring the dataset symbol/timeframe before searching saved backtests
    symbol = (
        str(df["Symbol"].dropna().iloc[0]).strip().upper()
        if "Symbol" in df.columns and not df["Symbol"].dropna().empty
        else None
    )
    timeframe = (
        str(df["Timeframe"].dropna().iloc[0]).strip().upper()
        if "Timeframe" in df.columns and not df["Timeframe"].dropna().empty
        else None
    )

    if not symbol or not timeframe:
        raise ValueError(
            f"{dataset_path.name} is missing Symbol/Timeframe columns required for trade-report enrichment."
        )

    #Scoring candidate trade reports by join coverage and shape match
    dataset_keys = normalize_join_keys(df[JOIN_KEY_COLUMNS]).drop_duplicates()
    candidates = sorted(backtest_dir.glob(f"{symbol}_{timeframe}_*/trades.csv"))
    if not candidates:
        raise ValueError(f"No backtest trade reports found for {symbol} {timeframe} under {backtest_dir}.")

    best_report: pd.DataFrame | None = None
    best_score: tuple[int, int, int, str] | None = None

    for path in candidates:
        report = _read_trade_report(path)
        merged = dataset_keys.merge(report[JOIN_KEY_COLUMNS], on=JOIN_KEY_COLUMNS, how="left", indicator=True)
        match_count = int((merged["_merge"] == "both").sum())
        score = (
            int(match_count == len(dataset_keys)),
            match_count,
            int(len(report) == len(df)),
            path.parent.name,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_report = report

    if best_report is None:
        raise ValueError(f"Unable to match {dataset_path.name} to a saved trade report.")

    #Merging the best matching report back into the dataset
    merged = normalize_join_keys(df).merge(
        best_report,
        on=JOIN_KEY_COLUMNS,
        how="left",
        suffixes=("", "_report"),
    )
    if merged["PnL_Cash"].isna().any() or merged["PnL_R"].isna().any():
        raise ValueError(f"Failed to enrich all outcomes for {dataset_path.name}.")
    return merged


def _load_dataset_file(path: Path, *, backtest_dir: Path, target: str) -> pd.DataFrame:
    #Loading one ML dataset file and enriching outcomes if needed
    df = pd.read_csv(path).copy()
    df["Dataset_Source"] = path.name
    df = normalize_join_keys(df)

    if "PnL_Cash" not in df.columns or "PnL_R" not in df.columns:
        df = _find_matching_trade_report(path, df, backtest_dir)

    if target not in df.columns:
        raise ValueError(f"{path.name} is missing target column {target!r}.")

    #Keeping only rows with a usable target label
    df = df[df[target].notna()].copy()
    df[target] = pd.to_numeric(df[target], errors="coerce").astype("Int64")
    df["PnL_Cash"] = pd.to_numeric(df["PnL_Cash"], errors="coerce")
    df["PnL_R"] = pd.to_numeric(df["PnL_R"], errors="coerce")

    if df["PnL_Cash"].isna().any() or df["PnL_R"].isna().any():
        raise ValueError(f"{path.name} has invalid PnL_Cash/PnL_R values.")

    return df


def load_training_data(
    *,
    data_dir: str | Path,
    pattern: str,
    backtest_dir: str | Path,
    target: str,
) -> pd.DataFrame:
    #Loading and combining all dataset files that match the requested pattern
    paths = sorted(Path(data_dir).glob(pattern))
    if not paths:
        raise ValueError(f"No datasets matched pattern {pattern!r} under {data_dir}.")

    frames = [
        _load_dataset_file(path, backtest_dir=Path(backtest_dir), target=target)
        for path in paths
    ]
    df = pd.concat(frames, ignore_index=True)

    #Validating chronological split requirements before training
    if "Trade_Date" not in df.columns:
        raise ValueError("Trade_Date is required for chronological splitting.")
    if df["Trade_Date"].isna().any():
        raise ValueError("Trade_Date contains invalid or missing values after loading.")

    sort_cols = [col for col in ("Trade_Date", "Entry_Bar", "Symbol", "Timeframe") if col in df.columns]
    return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
