from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ML_SCHEMA_VERSION = 1

ML_LABEL_COLUMNS = [
    "Label_PositivePnL",
    "Label_Hit2R",
    "Label_RunnerUsed",
]

ML_OUTCOME_COLUMNS = [
    "PnL_Cash",
    "PnL_R",
    "Outcome",
    "Duration_Bars",
    "Commission_Cash",
]

ML_METADATA_COLUMNS = [
    "Symbol",
    "Timeframe",
    "Trade_Date",
    "Entry_Bar",
    "Position",
]

ML_LEAKAGE_COLUMNS = {
    "Exit_Time",
    "Exit_Bar",
    "Exit_Price",
    "Final_Stop",
    "Partial_At_2R",
    "Partial_Price",
    "Runner_Active",
    "Runner_Stop_Price",
    "Runner_Exit_Reason",
    "Final_Exit_Type",
    "Balance_After",
    "Last_Partial_Reason",
    "Last_Partial_Time",
    "Exit_Adaptive_Spread_Pips",
    "Exit_Adaptive_Slippage_Pips",
    "Exit_Spread_Session_Mult",
    "Exit_Spread_Volatility_Mult",
    "exit_time",
    "close_bar",
    "exit_price",
    "remaining_size",
    "breakeven_moved",
    "is_breakeven_exit",
    "partial_taken",
    "partial_price",
    "partial_bar",
    "runner_active",
    "runner_stop_price",
    "runner_exit_reason",
    "final_exit_type",
    "result_r",
    "pnl",
    "commission_paid",
    "balance_after",
}


class BacktestEngineReportingMixin:
    def _record_equity_point(self, df: pd.DataFrame, i: int) -> None:
        #Recording realized balance at the current bar
        point = {"bar": i, "equity": self.balance}
        if "time" in df.columns:
            point["time"] = df["time"].iat[i]
        self.equity_points.append(point)

    def get_equity_curve_frame(self) -> pd.DataFrame:
        if not self.equity_points:
            return pd.DataFrame(columns=["bar", "time", "equity"])
        frame = pd.DataFrame(self.equity_points)
        if "time" not in frame.columns:
            frame["time"] = pd.NaT
        return frame[["bar", "time", "equity"]]

    def get_trade_report_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()

        #Flattening stored trade reports into a single table
        rows = [dict(t.report or {}) for t in self.trades]
        df = pd.DataFrame(rows)

        base_cols = [
            "Symbol",
            "Timeframe",
            "Trade_Date",
            "Exit_Time",
            "Entry_Bar",
            "Exit_Bar",
            "Position",
            "Entry_Price",
            "Exit_Price",
            "Initial_Stop",
            "Final_Stop",
            "Take_Profit",
            "Risk_Price",
            "Risk_Pips",
            "Target_Price_Distance",
            "Target_Pips",
            "Size",
            "Lots",
            "PnL_Cash",
            "PnL_R",
            "Outcome",
            "Duration_Bars",
            "Confluence_Count",
            "Partial_At_2R",
            "Partial_Price",
            "Runner_Qualified",
            "Runner_Active",
            "Runner_Reference_Level",
            "Runner_Target_Price",
            "Runner_Target_RR",
            "Runner_Stop_Price",
            "Runner_Exit_Reason",
            "Commission_Per_Lot",
            "Final_Exit_Type",
            "Label_PositivePnL",
            "Label_Hit2R",
            "Label_RunnerUsed",
        ]
        remaining = [c for c in df.columns if c not in base_cols]
        return df[base_cols + remaining]

    def export_trade_report(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.get_trade_report_frame().to_csv(output_path, index=False)
        return output_path

    def _build_ml_safe_frame(self) -> pd.DataFrame:
        df = self.get_trade_report_frame().copy()
        if df.empty:
            return df

        #Keeping labels/outcomes while dropping exit-time leakage fields
        keep_labels = [c for c in ML_LABEL_COLUMNS if c in df.columns]
        keep_meta = [c for c in ML_METADATA_COLUMNS if c in df.columns]
        keep_outcomes = [c for c in ML_OUTCOME_COLUMNS if c in df.columns]

        drop_cols = []
        for col in df.columns:
            if col in keep_labels:
                continue
            if col in keep_outcomes:
                continue
            if col in ML_LEAKAGE_COLUMNS:
                drop_cols.append(col)

        df = df.drop(columns=drop_cols, errors="ignore")

        #Normalizing label columns into nullable integer form
        for col in keep_labels:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        #Converting boolean fields into ML friendly integer flags
        for col in df.columns:
            if col in keep_labels:
                continue

            series = df[col]

            if pd.api.types.is_bool_dtype(series):
                df[col] = series.astype("Int64")
                continue

            non_na = series.dropna()
            if len(non_na) > 0 and non_na.isin([True, False]).all():
                df[col] = series.map(
                    lambda x: 1 if x is True else (0 if x is False else pd.NA)
                ).astype("Int64")

        df = df.replace([float("inf"), float("-inf")], pd.NA)

        df["ML_Schema_Version"] = ML_SCHEMA_VERSION

        ordered = (
            keep_meta
            + ["ML_Schema_Version"]
            + [c for c in df.columns if c not in keep_meta and c not in keep_outcomes and c not in keep_labels and c != "ML_Schema_Version"]
            + keep_outcomes
            + keep_labels
        )
        return df[ordered]

    def export_ml_dataset(self, output_path: str | Path) -> Path:
        #Saving a leakage safe ML dataset export
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self._build_ml_safe_frame()

        if df.empty:
            raise ValueError("ML dataset export failed: no trades available.")

        #Validating the required label columns before export
        for col in ML_LABEL_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing label column: {col}")
            if df[col].isna().all():
                raise ValueError(f"Label column {col} is entirely NaN")

        df.to_csv(output_path, index=False)
        return output_path

    def export_trade_log(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "direction",
            "entry_time",
            "exit_time",
            "open_bar",
            "close_bar",
            "entry",
            "stop",
            "initial_stop",
            "target",
            "exit_price",
            "size",
            "remaining_size",
            "risk",
            "breakeven_moved",
            "is_breakeven_exit",
            "partial_taken",
            "partial_price",
            "partial_bar",
            "runner_qualified",
            "runner_active",
            "runner_reference_level",
            "runner_target_price",
            "runner_target_rr",
            "runner_stop_price",
            "runner_exit_reason",
            "final_exit_type",
            "result_r",
            "pnl",
            "commission_paid",
            "balance_after",
            "entry_slippage_pips",
            "entry_spread_pips",
        ]

        #Writing one row per finalized trade state
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for t in self.trades:
                writer.writerow(
                    {
                        "direction": t.direction,
                        "entry_time": getattr(t, "entry_time", ""),
                        "exit_time": getattr(t, "exit_time", ""),
                        "open_bar": t.open_bar,
                        "close_bar": t.close_bar,
                        "entry": t.entry,
                        "stop": t.stop,
                        "initial_stop": getattr(t, "initial_stop", ""),
                        "target": t.target,
                        "exit_price": getattr(t, "exit_price", ""),
                        "size": t.size,
                        "remaining_size": getattr(t, "remaining_size", 0.0),
                        "risk": t.risk,
                        "breakeven_moved": t.breakeven_moved,
                        "is_breakeven_exit": getattr(t, "is_breakeven_exit", False),
                        "partial_taken": getattr(t, "partial_taken", False),
                        "partial_price": getattr(t, "partial_price", ""),
                        "partial_bar": getattr(t, "partial_bar", ""),
                        "runner_qualified": getattr(t, "runner_qualified", False),
                        "runner_active": getattr(t, "runner_active", False),
                        "runner_reference_level": getattr(t, "runner_reference_level", ""),
                        "runner_target_price": getattr(t, "runner_target_price", ""),
                        "runner_target_rr": getattr(t, "runner_target_rr", ""),
                        "runner_stop_price": getattr(t, "runner_stop_price", ""),
                        "runner_exit_reason": getattr(t, "runner_exit_reason", ""),
                        "final_exit_type": getattr(t, "final_exit_type", ""),
                        "result_r": t.result,
                        "pnl": getattr(t, "pnl", 0.0),
                        "commission_paid": getattr(t, "commission_paid", 0.0),
                        "balance_after": getattr(t, "balance_after", ""),
                        "entry_slippage_pips": getattr(t, "entry_slippage_pips", ""),
                        "entry_spread_pips": getattr(t, "entry_spread_pips", ""),
                    }
                )

        return output_path

    def get_confluence_trade_counts(self) -> dict[str, int]:
        #Counting how often each reported confluence was active on executed trades
        report_df = self.get_trade_report_frame()
        if report_df.empty:
            return {col: 0 for col in self.REPORT_CONFLUENCE_COLUMNS}

        counts: dict[str, int] = {}
        for col in self.REPORT_CONFLUENCE_COLUMNS:
            counts[col] = int(report_df[col].fillna(False).astype(bool).sum()) if col in report_df.columns else 0
        return counts

    def _max_drawdown(self) -> tuple[float, float]:
        #Measuring peak-to-trough drawdown from realized equity peaks
        equity = self.get_equity_curve_frame()
        if equity.empty:
            return 0.0, 0.0

        curve = equity["equity"].astype(float)
        running_peak = curve.cummax()
        drawdown_abs = running_peak - curve
        drawdown_pct = drawdown_abs / running_peak.replace(0.0, pd.NA)

        max_dd_abs = float(drawdown_abs.max()) if len(drawdown_abs) else 0.0
        max_dd_pct = float(drawdown_pct.fillna(0.0).max() * 100.0) if len(drawdown_pct) else 0.0
        return max_dd_abs, max_dd_pct

    def _initial_balance_drawdown(self) -> tuple[float, float]:
        #Measuring drawdown strictly against the starting balance
        equity = self.get_equity_curve_frame()
        if equity.empty or self.initial_balance <= 0:
            return 0.0, 0.0

        curve = equity["equity"].astype(float)
        drawdown_abs = (self.initial_balance - curve).clip(lower=0.0)
        drawdown_pct = drawdown_abs / float(self.initial_balance)

        max_dd_abs = float(drawdown_abs.max()) if len(drawdown_abs) else 0.0
        max_dd_pct = float(drawdown_pct.fillna(0.0).max() * 100.0) if len(drawdown_pct) else 0.0
        return max_dd_abs, max_dd_pct

    def _results(self) -> dict:
        #Aggregating the final realized backtest summary
        bes = [t for t in self.trades if getattr(t, "is_breakeven_exit", False)]
        wins = [t for t in self.trades if t.pnl > 0 and not getattr(t, "is_breakeven_exit", False)]
        losses = [t for t in self.trades if t.pnl < 0 and not getattr(t, "is_breakeven_exit", False)]

        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        expectancy_r = (sum(t.result for t in self.trades) / len(self.trades)) if self.trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        max_dd_abs, max_dd_pct = self._max_drawdown()
        initial_dd_abs, initial_dd_pct = self._initial_balance_drawdown()

        return {
            "initial_balance": self.initial_balance,
            "balance": self.balance,
            "net_pnl": self.balance - self.initial_balance,
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "breakevens": len(bes),
            "win_rate": (
                (len(wins) / (len(wins) + len(losses))) * 100.0
                if (len(wins) + len(losses))
                else 0.0
            ),
            "expectancy_r": expectancy_r,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "initial_balance_max_drawdown": initial_dd_abs,
            "initial_balance_max_drawdown_pct": initial_dd_pct,
            "max_drawdown": max_dd_abs,
            "max_drawdown_pct": max_dd_pct,
        }
