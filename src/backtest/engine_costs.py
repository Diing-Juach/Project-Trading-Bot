from __future__ import annotations

import pandas as pd

from src.backtest.models import Trade


def _find_snapshot_column(df: pd.DataFrame, col: str, aliases: dict[str, list[str]]) -> str:
    #Finding the first available column name for a snapshot field
    for candidate in aliases.get(col, [col]):
        if candidate in df.columns:
            return candidate
    return col


def _read_snapshot_number(df: pd.DataFrame, i: int, col: str):
    #Reading a numeric snapshot value if the column exists
    if col not in df.columns:
        return None
    value = df[col].iat[i]
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _read_snapshot_flag(df: pd.DataFrame, i: int, col: str, default: bool = False) -> bool:
    #Reading a boolean-like snapshot value with a fallback default
    if col not in df.columns:
        return default
    value = df[col].iat[i]
    if pd.isna(value):
        return default
    return bool(value)


class BacktestEngineCostsMixin:
    def _set_baseline_atr(self, df: pd.DataFrame) -> None:
        #Capturing a baseline ATR median for the spread/slippage
        atr_values: list[float] = []
        for col in ("atr_feature", "atr"):
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                series = series[series > 0]
                if not series.empty:
                    atr_values = series.tolist()
                    break

        self._baseline_atr_value = float(pd.Series(atr_values).median()) if atr_values else None

    def _bool_from_col(self, df: pd.DataFrame, i: int, col: str) -> bool:
        if col not in df.columns:
            return False
        value = df[col].iat[i]
        if pd.isna(value):
            return False
        return bool(value)

    def _get_atr_value(self, df: pd.DataFrame, i: int):
        #Reading the best available ATR value for the current bar
        for col in ("atr_feature", "atr"):
            if col in df.columns:
                v = df[col].iat[i]
                if pd.notna(v):
                    return float(v)
        for col in df.columns:
            if "atr" in col.lower():
                v = df[col].iat[i]
                if pd.notna(v):
                    return float(v)
        return None

    def _session_spread_multiplier(self, df: pd.DataFrame, i: int) -> float:
        #Adjusting spread/slippage by session liquidity 
        if self._bool_from_col(df, i, "is_london_newyork_overlap"):
            return self.session_liquid_mult
        if self._bool_from_col(df, i, "is_london_session") or self._bool_from_col(df, i, "is_newyork_session"):
            return self.session_neutral_mult
        if self._bool_from_col(df, i, "is_asia_session"):
            return self.session_thin_mult
        return self.session_neutral_mult

    def _volatility_spread_multiplier(self, df: pd.DataFrame, i: int) -> float:
        # Adjusting the spread/slippage by current ATR relative to baseline
        atr_value = self._get_atr_value(df, i)
        baseline_atr = self._baseline_atr_value

        if atr_value is None or atr_value <= 0:
            return 1.0
        if baseline_atr is None or baseline_atr <= 0:
            return 1.0

        atr_ratio = atr_value / baseline_atr

        if atr_ratio >= self.spread_vol_high_threshold:
            return self.spread_vol_high_mult
        if atr_ratio >= self.spread_vol_medium_threshold:
            return self.spread_vol_medium_mult
        return 1.0

    def _resolve_spread_pips(self, df: pd.DataFrame, i: int) -> float:
        #Combining session and volatility multipliers into the active spread
        session_mult = self._session_spread_multiplier(df, i)
        vol_mult = self._volatility_spread_multiplier(df, i)
        spread = self.base_spread_pips * session_mult * vol_mult
        return max(float(spread), 0.0)

    def _resolve_slippage_pips(self, df: pd.DataFrame, i: int) -> float:
        slippage = float(self.slippage_pips)

        if self.use_session_adaptive_slippage:
            slippage *= self._session_spread_multiplier(df, i)

        if self.use_spread_adaptive_slippage:
            slippage *= self._volatility_spread_multiplier(df, i)

        min_slippage = self.slippage_pips * self.slippage_min_mult
        max_slippage = self.slippage_pips * self.slippage_max_mult

        return max(float(min_slippage), min(float(slippage), float(max_slippage)))

    def _pips_to_price(self, value_pips: float) -> float:
        return float(value_pips) * self.pip_size

    def _make_trade_snapshot(
        self,
        df: pd.DataFrame,
        i: int,
        direction: str,
        stop: float,
        target: float,
        execution_bar: int | None = None,
    ) -> dict:
        is_long = direction == "long"

        #Capturing trade metadata and active adaptive execution settings
        entry_bar = i if execution_bar is None else int(execution_bar)
        atr_valid_col = "atr_long_valid" if is_long else "atr_short_valid"
        session_mult = self._session_spread_multiplier(df, i)
        vol_mult = self._volatility_spread_multiplier(df, i)
        adaptive_spread = self._resolve_spread_pips(df, i)
        adaptive_slippage = self._resolve_slippage_pips(df, i)

        #Storing the core trade context used by reporting and ML exports
        snapshot = {
            "Symbol": self.symbol,
            "Timeframe": self.timeframe,
            "Trade_Date": df["time"].iat[entry_bar] if "time" in df.columns else entry_bar,
            "Signal_Time": df["time"].iat[i] if "time" in df.columns else i,
            "Signal_Bar": i,
            "Entry_Bar": entry_bar,
            "Position": "Long" if is_long else "Short",
            "Stop_Loss": float(stop),
            "Take_Profit": float(target),
            "SupplyAndDemand_Valid": _read_snapshot_flag(
                df,
                i,
                "supply_demand_long_valid" if is_long else "supply_demand_short_valid",
            ),
            "Atr_Valid": _read_snapshot_flag(df, i, atr_valid_col, default=True),
            "Market_Structure_Valid": _read_snapshot_flag(
                df,
                i,
                "market_structure_long_valid" if is_long else "market_structure_short_valid",
            ),
            "Session_Liquidity_Valid": _read_snapshot_flag(
                df,
                i,
                "session_liquidity_long_valid" if is_long else "session_liquidity_short_valid",
            ),
            "Volume_Profile_Valid": _read_snapshot_flag(
                df,
                i,
                "volume_profile_long_valid" if is_long else "volume_profile_short_valid",
            ),
            "Imbalance_Valid": _read_snapshot_flag(df, i, "imbalance_long_valid" if is_long else "imbalance_short_valid"),
            "Fibonacci_Valid": _read_snapshot_flag(df, i, "fibonacci_long_valid" if is_long else "fibonacci_short_valid"),
            "Liquidity_Valid": _read_snapshot_flag(df, i, "liquidity_long_valid" if is_long else "liquidity_short_valid"),
            "Support_Resistance_Valid": _read_snapshot_flag(
                df,
                i,
                "support_resistance_long_valid" if is_long else "support_resistance_short_valid",
            ),
            "SR_Runner_Reference_Level": _read_snapshot_number(
                df,
                i,
                "sr_long_reference_level" if is_long else "sr_short_reference_level",
            ),
            "SR_Runner_Target": _read_snapshot_number(df, i, "sr_long_target" if is_long else "sr_short_target"),
            "SR_Runner_Target_RR": _read_snapshot_number(df, i, "sr_long_target_rr" if is_long else "sr_short_target_rr"),
            "Base_Spread_Pips": float(self.base_spread_pips),
            "Base_Slippage_Pips": float(self.slippage_pips),
            "Spread_Session_Mult": float(session_mult),
            "Spread_Volatility_Mult": float(vol_mult),
            "Entry_Adaptive_Spread_Pips": float(adaptive_spread),
            "Entry_Adaptive_Slippage_Pips": float(adaptive_slippage),
        }

        #Counting how many reportable confluences are active on entry
        snapshot["Confluence_Count"] = int(sum(bool(snapshot[col]) for col in self.REPORT_CONFLUENCE_COLUMNS))

        #Appending configured numeric and boolean snapshot fields
        for col in self.SNAPSHOT_NUMERIC_COLUMNS:
            resolved_col = _find_snapshot_column(df, col, self.SNAPSHOT_NUMERIC_ALIASES)
            if resolved_col in df.columns:
                snapshot[col] = _read_snapshot_number(df, i, resolved_col)

        for col in self.SNAPSHOT_BOOL_COLUMNS:
            resolved_col = _find_snapshot_column(df, col, self.SNAPSHOT_BOOL_ALIASES)
            if resolved_col in df.columns:
                snapshot[col] = _read_snapshot_flag(df, i, resolved_col)

        return snapshot

    def _attach_runner_plan_from_entry_snapshot(self, trade: Trade, df: pd.DataFrame, i: int) -> None:
        is_long = trade.direction == "long"

        #Reading support/resistance runner targets from the entry bar
        valid_key = "support_resistance_long_valid" if is_long else "support_resistance_short_valid"
        ref_key = "sr_long_reference_level" if is_long else "sr_short_reference_level"
        target_key = "sr_long_target" if is_long else "sr_short_target"
        rr_key = "sr_long_target_rr" if is_long else "sr_short_target_rr"

        valid_val = bool(df[valid_key].iat[i]) if valid_key in df.columns and pd.notna(df[valid_key].iat[i]) else False
        ref_val = df[ref_key].iat[i] if ref_key in df.columns else None
        target_val = df[target_key].iat[i] if target_key in df.columns else None
        rr_val = df[rr_key].iat[i] if rr_key in df.columns else None

        #Marking whether the runner plan is valid for this trade
        if valid_val and pd.notna(target_val) and pd.notna(rr_val):
            trade.runner_qualified = True
            trade.runner_reference_level = float(ref_val) if pd.notna(ref_val) else None
            trade.runner_target_price = float(target_val)
            trade.runner_target_rr = float(rr_val)
        else:
            trade.runner_qualified = False
            trade.runner_reference_level = None
            trade.runner_target_price = None
            trade.runner_target_rr = None

        trade.report["Runner_Qualified"] = int(bool(trade.runner_qualified))
        trade.report["Runner_Active"] = 0
        trade.report["Runner_Exit_Reason"] = ""
        trade.report["Final_Exit_Type"] = ""
        trade.report["Partial_At_2R"] = 0

    def _half_spread_price(self, df: pd.DataFrame, i: int) -> float:
        spread_pips = self._resolve_spread_pips(df, i)
        return self._pips_to_price(spread_pips) / 2.0

    def _slippage_price(self, df: pd.DataFrame, i: int) -> float:
        return self._pips_to_price(self._resolve_slippage_pips(df, i))

    def _apply_entry_costs(self, raw_entry: float, direction: str, df: pd.DataFrame, i: int) -> float:
        half_spread = self._half_spread_price(df, i)
        slippage = self._slippage_price(df, i)
        return raw_entry + half_spread + slippage if direction == "long" else raw_entry - half_spread - slippage

    def _apply_exit_costs(self, raw_exit: float, direction: str, df: pd.DataFrame, i: int) -> float:
        half_spread = self._half_spread_price(df, i)
        slippage = self._slippage_price(df, i)
        return raw_exit - half_spread - slippage if direction == "long" else raw_exit + half_spread + slippage

    def _reverse_exit_costs(self, executed_exit: float, direction: str, df: pd.DataFrame, i: int) -> float:
        half_spread = self._half_spread_price(df, i)
        slippage = self._slippage_price(df, i)
        return executed_exit + half_spread + slippage if direction == "long" else executed_exit - half_spread - slippage
