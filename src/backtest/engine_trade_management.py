from __future__ import annotations

import pandas as pd

from src.backtest.models import Trade
from src.strategy.exit import (
    breakeven_price_to_cover_costs,
    get_target_from_risk,
    runner_hit_status,
    runner_trailing_stop_candidate,
)
from src.strategy.position import should_move_be, value_per_price_unit_per_lot

class BacktestEngineTradeManagementMixin:
    def _cash_per_price_unit(self) -> float:
        #Resolving cash value per unit of price movement for one lot
        return value_per_price_unit_per_lot(
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            contract_size=self.contract_size,
        )

    def _cash_from_price_move(self, price_move: float, lots: float) -> float:
        return float(price_move) * float(lots) * self._cash_per_price_unit()

    def _total_trade_commission(self, trade: Trade) -> float:
        return max(self.commission_per_lot, 0.0) * float(trade.size)

    def _should_move_be_next_bar(self, trade: Trade, i: int, df: pd.DataFrame) -> bool:
        #Checking breakeven on the previous completed bar only
        if trade.breakeven_moved or trade.partial_taken:
            return False
        if i <= trade.open_bar:
            return False

        prev_i = i - 1
        high = float(df["high"].iat[prev_i])
        low = float(df["low"].iat[prev_i])
        return should_move_be(trade, high, low)

    def _target_before_exit_costs(self, entry: float, risk_price: float, direction: str, df: pd.DataFrame, i: int) -> float:
        #Converting the executed target back into a raw chart price
        executed_target = get_target_from_risk(entry, risk_price, direction, rr_multiple=self.target_rr)
        return self._reverse_exit_costs(executed_target, direction, df, i)

    def _breakeven_before_exit_costs(self, trade: Trade, df: pd.DataFrame, i: int) -> float:
        #Shifting breakeven enough to recover remaining commission
        remaining_size = max(trade.remaining_size, 1e-12)
        remaining_commission = max(self._total_trade_commission(trade) - trade.commission_paid, 0.0)
        executed_be_exit = breakeven_price_to_cover_costs(
            trade.entry,
            trade.direction,
            remaining_commission,
            remaining_size,
            self._cash_per_price_unit(),
        )

        return self._reverse_exit_costs(executed_be_exit, trade.direction, df, i)

    def _trade_risk_price(self, trade: Trade) -> float:
        #Reading the effective initial stop distance used for this trade
        effective_initial_risk = getattr(trade, "effective_initial_risk", None)
        if effective_initial_risk is not None and effective_initial_risk > 0:
            return effective_initial_risk

        initial_stop = trade.initial_stop if trade.initial_stop is not None else trade.stop
        return max(trade.entry - initial_stop, 1e-12) if trade.direction == "long" else max(initial_stop - trade.entry, 1e-12)

    def _risk_amount(self, trade: Trade) -> float:
        return self._cash_from_price_move(self._trade_risk_price(trade), trade.size)

    def _close_partial(self, trade: Trade, raw_price: float, fraction: float, i: int, df: pd.DataFrame, reason: str) -> None:
        #Closing part of the position and some to cover commission
        close_size = trade.size * fraction
        exit_price = self._apply_exit_costs(raw_price, trade.direction, df, i)
        commission_share = self.commission_per_lot * close_size

        price_move = (exit_price - trade.entry) if trade.direction == "long" else (trade.entry - exit_price)
        pnl = self._cash_from_price_move(price_move, close_size)
        pnl -= commission_share

        trade.realized_pnl += pnl
        trade.remaining_size = max(trade.remaining_size - close_size, 0.0)
        trade.commission_paid += commission_share

        #Marking the standard 2R partial in the trade report
        if abs(fraction - trade.partial_fraction) < 1e-12:
            trade.partial_taken = True
            trade.partial_price = exit_price
            trade.partial_bar = i
            trade.report["Partial_At_2R"] = 1

        trade.report["Last_Partial_Reason"] = reason
        trade.report["Last_Partial_Time"] = df["time"].iat[i] if "time" in df.columns else i

    def _finalize_trade(self, trade: Trade, raw_price: float, i: int, df: pd.DataFrame, exit_reason: str) -> None:
        #Closing the remaining size and finalizing the trade record
        exit_price = self._apply_exit_costs(raw_price, trade.direction, df, i)
        commission_share = max(self._total_trade_commission(trade) - trade.commission_paid, 0.0)

        price_move = (exit_price - trade.entry) if trade.direction == "long" else (trade.entry - exit_price)
        pnl = self._cash_from_price_move(price_move, trade.remaining_size)
        pnl -= commission_share

        trade.realized_pnl += pnl
        trade.commission_paid += commission_share
        trade.remaining_size = 0.0

        total_pnl = trade.realized_pnl
        risk_amount = max(self._risk_amount(trade), 1e-12)
        total_r = total_pnl / risk_amount

        trade.result = total_r
        trade.close_bar = i
        trade.pnl = total_pnl
        trade.entry_time = df["time"].iat[trade.open_bar] if "time" in df.columns else trade.open_bar
        trade.exit_time = df["time"].iat[i] if "time" in df.columns else i
        trade.exit_price = exit_price
        trade.final_exit_type = exit_reason

        raw_be_exit = trade.breakeven_moved and not trade.partial_taken and abs(raw_price - trade.stop) < 1e-12
        trade.is_breakeven_exit = raw_be_exit

        #Deriving the trade outcome label from realized PnL
        if trade.is_breakeven_exit:
            outcome = "Breakeven"
        elif total_pnl > 0:
            outcome = "Win"
        else:
            outcome = "Loss"

        report = trade.report if trade.report is not None else {}
        risk_price = self._trade_risk_price(trade)
        executed_initial_stop = getattr(trade, "executed_initial_stop", None)
        executed_target = getattr(trade, "executed_target", None)

        exit_spread_pips = self._resolve_spread_pips(df, i)
        exit_slippage_pips = self._resolve_slippage_pips(df, i)
        exit_session_mult = self._session_spread_multiplier(df, i)
        exit_vol_mult = self._volatility_spread_multiplier(df, i)

        #Backfilling executed stop/target values if they were not stored on entry
        if executed_initial_stop is None:
            executed_initial_stop = self._apply_exit_costs(
                trade.initial_stop if trade.initial_stop is not None else trade.stop,
                trade.direction,
                df,
                trade.open_bar,
            )

        if executed_target is None:
            executed_target = get_target_from_risk(trade.entry, risk_price, trade.direction, rr_multiple=self.target_rr)

        #Updating the flat trade report used by reports and ML exports
        report.update(
            {
                "Trade_Date": trade.entry_time,
                "Exit_Time": trade.exit_time,
                "Exit_Bar": i,
                "Entry_Price": float(trade.entry),
                "Exit_Price": float(exit_price),
                "Initial_Stop": float(trade.initial_stop if trade.initial_stop is not None else trade.stop),
                "Final_Stop": float(trade.stop),
                "Take_Profit": float(trade.target),
                "Risk_Price": float(risk_price),
                "Risk_Pips": float(risk_price / self.pip_size),
                "Target_Price_Distance": float(abs(executed_target - trade.entry)),
                "Target_Pips": float(abs(executed_target - trade.entry) / self.pip_size),
                "Initial_Stop_Executed": float(executed_initial_stop),
                "Take_Profit_Executed": float(executed_target),
                "Size": float(trade.size),
                "Lots": float(trade.size),
                "PnL_Cash": float(total_pnl),
                "PnL_R": float(total_r),
                "Outcome": outcome,
                "Duration_Bars": int(i - trade.open_bar),
                "Partial_At_2R": int(bool(trade.partial_taken)),
                "Partial_Price": trade.partial_price,
                "Runner_Qualified": int(bool(trade.runner_qualified)),
                "Runner_Active": int(bool(trade.runner_active)),
                "Runner_Reference_Level": trade.runner_reference_level,
                "Runner_Target_Price": trade.runner_target_price,
                "Runner_Target_RR": trade.runner_target_rr,
                "Runner_Stop_Price": trade.runner_stop_price,
                "Runner_Exit_Reason": trade.runner_exit_reason or "",
                "Final_Exit_Type": exit_reason,
                "Spread_Pips": float(self.base_spread_pips),
                "Slippage_Pips": float(self.slippage_pips),
                "Entry_Adaptive_Spread_Pips": float(getattr(trade, "entry_spread_pips", self.base_spread_pips)),
                "Entry_Adaptive_Slippage_Pips": float(getattr(trade, "entry_slippage_pips", self.slippage_pips)),
                "Exit_Adaptive_Spread_Pips": float(exit_spread_pips),
                "Exit_Adaptive_Slippage_Pips": float(exit_slippage_pips),
                "Exit_Spread_Session_Mult": float(exit_session_mult),
                "Exit_Spread_Volatility_Mult": float(exit_vol_mult),
                "Commission_Per_Lot": float(self.commission_per_lot),
                "Commission_Cash": float(trade.commission_paid),
                "Label_PositivePnL": int(total_pnl > 0),
                "Label_Hit2R": int(bool(trade.partial_taken)),
                "Label_RunnerUsed": int(bool(trade.runner_active)),
            }
        )
        trade.report = report

        #Applying the realized PnL to account balance
        self.balance += total_pnl
        trade.balance_after = self.balance
        trade.report["Balance_After"] = float(self.balance)

        self.trades.append(trade)

    def _activate_runner(self, trade: Trade, i: int) -> bool:
        #Activating the runner only when a valid target plan exists
        if not trade.runner_qualified or trade.runner_target_price is None:
            trade.runner_active = False
            return False

        trade.runner_active = True
        trade.runner_activation_bar = i
        trade.runner_stop_price = trade.stop
        trade.report["Runner_Active"] = 1
        return True

    def _manage_pre_2r(self, trade: Trade, i: int, df: pd.DataFrame) -> bool:
        #Managing the position before the first 2R target is reached
        high = float(df["high"].iat[i])
        low = float(df["low"].iat[i])

        if trade.direction == "long":
            stop_hit = low <= trade.stop
            target_hit = high >= trade.target
        else:
            stop_hit = high >= trade.stop
            target_hit = low <= trade.target

        if stop_hit:
            self._finalize_trade(trade, trade.stop, i, df, exit_reason="stop_before_2r")
            return True

        if target_hit:
            #Closing fully at 2R if runners are disabled
            if self.runner_exit_mode == "none":
                trade.runner_exit_reason = "runner_disabled"
                self._finalize_trade(trade, trade.target, i, df, exit_reason="full_close_at_2r")
                return True

            #Taking the 2R partial and handing the rest to runner logic
            self._close_partial(trade, trade.target, trade.partial_fraction, i, df, reason="partial_at_2r")

            runner_ok = self._activate_runner(trade, i)

            if runner_ok and self.runner_exit_mode == "sr_only":
                return False

            if runner_ok and self.runner_exit_mode == "sr_atr_trail":
                return False

            trade.runner_exit_reason = "no_valid_sr_runner"
            self._finalize_trade(trade, trade.target, i, df, exit_reason="close_rest_at_2r")
            return True

        return False

    def _update_runner_trailing_stop(self, trade: Trade, i: int, df: pd.DataFrame) -> None:
        #Trailing the runner stop from the previous bar close and ATR
        if i <= 0:
            return

        atr_prev = self._get_atr_value(df, i - 1)
        if atr_prev is None or atr_prev <= 0:
            return

        close_prev = float(df["close"].iat[i - 1])

        candidate = runner_trailing_stop_candidate(trade.direction, close_prev, atr_prev, self.runner_atr_multiple)
        if trade.direction == "long":
            trade.runner_stop_price = candidate if trade.runner_stop_price is None else max(trade.runner_stop_price, candidate)
        else:
            trade.runner_stop_price = candidate if trade.runner_stop_price is None else min(trade.runner_stop_price, candidate)

    def _manage_runner(self, trade: Trade, i: int, df: pd.DataFrame) -> bool:
        #Managing the post-partial runner until stop, target, or end of data
        if not trade.runner_active:
            self._finalize_trade(trade, trade.target, i, df, exit_reason="close_rest_at_2r")
            return True

        if trade.runner_activation_bar is not None and i <= trade.runner_activation_bar:
            return False

        #Updating the ATR trail before checking hits on the current bar
        if self.runner_exit_mode == "sr_atr_trail":
            self._update_runner_trailing_stop(trade, i, df)

        high = float(df["high"].iat[i])
        low = float(df["low"].iat[i])

        if self.runner_exit_mode == "sr_only":
            stop_price = trade.stop
        else:
            stop_price = trade.runner_stop_price if trade.runner_stop_price is not None else trade.stop

        target_price = trade.runner_target_price
        stop_hit, target_hit = runner_hit_status(
            trade.direction,
            high,
            low,
            stop_price,
            target_price,
        )

        #Prioritizing the runner stop when it is tagged
        if stop_hit:
            trade.runner_exit_reason = (
                "atr_trail_stop" if self.runner_exit_mode == "sr_atr_trail" else "initial_stop_after_partial"
            )
            self._finalize_trade(trade, stop_price, i, df, exit_reason="runner_stop")
            return True

        #Closing the runner at its support/resistance target
        if target_hit:
            trade.runner_exit_reason = "sr_target"
            self._finalize_trade(trade, target_price, i, df, exit_reason="runner_target")
            return True

        return False

    def _close_at_end_of_data(self, trade: Trade, i: int, df: pd.DataFrame) -> None:
        #Force-closing any remaining position on the final available bar
        raw_price = float(df["close"].iat[i])
        if trade.partial_taken:
            trade.runner_exit_reason = trade.runner_exit_reason or "end_of_data"
        self._finalize_trade(trade, raw_price, i, df, exit_reason="end_of_data")
