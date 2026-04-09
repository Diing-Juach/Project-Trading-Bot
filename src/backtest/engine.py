from __future__ import annotations

import pandas as pd

from src.backtest.engine_costs import BacktestEngineCostsMixin
from src.backtest.engine_reporting import BacktestEngineReportingMixin
from src.backtest.engine_trade_management import BacktestEngineTradeManagementMixin
from src.backtest.models import Trade
from src.mt5.symbol_spec import infer_pip_size
from src.strategy.entry import get_entry
from src.strategy.position import calculate_size, should_move_be


class BacktestEngine(
    BacktestEngineCostsMixin,
    BacktestEngineTradeManagementMixin,
    BacktestEngineReportingMixin,
):
    REPORT_CONFLUENCE_COLUMNS = [
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

    SNAPSHOT_NUMERIC_COLUMNS = [
        "atr_feature",
        "demand_zone_entry",
        "demand_zone_stop",
        "demand_zone_bottom",
        "demand_zone_top",
        "demand_zone_width_pips",
        "demand_zone_stop_pips",
        "demand_zone_retests",
        "demand_zone_age",
        "supply_zone_entry",
        "supply_zone_stop",
        "supply_zone_bottom",
        "supply_zone_top",
        "supply_zone_width_pips",
        "supply_zone_stop_pips",
        "supply_zone_retests",
        "supply_zone_age",
        "sr_nearest_resistance",
        "sr_nearest_support",
        "sr_long_reference_level",
        "sr_short_reference_level",
        "sr_long_target",
        "sr_short_target",
        "sr_long_target_rr",
        "sr_short_target_rr",
        "vp_reference_poc",
        "vp_distance_to_poc",
        "fib_bull_impulse_low",
        "fib_bull_impulse_high",
        "fib_bear_impulse_low",
        "fib_bear_impulse_high",
        "fib_long_band_low",
        "fib_long_band_high",
        "fib_short_band_low",
        "fib_short_band_high",
        "liquidity_pool_high",
        "liquidity_pool_low",
    ]

    SNAPSHOT_BOOL_COLUMNS = [
        "long_signal",
        "short_signal",
        "bos_up",
        "bos_down",
        "choch_up",
        "choch_down",
        "mss_up",
        "mss_down",
        "supply_demand_long_valid",
        "supply_demand_short_valid",
        "market_structure_long_valid",
        "market_structure_short_valid",
        "session_liquidity_long_valid",
        "session_liquidity_short_valid",
        "volume_profile_long_valid",
        "volume_profile_short_valid",
        "imbalance_long_valid",
        "imbalance_short_valid",
        "fibonacci_long_valid",
        "fibonacci_short_valid",
        "liquidity_long_valid",
        "liquidity_short_valid",
        "support_resistance_long_valid",
        "support_resistance_short_valid",
        "is_asia_session",
        "is_london_session",
        "is_newyork_session",
        "is_london_newyork_overlap",
    ]

    SNAPSHOT_NUMERIC_ALIASES = {
        "demand_zone_bottom": ["demand_zone_bottom", "demand_zone_low"],
        "demand_zone_top": ["demand_zone_top", "demand_zone_high"],
        "supply_zone_bottom": ["supply_zone_bottom", "supply_zone_low"],
        "supply_zone_top": ["supply_zone_top", "supply_zone_high"],
    }

    SNAPSHOT_BOOL_ALIASES = {
        "bos_up": ["bos_up", "bullish_bos"],
        "bos_down": ["bos_down", "bearish_bos"],
        "choch_up": ["choch_up", "bullish_choch"],
        "choch_down": ["choch_down", "bearish_choch"],
        "mss_up": ["mss_up", "bullish_mss"],
        "mss_down": ["mss_down", "bearish_mss"],
    }

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_pct: float = 0.01,
        target_rr: float = 2.0,
        commission_per_lot: float = 0.0,
        base_spread_pips: float = 0.0,
        slippage_pips: float = 0.2,
        symbol: str | None = None,
        timeframe: str | None = None,
        cooldown_bars: int = 0,
        runner_exit_mode: str = "sr_atr_trail",
        runner_atr_multiple: float = 1.0,
        session_liquid_mult: float = 0.85,
        session_neutral_mult: float = 1.00,
        session_thin_mult: float = 1.20,
        spread_vol_medium_threshold: float = 1.20,
        spread_vol_high_threshold: float = 1.50,
        spread_vol_medium_mult: float = 1.15,
        spread_vol_high_mult: float = 1.35,
        use_spread_adaptive_slippage: bool = True,
        use_session_adaptive_slippage: bool = True,
        slippage_min_mult: float = 0.75,
        slippage_max_mult: float = 2.50,
        pip_size: float | None = None,
        tick_size: float | None = None,
        tick_value: float | None = None,
        contract_size: float | None = None,
        volume_step: float | None = None,
        volume_min: float | None = None,
        volume_max: float | None = None,
    ):
        #Storing account and risk settings
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_pct = risk_pct
        self.target_rr = target_rr
        self.commission_per_lot = float(commission_per_lot)

        #Storing execution cost settings
        self.base_spread_pips = max(float(base_spread_pips), 0.0)
        self.slippage_pips = max(float(slippage_pips), 0.0)

        #Storing trade management settings
        self.symbol = symbol
        self.timeframe = timeframe
        self.cooldown_bars = cooldown_bars
        self.runner_exit_mode = runner_exit_mode
        self.runner_atr_multiple = runner_atr_multiple

        #Storing adaptive spread/slippage controls
        self.session_liquid_mult = float(session_liquid_mult)
        self.session_neutral_mult = float(session_neutral_mult)
        self.session_thin_mult = float(session_thin_mult)

        self.spread_vol_medium_threshold = float(spread_vol_medium_threshold)
        self.spread_vol_high_threshold = float(spread_vol_high_threshold)
        self.spread_vol_medium_mult = float(spread_vol_medium_mult)
        self.spread_vol_high_mult = float(spread_vol_high_mult)

        self.use_spread_adaptive_slippage = bool(use_spread_adaptive_slippage)
        self.use_session_adaptive_slippage = bool(use_session_adaptive_slippage)
        self.slippage_min_mult = float(slippage_min_mult)
        self.slippage_max_mult = float(slippage_max_mult)

        #Storing broker-derived symbol specification for sizing and pip conversion
        self.pip_size = float(pip_size) if pip_size is not None else infer_pip_size(symbol)
        self.tick_size = float(tick_size) if tick_size is not None else None
        self.tick_value = float(tick_value) if tick_value is not None else None
        self.contract_size = float(contract_size) if contract_size is not None else None
        self.volume_step = float(volume_step) if volume_step is not None else None
        self.volume_min = float(volume_min) if volume_min is not None else None
        self.volume_max = float(volume_max) if volume_max is not None else None
        self.trades: list[Trade] = []
        self.equity_points: list[dict] = []

        self._baseline_atr_value: float | None = None

    def run(self, df: pd.DataFrame) -> dict:
        #Resetting engine state for a fresh backtest pass
        self.trades = []
        self.equity_points = []
        self.balance = self.initial_balance

        #Preparing the ATR baseline used by adaptive spread/slippage
        self._set_baseline_atr(df)

        open_trade: Trade | None = None
        last_long_exit_bar = -10_000
        last_short_exit_bar = -10_000

        #Recording the starting equity point
        if len(df) > 0:
            self._record_equity_point(df, 0)

        #Walking bar by bar through the signal frame
        for i in range(len(df)):
            high = float(df["high"].iat[i])
            low = float(df["low"].iat[i])

            if open_trade is not None:
                #Managing open trades before looking for new entries
                if not open_trade.partial_taken:
                    if should_move_be(open_trade, high, low) and not open_trade.breakeven_moved:
                        open_trade.stop = self._breakeven_before_exit_costs(open_trade, df, i)
                        open_trade.breakeven_moved = True

                    closed = self._manage_pre_2r(open_trade, i, df)
                    if closed:
                        if open_trade.direction == "long":
                            last_long_exit_bar = i
                        else:
                            last_short_exit_bar = i
                        open_trade = None
                        self._record_equity_point(df, i)
                        continue
                else:
                    closed = self._manage_runner(open_trade, i, df)
                    if closed:
                        if open_trade.direction == "long":
                            last_long_exit_bar = i
                        else:
                            last_short_exit_bar = i
                        open_trade = None
                        self._record_equity_point(df, i)
                        continue

            if open_trade is None:
                #Respecting directional cooldowns after exits
                can_long = (i - last_long_exit_bar) > self.cooldown_bars
                can_short = (i - last_short_exit_bar) > self.cooldown_bars

                if bool(df["long_signal"].iat[i]) and can_long:
                    #Pricing and sizing a new long entry
                    raw_entry, raw_stop = get_entry(df, i, "long")
                    entry = self._apply_entry_costs(raw_entry, "long", df, i)
                    stop = raw_stop
                    executed_stop = self._apply_exit_costs(raw_stop, "long", df, i)
                    effective_risk = entry - executed_stop
                    target = self._target_before_exit_costs(entry, effective_risk, "long", df, i)
                    size = calculate_size(
                        self.balance,
                        self.risk_pct,
                        entry,
                        executed_stop,
                        tick_size=self.tick_size,
                        tick_value=self.tick_value,
                        contract_size=self.contract_size,
                        volume_step=self.volume_step,
                        volume_min=self.volume_min,
                        volume_max=self.volume_max,
                    )

                    if size > 0 and effective_risk > 0:
                        report = self._make_trade_snapshot(df, i, "long", stop, target)
                        trade = Trade(
                            direction="long",
                            entry=entry,
                            stop=stop,
                            target=target,
                            risk=self.risk_pct,
                            size=size,
                            remaining_size=size,
                            open_bar=i,
                            initial_stop=stop,
                            effective_initial_risk=effective_risk,
                            report=report,
                        )
                        trade.entry_slippage_pips = self._resolve_slippage_pips(df, i)
                        trade.entry_spread_pips = self._resolve_spread_pips(df, i)
                        trade.executed_initial_stop = executed_stop
                        trade.executed_target = entry + (effective_risk * self.target_rr)
                        self._attach_runner_plan_from_entry_snapshot(trade, df, i)
                        open_trade = trade

                elif bool(df["short_signal"].iat[i]) and can_short:
                    #Pricing and sizing a new short entry
                    raw_entry, raw_stop = get_entry(df, i, "short")
                    entry = self._apply_entry_costs(raw_entry, "short", df, i)
                    stop = raw_stop
                    executed_stop = self._apply_exit_costs(raw_stop, "short", df, i)
                    effective_risk = executed_stop - entry
                    target = self._target_before_exit_costs(entry, effective_risk, "short", df, i)
                    size = calculate_size(
                        self.balance,
                        self.risk_pct,
                        entry,
                        executed_stop,
                        tick_size=self.tick_size,
                        tick_value=self.tick_value,
                        contract_size=self.contract_size,
                        volume_step=self.volume_step,
                        volume_min=self.volume_min,
                        volume_max=self.volume_max,
                    )

                    if size > 0 and effective_risk > 0:
                        report = self._make_trade_snapshot(df, i, "short", stop, target)
                        trade = Trade(
                            direction="short",
                            entry=entry,
                            stop=stop,
                            target=target,
                            risk=self.risk_pct,
                            size=size,
                            remaining_size=size,
                            open_bar=i,
                            initial_stop=stop,
                            effective_initial_risk=effective_risk,
                            report=report,
                        )
                        trade.entry_slippage_pips = self._resolve_slippage_pips(df, i)
                        trade.entry_spread_pips = self._resolve_spread_pips(df, i)
                        trade.executed_initial_stop = executed_stop
                        trade.executed_target = entry - (effective_risk * self.target_rr)
                        self._attach_runner_plan_from_entry_snapshot(trade, df, i)
                        open_trade = trade

        #Closing any remaining open trade on the last bar
        if len(df) > 0 and open_trade is not None:
            self._close_at_end_of_data(open_trade, len(df) - 1, df)
            self._record_equity_point(df, len(df) - 1)

        #Ensuring the equity curve always includes the final bar
        if len(df) > 0 and (not self.equity_points or self.equity_points[-1]["bar"] != len(df) - 1):
            self._record_equity_point(df, len(df) - 1)

        return self._results()
