from __future__ import annotations

import time
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

#Ensuring that the repo root is importable when running the script directly
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mt5.config import (
    SYMBOLS,
    TIMEFRAMES,
    START_DATE,
    END_DATE,
    RAW_DATA_DIR,
    FEATURE_CACHE_DIR,
    BACKTEST_OUTPUT_DIR,
    ML_DATASET_OUTPUT_DIR,
    INITIAL_CASH,
    RISK_PCT,
    TARGET_RR,
    COMMISSION_PER_LOT,
    RUNNER_EXIT_MODE,
    RUNNER_ATR_MULTIPLE,
    BASE_SPREAD_PIPS,
    SLIPPAGE_PIPS,
    USE_SPREAD_ADAPTIVE_SLIPPAGE,
    USE_SESSION_ADAPTIVE_SLIPPAGE,
    SLIPPAGE_MIN_MULT,
    SLIPPAGE_MAX_MULT,
    SPREAD_SESSION_LIQUID_MULT,
    SPREAD_SESSION_NEUTRAL_MULT,
    SPREAD_SESSION_THIN_MULT,
    SPREAD_VOL_MEDIUM_THRESHOLD,
    SPREAD_VOL_HIGH_THRESHOLD,
    SPREAD_VOL_MEDIUM_MULT,
    SPREAD_VOL_HIGH_MULT,
    USE_RAW_CACHE,
    USE_FEATURE_CACHE,
    FORCE_REBUILD_FEATURES,
    PRINT_STAGE_TIMINGS,
    PIPELINE_TOGGLES,
)
from src.mt5.data_fetcher import fetch_data
from src.mt5.symbol_spec import load_symbol_spec
from src.features.pipeline import FeaturePipelineConfig, build_feature_frame
from src.strategy.rules import generate_signals
from src.backtest.engine import BacktestEngine
from src.backtest.formatting import format_stats_payload
from src.backtest.reports import save_backtest_artifacts


def _make_feature_cache_key(symbol: str, timeframe: str) -> str:
    return f"{symbol}_{timeframe}_{START_DATE.date()}_{END_DATE.date()}"


def _print_timing(label: str, started_at: float) -> None:
    if PRINT_STAGE_TIMINGS:
        print(f"{label}: {time.perf_counter() - started_at:.3f}s")


def run():
    #Ensuring that runtime output/cache directories exist
    BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ML_DATASET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    #Running the configured backtest for each symbol/timeframe pair
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"\n=== {symbol} {tf} | {START_DATE.date()} -> {END_DATE.date()} ===")
            total_start = time.perf_counter()

            #Loading broker-aware symbol metadata once for the run
            symbol_spec = load_symbol_spec(symbol)
            print(
                "Symbol spec:"
                f" broker_symbol={symbol_spec.broker_symbol or symbol}"
                f" pip_size={symbol_spec.pip_size}"
                f" source={symbol_spec.source}"
            )

            #Building the feature pipeline config from current toggles
            pipeline_cfg = FeaturePipelineConfig(
                pip_size=symbol_spec.pip_size,
                use_session_liquidity=PIPELINE_TOGGLES.use_session_liquidity,
                use_volume_profile=PIPELINE_TOGGLES.use_volume_profile,
                use_imbalance=PIPELINE_TOGGLES.use_imbalance,
                use_fibonacci=PIPELINE_TOGGLES.use_fibonacci,
                use_liquidity=PIPELINE_TOGGLES.use_liquidity,
                use_support_resistance=PIPELINE_TOGGLES.use_support_resistance,
            )

            t0 = time.perf_counter()

            #Fetch raw MT5 bars
            df = fetch_data(
                symbol,
                tf,
                START_DATE,
                END_DATE,
                data_dir=RAW_DATA_DIR,
                overwrite=not USE_RAW_CACHE,
                use_pickle_cache=USE_RAW_CACHE,
                broker_symbol=symbol_spec.broker_symbol,
            )
            _print_timing("fetch/load raw", t0)

            t1 = time.perf_counter()

            df = build_feature_frame(
                df,
                symbol=symbol,
                config=pipeline_cfg,
                cache_dir=FEATURE_CACHE_DIR,
                cache_key=_make_feature_cache_key(symbol, tf),
                use_cache=USE_FEATURE_CACHE,
                force_rebuild=FORCE_REBUILD_FEATURES,
            )
            _print_timing("build/load features", t1)

            t2 = time.perf_counter()

            #Applying strategy rules to produce entry signals
            df = generate_signals(df)
            _print_timing("generate signals", t2)

            t3 = time.perf_counter()

            #Initializing the execution engine with broker-aware sizing inputs
            engine = BacktestEngine(
                initial_balance=INITIAL_CASH,
                risk_pct=RISK_PCT,
                target_rr=TARGET_RR,
                commission_per_lot=COMMISSION_PER_LOT,
                base_spread_pips=BASE_SPREAD_PIPS,
                slippage_pips=SLIPPAGE_PIPS,
                symbol=symbol,
                timeframe=tf,
                runner_exit_mode=RUNNER_EXIT_MODE,
                runner_atr_multiple=RUNNER_ATR_MULTIPLE,
                session_liquid_mult=SPREAD_SESSION_LIQUID_MULT,
                session_neutral_mult=SPREAD_SESSION_NEUTRAL_MULT,
                session_thin_mult=SPREAD_SESSION_THIN_MULT,
                spread_vol_medium_threshold=SPREAD_VOL_MEDIUM_THRESHOLD,
                spread_vol_high_threshold=SPREAD_VOL_HIGH_THRESHOLD,
                spread_vol_medium_mult=SPREAD_VOL_MEDIUM_MULT,
                spread_vol_high_mult=SPREAD_VOL_HIGH_MULT,
                use_spread_adaptive_slippage=USE_SPREAD_ADAPTIVE_SLIPPAGE,
                use_session_adaptive_slippage=USE_SESSION_ADAPTIVE_SLIPPAGE,
                slippage_min_mult=SLIPPAGE_MIN_MULT,
                slippage_max_mult=SLIPPAGE_MAX_MULT,
                pip_size=symbol_spec.pip_size,
                tick_size=symbol_spec.tick_size,
                tick_value=symbol_spec.tick_value,
                contract_size=symbol_spec.contract_size,
                volume_step=symbol_spec.volume_step,
                volume_min=symbol_spec.volume_min,
                volume_max=symbol_spec.volume_max,
            )
            results = engine.run(df)
            _print_timing("backtest", t3)

            ml_dataset_path = ML_DATASET_OUTPUT_DIR / f"{symbol}_{tf}_ml_dataset.csv"

            t4 = time.perf_counter()

            #Saving backtest reports and refreshing the ML dataset export
            artifacts = save_backtest_artifacts(
                summary=results,
                trades=engine.trades,
                engine=engine,
                output_dir=BACKTEST_OUTPUT_DIR,
                symbol=symbol,
                timeframe=tf,
                start=START_DATE,
                end=END_DATE,
            )
            ml_dataset_exported = False
            if engine.trades:
                engine.export_ml_dataset(ml_dataset_path)
                ml_dataset_exported = True
            elif ml_dataset_path.exists():
                ml_dataset_path.unlink()
            _print_timing("export datasets", t4)

            #Printing the run summary and confluence counts
            print(f"{symbol} {tf} {format_stats_payload(results)}")
            print(f"Artifacts directory: {artifacts['run_dir']}")
            if ml_dataset_exported:
                print(f"ML dataset saved to: {ml_dataset_path}")
            else:
                print("ML dataset export skipped: no trades were generated.")

            confluence_counts = engine.get_confluence_trade_counts()
            print("Trade-level confluence validity counts:")
            for name, count in confluence_counts.items():
                print(f"  {name}: {count}")

            _print_timing("total", total_start)


if __name__ == "__main__":
    run()
