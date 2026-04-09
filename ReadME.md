# TBotFixed

Rule-based FX and metals backtesting project with a post-signal XGBoost trade filter and a Streamlit research dashboard.

This repository was built for dissertation-style strategy research. It combines:
- broker-aware MT5 data loading
- a multi-feature rule-based strategy
- cost-adjusted backtesting
- trade-level ML dataset export
- walk-forward XGBoost filtering
- a local dashboard for analysis and reporting

## What The Project Does

The research workflow is:

1. Load historical OHLCV data from MetaTrader 5
2. Build feature layers such as market structure, supply and demand, imbalance, liquidity, Fibonacci, and support/resistance
3. Generate strategy signals
4. Execute a cost-adjusted backtest with spread, slippage, and commission
5. Export trade artifacts and an ML-ready trade dataset
6. Train a walk-forward XGBoost model to rank and filter trades
7. Review the outputs in the Streamlit dashboard

## Strategy Components

Core strategy layers:
- ATR
- Market Structure
- Supply & Demand

Optional confluence layers:
- Session Liquidity
- Volume Profile
- Imbalance (FVG)
- Fibonacci
- Liquidity
- Support & Resistance

## Repository Structure

```text
TBotFixed/
+-- dashboard/         # Streamlit dashboard
+-- scripts/           # Entrypoints for backtesting and ML training
+-- src/
|   +-- backtest/      # Engine, reporting, cost modelling
|   +-- features/      # Feature/confluence generation
|   +-- ml/            # Dataset loading, ranking, training
|   +-- mt5/           # MT5 integration, symbol resolution, config
|   `-- strategy/      # Signal, position, and exit logic
+-- data/              # Local raw/cache data (ignored by git)
+-- outputs/           # Local artifacts and model outputs (ignored by git)
+-- requirements.txt
`-- README.md
```

## Requirements

- Python `3.12`
- Windows
- MetaTrader 5 installed locally
- Access to an MT5 terminal/account for historical data retrieval

Python packages are listed in `requirements.txt`.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create your local environment file from the example template:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and add your own MT5 credentials.

## Configuration

Project defaults live in `src/mt5/config.py`.

Key defaults include:
- symbols
- timeframes
- date range
- initial cash
- risk percentage
- target RR
- commission, spread, and slippage
- cache settings
- feature toggles

Environment setup:
- `.env.example` is the safe template committed to the repository
- `.env` is your private local file and is ignored by git
- the MT5 connector reads `MT5_LOGIN`, `MT5_PASSWORD`, and `MT5_SERVER` from `.env`


Example one-off run in PowerShell:

```powershell
$env:TBOT_SYMBOLS = "EURUSD"
$env:TBOT_TIMEFRAMES = "M30"
$env:TBOT_START_DATE = "2020-01-01"
$env:TBOT_END_DATE = "2026-01-01"
$env:TBOT_FORCE_REBUILD_FEATURES = "true"
.\venv\Scripts\python.exe scripts\run_backtest.py
```

## Run A Backtest

Run the full research pipeline:

```powershell
.\venv\Scripts\python.exe scripts\run_backtest.py
```

This will:
- fetch or load MT5 price data
- build feature layers
- generate entry signals
- run the backtest engine
- export artifact folders
- export or refresh the matching ML dataset

Main backtest outputs are written to:
- `outputs/backtests`
- `outputs/ml_datasets`

Each backtest artifact folder typically includes:
- `account_info.json`
- `trade_stats.json`
- `trades.csv`
- `equity_curve.csv`

## Train The ML Filter

Train the walk-forward XGBoost filter on all available ML datasets:

```powershell
.\venv\Scripts\python.exe scripts\train_ml.py
```

Train on a single dataset:

```powershell
.\venv\Scripts\python.exe scripts\train_ml.py --pattern EURUSD_M15_ml_dataset.csv --output-dir outputs\ml_models\eurusd_m15_run
```

The ML pipeline:
- loads trade-level datasets
- engineers trade-quality features
- builds walk-forward folds
- selects a keep ratio on validation data
- applies the chosen filter to test data
- exports metrics, predictions, feature importance, and fold summaries

Typical ML outputs:
- `training_summary.json`
- `validation_predictions.csv`
- `test_predictions.csv`
- `validation_keep_ratio_sweep.csv`
- `walk_forward_fold_summary.csv`
- `feature_importance.csv`
- `model_bundle.pkl`

These are written under `outputs/ml_models`.

## Run The Dashboard

Start the Streamlit dashboard:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Current dashboard sections:
- `Overview`
  - saved backtest selection
  - trade metrics
  - account metrics
  - equity curve
- `Backtest`
  - run configuration
  - strategy toggles
  - engine controls
  - run status
- `Confluence Analysis`
  - confluence summary
  - with-vs-without impact
  - best confluence-set analysis
- `ML Analysis`
  - train on trade dataset
  - validation metrics
  - test metrics
  - training summary

## Notes On Data And Outputs

This repository does not commit local runtime data by default.

Ignored by git:
- `venv/`
- `outputs/`
- `data/raw/`
- `data/feature_cache/`
- `data/processed/`
- `.env`

That means anyone cloning the repo will need to:
- create their own virtual environment
- install dependencies
- connect to MT5 locally
- regenerate raw data, caches, outputs, and models

## Notes On Instruments

The project supports broker-resolved symbol names from MT5, including broker-specific suffixes such as `.sml`. Dashboard and artifact labels are normalized for display, but the underlying MT5 connector still resolves the actual broker symbol automatically.

Pip-size overrides are included for:
- `XAUUSD`
- `XAGUSD`
