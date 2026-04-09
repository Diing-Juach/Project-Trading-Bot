from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_datetime(name: str, default: datetime) -> datetime:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return default


# Data directories
# -----------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_CACHE_DIR = DATA_DIR / "feature_cache"


# Output directories
# -----------------------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BACKTEST_OUTPUT_DIR = OUTPUTS_DIR / "backtests"
ML_DATASET_OUTPUT_DIR = OUTPUTS_DIR / "ml_datasets"


# Backtest universe
# -----------------------------
SYMBOLS = _env_csv("TBOT_SYMBOLS", ["GBPUSD"])
TIMEFRAMES = _env_csv("TBOT_TIMEFRAMES", ["M30"])
START_DATE = _env_datetime("TBOT_START_DATE", datetime(2008, 1, 1))
END_DATE = _env_datetime("TBOT_END_DATE", datetime(2009, 1, 1))


# Backtest settings
# -----------------------------
INITIAL_CASH = _env_float("TBOT_INITIAL_CASH", 10_000.0)
RISK_PCT = _env_float("TBOT_RISK_PCT", 0.01)
TARGET_RR = _env_float("TBOT_TARGET_RR", 2.0)


# Execution costs
# -----------------------------
# Round-turn commission per 1.00 lot in account currency.
COMMISSION_PER_LOT = _env_float("TBOT_COMMISSION_PER_LOT", 6.0)
BASE_SPREAD_PIPS = _env_float("TBOT_BASE_SPREAD_PIPS", 0.7)
SLIPPAGE_PIPS = _env_float("TBOT_SLIPPAGE_PIPS", 0.5)

# Pip-size overrides
# -----------------------------
# MT5 exposes point/tick precision, but some non-FX instruments do not have a
# universally meaningful "pip" definition. These overrides let the strategy use
# the pip convention its zone-width and stop-width thresholds were tuned for.
PIP_SIZE_OVERRIDES = {
    "XAUUSD": 0.10,
    "GOLD": 0.10,
    "XAGUSD": 0.01,
    "SILVER": 0.01,
}

# Runner / exit mode
RUNNER_EXIT_MODE = "sr_atr_trail"   # options: "none", "sr_only", "sr_atr_trail"
RUNNER_ATR_MULTIPLE = 1.0

# Market-adaptive slippage settings
USE_SPREAD_ADAPTIVE_SLIPPAGE = True
USE_SESSION_ADAPTIVE_SLIPPAGE = True

SLIPPAGE_MIN_MULT = 0.75
SLIPPAGE_MAX_MULT = 2.50

SESSION_LIQUID_MULT = 0.90
SESSION_NEUTRAL_MULT = 1.00
SESSION_THIN_MULT = 1.20

# Session multipliers
SPREAD_SESSION_LIQUID_MULT = 0.85
SPREAD_SESSION_NEUTRAL_MULT = 1.00
SPREAD_SESSION_THIN_MULT = 1.25

# Volatility multipliers
SPREAD_VOL_MEDIUM_THRESHOLD = 1.20
SPREAD_VOL_HIGH_THRESHOLD = 1.50
SPREAD_VOL_MEDIUM_MULT = 1.20
SPREAD_VOL_HIGH_MULT = 1.70


# Runtime / cache settings
# -----------------------------
USE_RAW_CACHE = _env_bool("TBOT_USE_RAW_CACHE", True)
USE_FEATURE_CACHE = _env_bool("TBOT_USE_FEATURE_CACHE", True)
FORCE_REBUILD_FEATURES = _env_bool("TBOT_FORCE_REBUILD_FEATURES", False)
PRINT_STAGE_TIMINGS = _env_bool("TBOT_PRINT_STAGE_TIMINGS", True)

# Use smaller windows during development if needed.
DEVELOPMENT_MODE = False


# Feature toggles
# -----------------------------
@dataclass(slots=True)
class PipelineToggleConfig:
    use_session_liquidity: bool = True
    use_volume_profile: bool = True
    use_imbalance: bool = True
    use_fibonacci: bool = True
    use_liquidity: bool = True
    use_support_resistance: bool = True


PIPELINE_TOGGLES = PipelineToggleConfig(
    use_session_liquidity=_env_bool("TBOT_USE_SESSION_LIQUIDITY", True),
    use_volume_profile=_env_bool("TBOT_USE_VOLUME_PROFILE", True),
    use_imbalance=_env_bool("TBOT_USE_IMBALANCE", True),
    use_fibonacci=_env_bool("TBOT_USE_FIBONACCI", True),
    use_liquidity=_env_bool("TBOT_USE_LIQUIDITY", True),
    use_support_resistance=_env_bool("TBOT_USE_SUPPORT_RESISTANCE", True),
)
