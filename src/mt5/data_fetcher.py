from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Literal
import logging
import pathlib

import pandas as pd
from src.mt5.connector import resolve_symbol_name

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


log = logging.getLogger(__name__)

#Allowed timeframe inputs
TimeframeStr = Literal["M5", "M10", "M15", "M30", "H1"]


@dataclass(slots=True)
class FetchConfig:
    data_dir: pathlib.Path = pathlib.Path("data/raw")
    overwrite: bool = False
    chunk_days: int = 180
    use_pickle_cache: bool = True


class MT5DataFetcher:
    def __init__(self, config: FetchConfig):
        self.config = config

        #Ensuring raw data directory exists
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        symbol: str,
        timeframe: TimeframeStr,
        start: datetime,
        end: datetime,
        *,
        broker_symbol: str | None = None,
    ) -> pd.DataFrame:

        #Resolving cache file paths
        csv_path = self._get_csv_path(symbol, timeframe, start, end)
        pkl_path = self._get_pickle_path(symbol, timeframe, start, end)

        #Normalizing timestamps to UTC
        start_utc = self._coerce_utc(start)
        end_utc = self._coerce_utc(end)

        #Checking cache before hitting MT5
        if not self.config.overwrite:

            #Loading pickle cache (fast path)
            if self.config.use_pickle_cache and pkl_path.exists():
                log.info("Raw pickle cache hit: %s", pkl_path)
                cached = self._standardize(pd.read_pickle(pkl_path))

                #Validating cache covers requested window
                if self._cache_matches_window(cached, start_utc, end_utc):
                    return cached

                log.warning("Ignoring out-of-window raw pickle cache: %s", pkl_path)

            #Fallback to CSV cache
            if csv_path.exists():
                log.info("Raw CSV cache hit: %s", csv_path)
                cached = self._standardize(pd.read_csv(csv_path, parse_dates=["time"]))

                #Validating CSV cache window
                if self._cache_matches_window(cached, start_utc, end_utc):

                    #Upgrading CSV → pickle for faster future loads
                    if self.config.use_pickle_cache:
                        cached.to_pickle(pkl_path)

                    return cached

                log.warning("Ignoring out-of-window raw CSV cache: %s", csv_path)

        #Ensuring MT5 is available
        if mt5 is None:
            raise RuntimeError("MetaTrader5 not installed")

        #Initializing MT5 session
        if not mt5.initialize():
            raise RuntimeError("MT5 initialize() failed")

        try:
            #Resolving broker-specific symbol naming
            resolved = broker_symbol or resolve_symbol_name(symbol)

            #Mapping timeframe string → MT5 constant
            tf = self._resolve_timeframe(timeframe)

            #Fetching data in chunks (avoiding MT5 limits)
            df = self._fetch_chunked(resolved, tf, start_utc, end_utc)

            #Falling back to terminal cache if API fails
            if df.empty:
                code, msg = mt5.last_error()
                log.warning(
                    "Chunked fetch failed for %s %s: %s:%s. Falling back to terminal cache.",
                    resolved,
                    timeframe,
                    code,
                    msg,
                )

                df = self._load_from_terminal_cache(resolved, tf)

                #Failing hard if no data available
                if df.empty:
                    raise RuntimeError(f"No data available for {resolved} {timeframe}")

            #Normalizing raw MT5 structure
            df = self._standardize(df)

            #Clipping strictly to requested time window
            df = self._clip_to_window(df, start_utc, end_utc)

            #Ensuring dataset is not empty post-filter
            if df.empty:
                raise RuntimeError(
                    f"No data available for {resolved} {timeframe} within requested window "
                    f"{start_utc.isoformat()} -> {end_utc.isoformat()}"
                )

            #Logging final dataset coverage
            actual_start = df["time"].min()
            actual_end = df["time"].max()

            log.info(
                "Fetched %s %s: %d rows | %s -> %s",
                resolved,
                timeframe,
                len(df),
                actual_start,
                actual_end,
            )

            #Persisting CSV cache
            df.to_csv(csv_path, index=False)

            #Persisting pickle cache (fast reload path)
            if self.config.use_pickle_cache:
                df.to_pickle(pkl_path)

            return df

        finally:
            #Ensuring MT5 shutdown (even on failure)
            try:
                mt5.shutdown()
            except Exception:
                pass

    def _fetch_chunked(self, symbol: str, tf, start: datetime, end: datetime) -> pd.DataFrame:

        #Chunk accumulator
        chunks: list[pd.DataFrame] = []
        cur = start

        while cur < end:
            nxt = min(cur + timedelta(days=self.config.chunk_days), end)

            #Requesting chunk from MT5
            rates = mt5.copy_rates_range(symbol, tf, cur, nxt)

            #Appending valid chunk
            if rates is not None and len(rates) > 0:
                chunks.append(pd.DataFrame(rates))
            else:
                code, msg = mt5.last_error()
                log.debug("Chunk failed %s -> %s (%s:%s)", cur, nxt, code, msg)

            cur = nxt

        #Returning empty if no data retrieved
        if not chunks:
            return pd.DataFrame()

        #Merging chunks + removing duplicates
        df = pd.concat(chunks, ignore_index=True)
        df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

        return df

    def _resolve_timeframe(self, tf: TimeframeStr):

        #Mapping timeframe string → MT5 enum
        return {
            "M5": mt5.TIMEFRAME_M5,
            "M10": mt5.TIMEFRAME_M10,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
        }[tf]

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:

        #Ensuring required column exists
        if "time" not in df.columns:
            raise ValueError("Expected a 'time' column in raw MT5 data.")

        df = df.copy()

        #Converting timestamps to UTC datetime
        if pd.api.types.is_numeric_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
        else:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

        #Selecting standard OHLCV schema
        out = df[["time", "open", "high", "low", "close", "tick_volume"]].copy()

        #Cleaning invalid timestamps + duplicates
        out = out.dropna(subset=["time"])
        out = out.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

        return out

    def _coerce_utc(self, value: datetime) -> pd.Timestamp:

        #Ensuring datetime is UTC-aware
        ts = pd.Timestamp(value)

        if ts.tzinfo is None:
            return ts.tz_localize(timezone.utc)

        return ts.tz_convert(timezone.utc)

    def _clip_to_window(self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:

        #Filtering strictly within requested window
        clipped = df[(df["time"] >= start) & (df["time"] <= end)].copy()
        return clipped.reset_index(drop=True)

    def _cache_matches_window(self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:

        #Rejecting empty cache
        if df.empty:
            return False

        actual_start = df["time"].min()
        actual_end = df["time"].max()

        #Rejecting partial or misaligned cache
        if actual_start < start or actual_end > end:
            return False

        return True

    def _base_name(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> str:

        #Standardized cache file naming
        return f"{symbol}_{timeframe}_{start.date()}_{end.date()}"

    def _get_csv_path(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pathlib.Path:
        return self.config.data_dir / f"{self._base_name(symbol, timeframe, start, end)}.csv"

    def _get_pickle_path(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pathlib.Path:
        return self.config.data_dir / f"{self._base_name(symbol, timeframe, start, end)}.pkl"

    def _load_from_terminal_cache(self, symbol: str, tf) -> pd.DataFrame:

        #Pulling latest available data from MT5 terminal cache
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, 200000)

        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        return pd.DataFrame(rates)


def fetch_data(
    symbol,
    timeframe,
    start,
    end,
    *,
    data_dir=None,
    overwrite=False,
    use_pickle_cache=True,
    broker_symbol=None,
):

    #Creating fetcher with runtime config
    fetcher = MT5DataFetcher(
        FetchConfig(
            data_dir=pathlib.Path(data_dir) if data_dir is not None else pathlib.Path("data/raw"),
            overwrite=overwrite,
            use_pickle_cache=use_pickle_cache,
        )
    )

    #Delegating to core fetch logic
    return fetcher.fetch(symbol, timeframe, start, end, broker_symbol=broker_symbol)