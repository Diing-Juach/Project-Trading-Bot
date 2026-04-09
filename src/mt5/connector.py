from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import os
import time
import logging
import difflib

try:
    import MetaTrader5 as mt5
except Exception as e:  # pragma: no cover
    mt5 = None  # type: ignore
    _MT5_IMPORT_ERROR = e
else:
    _MT5_IMPORT_ERROR = None


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MT5Credentials:
    login: int
    password: str
    server: str


def resolve_symbol_name(symbol: str) -> str:
    # Ensuring MT5 package is available
    if mt5 is None:  # pragma: no cover
        raise ImportError("MetaTrader5 package is not available.")

    # Trying exact symbol match first
    if mt5.symbol_info(symbol) is not None:
        if not mt5.symbol_select(symbol, True):
            code, msg = mt5.last_error()
            raise RuntimeError(f"Failed to select symbol {symbol!r}. last_error={code}:{msg}")
        return symbol

    # Searching all broker symbols containing the requested base
    base = symbol.upper()
    symbols = mt5.symbols_get()
    if symbols is None:
        code, msg = mt5.last_error()
        raise RuntimeError(f"Failed to list symbols. last_error={code}:{msg}")

    candidates = [item.name for item in symbols if base in item.name.upper()]

    # Returning a single unambiguous match
    if len(candidates) == 1:
        if not mt5.symbol_select(candidates[0], True):
            code, msg = mt5.last_error()
            raise RuntimeError(f"Failed to select symbol {candidates[0]!r}. last_error={code}:{msg}")
        return candidates[0]

    # Preferring symbols that start with the requested base
    if len(candidates) > 1:
        starts = [name for name in candidates if name.upper().startswith(base)]
        if len(starts) == 1:
            if not mt5.symbol_select(starts[0], True):
                code, msg = mt5.last_error()
                raise RuntimeError(f"Failed to select symbol {starts[0]!r}. last_error={code}:{msg}")
            return starts[0]

        # Picking the closest broker variant when several remain
        if len(starts) > 1:
            best = difflib.get_close_matches(base, starts, n=1)
            resolved = best[0] if best else starts[0]
            if not mt5.symbol_select(resolved, True):
                code, msg = mt5.last_error()
                raise RuntimeError(f"Failed to select symbol {resolved!r}. last_error={code}:{msg}")
            return resolved

        # Falling back to the closest partial match
        best = difflib.get_close_matches(base, candidates, n=1)
        resolved = best[0] if best else candidates[0]
        if not mt5.symbol_select(resolved, True):
            code, msg = mt5.last_error()
            raise RuntimeError(f"Failed to select symbol {resolved!r}. last_error={code}:{msg}")
        return resolved

    # Raising if the broker exposes no matching symbol
    raise ValueError(
        f"Unknown symbol: {symbol!r}. Broker symbols differ. "
        "Try listing symbols via connector.list_symbols('EURUSD*')."
    )


class MT5Connector:
    def __init__(
        self,
        creds: Optional[MT5Credentials] = None,
        *,
        path: Optional[str] = None,
        portable: bool = False,
        timeout_sec: float = 10.0,
        retries: int = 3,
        retry_backoff_sec: float = 1.0,
    ) -> None:
        if mt5 is None:  # pragma: no cover
            raise ImportError(
                "MetaTrader5 package is not available. Install it first."
            ) from _MT5_IMPORT_ERROR

        # Storing connection + retry settings
        self.creds = creds
        self.path = path
        self.portable = portable
        self.timeout_sec = float(timeout_sec)
        self.retries = int(retries)
        self.retry_backoff_sec = float(retry_backoff_sec)

        self._connected = False

    # ---------------------------
    # Construction helpers
    # ---------------------------
    @staticmethod
    def creds_from_env(
        login_key: str = "MT5_LOGIN",
        password_key: str = "MT5_PASSWORD",
        server_key: str = "MT5_SERVER",
    ) -> Optional[MT5Credentials]:
        # Reading broker credentials from environment variables
        login = os.getenv(login_key)
        password = os.getenv(password_key)
        server = os.getenv(server_key)

        # Returning no credentials if any field is missing
        if not login or not password or not server:
            return None

        # Validating MT5 login is numeric
        try:
            login_int = int(login)
        except ValueError as e:
            raise ValueError(f"{login_key} must be an integer, got {login!r}") from e
        return MT5Credentials(login=login_int, password=password, server=server)

    # ---------------------------
    # Connection lifecycle
    # ---------------------------
    def connect(self) -> None:
        if self._connected:
            return

        # Ensuring MT5 bindings are available
        self._assert_mt5()

        last_err: Optional[Tuple[int, str]] = None

        # Retrying terminal initialize/login a few times before failing
        for attempt in range(1, self.retries + 1):
            ok = self._initialize_terminal()
            if ok:
                if self.creds is not None:
                    ok = self._login(self.creds)

            if ok:
                self._connected = True
                return

            last_err = mt5.last_error()

            # Logging the last MT5 error for each failed attempt
            log.warning(
                "MT5 connect attempt %s/%s failed: %s",
                attempt,
                self.retries,
                last_err,
            )
            time.sleep(self.retry_backoff_sec * attempt)

        # Resetting terminal state before raising
        try:
            mt5.shutdown()
        except Exception:
            pass

        if last_err:
            code, msg = last_err
            raise RuntimeError(f"Failed to connect to MT5. last_error={code}:{msg}")
        raise RuntimeError("Failed to connect to MT5 (unknown error).")

    def disconnect(self) -> None:
        # Shutting down the MT5 connection cleanly
        if mt5 is None:
            return
        try:
            mt5.shutdown()
        finally:
            self._connected = False

    def __enter__(self) -> "MT5Connector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    @property
    def connected(self) -> bool:
        return self._connected

    # ---------------------------
    # Symbol utilities
    # ---------------------------
    def ensure_symbol(self, symbol: str) -> None:
        self._require_connected()

        # Resolving broker-specific symbol name
        symbol = self.resolve_symbol(symbol)
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Unknown symbol: {symbol!r}")

        # Selecting symbol into Market Watch if needed
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                code, msg = mt5.last_error()
                raise RuntimeError(
                    f"Failed to select symbol {symbol!r}. last_error={code}:{msg}"
                )

    def symbol_info(self, symbol: str) -> Dict[str, Any]:
        # Returning symbol metadata as a plain dictionary
        self._require_connected()
        self.ensure_symbol(symbol)
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Unknown symbol: {symbol!r}")
        return info._asdict()

    def list_symbols(self, pattern: Optional[str] = None) -> List[str]:
        # Listing terminal symbols, optionally with MT5 pattern filtering
        self._require_connected()
        syms = mt5.symbols_get(pattern) if pattern else mt5.symbols_get()
        if syms is None:
            code, msg = mt5.last_error()
            raise RuntimeError(f"Failed to list symbols. last_error={code}:{msg}")
        return [s.name for s in syms]

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _assert_mt5(self) -> None:
        # Guarding methods that require the MT5 package
        if mt5 is None:  # pragma: no cover
            raise ImportError("MetaTrader5 package is not available.")

    def _require_connected(self) -> None:
        # Guarding methods that require an active MT5 connection
        self._assert_mt5()
        if not self._connected:
            raise RuntimeError("MT5 is not connected. Call connect() first.")

    def _initialize_terminal(self) -> bool:
        # Starting the configured MT5 terminal instance
        if self.path:
            ok = mt5.initialize(path=self.path, portable=self.portable, timeout=int(self.timeout_sec * 1000))
        else:
            ok = mt5.initialize(timeout=int(self.timeout_sec * 1000))
        return bool(ok)

    def _login(self, creds: MT5Credentials) -> bool:
        # Logging into the initialized terminal session
        ok = mt5.login(creds.login, password=creds.password, server=creds.server, timeout=int(self.timeout_sec * 1000))
        return bool(ok)

    def resolve_symbol(self, symbol: str) -> str:
        # Resolving the requested symbol while connected
        self._require_connected()
        return resolve_symbol_name(symbol)
