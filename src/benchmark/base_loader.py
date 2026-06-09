"""
Base Loader — Abstract base class für alle Datenquellen.

Jeder konkrete Loader (yfinance, ICCO, ERA5, ...) erbt von BaseLoader und
implementiert mindestens fetch() und parse().

Design-Prinzip: Alle Loader liefern ein pandas.DataFrame mit DatetimeIndex
(timezone-aware, UTC) und konsistenten Spaltennamen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class BaseLoader(ABC):
    """
    Abstract base class für Daten-Loader.

    Subclasses müssen implementieren:
        - fetch(): Holt Rohdaten von der Quelle
        - parse(): Konvertiert Rohdaten in standardisiertes DataFrame

    Convenience: load() kombiniert fetch + parse + cache + validate.
    """

    SOURCE_NAME: str = "base"
    REQUIRED_COLUMNS: tuple[str, ...] = ("close",)

    def __init__(
        self,
        cache_dir: Path | str = "data/raw",
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.force_refresh = force_refresh

    # --------------------------------------------------------------------------
    # Abstract Methods
    # --------------------------------------------------------------------------
    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime | None = None,
        **kwargs: Any,
    ) -> Any:
        """Holt Rohdaten von der Quelle. Return-Typ ist quellspezifisch."""
        ...

    @abstractmethod
    def parse(self, raw_data: Any) -> pd.DataFrame:
        """
        Konvertiert Rohdaten in standardisiertes DataFrame.

        Returns:
            DataFrame mit DatetimeIndex (UTC) und mindestens REQUIRED_COLUMNS.
        """
        ...

    # --------------------------------------------------------------------------
    # Concrete Methods
    # --------------------------------------------------------------------------
    def load(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Lädt Daten: Cache → Fetch → Parse → Validate → Cache.

        Args:
            symbol: Asset/Series-Identifier (z.B. 'CC=F').
            start: Start-Datum (ISO-String oder datetime).
            end: End-Datum. None = heute.
            **kwargs: Weitere Argumente, an fetch() durchgereicht.

        Returns:
            Standardisiertes DataFrame.
        """
        cache_path = self._cache_path(symbol, start, end)

        if self.use_cache and not self.force_refresh and cache_path.exists():
            logger.info(f"[{self.SOURCE_NAME}] Cache-Hit: {cache_path}")
            df = pd.read_parquet(cache_path)
            return self._validate(df)

        logger.info(f"[{self.SOURCE_NAME}] Fetching {symbol} from {start} to {end}")
        raw = self.fetch(symbol, start, end, **kwargs)
        df = self.parse(raw)
        df = self._validate(df)

        if self.use_cache:
            df.to_parquet(cache_path, compression="snappy")
            logger.info(f"[{self.SOURCE_NAME}] Cached: {cache_path}")

        return df

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
    def _cache_path(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime | None,
    ) -> Path:
        """Eindeutiger Cache-Pfad pro (source, symbol, date-range)."""
        safe_symbol = symbol.replace("=", "_").replace("/", "_").replace("^", "")
        start_str = pd.Timestamp(start).strftime("%Y%m%d")
        end_str = pd.Timestamp(end).strftime("%Y%m%d") if end else "latest"
        filename = f"{self.SOURCE_NAME}_{safe_symbol}_{start_str}_{end_str}.parquet"
        return self.cache_dir / filename

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Überprüft Schema, Index, Spalten."""
        if df.empty:
            raise ValueError(f"[{self.SOURCE_NAME}] DataFrame ist leer.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                f"[{self.SOURCE_NAME}] Index muss DatetimeIndex sein, "
                f"ist aber {type(df.index).__name__}"
            )

        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"[{self.SOURCE_NAME}] Pflicht-Spalten fehlen: {missing}. "
                f"Vorhanden: {list(df.columns)}"
            )

        # Index sortieren, Duplikate entfernen
        df = df.sort_index()
        n_dup = df.index.duplicated().sum()
        if n_dup > 0:
            logger.warning(f"[{self.SOURCE_NAME}] {n_dup} duplizierte Indizes entfernt.")
            df = df[~df.index.duplicated(keep="last")]

        # NaN-Check auf Close
        n_nan = df["close"].isna().sum()
        if n_nan > 0:
            logger.warning(
                f"[{self.SOURCE_NAME}] {n_nan} NaN-Werte in 'close' "
                f"({100*n_nan/len(df):.2f}%)"
            )

        return df

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(cache_dir={self.cache_dir})>"
