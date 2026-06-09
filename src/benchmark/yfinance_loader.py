"""
yfinance Loader — ICE Futures via Yahoo Finance.

Symbole:
    CC=F : ICE NY Cocoa Futures (Continuous Front Month, USD/tonne)
    KC=F : ICE NY Coffee C / Arabica (Continuous Front Month, cents/lb)
    CT=F : ICE NY Cotton (für später)
    SB=F : ICE NY Sugar #11 (für später)

CLI Usage:
    python -m benchmark.yfinance_loader --asset cocoa --start 2000-01-01
    python -m benchmark.yfinance_loader --asset coffee --start 2000-01-01
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from benchmark.base_loader import BaseLoader


class YFinanceLoader(BaseLoader):
    """Lädt OHLCV-Daten von Yahoo Finance."""

    SOURCE_NAME = "yfinance"
    REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

    ASSET_SYMBOLS: dict[str, str] = {
        "cocoa": "CC=F",
        "coffee": "KC=F",
        "coffee_arabica": "KC=F",
        "cotton": "CT=F",
        "sugar": "SB=F",
    }

    def __init__(
        self,
        cache_dir: Path | str = "data/raw/prices",
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> None:
        super().__init__(cache_dir=cache_dir, use_cache=use_cache, force_refresh=force_refresh)

    # --------------------------------------------------------------------------
    # Fetch
    # --------------------------------------------------------------------------
    def fetch(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Holt OHLCV von Yahoo Finance.

        Args:
            symbol: yfinance-Symbol (z.B. 'CC=F') ODER unser asset-key ('cocoa').
            start: Start-Datum.
            end: End-Datum oder None (= heute).
            interval: '1d', '1wk', '1mo'. Default daily.

        Returns:
            Roh-DataFrame von yfinance.
        """
        # Asset-Key → Symbol auflösen
        symbol_resolved = self.ASSET_SYMBOLS.get(symbol.lower(), symbol)

        ticker = yf.Ticker(symbol_resolved)
        df = ticker.history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            actions=False,
        )

        if df.empty:
            raise ValueError(
                f"[yfinance] Keine Daten für {symbol_resolved} "
                f"({start} → {end}). Symbol korrekt?"
            )

        logger.info(
            f"[yfinance] {symbol_resolved}: {len(df)} Datenpunkte "
            f"({df.index.min().date()} → {df.index.max().date()})"
        )

        return df

    # --------------------------------------------------------------------------
    # Parse
    # --------------------------------------------------------------------------
    def parse(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardisiert yfinance OHLCV in unser Schema.

        Spaltennamen werden auf lowercase normalisiert.
        Index wird timezone-aware (UTC).
        """
        df = raw_data.copy()

        # Multi-Index-Spalten flatten (kommt vor bei manchen yfinance-Versionen)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Spalten lowercase
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Index → UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "date"

        # Nur Pflicht-Spalten behalten
        keep_cols = [c for c in self.REQUIRED_COLUMNS if c in df.columns]
        df = df[keep_cols]

        # Returns berechnen (Convenience — Modelle können auch selbst rechnen)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["simple_return"] = df["close"].pct_change()

        return df


# ==============================================================================
# CLI
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Lade Futures-Preise via yfinance.")
    parser.add_argument(
        "--asset",
        type=str,
        required=True,
        choices=list(YFinanceLoader.ASSET_SYMBOLS.keys()) + ["all"],
        help="Asset-Key (cocoa, coffee, ...) oder 'all'.",
    )
    parser.add_argument("--start", type=str, default="2000-01-01", help="Start-Datum YYYY-MM-DD.")
    parser.add_argument("--end", type=str, default=None, help="End-Datum YYYY-MM-DD.")
    parser.add_argument("--interval", type=str, default="1d", help="1d, 1wk, 1mo.")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Cache ignorieren und neu von yfinance laden.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/raw/prices",
        help="Cache-Verzeichnis.",
    )

    args = parser.parse_args()

    loader = YFinanceLoader(
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh,
    )

    assets = (
        list(YFinanceLoader.ASSET_SYMBOLS.keys())
        if args.asset == "all"
        else [args.asset]
    )

    for asset in assets:
        try:
            df = loader.load(
                symbol=asset,
                start=args.start,
                end=args.end,
                interval=args.interval,
            )
            logger.success(
                f"✓ {asset}: {len(df)} Zeilen, "
                f"{df.index.min().date()} → {df.index.max().date()}"
            )
            logger.info(f"  Spalten: {list(df.columns)}")
            logger.info(f"  Last Close: {df['close'].iloc[-1]:.2f}")
        except Exception as e:
            logger.error(f"✗ {asset}: {e}")


if __name__ == "__main__":
    main()
