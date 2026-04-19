# core/historical_data_updater.py
"""
Historical Data Updater - Alpaca-specific data fetching

LEGACY MODULE: Consider using UnifiedDataPipeline instead for new code.

    from core.unified_data_pipeline import UnifiedDataPipeline
    pipeline = UnifiedDataPipeline()
    await pipeline.update_symbols(['AAPL'], source='alpaca')

This module is maintained for:
- BaseLiveRunner.seed() which uses Alpaca-specific functionality
- Backward compatibility with existing code

Features:
- Fetches historical bars from Alpaca API
- Saves to JSON format compatible with HistoricalBarLoader
- Supports incremental updates (only fetch new bars)
- Thread-safe file operations
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from loggers.logger import Logger


class HistoricalDataUpdater:
    """
    Fetches and stores historical bar data from Alpaca.

    Designed to work with HistoricalBarLoader which reads from
    proc_{SYMBOL}_file.json files.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        data_path: str | None = None,
        timeframe: TimeFrame = TimeFrame.Minute,
    ):
        """
        Initialize the historical data updater.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            data_path: Path to store data files. Defaults to data/data_storage/proc_data
            timeframe: Bar timeframe (default: 1 minute)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeframe = timeframe

        # Setup data path
        if data_path is None:
            root = Path(__file__).resolve().parents[1]
            data_path = root / "data" / "data_storage" / "proc_data"
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Initialize Alpaca client
        self.client = StockHistoricalDataClient(api_key, api_secret)

        # File lock for thread-safe writes
        self._file_lock = Lock()

        # Logger
        self.logger = Logger("historical_updater.log", "HistoricalDataUpdater", propagate=True).get_logger()

        self.logger.info(f"HistoricalDataUpdater initialized. Data path: {self.data_path}")

    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================

    async def update_symbol(self, symbol: str, days: int = 30, force_full: bool = False) -> int:
        """
        Update historical data for a single symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days: Number of days of history to fetch
            force_full: If True, fetch full history. Otherwise, incremental.

        Returns:
            Number of bars fetched
        """
        self.logger.info(f"[{symbol}] Updating historical data ({days} days)")

        try:
            # Determine start date
            if force_full:
                start = datetime.now(timezone.utc) - timedelta(days=days)
            else:
                # Check existing data for incremental update
                last_ts = self._get_last_timestamp(symbol)
                if last_ts:
                    # Fetch from last timestamp + 1 minute
                    start = last_ts + timedelta(minutes=1)
                    self.logger.debug(f"[{symbol}] Incremental from {start}")
                else:
                    start = datetime.now(timezone.utc) - timedelta(days=days)

            end = datetime.now(timezone.utc)

            # Skip if already up to date
            if start >= end:
                self.logger.info(f"[{symbol}] Already up to date")
                return 0

            # Fetch bars from Alpaca
            bars = await self._fetch_bars(symbol, start, end)

            if not bars:
                self.logger.warning(f"[{symbol}] No new bars fetched")
                return 0

            # Save to file
            self._save_bars(symbol, bars, append=not force_full)

            self.logger.info(f"[{symbol}] Updated with {len(bars)} bars")
            return len(bars)

        except Exception as e:
            self.logger.exception(f"[{symbol}] Failed to update: {e}")
            return 0

    async def update_symbols(
        self, symbols: list[str], days: int = 30, force_full: bool = False, max_concurrent: int = 5
    ) -> dict[str, int]:
        """
        Update historical data for multiple symbols.

        Args:
            symbols: List of stock symbols
            days: Number of days of history
            force_full: Force full refresh
            max_concurrent: Max concurrent API requests

        Returns:
            Dict of symbol -> bars fetched
        """
        self.logger.info(f"Updating {len(symbols)} symbols: {symbols}")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def update_with_semaphore(symbol: str) -> tuple[str, int]:
            async with semaphore:
                count = await self.update_symbol(symbol, days, force_full)
                return symbol, count

        tasks = [update_with_semaphore(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        output = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Update failed: {result}")
            else:
                symbol, count = result
                output[symbol] = count

        total = sum(output.values())
        self.logger.info(f"Update complete. Total bars: {total}")

        return output

    async def run_periodic(self, symbols: list[str], interval_minutes: int = 60, days: int = 5) -> None:
        """
        Run periodic updates as a background task.

        Args:
            symbols: List of symbols to update
            interval_minutes: Update interval in minutes
            days: Days of history to maintain
        """
        self.logger.info(f"Starting periodic updater: {len(symbols)} symbols, every {interval_minutes} minutes")

        while True:
            try:
                await self.update_symbols(symbols, days=days)
            except Exception as e:
                self.logger.exception(f"Periodic update failed: {e}")

            await asyncio.sleep(interval_minutes * 60)

    def get_data_freshness(self, symbol: str) -> dict[str, Any] | None:
        """
        Get information about data freshness for a symbol.

        Returns:
            Dict with last_timestamp, bar_count, age_minutes, or None if no data
        """
        file_path = self.data_path / f"proc_{symbol}_file.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                data = json.load(f)

            bars = data.get("bars", data) if isinstance(data, dict) else data
            if not bars:
                return None

            last_bar = bars[-1]
            last_ts_raw = last_bar.get("Date") or last_bar.get("timestamp")

            if isinstance(last_ts_raw, (int, float)) and last_ts_raw > 10_000_000_000:
                last_ts = datetime.fromtimestamp(last_ts_raw / 1000.0, tz=timezone.utc)
            else:
                last_ts = datetime.fromisoformat(str(last_ts_raw))
                # Ensure timezone-aware for comparison
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)

            age = datetime.now(timezone.utc) - last_ts

            return {
                "symbol": symbol,
                "last_timestamp": last_ts.isoformat(),
                "bar_count": len(bars),
                "age_minutes": int(age.total_seconds() / 60),
                "is_stale": age > timedelta(hours=1),
            }

        except Exception as e:
            self.logger.error(f"[{symbol}] Failed to get freshness: {e}")
            return None

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    async def _fetch_bars(self, symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """Fetch bars from Alpaca API."""

        def _do_fetch():
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=self.timeframe,
                start=start,
                end=end,
                feed=DataFeed.IEX,  # Use IEX feed (free tier compatible)
            )

            bars_response = self.client.get_stock_bars(request)

            # Convert to list of dicts
            bars = []
            # BarSet uses dict-like access with []
            symbol_bars = bars_response[symbol] if symbol in bars_response.data else []

            for bar in symbol_bars:
                bars.append(
                    {
                        "Date": int(bar.timestamp.timestamp() * 1000),  # epoch ms
                        "Symbol": symbol,
                        "Open": float(bar.open),
                        "High": float(bar.high),
                        "Low": float(bar.low),
                        "Close": float(bar.close),
                        "Volume": int(bar.volume),
                        "VWAP": float(bar.vwap) if hasattr(bar, "vwap") and bar.vwap else None,
                        "trade_count": int(bar.trade_count)
                        if hasattr(bar, "trade_count") and bar.trade_count
                        else None,
                    }
                )

            return bars

        # Run in thread pool to not block event loop
        return await asyncio.to_thread(_do_fetch)

    def _get_last_timestamp(self, symbol: str) -> datetime | None:
        """Get the last timestamp from existing data file."""
        file_path = self.data_path / f"proc_{symbol}_file.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                data = json.load(f)

            bars = data.get("bars", data) if isinstance(data, dict) else data
            if not bars:
                return None

            last_bar = bars[-1]
            ts_raw = last_bar.get("Date") or last_bar.get("timestamp")

            if isinstance(ts_raw, (int, float)) and ts_raw > 10_000_000_000:
                return datetime.fromtimestamp(ts_raw / 1000.0, tz=timezone.utc)
            else:
                return datetime.fromisoformat(str(ts_raw)).replace(tzinfo=timezone.utc)

        except Exception as e:
            self.logger.warning(f"[{symbol}] Failed to read last timestamp: {e}")
            return None

    def _save_bars(self, symbol: str, new_bars: list[dict[str, Any]], append: bool = True) -> None:
        """Save bars to JSON file."""
        file_path = self.data_path / f"proc_{symbol}_file.json"

        with self._file_lock:
            try:
                existing_bars = []

                if append and file_path.exists():
                    with open(file_path) as f:
                        data = json.load(f)
                    existing_bars = data.get("bars", data) if isinstance(data, dict) else data

                # Merge and deduplicate by timestamp
                all_bars = existing_bars + new_bars
                seen_timestamps = set()
                unique_bars = []

                for bar in all_bars:
                    ts = bar.get("Date") or bar.get("timestamp")
                    if ts not in seen_timestamps:
                        seen_timestamps.add(ts)
                        unique_bars.append(bar)

                # Sort by timestamp
                unique_bars.sort(key=lambda b: b.get("Date") or b.get("timestamp") or 0)

                # Keep only last N bars to prevent unbounded growth
                max_bars = 50000  # ~34 days of 1-min bars
                if len(unique_bars) > max_bars:
                    unique_bars = unique_bars[-max_bars:]

                # Save
                with open(file_path, "w") as f:
                    json.dump({"bars": unique_bars}, f)

                self.logger.debug(f"[{symbol}] Saved {len(unique_bars)} bars to {file_path}")

            except Exception as e:
                self.logger.exception(f"[{symbol}] Failed to save bars: {e}")
                raise


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================


async def main():
    """Standalone entry point for historical data updates."""
    import argparse

    from dotenv import load_dotenv

    # Load environment
    load_dotenv()

    parser = argparse.ArgumentParser(description="Update historical data from Alpaca")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"], help="Symbols to update")
    parser.add_argument("--days", type=int, default=30, help="Days of history to fetch")
    parser.add_argument("--force", action="store_true", help="Force full refresh (not incremental)")
    parser.add_argument("--periodic", type=int, default=0, help="Run periodically every N minutes (0 = one-shot)")

    args = parser.parse_args()

    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID")
    api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET")

    if not api_key or not api_secret:
        print("Error: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        return

    updater = HistoricalDataUpdater(api_key, api_secret)

    # Check current freshness
    print("\nCurrent data freshness:")
    for symbol in args.symbols:
        freshness = updater.get_data_freshness(symbol)
        if freshness:
            print(
                f"  {symbol}: {freshness['bar_count']} bars, "
                f"age={freshness['age_minutes']} min, "
                f"stale={freshness['is_stale']}"
            )
        else:
            print(f"  {symbol}: No data")

    print()

    if args.periodic > 0:
        print(f"Running periodic updates every {args.periodic} minutes...")
        await updater.run_periodic(args.symbols, args.periodic, args.days)
    else:
        results = await updater.update_symbols(args.symbols, args.days, args.force)
        print("\nUpdate results:")
        for symbol, count in results.items():
            print(f"  {symbol}: {count} bars fetched")


if __name__ == "__main__":
    asyncio.run(main())
