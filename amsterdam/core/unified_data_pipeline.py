# core/unified_data_pipeline.py
"""
Unified Data Pipeline - Fetches and processes historical data from multiple sources

Features:
- Supports Alpaca and Schwab as data sources
- Automatic fallback if one source is unavailable
- Multi-timeframe support (1min, 5min, 15min, 30min, 1hour, day)
- Same processing pipeline for both sources
- Stores to both JSON files AND SQLite database
- Updates cache for tracking processed dates
- Compatible with existing HistoricalBarLoader format

Usage:
    pipeline = UnifiedDataPipeline()

    # Auto-select best source
    await pipeline.update_symbols(['AAPL', 'MSFT'])

    # Multi-timeframe fetch
    await pipeline.update_symbols(
        ['AAPL', 'TSLA'],
        timeframes=['1min', '5min', '15min', '30min', '1hour']
    )

    # Force specific source
    await pipeline.update_symbols(['AAPL'], source='alpaca')
    await pipeline.update_symbols(['AAPL'], source='schwab')

    # Run as periodic background task
    await pipeline.run_periodic(symbols=['AAPL', 'MSFT'], interval_minutes=60)
"""

from __future__ import annotations

import os
import json
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from loggers.logger import Logger
from core.credential_validator import CredentialValidator, CredentialStatus
from data.datastorage import DataStore
from cache.cache import CacheManager


class UnifiedDataPipeline:
    """
    Unified data pipeline that fetches from Alpaca or Schwab
    and processes through the same pipeline.

    Storage:
    - JSON files (proc_{SYMBOL}_{TIMEFRAME}.json, raw_{SYMBOL}_{TIMEFRAME}.json)
    - SQLite database (stock_base.db with timeframe column)
    - Cache tracking (cache/system_cache.json)

    Supported Timeframes:
    - 1min, 5min, 15min, 30min, 1hour (intraday)
    - day (daily)
    """

    # Supported timeframes mapping
    SUPPORTED_TIMEFRAMES = {
        '1min': ('Minute', 1),
        '5min': ('Minute', 5),
        '15min': ('Minute', 15),
        '30min': ('Minute', 30),
        '1hour': ('Hour', 1),
        'day': ('Day', 1),
    }

    def __init__(
        self,
        data_path: Optional[str] = None,
        raw_data_path: Optional[str] = None,
        database: str = 'stock_base.db',
        store_to_db: bool = True,
        store_to_files: bool = True,
    ):
        """
        Initialize the unified data pipeline.

        Args:
            data_path: Path for processed data. Defaults to data/data_storage/proc_data
            raw_data_path: Path for raw data. Defaults to data/data_storage/raw_data
            database: SQLite database filename
            store_to_db: Whether to store data in SQLite database
            store_to_files: Whether to store data in JSON files
        """
        self.root = Path(__file__).resolve().parents[1]

        # Storage options
        self.store_to_db = store_to_db
        self.store_to_files = store_to_files

        # Setup paths
        if data_path is None:
            data_path = self.root / "data" / "data_storage" / "proc_data"
        if raw_data_path is None:
            raw_data_path = self.root / "data" / "data_storage" / "raw_data"

        self.data_path = Path(data_path)
        self.raw_data_path = Path(raw_data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        # Logger - dedicated file with propagation to app.log
        self.logger = Logger(
            "unified_pipeline.log",
            "UnifiedDataPipeline",
            propagate=True,  # Also logs to app.log for centralized tracking
            level=10,  # DEBUG level for detailed tracking
        ).get_logger()

        # Database storage
        self.database = database
        self.table_name = 'stock_table'
        self._datastore: Optional[DataStore] = None

        # Cache manager for tracking processed dates
        try:
            self.cache = CacheManager()
        except Exception as e:
            self.logger.warning(f"Cache manager not available: {e}")
            self.cache = None

        # Credential validator
        self.validator = CredentialValidator()

        # Clients (lazy initialized)
        self._alpaca_data_client = None
        self._schwab_client = None

        self.logger.info(
            f"UnifiedDataPipeline initialized. "
            f"Data: {self.data_path}, DB: {database}, "
            f"store_to_db={store_to_db}, store_to_files={store_to_files}"
        )

    async def check_and_warn_credentials(self) -> Dict[str, Any]:
        """
        Check credentials and log warnings for expiring tokens.

        Call this at startup to get proactive warnings about Schwab tokens.

        Returns:
            Dict with credential status for each broker
        """
        results = await self.validator.validate_all()
        status = {}

        # Check Schwab
        schwab = results.get('schwab')
        if schwab:
            status['schwab'] = {
                'status': schwab.status.value,
                'can_fetch_data': schwab.can_fetch_data,
                'message': schwab.message,
            }

            if schwab.status == CredentialStatus.EXPIRED:
                self.logger.error(
                    "=" * 60 + "\n"
                    "SCHWAB TOKENS EXPIRED - Manual re-authentication required!\n"
                    "Run: python -m data.streaming.authenticator\n"
                    "=" * 60
                )
            elif schwab.status == CredentialStatus.EXPIRING_SOON:
                hours = schwab.expires_in // 3600 if schwab.expires_in else 0
                self.logger.warning(
                    f"SCHWAB TOKEN EXPIRING in {hours} hours. "
                    f"Consider re-authenticating soon: python -m data.streaming.authenticator"
                )
            elif schwab.status == CredentialStatus.MISSING:
                self.logger.info(
                    "Schwab not configured. Using Alpaca as primary data source."
                )

        # Check Alpaca
        alpaca = results.get('alpaca')
        if alpaca:
            status['alpaca'] = {
                'status': alpaca.status.value,
                'can_fetch_data': alpaca.can_fetch_data,
                'message': alpaca.message,
            }

            if not alpaca.can_fetch_data:
                self.logger.warning(f"Alpaca unavailable: {alpaca.message}")

        # Determine recommended source
        recommended = self.validator.get_best_data_source(results)
        status['recommended'] = recommended

        if recommended == 'none':
            self.logger.error(
                "NO DATA SOURCES AVAILABLE! Check your credentials:\n"
                "  - Alpaca: Set ALPACA_API_KEY and ALPACA_SECRET_KEY\n"
                "  - Schwab: Run python -m data.streaming.authenticator"
            )
        else:
            self.logger.info(f"Using {recommended.upper()} as primary data source")

        return status

    @property
    def datastore(self) -> DataStore:
        """Lazy-load the database connection."""
        if self._datastore is None:
            self._datastore = DataStore(self.database, use_config=False)
        return self._datastore

    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================

    async def update_symbols(
        self,
        symbols: List[str],
        days: int = 30,
        source: Optional[str] = None,
        force_reprocess: bool = False,
        process_data: bool = True,
        timeframes: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Update historical data for multiple symbols at specified timeframes.

        Data Source Hierarchy (when source=None or 'auto'):
        - Daily data: Schwab (primary) → Alpaca (fallback)
        - Intraday data: Alpaca (primary) → Schwab (fallback)

        Rationale:
        - Schwab: Excellent daily data (years of history)
        - Alpaca: Excellent intraday data (750+ days vs Schwab's 10 days)
        - Automatic fallback ensures reliability

        Args:
            symbols: List of stock symbols
            days: Number of days of history
            source: Data source selection:
                   - None or 'auto': Use intelligent hierarchy (recommended)
                   - 'alpaca': Force Alpaca only
                   - 'schwab': Force Schwab only
            force_reprocess: Force reprocessing even if data is up-to-date
            process_data: Whether to process through ML pipeline
            timeframes: List of timeframes to fetch (e.g., ['15min', '30min', '1hour', 'day'])
                       If None, defaults to ['day'] for backward compatibility

        Returns:
            Dict of (symbol, timeframe) -> bars fetched

        Example:
            # Intelligent auto-selection (recommended)
            await pipeline.update_symbols(
                ['AAPL', 'TSLA'],
                timeframes=['15min', '30min', '1hour', 'day']
            )
            # → Daily from Schwab, intraday from Alpaca

            # Force specific source
            await pipeline.update_symbols(
                ['AAPL'],
                source='alpaca',
                timeframes=['day']
            )
            # → All from Alpaca, no fallback
        """
        # Default to daily data for backward compatibility
        if timeframes is None:
            timeframes = ['day']
        # Validate timeframes
        valid_timeframes = [tf for tf in timeframes if tf in self.SUPPORTED_TIMEFRAMES]
        if not valid_timeframes:
            self.logger.error(f"No valid timeframes in {timeframes}. Supported: {list(self.SUPPORTED_TIMEFRAMES.keys())}")
            return {}

        invalid_timeframes = set(timeframes) - set(valid_timeframes)
        if invalid_timeframes:
            self.logger.warning(f"Skipping unsupported timeframes: {invalid_timeframes}")

        self.logger.info("=" * 60)
        self.logger.info(f"UPDATE SYMBOLS: {symbols}")
        self.logger.info(f"Timeframes: {valid_timeframes}")
        self.logger.info(f"Parameters: days={days}, source={source}, force_reprocess={force_reprocess}, process_data={process_data}")

        # Check credentials and warn about expiring tokens
        await self.check_and_warn_credentials()

        # Determine source mode
        if source is None:
            source = 'auto'
            self.logger.info("Source mode: AUTO (intelligent hierarchy per timeframe)")
        else:
            self.logger.info(f"Source mode: {source.upper()} (user-specified, no fallback)")

        # Display hierarchy being used
        if source == 'auto':
            daily_primary, daily_fallback = self._select_source_for_timeframe('day')
            intraday_primary, intraday_fallback = self._select_source_for_timeframe('15min')
            self.logger.info(
                f"Data source hierarchy:\n"
                f"  Daily:    {daily_primary.upper()} → {daily_fallback.upper()}\n"
                f"  Intraday: {intraday_primary.upper()} → {intraday_fallback.upper()}"
            )

        self.logger.info(f"Updating {len(symbols)} symbols × {len(valid_timeframes)} timeframes")

        results = {}
        start_time = datetime.now(timezone.utc)

        # Parallel updates with rate limiting (increased for scaling)
        semaphore = asyncio.Semaphore(20)  # Max 20 concurrent API requests

        async def update_with_limit(symbol: str, timeframe: str) -> tuple[str, str, int]:
            """Update a single symbol at specific timeframe with rate limiting."""
            async with semaphore:
                self.logger.info(f"[{symbol}/{timeframe}] Processing...")
                try:
                    count = await self._update_symbol_at_timeframe(
                        symbol, timeframe, days, source, force_reprocess, process_data
                    )
                    self.logger.info(f"[{symbol}/{timeframe}] SUCCESS: {count} bars fetched")
                    return (symbol, timeframe, count)
                except Exception as e:
                    self.logger.exception(f"[{symbol}/{timeframe}] FAILED: {e}")
                    return (symbol, timeframe, 0)

        # Run all updates in parallel with exception handling
        tasks = [update_with_limit(s, tf) for s in symbols for tf in valid_timeframes]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in task_results:
            if isinstance(result, Exception):
                self.logger.exception(f"Task failed: {result}")
            else:
                symbol, timeframe, count = result
                results[f"{symbol}_{timeframe}"] = count

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        total = sum(results.values())
        success_count = sum(1 for c in results.values() if c > 0)

        self.logger.info("=" * 60)
        self.logger.info(f"UPDATE COMPLETE in {elapsed:.1f}s")
        self.logger.info(f"Total bars: {total}, Success: {success_count}/{len(results)}")
        self.logger.info("=" * 60)

        return results

    async def run_periodic(
        self,
        symbols: List[str],
        interval_minutes: int = 60,
        days: int = 5,
        source: Optional[str] = None,
    ) -> None:
        """
        Run periodic updates as a background task.

        Args:
            symbols: List of symbols to update
            interval_minutes: Update interval in minutes
            days: Days of history to maintain
            source: Data source (None = auto-select)
        """
        self.logger.info(
            f"Starting periodic pipeline: {len(symbols)} symbols, "
            f"every {interval_minutes} minutes"
        )

        while True:
            try:
                await self.update_symbols(symbols, days=days, source=source)
            except Exception as e:
                self.logger.exception(f"Periodic update failed: {e}")

            await asyncio.sleep(interval_minutes * 60)

    async def check_sources(self) -> Dict[str, Any]:
        """
        Check available data sources.

        Returns:
            Dict with source availability and recommendations
        """
        results = await self.validator.validate_all()

        return {
            'alpaca': {
                'available': results['alpaca'].can_fetch_data,
                'status': results['alpaca'].status.value,
                'message': results['alpaca'].message,
            },
            'schwab': {
                'available': results['schwab'].can_fetch_data,
                'status': results['schwab'].status.value,
                'message': results['schwab'].message,
                'expires_in_hours': results['schwab'].expires_in // 3600 if results['schwab'].expires_in else None,
            },
            'recommended': self.validator.get_best_data_source(results),
        }

    def get_data_from_db(self, symbol: str, timeframe: str = 'day') -> pd.DataFrame:
        """
        Retrieve processed data from database for a symbol at specific timeframe.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe identifier

        Returns:
            DataFrame with processed data
        """
        try:
            with self.datastore as store:
                # Get all data for symbol, then filter by timeframe
                df = store.get_data_by_symbol(self.table_name, symbol)
                if not df.empty and 'timeframe' in df.columns:
                    df = df[df['timeframe'] == timeframe]
                if not df.empty:
                    self.logger.debug(f"[{symbol}/{timeframe}] Retrieved {len(df)} rows from database")
                return df
        except Exception as e:
            self.logger.error(f"[{symbol}/{timeframe}] Failed to retrieve from database: {e}")
            return pd.DataFrame()

    def get_data_from_file(self, symbol: str, timeframe: str = 'day') -> pd.DataFrame:
        """
        Retrieve processed data from JSON file for a symbol at specific timeframe.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe identifier

        Returns:
            DataFrame with processed data
        """
        if timeframe == 'day':
            file_path = self.data_path / f"proc_{symbol}_file.json"
        else:
            file_path = self.data_path / f"proc_{symbol}_{timeframe}.json"

        try:
            if not file_path.exists():
                self.logger.warning(f"[{symbol}/{timeframe}] File not found: {file_path}")
                return pd.DataFrame()

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Handle both 'bars' key and direct list format
            records = data.get('bars', data) if isinstance(data, dict) else data

            df = pd.DataFrame(records)
            self.logger.debug(f"[{symbol}/{timeframe}] Retrieved {len(df)} rows from file")
            return df

        except Exception as e:
            self.logger.error(f"[{symbol}/{timeframe}] Failed to retrieve from file: {e}")
            return pd.DataFrame()

    def get_data(self, symbol: str, timeframe: str = 'day', prefer_db: bool = False) -> pd.DataFrame:
        """
        Retrieve processed data from best available source for specific timeframe.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe identifier ('1min', '5min', '15min', '30min', '1hour', 'day')
            prefer_db: If True, prefer database over files

        Returns:
            DataFrame with processed data

        Example:
            # Get 5-minute bars for AAPL
            bars_5min = pipeline.get_data('AAPL', timeframe='5min')

            # Get daily bars (default)
            bars_daily = pipeline.get_data('AAPL')
        """
        if prefer_db:
            df = self.get_data_from_db(symbol, timeframe)
            if not df.empty:
                return df
            return self.get_data_from_file(symbol, timeframe)
        else:
            df = self.get_data_from_file(symbol, timeframe)
            if not df.empty:
                return df
            return self.get_data_from_db(symbol, timeframe)

    def load_symbol_data(self, symbol: str, timeframe: str = 'day', prefer_db: bool = False) -> pd.DataFrame:
        """
        Alias for get_data() - Load processed data for a symbol at specific timeframe.

        This method exists for backward compatibility with code that uses
        load_symbol_data() instead of get_data().

        Args:
            symbol: Stock symbol
            timeframe: Timeframe identifier
            prefer_db: If True, prefer database over files

        Returns:
            DataFrame with processed data
        """
        return self.get_data(symbol, timeframe=timeframe, prefer_db=prefer_db)

    def list_available_symbols(self, source: str = 'file', timeframe: Optional[str] = None) -> List[str]:
        """
        List symbols with available data, optionally filtered by timeframe.

        Args:
            source: 'file' or 'db'
            timeframe: Optional timeframe filter (e.g., '5min', '15min')

        Returns:
            List of symbol names
        """
        if source == 'file':
            if timeframe:
                # Look for specific timeframe files
                pattern = f"proc_*_{timeframe}.json"
            else:
                # Look for all processed files
                pattern = "proc_*.json"

            files = list(self.data_path.glob(pattern))
            symbols = []
            for f in files:
                name = f.stem  # proc_AAPL_5min or proc_AAPL_file
                parts = name.split('_')
                if len(parts) >= 2:
                    # Extract symbol (second part)
                    symbols.append(parts[1])
            return sorted(set(symbols))  # Remove duplicates

        elif source == 'db':
            try:
                with self.datastore as store:
                    if not store.table_exists(self.table_name):
                        return []
                    df = store.get_data_base(self.table_name)
                    if 'symbol' in df.columns:
                        if timeframe and 'timeframe' in df.columns:
                            df = df[df['timeframe'] == timeframe]
                        return sorted(df['symbol'].unique().tolist())
                    return []
            except Exception as e:
                self.logger.error(f"Failed to list symbols from database: {e}")
                return []

        return []

    def list_available_timeframes(self, symbol: str, source: str = 'file') -> List[str]:
        """
        List available timeframes for a specific symbol.

        Args:
            symbol: Stock symbol
            source: 'file' or 'db'

        Returns:
            List of timeframe identifiers

        Example:
            timeframes = pipeline.list_available_timeframes('AAPL')
            # Returns: ['1min', '5min', '15min', 'day']
        """
        if source == 'file':
            files = list(self.data_path.glob(f"proc_{symbol}_*.json"))
            timeframes = []
            for f in files:
                name = f.stem  # proc_AAPL_5min or proc_AAPL_file
                parts = name.split('_')
                if len(parts) >= 3:
                    # Extract timeframe (third part onwards, excluding 'file')
                    tf = '_'.join(parts[2:])
                    if tf != 'file':  # Skip old format
                        timeframes.append(tf)
                    else:
                        timeframes.append('day')  # Old format is daily
            return sorted(set(timeframes))

        elif source == 'db':
            try:
                with self.datastore as store:
                    if not store.table_exists(self.table_name):
                        return []
                    df = store.get_data_by_symbol(self.table_name, symbol)
                    if 'timeframe' in df.columns:
                        return sorted(df['timeframe'].unique().tolist())
                    return []
            except Exception as e:
                self.logger.error(f"Failed to list timeframes from database: {e}")
                return []

        return []

    def get_cache_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cache information for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with cache info or None
        """
        if self.cache is None:
            return None

        try:
            last_date = self.cache.get_last_processed_date('stock_files', symbol)
            if last_date:
                # Convert epoch ms to datetime
                dt = datetime.fromtimestamp(last_date / 1000, tz=timezone.utc)
                return {
                    'symbol': symbol,
                    'last_processed_ms': last_date,
                    'last_processed_date': dt.isoformat(),
                    'age_hours': (datetime.now(timezone.utc) - dt).total_seconds() / 3600,
                }
            return None
        except Exception as e:
            self.logger.warning(f"[{symbol}] Failed to get cache info: {e}")
            return None

    # ========================================================================
    # PRIVATE METHODS - SOURCE SELECTION
    # ========================================================================

    async def _select_best_source(self) -> str:
        """Select the best available data source."""
        results = await self.validator.validate_all()
        return self.validator.get_best_data_source(results)

    def _select_source_for_timeframe(self, timeframe: str) -> tuple[str, str]:
        """
        Select optimal data source based on timeframe hierarchy.

        Hierarchy:
        - Daily data: Schwab (primary), Alpaca (fallback)
        - Intraday data: Alpaca (primary), Schwab (fallback)

        Args:
            timeframe: Timeframe to fetch (e.g., 'day', '15min', '1hour')

        Returns:
            Tuple of (primary_source, fallback_source)
        """
        if timeframe == 'day':
            # Daily: Schwab has excellent history, Alpaca as backup
            return ('schwab', 'alpaca')
        else:
            # Intraday: Alpaca has 750+ days, Schwab only has 10 days
            return ('alpaca', 'schwab')

    # ========================================================================
    # PRIVATE METHODS - DATA FETCHING
    # ========================================================================

    async def _update_symbol(
        self,
        symbol: str,
        days: int,
        source: str,
        force_reprocess: bool,
        process_data: bool,
    ) -> int:
        """Update data for a single symbol (backward compatibility - uses 'day' timeframe)."""
        return await self._update_symbol_at_timeframe(
            symbol, 'day', days, source, force_reprocess, process_data
        )

    async def _update_symbol_at_timeframe(
        self,
        symbol: str,
        timeframe: str,
        days: int,
        source: str,
        force_reprocess: bool,
        process_data: bool,
    ) -> int:
        """
        Update data for a single symbol at specific timeframe.

        Uses intelligent source hierarchy with automatic fallback:
        - Daily: Schwab (primary) → Alpaca (fallback)
        - Intraday: Alpaca (primary) → Schwab (fallback)
        """

        # Determine source hierarchy
        if source in ['auto', None]:
            primary_source, fallback_source = self._select_source_for_timeframe(timeframe)
            self.logger.info(
                f"[{symbol}/{timeframe}] Using source hierarchy: "
                f"{primary_source.upper()} (primary) → {fallback_source.upper()} (fallback)"
            )
        else:
            # User explicitly specified source - use it without fallback
            primary_source = source
            fallback_source = None
            self.logger.info(f"[{symbol}/{timeframe}] Using user-specified source: {source.upper()}")

        # Try primary source
        raw_bars = None
        try:
            if primary_source == 'alpaca':
                raw_bars = await self._fetch_alpaca(symbol, days, timeframe)
            elif primary_source == 'schwab':
                raw_bars = await self._fetch_schwab(symbol, days, timeframe)
            else:
                raise ValueError(f"Unknown source: {primary_source}")

            if raw_bars and len(raw_bars) > 0:
                self.logger.info(
                    f"[{symbol}/{timeframe}] ✓ Fetched {len(raw_bars)} bars from {primary_source.upper()}"
                )
            else:
                self.logger.warning(
                    f"[{symbol}/{timeframe}] {primary_source.upper()} returned no data"
                )
                raw_bars = None

        except Exception as e:
            self.logger.error(
                f"[{symbol}/{timeframe}] {primary_source.upper()} fetch failed: {e}"
            )
            raw_bars = None

        # Try fallback source if primary failed and fallback is available
        if not raw_bars and fallback_source:
            self.logger.warning(
                f"[{symbol}/{timeframe}] Trying fallback source: {fallback_source.upper()}"
            )
            try:
                if fallback_source == 'alpaca':
                    raw_bars = await self._fetch_alpaca(symbol, days, timeframe)
                elif fallback_source == 'schwab':
                    raw_bars = await self._fetch_schwab(symbol, days, timeframe)

                if raw_bars and len(raw_bars) > 0:
                    self.logger.info(
                        f"[{symbol}/{timeframe}] ✓ Fallback successful! "
                        f"Fetched {len(raw_bars)} bars from {fallback_source.upper()}"
                    )
                else:
                    self.logger.error(
                        f"[{symbol}/{timeframe}] Fallback {fallback_source.upper()} also returned no data"
                    )

            except Exception as e:
                self.logger.error(
                    f"[{symbol}/{timeframe}] Fallback {fallback_source.upper()} also failed: {e}"
                )

        # Final check
        if not raw_bars or len(raw_bars) == 0:
            self.logger.error(
                f"[{symbol}/{timeframe}] ✗ All sources failed - no data available"
            )
            return 0

        # Save raw data with timeframe
        self._save_raw_data(symbol, raw_bars, timeframe)

        # Process data if requested
        if process_data:
            await self._process_and_save(symbol, raw_bars, timeframe, force_reprocess=force_reprocess)

        self.logger.info(
            f"[{symbol}/{timeframe}] ✓ COMPLETE: {len(raw_bars)} bars saved and processed"
        )
        return len(raw_bars)

    async def _fetch_alpaca(self, symbol: str, days: int, timeframe: str = 'day') -> List[Dict[str, Any]]:
        """Fetch historical bars from Alpaca at specified timeframe."""
        try:
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            from alpaca.data.enums import DataFeed

            if self._alpaca_data_client is None:
                api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_KEY_ID')
                api_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_SECRET')
                self._alpaca_data_client = StockHistoricalDataClient(api_key, api_secret)

            start = datetime.now(timezone.utc) - timedelta(days=days)
            end = datetime.now(timezone.utc)

            # Convert timeframe string to Alpaca TimeFrame
            tf_unit, tf_amount = self.SUPPORTED_TIMEFRAMES[timeframe]
            if tf_unit == 'Day':
                alpaca_tf = TimeFrame.Day
            elif tf_unit == 'Hour':
                alpaca_tf = TimeFrame.Hour
            elif tf_unit == 'Minute':
                alpaca_tf = TimeFrame(tf_amount, TimeFrameUnit.Minute)
            else:
                raise ValueError(f"Unsupported timeframe unit: {tf_unit}")

            def _do_fetch():
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=alpaca_tf,
                    start=start,
                    end=end,
                    feed=DataFeed.IEX,  # Use IEX feed (free tier compatible)
                )
                response = self._alpaca_data_client.get_stock_bars(request)
                # BarSet uses dict-like access with []
                symbol_bars = response[symbol] if symbol in response.data else []

                # Convert to Schwab-compatible format
                bars = []
                for bar in symbol_bars:
                    bars.append({
                        'datetime': int(bar.timestamp.timestamp() * 1000),  # epoch ms
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': int(bar.volume),
                    })
                return bars

            return await asyncio.to_thread(_do_fetch)

        except Exception as e:
            self.logger.exception(f"[{symbol}/{timeframe}] Alpaca fetch failed: {e}")
            return []

    async def _fetch_schwab(self, symbol: str, days: int, timeframe: str = 'day') -> List[Dict[str, Any]]:
        """Fetch historical bars from Schwab at specified timeframe."""
        try:
            from data.streaming.schwab_client import SchwabClient
            from data.streaming.authenticator import Authenticator

            auth = Authenticator()

            # Ensure token is fresh (auto-refresh if needed)
            token = auth.access_token()
            if not token:
                self.logger.error("Failed to get valid Schwab access token")
                return []

            if self._schwab_client is None:
                self._schwab_client = SchwabClient(auth.apikey, auth.secret)

            def _do_fetch():
                # Calculate start and end dates (epoch ms)
                end_dt = datetime.now(timezone.utc)
                start_dt = end_dt - timedelta(days=days)
                start_ms = int(start_dt.timestamp() * 1000)
                end_ms = int(end_dt.timestamp() * 1000)

                # Map timeframe to Schwab API parameters
                if timeframe == 'day':
                    data = self._schwab_client.daily_price_history(symbol, start=start_ms, end=end_ms)

                elif timeframe in ['1min', '5min', '10min', '15min', '30min']:
                    # Extract minute value
                    minutes = int(timeframe.replace('min', ''))

                    # Schwab API limits for minute data:
                    # - Can only request up to 31 days of intraday data
                    # - frequencyType: 'minute', frequency: 1, 5, 10, 15, or 30
                    # - periodType: 'day', period: 1, 2, 3, 4, 5, or 10

                    # Limit days for intraday requests (Schwab API allows 1-10 days)
                    actual_days = min(days, 10)

                    if actual_days < days:
                        self.logger.info(
                            f"[{symbol}/{timeframe}] Schwab intraday limited to {actual_days} days "
                            f"(requested {days})"
                        )

                    # Use period-based approach (NOT date-based) per Schwab API docs
                    data = self._schwab_client.custom_price_history(
                        symbol=symbol,
                        periodType='day',
                        period=actual_days,
                        frequencyType='minute',
                        frequency=minutes
                    )

                elif timeframe == '1hour':
                    # For hourly data, use 60-minute frequency (Schwab allows 1-10 days)
                    actual_days = min(days, 10)

                    if actual_days < days:
                        self.logger.info(
                            f"[{symbol}/{timeframe}] Schwab hourly limited to {actual_days} days "
                            f"(requested {days})"
                        )

                    # Use period-based approach (NOT date-based) per Schwab API docs
                    data = self._schwab_client.custom_price_history(
                        symbol=symbol,
                        periodType='day',
                        period=actual_days,
                        frequencyType='minute',
                        frequency=60
                    )

                else:
                    self.logger.error(f"[{symbol}/{timeframe}] Unsupported timeframe: {timeframe}")
                    return []

                return data.get('candles', [])

            return await asyncio.to_thread(_do_fetch)

        except Exception as e:
            self.logger.exception(f"[{symbol}/{timeframe}] Schwab fetch failed: {e}")
            return []

    # ========================================================================
    # PRIVATE METHODS - DATA STORAGE
    # ========================================================================

    def _save_raw_data(self, symbol: str, bars: List[Dict[str, Any]], timeframe: str = 'day') -> None:
        """Save raw bar data to JSON file with timeframe identifier."""
        file_path = self.raw_data_path / f"raw_{symbol}_{timeframe}.json"

        try:
            # Load existing data if present
            existing_bars = []
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    existing_bars = data.get('candles', [])

            # Merge and deduplicate
            all_bars = existing_bars + bars
            seen = set()
            unique_bars = []
            for bar in all_bars:
                dt = bar.get('datetime')
                if dt not in seen:
                    seen.add(dt)
                    unique_bars.append(bar)

            # Sort by datetime
            unique_bars.sort(key=lambda b: b.get('datetime', 0))

            # Save
            with open(file_path, 'w') as f:
                json.dump({'candles': unique_bars, 'timeframe': timeframe}, f, indent=2)

            self.logger.debug(f"[{symbol}/{timeframe}] Raw data saved: {len(unique_bars)} bars")

        except Exception as e:
            self.logger.exception(f"[{symbol}/{timeframe}] Failed to save raw data: {e}")

    def _load_all_raw_data(self, symbol: str, timeframe: str = 'day') -> List[Dict[str, Any]]:
        """
        Load ALL raw data for a symbol at specific timeframe from the raw data file.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe identifier

        Returns:
            List of all raw bar dictionaries
        """
        file_path = self.raw_data_path / f"raw_{symbol}_{timeframe}.json"

        if not file_path.exists():
            # Try backward compatibility with old format (no timeframe suffix)
            old_file_path = self.raw_data_path / f"raw_{symbol}_file.json"
            if old_file_path.exists() and timeframe == 'day':
                file_path = old_file_path
            else:
                return []

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data.get('candles', [])
        except Exception as e:
            self.logger.error(f"[{symbol}/{timeframe}] Failed to load raw data: {e}")
            return []

    def _get_last_processed_timestamp(self, symbol: str, timeframe: str = 'day') -> Optional[int]:
        """
        Get the latest timestamp from processed data file for specific timeframe.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe identifier

        Returns:
            Latest timestamp (epoch ms) or None if no processed data
        """
        if timeframe == 'day':
            file_path = self.data_path / f"proc_{symbol}_file.json"
        else:
            file_path = self.data_path / f"proc_{symbol}_{timeframe}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                bars = data.get('bars', data) if isinstance(data, dict) else data

            if not bars:
                return None

            # Find the latest timestamp
            # Processed data uses 'Date' field (ISO string format)
            last_bar = bars[-1]
            date_str = last_bar.get('Date')

            if date_str:
                # Convert ISO string to epoch ms for comparison
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return int(dt.timestamp() * 1000)

            return None

        except Exception as e:
            self.logger.debug(f"[{symbol}/{timeframe}] Could not read processed timestamp: {e}")
            return None

    def _get_last_raw_timestamp(self, symbol: str, timeframe: str = 'day') -> Optional[int]:
        """
        Get the latest timestamp from raw data file for specific timeframe.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe identifier

        Returns:
            Latest timestamp (epoch ms) or None if no raw data
        """
        file_path = self.raw_data_path / f"raw_{symbol}_{timeframe}.json"

        if not file_path.exists():
            # Try backward compatibility
            old_file_path = self.raw_data_path / f"raw_{symbol}_file.json"
            if old_file_path.exists() and timeframe == 'day':
                file_path = old_file_path
            else:
                return None

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                candles = data.get('candles', [])

            if not candles:
                return None

            # Raw data uses 'datetime' field (epoch ms)
            return max(c.get('datetime', 0) for c in candles)

        except Exception as e:
            self.logger.debug(f"[{symbol}/{timeframe}] Could not read raw timestamp: {e}")
            return None

    def _is_processed_data_current(self, symbol: str, timeframe: str = 'day') -> bool:
        """
        Check if processed data is up-to-date with raw data for specific timeframe.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe identifier

        Returns:
            True if processed data contains the latest raw data
        """
        raw_ts = self._get_last_raw_timestamp(symbol, timeframe)
        proc_ts = self._get_last_processed_timestamp(symbol, timeframe)

        if raw_ts is None:
            return True  # No raw data, nothing to process

        if proc_ts is None:
            return False  # No processed data, needs processing

        # Allow 1 day tolerance (86400000 ms) for timezone differences
        return proc_ts >= (raw_ts - 86400000)

    async def _process_and_save(
        self,
        symbol: str,
        bars: List[Dict[str, Any]],
        timeframe: str = 'day',
        force_reprocess: bool = False,
    ) -> None:
        """
        Process raw data through the ML pipeline and save to all storage targets.

        This method checks if processed data is already up-to-date before reprocessing.
        When reprocessing is needed, it loads the COMPLETE raw data file to ensure
        indicators like SMA_200 are computed correctly.

        Args:
            symbol: Stock symbol
            bars: Newly fetched bars (used for cache update)
            timeframe: Timeframe identifier
            force_reprocess: If True, reprocess even if data appears current

        Storage targets:
        - JSON file (proc_{SYMBOL}_{TIMEFRAME}.json)
        - SQLite database (stock_table with timeframe column)
        - Cache update (system_cache.json)
        """
        try:
            from data.processor import Processor

            # Check if processed data is already current (unless forced)
            if not force_reprocess and self._is_processed_data_current(symbol, timeframe):
                self.logger.info(f"[{symbol}/{timeframe}] Processed data is up-to-date, skipping reprocess")
                return

            # Load ALL raw data from file (not just the passed-in bars)
            # This ensures indicators like SMA_200 are computed correctly
            all_bars = self._load_all_raw_data(symbol, timeframe)

            if not all_bars:
                self.logger.warning(f"[{symbol}/{timeframe}] No raw data to process")
                return

            self.logger.info(f"[{symbol}/{timeframe}] Processing {len(all_bars)} total bars (new data detected)")

            # Convert bars to DataFrame
            df = pd.DataFrame(all_bars)

            if df.empty:
                self.logger.warning(f"[{symbol}/{timeframe}] No data to process")
                return

            # Rename columns for processor (expects lowercase)
            # Schwab format: datetime, open, high, low, close, volume
            df = df.rename(columns={
                'datetime': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
            })

            # Ensure datetime column exists
            if 'datetime' not in df.columns:
                self.logger.error(f"[{symbol}/{timeframe}] Missing datetime column")
                return

            # Process using the Processor
            processor = Processor(stock=symbol, frame=df)

            # Run ML processing
            processed_df = processor.ml_process(
                sma_window=200,
                ema_window=50,
                scaling_method="standard",
                include_scaled=True,
                include_pca=False,
            )

            if processed_df is None or processed_df.empty:
                self.logger.warning(f"[{symbol}/{timeframe}] Processing returned empty DataFrame")
                return

            # Save to JSON files
            if self.store_to_files:
                self._save_processed_data_file(symbol, processed_df, timeframe)

            # Save to database
            if self.store_to_db:
                self._save_processed_data_db(symbol, processed_df, timeframe)

            # Update cache with latest timestamp
            self._update_cache(symbol, bars, timeframe)

            self.logger.info(
                f"[{symbol}/{timeframe}] Data saved: {len(processed_df)} rows "
                f"(files={self.store_to_files}, db={self.store_to_db})"
            )

        except Exception as e:
            self.logger.exception(f"[{symbol}/{timeframe}] Processing failed: {e}")

    def _save_processed_data_file(self, symbol: str, df: pd.DataFrame, timeframe: str = 'day') -> None:
        """Save processed DataFrame to JSON file with timeframe identifier."""
        # Use new format with timeframe, with backward compatibility for 'day'
        if timeframe == 'day':
            # For backward compatibility, also save to old filename format
            file_path = self.data_path / f"proc_{symbol}_file.json"
        else:
            file_path = self.data_path / f"proc_{symbol}_{timeframe}.json"

        try:
            # Make a copy to avoid modifying original
            df_copy = df.copy()

            # Reset index if it's a DatetimeIndex (processor sets Date as index)
            if isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy = df_copy.reset_index()

            # Convert datetime/Timestamp columns to ISO strings for JSON serialization
            for col in df_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
                elif df_copy[col].dtype == 'object':
                    # Handle any Timestamp objects in object columns
                    df_copy[col] = df_copy[col].apply(
                        lambda x: x.isoformat() if isinstance(x, (pd.Timestamp, datetime)) else x
                    )

            # Handle NaN and infinity values
            df_copy = df_copy.where(df_copy.notna(), None)
            df_copy = df_copy.replace({float('inf'): None, float('-inf'): None})

            # Convert to records
            data = df_copy.to_dict(orient='records')

            # Save with timeframe metadata
            with open(file_path, 'w') as f:
                json.dump({'bars': data, 'timeframe': timeframe}, f)

            self.logger.debug(f"[{symbol}/{timeframe}] Saved to file: {len(data)} rows")

        except Exception as e:
            self.logger.exception(f"[{symbol}/{timeframe}] Failed to save to file: {e}")

    def _save_processed_data_db(self, symbol: str, df: pd.DataFrame, timeframe: str = 'day') -> None:
        """Save processed DataFrame to SQLite database with timeframe column."""
        try:
            # Make a copy and add symbol and timeframe columns
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            df_copy['timeframe'] = timeframe

            # Handle NaN and infinity values
            df_copy = df_copy.where(df_copy.notna(), None)
            df_copy = df_copy.replace({float('inf'): None, float('-inf'): None})

            # Use context manager for database operations
            with self.datastore as store:
                # Use upsert to avoid duplicates (key on symbol + Date + timeframe)
                if 'Date' in df_copy.columns:
                    store.upsert_data(
                        self.table_name,
                        df_copy,
                        key_columns=['symbol', 'Date', 'timeframe']
                    )
                else:
                    # Fallback to regular insert
                    store.fill_database(self.table_name, df_copy)

            self.logger.debug(f"[{symbol}/{timeframe}] Saved to database: {len(df_copy)} rows")

        except Exception as e:
            self.logger.exception(f"[{symbol}/{timeframe}] Failed to save to database: {e}")

    def _update_cache(self, symbol: str, bars: List[Dict[str, Any]], timeframe: str = 'day') -> None:
        """Update cache with latest processed timestamp for specific timeframe."""
        if self.cache is None or not bars:
            return

        try:
            # Get the latest timestamp from bars
            latest_ts = max(bar.get('datetime', 0) for bar in bars)

            if latest_ts > 0:
                # Use composite key for cache: symbol_timeframe
                cache_key = f"{symbol}_{timeframe}"
                self.cache.update('stock_files', cache_key, latest_ts)
                self.logger.debug(f"[{symbol}/{timeframe}] Cache updated: {latest_ts}")

        except Exception as e:
            self.logger.warning(f"[{symbol}/{timeframe}] Failed to update cache: {e}")


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================

async def main():
    """Standalone entry point for unified data pipeline."""
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description='Unified Data Pipeline - Fetch and process historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update symbols with auto-source selection
  python -m core.unified_data_pipeline --symbols AAPL MSFT

  # Fetch multiple timeframes
  python -m core.unified_data_pipeline --symbols AAPL TSLA --timeframes 5min 15min 30min

  # Fetch intraday data for optimization
  python -m core.unified_data_pipeline --symbols AAPL --timeframes 5min 15min 30min 1hour --days 750

  # Force Alpaca as source
  python -m core.unified_data_pipeline --symbols AAPL --source alpaca

  # Store to files only (no database)
  python -m core.unified_data_pipeline --symbols AAPL --no-db

  # List available data
  python -m core.unified_data_pipeline --list

  # Show cache info
  python -m core.unified_data_pipeline --symbols AAPL --cache-info
        """
    )
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT'],
                        help='Symbols to update')
    parser.add_argument('--days', type=int, default=30,
                        help='Days of history to fetch')
    parser.add_argument('--timeframes', nargs='+', default=None,
                        help='Timeframes to fetch (e.g., 5min 15min 30min 1hour day)')
    parser.add_argument('--source', choices=['alpaca', 'schwab', 'auto'],
                        default='auto', help='Data source')
    parser.add_argument('--check', action='store_true',
                        help='Only check source availability')
    parser.add_argument('--periodic', type=int, default=0,
                        help='Run periodically every N minutes (0 = one-shot)')
    parser.add_argument('--no-process', action='store_true',
                        help='Skip processing (save raw only)')
    parser.add_argument('--no-db', action='store_true',
                        help='Skip database storage (files only)')
    parser.add_argument('--no-files', action='store_true',
                        help='Skip file storage (database only)')
    parser.add_argument('--list', action='store_true',
                        help='List available symbols')
    parser.add_argument('--cache-info', action='store_true',
                        help='Show cache information for symbols')
    parser.add_argument('--read', action='store_true',
                        help='Read existing data instead of updating')
    parser.add_argument('--force-reprocess', '-f', action='store_true',
                        help='Force reprocessing even if data is up-to-date')

    args = parser.parse_args()

    pipeline = UnifiedDataPipeline(
        store_to_db=not args.no_db,
        store_to_files=not args.no_files,
    )

    # List available symbols
    if args.list:
        print("\n=== Available Symbols ===")
        file_symbols = pipeline.list_available_symbols('file')
        db_symbols = pipeline.list_available_symbols('db')
        print(f"Files: {', '.join(file_symbols) if file_symbols else 'None'}")
        print(f"Database: {', '.join(db_symbols) if db_symbols else 'None'}")
        return

    # Show cache info
    if args.cache_info:
        print("\n=== Cache Information ===")
        for symbol in args.symbols:
            info = pipeline.get_cache_info(symbol)
            if info:
                print(f"{symbol}:")
                print(f"  Last processed: {info['last_processed_date']}")
                print(f"  Age: {info['age_hours']:.1f} hours")
            else:
                print(f"{symbol}: No cache entry")
        return

    # Read existing data
    if args.read:
        print("\n=== Reading Existing Data ===")
        for symbol in args.symbols:
            df = pipeline.get_data(symbol)
            if not df.empty:
                print(f"{symbol}: {len(df)} rows, columns: {list(df.columns)[:5]}...")
            else:
                print(f"{symbol}: No data found")
        return

    # Check sources
    print("\n=== Data Source Status ===")
    sources = await pipeline.check_sources()
    for name, info in sources.items():
        if name == 'recommended':
            print(f"\nRecommended source: {info}")
        else:
            status_icon = "✓" if info['available'] else "✗"
            print(f"{status_icon} {name.upper()}: {info['status']} - {info['message']}")
            if name == 'schwab' and info.get('expires_in_hours'):
                print(f"  └─ Expires in {info['expires_in_hours']} hours")

    if args.check:
        return

    # Show storage targets
    print(f"\nStorage targets: files={not args.no_files}, database={not args.no_db}")

    # Determine source
    source = None if args.source == 'auto' else args.source

    if args.periodic > 0:
        print(f"Running periodic updates every {args.periodic} minutes...")
        await pipeline.run_periodic(
            args.symbols,
            args.periodic,
            args.days,
            source
        )
    else:
        results = await pipeline.update_symbols(
            args.symbols,
            args.days,
            source,
            force_reprocess=args.force_reprocess,
            process_data=not args.no_process,
            timeframes=args.timeframes,
        )

        print("\n=== Update Results ===")
        for symbol, count in results.items():
            status_icon = "✓" if count > 0 else "✗"
            print(f"{status_icon} {symbol}: {count} bars")


if __name__ == '__main__':
    asyncio.run(main())
