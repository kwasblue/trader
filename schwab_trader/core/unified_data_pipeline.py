# core/unified_data_pipeline.py
"""
Unified Data Pipeline - Fetches and processes historical data from multiple sources

Features:
- Supports Alpaca and Schwab as data sources
- Automatic fallback if one source is unavailable
- Same processing pipeline for both sources
- Stores to both JSON files AND SQLite database
- Updates cache for tracking processed dates
- Compatible with existing HistoricalBarLoader format

Usage:
    pipeline = UnifiedDataPipeline()

    # Auto-select best source
    await pipeline.update_symbols(['AAPL', 'MSFT'])

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

from loggers.logger import Logger
from core.credential_validator import CredentialValidator, CredentialStatus
from data.datastorage import DataStore
from cache.cache import CacheManager


class UnifiedDataPipeline:
    """
    Unified data pipeline that fetches from Alpaca or Schwab
    and processes through the same pipeline.

    Storage:
    - JSON files (proc_{SYMBOL}_file.json, raw_{SYMBOL}_file.json)
    - SQLite database (stock_base.db)
    - Cache tracking (cache/system_cache.json)
    """

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
        force_full: bool = False,
        process_data: bool = True,
    ) -> Dict[str, int]:
        """
        Update historical data for multiple symbols.

        Args:
            symbols: List of stock symbols
            days: Number of days of history
            source: 'alpaca', 'schwab', or None (auto-select)
            force_full: Force full refresh (not incremental)
            process_data: Whether to process through ML pipeline

        Returns:
            Dict of symbol -> bars fetched
        """
        self.logger.info("=" * 60)
        self.logger.info(f"UPDATE SYMBOLS: {symbols}")
        self.logger.info(f"Parameters: days={days}, source={source}, force_full={force_full}, process_data={process_data}")

        # Determine source
        if source is None:
            self.logger.debug("Auto-selecting data source...")
            source = await self._select_best_source()
            self.logger.info(f"Auto-selected source: {source}")

        if source == 'none':
            self.logger.error("FATAL: No valid data source available. Check credentials.")
            return {s: 0 for s in symbols}

        self.logger.info(f"Updating {len(symbols)} symbols from {source.upper()}")

        results = {}
        start_time = datetime.now(timezone.utc)

        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"[{symbol}] Processing ({i}/{len(symbols)})...")
            try:
                count = await self._update_symbol(
                    symbol, days, source, force_full, process_data
                )
                results[symbol] = count
                self.logger.info(f"[{symbol}] SUCCESS: {count} bars fetched")
            except Exception as e:
                self.logger.exception(f"[{symbol}] FAILED: {e}")
                results[symbol] = 0

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        total = sum(results.values())
        success_count = sum(1 for c in results.values() if c > 0)

        self.logger.info("=" * 60)
        self.logger.info(f"UPDATE COMPLETE in {elapsed:.1f}s")
        self.logger.info(f"Total bars: {total}, Success: {success_count}/{len(symbols)}")
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

    def get_data_from_db(self, symbol: str) -> pd.DataFrame:
        """
        Retrieve processed data from database for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with processed data
        """
        try:
            with self.datastore as store:
                df = store.get_data_by_symbol(self.table_name, symbol)
                if not df.empty:
                    self.logger.debug(f"[{symbol}] Retrieved {len(df)} rows from database")
                return df
        except Exception as e:
            self.logger.error(f"[{symbol}] Failed to retrieve from database: {e}")
            return pd.DataFrame()

    def get_data_from_file(self, symbol: str) -> pd.DataFrame:
        """
        Retrieve processed data from JSON file for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with processed data
        """
        file_path = self.data_path / f"proc_{symbol}_file.json"

        try:
            if not file_path.exists():
                self.logger.warning(f"[{symbol}] File not found: {file_path}")
                return pd.DataFrame()

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Handle both 'bars' key and direct list format
            records = data.get('bars', data) if isinstance(data, dict) else data

            df = pd.DataFrame(records)
            self.logger.debug(f"[{symbol}] Retrieved {len(df)} rows from file")
            return df

        except Exception as e:
            self.logger.error(f"[{symbol}] Failed to retrieve from file: {e}")
            return pd.DataFrame()

    def get_data(self, symbol: str, prefer_db: bool = False) -> pd.DataFrame:
        """
        Retrieve processed data from best available source.

        Args:
            symbol: Stock symbol
            prefer_db: If True, prefer database over files

        Returns:
            DataFrame with processed data
        """
        if prefer_db:
            df = self.get_data_from_db(symbol)
            if not df.empty:
                return df
            return self.get_data_from_file(symbol)
        else:
            df = self.get_data_from_file(symbol)
            if not df.empty:
                return df
            return self.get_data_from_db(symbol)

    def list_available_symbols(self, source: str = 'file') -> List[str]:
        """
        List symbols with available data.

        Args:
            source: 'file' or 'db'

        Returns:
            List of symbol names
        """
        if source == 'file':
            files = list(self.data_path.glob("proc_*_file.json"))
            # Extract symbol from filename: proc_AAPL_file.json -> AAPL
            symbols = []
            for f in files:
                name = f.stem  # proc_AAPL_file
                parts = name.split('_')
                if len(parts) >= 2:
                    symbols.append(parts[1])
            return sorted(symbols)

        elif source == 'db':
            try:
                with self.datastore as store:
                    if not store.table_exists(self.table_name):
                        return []
                    df = store.get_data_base(self.table_name)
                    if 'symbol' in df.columns:
                        return sorted(df['symbol'].unique().tolist())
                    return []
            except Exception as e:
                self.logger.error(f"Failed to list symbols from database: {e}")
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

    # ========================================================================
    # PRIVATE METHODS - DATA FETCHING
    # ========================================================================

    async def _update_symbol(
        self,
        symbol: str,
        days: int,
        source: str,
        force_full: bool,
        process_data: bool,
    ) -> int:
        """Update data for a single symbol."""

        if source == 'alpaca':
            raw_bars = await self._fetch_alpaca(symbol, days)
        elif source == 'schwab':
            raw_bars = await self._fetch_schwab(symbol, days)
        else:
            raise ValueError(f"Unknown source: {source}")

        if not raw_bars:
            self.logger.warning(f"[{symbol}] No bars fetched from {source}")
            return 0

        # Save raw data
        self._save_raw_data(symbol, raw_bars)

        # Process data if requested
        if process_data:
            await self._process_and_save(symbol, raw_bars)

        self.logger.info(f"[{symbol}] Updated with {len(raw_bars)} bars from {source}")
        return len(raw_bars)

    async def _fetch_alpaca(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """Fetch historical bars from Alpaca."""
        try:
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            if self._alpaca_data_client is None:
                api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_KEY_ID')
                api_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_SECRET')
                self._alpaca_data_client = StockHistoricalDataClient(api_key, api_secret)

            start = datetime.now(timezone.utc) - timedelta(days=days)
            end = datetime.now(timezone.utc)

            def _do_fetch():
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,  # Daily bars for processing
                    start=start,
                    end=end,
                )
                response = self._alpaca_data_client.get_stock_bars(request)
                symbol_bars = response.get(symbol, [])

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
            self.logger.exception(f"[{symbol}] Alpaca fetch failed: {e}")
            return []

    async def _fetch_schwab(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """Fetch historical bars from Schwab."""
        try:
            from data.streaming.schwab_client import SchwabClient
            from data.streaming.authenticator import Authenticator

            auth = Authenticator()

            if self._schwab_client is None:
                self._schwab_client = SchwabClient(auth.apikey, auth.secret)

            def _do_fetch():
                # Calculate start date (epoch ms)
                start_dt = datetime.now(timezone.utc) - timedelta(days=days)
                start_ms = int(start_dt.timestamp() * 1000)

                data = self._schwab_client.daily_price_history(symbol, start=start_ms)
                return data.get('candles', [])

            return await asyncio.to_thread(_do_fetch)

        except Exception as e:
            self.logger.exception(f"[{symbol}] Schwab fetch failed: {e}")
            return []

    # ========================================================================
    # PRIVATE METHODS - DATA STORAGE
    # ========================================================================

    def _save_raw_data(self, symbol: str, bars: List[Dict[str, Any]]) -> None:
        """Save raw bar data to JSON file."""
        file_path = self.raw_data_path / f"raw_{symbol}_file.json"

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
                json.dump({'candles': unique_bars}, f, indent=2)

            self.logger.debug(f"[{symbol}] Raw data saved: {len(unique_bars)} bars")

        except Exception as e:
            self.logger.exception(f"[{symbol}] Failed to save raw data: {e}")

    async def _process_and_save(self, symbol: str, bars: List[Dict[str, Any]]) -> None:
        """
        Process raw bars through the ML pipeline and save to all storage targets.

        Storage targets:
        - JSON file (proc_{SYMBOL}_file.json)
        - SQLite database (stock_table)
        - Cache update (system_cache.json)
        """
        try:
            from data.processor import Processor

            # Convert bars to DataFrame
            df = pd.DataFrame(bars)

            if df.empty:
                self.logger.warning(f"[{symbol}] No data to process")
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
                self.logger.error(f"[{symbol}] Missing datetime column")
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
                self.logger.warning(f"[{symbol}] Processing returned empty DataFrame")
                return

            # Save to JSON files
            if self.store_to_files:
                self._save_processed_data_file(symbol, processed_df)

            # Save to database
            if self.store_to_db:
                self._save_processed_data_db(symbol, processed_df)

            # Update cache with latest timestamp
            self._update_cache(symbol, bars)

            self.logger.info(
                f"[{symbol}] Data saved: {len(processed_df)} rows "
                f"(files={self.store_to_files}, db={self.store_to_db})"
            )

        except Exception as e:
            self.logger.exception(f"[{symbol}] Processing failed: {e}")

    def _save_processed_data_file(self, symbol: str, df: pd.DataFrame) -> None:
        """Save processed DataFrame to JSON file."""
        file_path = self.data_path / f"proc_{symbol}_file.json"

        try:
            # Make a copy to avoid modifying original
            df_copy = df.copy()

            # Handle NaN and infinity values
            df_copy = df_copy.where(df_copy.notna(), None)
            df_copy = df_copy.replace({float('inf'): None, float('-inf'): None})

            # Convert to records
            data = df_copy.to_dict(orient='records')

            # Save
            with open(file_path, 'w') as f:
                json.dump({'bars': data}, f)

            self.logger.debug(f"[{symbol}] Saved to file: {len(data)} rows")

        except Exception as e:
            self.logger.exception(f"[{symbol}] Failed to save to file: {e}")

    def _save_processed_data_db(self, symbol: str, df: pd.DataFrame) -> None:
        """Save processed DataFrame to SQLite database."""
        try:
            # Make a copy and add symbol column
            df_copy = df.copy()
            df_copy['symbol'] = symbol

            # Handle NaN and infinity values
            df_copy = df_copy.where(df_copy.notna(), None)
            df_copy = df_copy.replace({float('inf'): None, float('-inf'): None})

            # Use context manager for database operations
            with self.datastore as store:
                # Use upsert to avoid duplicates (key on symbol + Date)
                if 'Date' in df_copy.columns:
                    store.upsert_data(
                        self.table_name,
                        df_copy,
                        key_columns=['symbol', 'Date']
                    )
                else:
                    # Fallback to regular insert
                    store.fill_database(self.table_name, df_copy)

            self.logger.debug(f"[{symbol}] Saved to database: {len(df_copy)} rows")

        except Exception as e:
            self.logger.exception(f"[{symbol}] Failed to save to database: {e}")

    def _update_cache(self, symbol: str, bars: List[Dict[str, Any]]) -> None:
        """Update cache with latest processed timestamp."""
        if self.cache is None or not bars:
            return

        try:
            # Get the latest timestamp from bars
            latest_ts = max(bar.get('datetime', 0) for bar in bars)

            if latest_ts > 0:
                self.cache.update('stock_files', symbol, latest_ts)
                self.logger.debug(f"[{symbol}] Cache updated: {latest_ts}")

        except Exception as e:
            self.logger.warning(f"[{symbol}] Failed to update cache: {e}")


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
            process_data=not args.no_process
        )

        print("\n=== Update Results ===")
        for symbol, count in results.items():
            status_icon = "✓" if count > 0 else "✗"
            print(f"{status_icon} {symbol}: {count} bars")


if __name__ == '__main__':
    asyncio.run(main())
