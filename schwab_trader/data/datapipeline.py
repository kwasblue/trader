"""
Stock Data Pipeline - Deprecated

This module is deprecated. Use core.unified_data_pipeline.UnifiedDataPipeline instead.

The UnifiedDataPipeline provides:
- Multi-source support (Alpaca, Schwab)
- Async-first design
- Better error handling
- Storage to both JSON and SQLite

Example:
    # Old (deprecated):
    from data.datapipeline import StockDataPipeline
    pipeline = StockDataPipeline(watch_list=['AAPL'])
    pipeline.run()

    # New (recommended):
    from core.unified_data_pipeline import UnifiedDataPipeline
    pipeline = UnifiedDataPipeline()
    await pipeline.update_symbols(['AAPL'])
"""
import warnings
import asyncio
from typing import List, Optional, Dict, Any

warnings.warn(
    "data.datapipeline.StockDataPipeline is deprecated. "
    "Use core.unified_data_pipeline.UnifiedDataPipeline instead.",
    DeprecationWarning,
    stacklevel=2
)


class StockDataPipeline:
    """
    Legacy data pipeline - DEPRECATED.

    This class is a compatibility wrapper around UnifiedDataPipeline.
    Use UnifiedDataPipeline directly for new code.
    """

    def __init__(
        self,
        watch_list: Optional[List[str]] = None,
        max_workers: int = 4,
        max_retries: int = 3
    ):
        """
        Initialize the legacy pipeline.

        Args:
            watch_list: List of symbols to process
            max_workers: Not used (kept for compatibility)
            max_retries: Not used (kept for compatibility)
        """
        warnings.warn(
            "StockDataPipeline is deprecated. Use UnifiedDataPipeline instead.",
            DeprecationWarning,
            stacklevel=2
        )

        self.watch_list = watch_list or []
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.skipped_stocks: List[str] = []

        # Lazy import to avoid circular imports
        from core.unified_data_pipeline import UnifiedDataPipeline
        self._pipeline = UnifiedDataPipeline()

    def fetch_data(self) -> Dict[str, str]:
        """
        Fetch raw data (deprecated).

        Returns:
            Dictionary of stock -> result
        """
        warnings.warn(
            "fetch_data() is deprecated. Use UnifiedDataPipeline.update_symbols() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Run the async pipeline synchronously
        asyncio.run(self._pipeline.update_symbols(self.watch_list, source='schwab'))
        return {stock: "File Generated!" for stock in self.watch_list}

    def gather_data(self) -> Dict[str, Any]:
        """
        Gather stock data (deprecated).

        Returns:
            Dictionary of stock -> DataFrame
        """
        warnings.warn(
            "gather_data() is deprecated. Use UnifiedDataPipeline.update_symbols() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return {}

    def process_data(self, gathered_data: Dict) -> Dict[str, Any]:
        """
        Process stock data (deprecated).

        Returns:
            Dictionary of stock -> processed data
        """
        warnings.warn(
            "process_data() is deprecated. Use UnifiedDataPipeline.update_symbols() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return {}

    def store_data(self, processed_data: Dict) -> None:
        """
        Store processed data (deprecated).
        """
        warnings.warn(
            "store_data() is deprecated. Use UnifiedDataPipeline.update_symbols() instead.",
            DeprecationWarning,
            stacklevel=2
        )

    def run(self) -> None:
        """
        Run the full pipeline (deprecated).

        Use UnifiedDataPipeline.update_symbols() instead.
        """
        warnings.warn(
            "run() is deprecated. Use UnifiedDataPipeline.update_symbols() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        if not self.watch_list:
            return

        # Run the async pipeline synchronously
        asyncio.run(self._pipeline.update_symbols(self.watch_list, source='schwab'))


__all__ = ['StockDataPipeline']
