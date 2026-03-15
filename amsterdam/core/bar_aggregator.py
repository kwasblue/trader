# core/bar_aggregator.py
"""
Multi-Timeframe Bar Aggregator for Live Streaming Data

Aggregates high-frequency streaming bars (from Schwab websocket) into target
timeframes (1min, 5min, 15min, 30min, 1hour) for different symbols.

Features:
- Supports multiple symbols with different timeframes simultaneously
- Seamless timeframe switching on regime changes
- Automatic window alignment to interval boundaries
- Force completion at market close
- Callback system for completed bars

Usage:
    aggregator = BarAggregator()

    # Configure timeframes
    aggregator.set_timeframe("AAPL", "5min")
    aggregator.set_timeframe("TSLA", "15min")

    # Register callback
    def on_bar(bar):
        strategy.process_bar(bar)
    aggregator.register_callback(on_bar)

    # Process streaming bars
    for raw_bar in websocket_stream:
        aggregator.process_bar(raw_bar)

    # On regime change
    aggregator.set_timeframe("AAPL", "15min")  # Seamlessly switches
"""

from __future__ import annotations

from typing import Dict, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from loggers.logger import Logger


@dataclass
class Bar:
    """
    OHLCV bar data structure.

    Attributes:
        timestamp: Bar timestamp (aligned to interval start)
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Total volume
        symbol: Stock symbol
        timeframe: Timeframe identifier (e.g., "5min", "15min")
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str
    timeframe: str

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
        }


class TimeframeWindow:
    """
    Manages aggregation window for a single (symbol, timeframe) combination.

    Accumulates incoming bars and emits completed bar when interval completes.
    """

    def __init__(self, symbol: str, timeframe: str, logger: Optional[logging.Logger] = None):
        """
        Initialize timeframe window.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe identifier (e.g., "5min", "15min")
            logger: Optional logger instance
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.interval_minutes = self._parse_timeframe(timeframe)
        self.logger = logger or logging.getLogger(__name__)

        # Current aggregation state
        self.current_window_start: Optional[datetime] = None
        self.bars_in_window: List[Bar] = []

    def _parse_timeframe(self, timeframe: str) -> int:
        """
        Convert timeframe string to minutes.

        Args:
            timeframe: Timeframe string (e.g., "5min", "1hour")

        Returns:
            Number of minutes in the interval

        Raises:
            ValueError: If timeframe is not supported
        """
        timeframe_map = {
            "1min": 1,
            "5min": 5,
            "15min": 15,
            "30min": 30,
            "1hour": 60,
        }

        if timeframe not in timeframe_map:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {list(timeframe_map.keys())}"
            )

        return timeframe_map[timeframe]

    def _get_window_start(self, timestamp: datetime) -> datetime:
        """
        Get the start of the aggregation window for a timestamp.

        Aligns to interval boundaries (e.g., 9:30, 9:35, 9:40 for 5min).

        Args:
            timestamp: Bar timestamp

        Returns:
            Window start timestamp (aligned to interval boundary)
        """
        # Calculate minutes since midnight
        minutes_since_midnight = timestamp.hour * 60 + timestamp.minute

        # Find which window this falls into
        window_index = minutes_since_midnight // self.interval_minutes

        # Calculate window start minutes
        window_start_minutes = window_index * self.interval_minutes

        # Create window start timestamp
        return timestamp.replace(
            hour=window_start_minutes // 60,
            minute=window_start_minutes % 60,
            second=0,
            microsecond=0
        )

    def add_bar(self, bar: Bar) -> Optional[Bar]:
        """
        Add incoming bar to aggregation window.

        Args:
            bar: Incoming bar to aggregate

        Returns:
            Completed aggregated bar if window finished, None otherwise
        """
        window_start = self._get_window_start(bar.timestamp)

        # First bar or new window starting
        if self.current_window_start is None:
            self.current_window_start = window_start
            self.bars_in_window = [bar]
            self.logger.debug(
                f"[{self.symbol}/{self.timeframe}] Started new window at {window_start}"
            )
            return None

        # Bar belongs to current window
        if window_start == self.current_window_start:
            self.bars_in_window.append(bar)
            return None

        # Bar is in next window - complete current window
        completed = self._complete_window()

        # Start new window
        self.current_window_start = window_start
        self.bars_in_window = [bar]

        return completed

    def _complete_window(self) -> Optional[Bar]:
        """
        Aggregate bars in current window into single bar.

        Returns:
            Aggregated bar, or None if no bars in window
        """
        if not self.bars_in_window:
            return None

        aggregated = Bar(
            timestamp=self.current_window_start,
            open=self.bars_in_window[0].open,
            high=max(b.high for b in self.bars_in_window),
            low=min(b.low for b in self.bars_in_window),
            close=self.bars_in_window[-1].close,
            volume=sum(b.volume for b in self.bars_in_window),
            symbol=self.symbol,
            timeframe=self.timeframe
        )

        self.logger.debug(
            f"[{self.symbol}/{self.timeframe}] Completed window at {self.current_window_start}: "
            f"{len(self.bars_in_window)} bars → OHLCV({aggregated.open:.2f}, "
            f"{aggregated.high:.2f}, {aggregated.low:.2f}, {aggregated.close:.2f}, "
            f"{aggregated.volume})"
        )

        return aggregated

    def force_complete(self) -> Optional[Bar]:
        """
        Force completion of current window (e.g., at market close).

        Returns:
            Completed aggregated bar, or None if no bars in window
        """
        if self.bars_in_window:
            self.logger.info(
                f"[{self.symbol}/{self.timeframe}] Force completing window with "
                f"{len(self.bars_in_window)} bars"
            )
            return self._complete_window()
        return None


class BarAggregator:
    """
    Multi-timeframe bar aggregator for live streaming data.

    Manages aggregation windows for multiple symbols, each potentially
    using different timeframes. Supports seamless timeframe switching
    on regime changes.

    Usage:
        aggregator = BarAggregator()

        # Configure timeframes
        aggregator.set_timeframe("AAPL", "5min")
        aggregator.set_timeframe("TSLA", "15min")

        # Process incoming bars from websocket
        for bar in websocket_stream:
            completed_bars = aggregator.process_bar(bar)
            for completed in completed_bars:
                strategy.on_bar(completed)

        # On regime change
        aggregator.set_timeframe("AAPL", "15min")  # Seamlessly switches
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize bar aggregator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or Logger(
            "bar_aggregator.log",
            "BarAggregator",
            propagate=True,
            level=10  # DEBUG
        ).get_logger()

        # Map: symbol → TimeframeWindow
        self.windows: Dict[str, TimeframeWindow] = {}

        # Callbacks for completed bars
        self.bar_callbacks: List[Callable[[Bar], None]] = []

        # Stats
        self.stats = {
            'bars_received': 0,
            'bars_emitted': 0,
            'timeframe_changes': 0,
        }

        self.logger.info("BarAggregator initialized")

    def set_timeframe(self, symbol: str, timeframe: str):
        """
        Set target timeframe for a symbol.

        If timeframe changes for existing symbol, current window is
        completed and new window starts immediately.

        Args:
            symbol: Stock symbol
            timeframe: Target timeframe (e.g., "5min", "15min")
        """
        # If symbol exists and timeframe is changing
        if symbol in self.windows:
            old_timeframe = self.windows[symbol].timeframe
            if old_timeframe != timeframe:
                # Complete current window before switching
                completed = self.windows[symbol].force_complete()
                if completed:
                    self._emit_bar(completed)

                self.stats['timeframe_changes'] += 1
                self.logger.info(
                    f"[{symbol}] Timeframe changed: {old_timeframe} → {timeframe}"
                )

        # Create new window with target timeframe
        self.windows[symbol] = TimeframeWindow(symbol, timeframe, self.logger)
        self.logger.info(f"[{symbol}] Configured for {timeframe} bars")

    def process_bar(self, bar: Bar) -> List[Bar]:
        """
        Process incoming bar from websocket.

        Args:
            bar: Incoming bar to process

        Returns:
            List of completed aggregated bars (usually 0 or 1)
        """
        self.stats['bars_received'] += 1

        # Unknown symbol - skip (not configured for trading)
        if bar.symbol not in self.windows:
            return []

        window = self.windows[bar.symbol]
        completed = window.add_bar(bar)

        if completed:
            self.stats['bars_emitted'] += 1
            self._emit_bar(completed)
            return [completed]

        return []

    def _emit_bar(self, bar: Bar):
        """
        Emit completed bar to all callbacks.

        Args:
            bar: Completed aggregated bar
        """
        self.logger.debug(
            f"[{bar.symbol}/{bar.timeframe}] Emitting bar: "
            f"{bar.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        for callback in self.bar_callbacks:
            try:
                callback(bar)
            except Exception as e:
                self.logger.exception(f"Callback error: {e}")

    def register_callback(self, callback: Callable[[Bar], None]):
        """
        Register callback to receive completed bars.

        Args:
            callback: Function to call with completed bars
                     Signature: callback(bar: Bar) -> None

        Example:
            def on_bar(bar):
                print(f"Received {bar.symbol} bar at {bar.timestamp}")

            aggregator.register_callback(on_bar)
        """
        self.bar_callbacks.append(callback)
        self.logger.info(f"Registered callback: {callback.__name__}")

    def get_stats(self) -> dict:
        """
        Get aggregation statistics.

        Returns:
            Dict with statistics including:
            - bars_received: Total bars received
            - bars_emitted: Total aggregated bars emitted
            - timeframe_changes: Number of timeframe switches
            - active_symbols: Number of symbols being tracked
            - timeframes: Dict of symbol → timeframe mappings
        """
        return {
            **self.stats,
            'active_symbols': len(self.windows),
            'timeframes': {
                symbol: window.timeframe
                for symbol, window in self.windows.items()
            }
        }

    def force_complete_all(self):
        """
        Force completion of all windows (e.g., at market close).

        Useful for ensuring all partial windows are emitted before
        market close or system shutdown.
        """
        self.logger.info("Force completing all windows")

        for symbol, window in self.windows.items():
            completed = window.force_complete()
            if completed:
                self.stats['bars_emitted'] += 1
                self._emit_bar(completed)

    def remove_symbol(self, symbol: str):
        """
        Remove symbol from tracking.

        Completes any partial window before removal.

        Args:
            symbol: Stock symbol to remove
        """
        if symbol in self.windows:
            # Complete partial window
            completed = self.windows[symbol].force_complete()
            if completed:
                self.stats['bars_emitted'] += 1
                self._emit_bar(completed)

            # Remove window
            del self.windows[symbol]
            self.logger.info(f"[{symbol}] Removed from tracking")

    def clear_all(self):
        """
        Clear all windows and reset state.

        Completes all partial windows before clearing.
        """
        self.force_complete_all()
        self.windows.clear()
        self.logger.info("All windows cleared")

    def get_configured_symbols(self) -> List[str]:
        """
        Get list of symbols currently configured.

        Returns:
            List of symbol names
        """
        return list(self.windows.keys())

    def get_timeframe(self, symbol: str) -> Optional[str]:
        """
        Get current timeframe for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Timeframe string, or None if symbol not configured
        """
        if symbol in self.windows:
            return self.windows[symbol].timeframe
        return None


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == '__main__':
    """Standalone testing and demonstration."""
    import sys
    from datetime import datetime, timedelta

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create aggregator
    aggregator = BarAggregator()

    # Configure symbols
    aggregator.set_timeframe("AAPL", "5min")
    aggregator.set_timeframe("TSLA", "15min")

    # Register callback
    def on_bar(bar: Bar):
        print(f"✓ [{bar.symbol}/{bar.timeframe}] {bar.timestamp.strftime('%H:%M:%S')} "
              f"OHLCV({bar.open:.2f}, {bar.high:.2f}, {bar.low:.2f}, {bar.close:.2f}, {bar.volume})")

    aggregator.register_callback(on_bar)

    # Simulate streaming bars
    print("\n=== Simulating 1-minute bars ===\n")
    base_time = datetime(2026, 3, 14, 9, 30, 0)

    for i in range(20):
        # Create 1-minute bars for AAPL
        bar = Bar(
            timestamp=base_time + timedelta(minutes=i),
            open=150.0 + i * 0.1,
            high=150.5 + i * 0.1,
            low=149.5 + i * 0.1,
            close=150.2 + i * 0.1,
            volume=1000 * (i + 1),
            symbol="AAPL",
            timeframe="1min"
        )
        aggregator.process_bar(bar)

        # Create 1-minute bars for TSLA
        bar = Bar(
            timestamp=base_time + timedelta(minutes=i),
            open=200.0 + i * 0.2,
            high=200.5 + i * 0.2,
            low=199.5 + i * 0.2,
            close=200.1 + i * 0.2,
            volume=2000 * (i + 1),
            symbol="TSLA",
            timeframe="1min"
        )
        aggregator.process_bar(bar)

    # Force complete remaining windows
    print("\n=== Force completing remaining windows ===\n")
    aggregator.force_complete_all()

    # Show stats
    print("\n=== Statistics ===")
    stats = aggregator.get_stats()
    print(f"Bars received: {stats['bars_received']}")
    print(f"Bars emitted: {stats['bars_emitted']}")
    print(f"Active symbols: {stats['active_symbols']}")
    print(f"Timeframes: {stats['timeframes']}")

    print("\n=== Test timeframe change ===")
    aggregator.set_timeframe("AAPL", "15min")
    print(f"AAPL now using: {aggregator.get_timeframe('AAPL')}")
    print(f"Timeframe changes: {aggregator.get_stats()['timeframe_changes']}")
