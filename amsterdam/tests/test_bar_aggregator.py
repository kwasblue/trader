# tests/test_bar_aggregator.py
"""
Unit tests for BarAggregator - Multi-timeframe bar aggregation

Tests:
- Basic bar aggregation for different timeframes
- Window boundary alignment
- Multiple symbols with different timeframes
- Timeframe switching (regime changes)
- Force completion
- Edge cases (market open/close, partial windows)
"""

import unittest
from datetime import datetime, timedelta

from core.bar_aggregator import Bar, BarAggregator, TimeframeWindow


class TestBar(unittest.TestCase):
    """Test Bar dataclass."""

    def test_bar_creation(self):
        """Test creating a Bar instance."""
        bar = Bar(
            timestamp=datetime(2026, 3, 14, 9, 30, 0),
            open=150.0,
            high=151.0,
            low=149.5,
            close=150.5,
            volume=10000,
            symbol="AAPL",
            timeframe="5min",
        )

        self.assertEqual(bar.symbol, "AAPL")
        self.assertEqual(bar.timeframe, "5min")
        self.assertEqual(bar.open, 150.0)
        self.assertEqual(bar.volume, 10000)

    def test_bar_to_dict(self):
        """Test converting Bar to dictionary."""
        bar = Bar(
            timestamp=datetime(2026, 3, 14, 9, 30, 0),
            open=150.0,
            high=151.0,
            low=149.5,
            close=150.5,
            volume=10000,
            symbol="AAPL",
            timeframe="5min",
        )

        bar_dict = bar.to_dict()
        self.assertIn("symbol", bar_dict)
        self.assertIn("timestamp", bar_dict)
        self.assertEqual(bar_dict["symbol"], "AAPL")
        self.assertEqual(bar_dict["timeframe"], "5min")


class TestTimeframeWindow(unittest.TestCase):
    """Test TimeframeWindow aggregation logic."""

    def test_parse_timeframe(self):
        """Test timeframe parsing to minutes."""
        window_1min = TimeframeWindow("AAPL", "1min")
        self.assertEqual(window_1min.interval_minutes, 1)

        window_5min = TimeframeWindow("AAPL", "5min")
        self.assertEqual(window_5min.interval_minutes, 5)

        window_15min = TimeframeWindow("AAPL", "15min")
        self.assertEqual(window_15min.interval_minutes, 15)

        window_1hour = TimeframeWindow("AAPL", "1hour")
        self.assertEqual(window_1hour.interval_minutes, 60)

    def test_parse_invalid_timeframe(self):
        """Test invalid timeframe raises error."""
        with self.assertRaises(ValueError):
            TimeframeWindow("AAPL", "invalid")

    def test_window_alignment_5min(self):
        """Test window alignment for 5-minute intervals."""
        window = TimeframeWindow("AAPL", "5min")

        # 9:32 should align to 9:30
        ts1 = datetime(2026, 3, 14, 9, 32, 0)
        aligned1 = window._get_window_start(ts1)
        self.assertEqual(aligned1, datetime(2026, 3, 14, 9, 30, 0))

        # 9:37 should align to 9:35
        ts2 = datetime(2026, 3, 14, 9, 37, 0)
        aligned2 = window._get_window_start(ts2)
        self.assertEqual(aligned2, datetime(2026, 3, 14, 9, 35, 0))

        # 9:40 should align to 9:40
        ts3 = datetime(2026, 3, 14, 9, 40, 0)
        aligned3 = window._get_window_start(ts3)
        self.assertEqual(aligned3, datetime(2026, 3, 14, 9, 40, 0))

    def test_window_alignment_15min(self):
        """Test window alignment for 15-minute intervals."""
        window = TimeframeWindow("AAPL", "15min")

        # 9:37 should align to 9:30
        ts1 = datetime(2026, 3, 14, 9, 37, 0)
        aligned1 = window._get_window_start(ts1)
        self.assertEqual(aligned1, datetime(2026, 3, 14, 9, 30, 0))

        # 9:47 should align to 9:45
        ts2 = datetime(2026, 3, 14, 9, 47, 0)
        aligned2 = window._get_window_start(ts2)
        self.assertEqual(aligned2, datetime(2026, 3, 14, 9, 45, 0))

    def test_5min_aggregation(self):
        """Test aggregating five 1-minute bars into one 5-minute bar."""
        window = TimeframeWindow("AAPL", "5min")
        base_time = datetime(2026, 3, 14, 9, 30, 0)

        # Add 5 one-minute bars (9:30 - 9:34)
        for i in range(5):
            bar = Bar(
                timestamp=base_time + timedelta(minutes=i),
                open=150.0 + i * 0.1,
                high=150.5 + i * 0.1,
                low=149.5 + i * 0.1,
                close=150.2 + i * 0.1,
                volume=1000 * (i + 1),
                symbol="AAPL",
                timeframe="1min",
            )
            result = window.add_bar(bar)

            # First 4 bars should return None (window incomplete)
            if i < 4:
                self.assertIsNone(result)
            # 5th bar should return None (triggers on next bar)
            else:
                self.assertIsNone(result)

        # Add bar from next window - this triggers completion
        next_bar = Bar(
            timestamp=base_time + timedelta(minutes=5),
            open=150.5,
            high=151.0,
            low=150.0,
            close=150.7,
            volume=5000,
            symbol="AAPL",
            timeframe="1min",
        )
        completed = window.add_bar(next_bar)

        # Should get completed 5-minute bar
        self.assertIsNotNone(completed)
        self.assertEqual(completed.timestamp, datetime(2026, 3, 14, 9, 30, 0))
        self.assertEqual(completed.timeframe, "5min")
        self.assertEqual(completed.open, 150.0)  # First bar's open
        self.assertEqual(completed.close, 150.6)  # Last bar's close (4th bar)
        self.assertEqual(completed.high, 150.9)  # Max high
        self.assertEqual(completed.low, 149.5)  # Min low
        self.assertEqual(completed.volume, 1000 + 2000 + 3000 + 4000 + 5000)  # Sum

    def test_force_complete(self):
        """Test force completion of partial window."""
        window = TimeframeWindow("AAPL", "5min")
        base_time = datetime(2026, 3, 14, 9, 30, 0)

        # Add only 3 bars (partial window)
        for i in range(3):
            bar = Bar(
                timestamp=base_time + timedelta(minutes=i),
                open=150.0 + i * 0.1,
                high=150.5 + i * 0.1,
                low=149.5 + i * 0.1,
                close=150.2 + i * 0.1,
                volume=1000,
                symbol="AAPL",
                timeframe="1min",
            )
            window.add_bar(bar)

        # Force complete
        completed = window.force_complete()

        self.assertIsNotNone(completed)
        self.assertEqual(completed.timestamp, datetime(2026, 3, 14, 9, 30, 0))
        self.assertEqual(completed.volume, 3000)  # Only 3 bars


class TestBarAggregator(unittest.TestCase):
    """Test BarAggregator multi-symbol aggregation."""

    def setUp(self):
        """Set up aggregator for each test."""
        self.aggregator = BarAggregator()
        self.received_bars = []

        def callback(bar):
            self.received_bars.append(bar)

        self.aggregator.register_callback(callback)

    def test_single_symbol_5min(self):
        """Test aggregating single symbol to 5-minute bars."""
        self.aggregator.set_timeframe("AAPL", "5min")
        base_time = datetime(2026, 3, 14, 9, 30, 0)

        # Send 10 one-minute bars (should produce 2 five-minute bars)
        for i in range(10):
            bar = Bar(
                timestamp=base_time + timedelta(minutes=i),
                open=150.0,
                high=151.0,
                low=149.0,
                close=150.5,
                volume=1000,
                symbol="AAPL",
                timeframe="1min",
            )
            self.aggregator.process_bar(bar)

        # Should have 1 completed 5-minute bar (after 5th bar)
        # Second window still open (bars 5-9)
        self.assertEqual(len(self.received_bars), 1)
        self.assertEqual(self.received_bars[0].timeframe, "5min")
        self.assertEqual(self.received_bars[0].timestamp, datetime(2026, 3, 14, 9, 30, 0))

        # Force complete to get second bar
        self.aggregator.force_complete_all()
        self.assertEqual(len(self.received_bars), 2)
        self.assertEqual(self.received_bars[1].timestamp, datetime(2026, 3, 14, 9, 35, 0))

    def test_multiple_symbols_different_timeframes(self):
        """Test multiple symbols with different timeframes."""
        self.aggregator.set_timeframe("AAPL", "5min")
        self.aggregator.set_timeframe("TSLA", "15min")

        base_time = datetime(2026, 3, 14, 9, 30, 0)

        # Send 20 one-minute bars for each symbol
        for i in range(20):
            # AAPL bar
            aapl_bar = Bar(
                timestamp=base_time + timedelta(minutes=i),
                open=150.0,
                high=151.0,
                low=149.0,
                close=150.5,
                volume=1000,
                symbol="AAPL",
                timeframe="1min",
            )
            self.aggregator.process_bar(aapl_bar)

            # TSLA bar
            tsla_bar = Bar(
                timestamp=base_time + timedelta(minutes=i),
                open=200.0,
                high=201.0,
                low=199.0,
                close=200.5,
                volume=2000,
                symbol="TSLA",
                timeframe="1min",
            )
            self.aggregator.process_bar(tsla_bar)

        # AAPL should have 3 completed bars (5, 10, 15 min mark)
        # TSLA should have 1 completed bar (15 min mark)
        aapl_bars = [b for b in self.received_bars if b.symbol == "AAPL"]
        tsla_bars = [b for b in self.received_bars if b.symbol == "TSLA"]

        self.assertEqual(len(aapl_bars), 3)
        self.assertEqual(len(tsla_bars), 1)

        # Check TSLA bar aggregated 15 minutes of data
        tsla_bar = tsla_bars[0]
        self.assertEqual(tsla_bar.timeframe, "15min")
        self.assertEqual(tsla_bar.volume, 2000 * 15)  # 15 bars * 2000

    def test_timeframe_switch(self):
        """Test switching timeframe mid-stream (regime change)."""
        self.aggregator.set_timeframe("AAPL", "5min")
        base_time = datetime(2026, 3, 14, 9, 30, 0)

        # Send 3 bars into a 5-minute window
        for i in range(3):
            bar = Bar(
                timestamp=base_time + timedelta(minutes=i),
                open=150.0,
                high=151.0,
                low=149.0,
                close=150.5,
                volume=1000,
                symbol="AAPL",
                timeframe="1min",
            )
            self.aggregator.process_bar(bar)

        # No completed bars yet
        self.assertEqual(len(self.received_bars), 0)

        # Switch to 15min (should force complete partial 5min window)
        self.aggregator.set_timeframe("AAPL", "15min")

        # Should have 1 completed bar (partial 5min window)
        self.assertEqual(len(self.received_bars), 1)
        self.assertEqual(self.received_bars[0].timeframe, "5min")
        self.assertEqual(self.received_bars[0].volume, 3000)  # Only 3 bars

        # Check stats
        stats = self.aggregator.get_stats()
        self.assertEqual(stats["timeframe_changes"], 1)

    def test_unknown_symbol_ignored(self):
        """Test that bars for unconfigured symbols are ignored."""
        self.aggregator.set_timeframe("AAPL", "5min")

        # Send bar for unconfigured symbol
        bar = Bar(
            timestamp=datetime(2026, 3, 14, 9, 30, 0),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000,
            symbol="MSFT",  # Not configured
            timeframe="1min",
        )
        result = self.aggregator.process_bar(bar)

        # Should return empty list
        self.assertEqual(result, [])
        self.assertEqual(len(self.received_bars), 0)

    def test_remove_symbol(self):
        """Test removing a symbol completes partial window."""
        self.aggregator.set_timeframe("AAPL", "5min")
        base_time = datetime(2026, 3, 14, 9, 30, 0)

        # Add 3 bars
        for i in range(3):
            bar = Bar(
                timestamp=base_time + timedelta(minutes=i),
                open=150.0,
                high=151.0,
                low=149.0,
                close=150.5,
                volume=1000,
                symbol="AAPL",
                timeframe="1min",
            )
            self.aggregator.process_bar(bar)

        # Remove symbol
        self.aggregator.remove_symbol("AAPL")

        # Should have completed partial window
        self.assertEqual(len(self.received_bars), 1)
        self.assertEqual(self.received_bars[0].volume, 3000)

        # Symbol should be gone
        self.assertNotIn("AAPL", self.aggregator.get_configured_symbols())

    def test_get_stats(self):
        """Test statistics collection."""
        self.aggregator.set_timeframe("AAPL", "5min")
        self.aggregator.set_timeframe("TSLA", "15min")

        base_time = datetime(2026, 3, 14, 9, 30, 0)

        # Send 10 bars for each
        for i in range(10):
            for symbol in ["AAPL", "TSLA"]:
                bar = Bar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=150.0,
                    high=151.0,
                    low=149.0,
                    close=150.5,
                    volume=1000,
                    symbol=symbol,
                    timeframe="1min",
                )
                self.aggregator.process_bar(bar)

        stats = self.aggregator.get_stats()

        self.assertEqual(stats["bars_received"], 20)  # 10 * 2 symbols
        self.assertEqual(stats["active_symbols"], 2)
        self.assertEqual(stats["timeframes"]["AAPL"], "5min")
        self.assertEqual(stats["timeframes"]["TSLA"], "15min")

    def test_clear_all(self):
        """Test clearing all windows."""
        self.aggregator.set_timeframe("AAPL", "5min")
        self.aggregator.set_timeframe("TSLA", "15min")

        base_time = datetime(2026, 3, 14, 9, 30, 0)

        # Add some bars
        for i in range(3):
            bar = Bar(
                timestamp=base_time + timedelta(minutes=i),
                open=150.0,
                high=151.0,
                low=149.0,
                close=150.5,
                volume=1000,
                symbol="AAPL",
                timeframe="1min",
            )
            self.aggregator.process_bar(bar)

        # Clear all
        self.aggregator.clear_all()

        # Should have completed partial windows
        self.assertEqual(len(self.received_bars), 1)

        # No symbols left
        self.assertEqual(len(self.aggregator.get_configured_symbols()), 0)

    def test_get_timeframe(self):
        """Test retrieving configured timeframe for symbol."""
        self.aggregator.set_timeframe("AAPL", "5min")
        self.aggregator.set_timeframe("TSLA", "15min")

        self.assertEqual(self.aggregator.get_timeframe("AAPL"), "5min")
        self.assertEqual(self.aggregator.get_timeframe("TSLA"), "15min")
        self.assertIsNone(self.aggregator.get_timeframe("MSFT"))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_market_close_force_complete(self):
        """Test force completing all windows at market close."""
        aggregator = BarAggregator()
        received = []
        aggregator.register_callback(lambda b: received.append(b))

        # Configure multiple symbols
        aggregator.set_timeframe("AAPL", "5min")
        aggregator.set_timeframe("TSLA", "15min")
        aggregator.set_timeframe("MSFT", "30min")

        base_time = datetime(2026, 3, 14, 15, 55, 0)  # Near market close

        # Add partial windows for each
        for i in range(3):
            for symbol in ["AAPL", "TSLA", "MSFT"]:
                bar = Bar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=150.0,
                    high=151.0,
                    low=149.0,
                    close=150.5,
                    volume=1000,
                    symbol=symbol,
                    timeframe="1min",
                )
                aggregator.process_bar(bar)

        # Force complete all at market close
        aggregator.force_complete_all()

        # Should have 3 partial bars (one per symbol)
        self.assertEqual(len(received), 3)

        # All should have 3 bars worth of volume
        for bar in received:
            self.assertEqual(bar.volume, 3000)

    def test_empty_window_force_complete(self):
        """Test force completing empty window does nothing."""
        aggregator = BarAggregator()
        received = []
        aggregator.register_callback(lambda b: received.append(b))

        aggregator.set_timeframe("AAPL", "5min")

        # Force complete without any bars
        aggregator.force_complete_all()

        # Should have no bars
        self.assertEqual(len(received), 0)


if __name__ == "__main__":
    unittest.main()
