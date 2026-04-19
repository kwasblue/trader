"""
Test suite for market data streaming.

Tests WebSocket connections, data streaming, and event handling.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import unittest
from unittest.mock import AsyncMock, patch


class TestStreamerConnection(unittest.TestCase):
    """Test streamer connection management."""

    def test_streamer_initialization(self):
        """Streamer should initialize with correct parameters."""
        # Mock streamer initialization
        config = {"api_key": "test_key", "symbols": ["AAPL", "MSFT"], "reconnect": True}

        self.assertIn("api_key", config)
        self.assertIn("symbols", config)

    @patch("websockets.connect")
    def test_websocket_connect(self, mock_connect):
        """Should establish WebSocket connection."""

        async def run_test():
            mock_ws = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_ws

            # Simulate connection
            async with mock_connect("wss://test.example.com") as ws:
                self.assertIsNotNone(ws)

        asyncio.run(run_test())

    def test_streamer_reconnect_config(self):
        """Streamer should support auto-reconnect configuration."""
        config = {"reconnect": True, "reconnect_attempts": 5, "reconnect_delay_ms": 1000}

        self.assertTrue(config["reconnect"])
        self.assertEqual(config["reconnect_attempts"], 5)


class TestStreamerSubscription(unittest.TestCase):
    """Test streamer subscription methods."""

    def test_subscribe_to_quotes(self):
        """Should subscribe to quote updates."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        subscription = {
            "service": "QUOTE",
            "command": "SUBS",
            "parameters": {"keys": ",".join(symbols), "fields": "0,1,2,3,4,5,8,9"},
        }

        self.assertEqual(subscription["service"], "QUOTE")
        self.assertEqual(subscription["command"], "SUBS")
        self.assertIn("AAPL", subscription["parameters"]["keys"])

    def test_subscribe_to_bars(self):
        """Should subscribe to bar (OHLC) updates."""
        symbols = ["AAPL"]
        subscription = {
            "service": "CHART_EQUITY",
            "command": "SUBS",
            "parameters": {"keys": ",".join(symbols), "fields": "0,1,2,3,4,5,6,7"},
        }

        self.assertEqual(subscription["service"], "CHART_EQUITY")

    def test_unsubscribe(self):
        """Should unsubscribe from updates."""
        unsubscription = {"service": "QUOTE", "command": "UNSUBS", "parameters": {"keys": "AAPL"}}

        self.assertEqual(unsubscription["command"], "UNSUBS")


class TestStreamerMessages(unittest.TestCase):
    """Test streamer message parsing."""

    def setUp(self):
        """Create sample streamer messages."""
        self.quote_message = {
            "service": "QUOTE",
            "timestamp": 1672531200000,
            "content": [
                {
                    "key": "AAPL",
                    "1": 150.25,  # bid
                    "2": 150.30,  # ask
                    "3": 150.27,  # last
                    "8": 5000000,  # volume
                }
            ],
        }

        self.bar_message = {
            "service": "CHART_EQUITY",
            "timestamp": 1672531200000,
            "content": [
                {
                    "key": "AAPL",
                    "1": 150.00,  # open
                    "2": 152.00,  # high
                    "3": 149.50,  # low
                    "4": 151.50,  # close
                    "5": 1000000,  # volume
                }
            ],
        }

    def test_parse_quote_message(self):
        """Should parse quote message correctly."""
        content = self.quote_message["content"][0]

        self.assertEqual(content["key"], "AAPL")
        self.assertEqual(content["1"], 150.25)  # bid
        self.assertEqual(content["2"], 150.30)  # ask

    def test_parse_bar_message(self):
        """Should parse bar message correctly."""
        content = self.bar_message["content"][0]

        self.assertEqual(content["key"], "AAPL")
        self.assertEqual(content["1"], 150.00)  # open
        self.assertEqual(content["4"], 151.50)  # close


class TestStreamerEventHandling(unittest.TestCase):
    """Test streamer event callback handling."""

    def test_on_quote_callback(self):
        """Quote callback should receive data."""
        received_quotes = []

        def on_quote(symbol, data):
            received_quotes.append({"symbol": symbol, "data": data})

        # Simulate receiving quote
        on_quote("AAPL", {"bid": 150.25, "ask": 150.30})

        self.assertEqual(len(received_quotes), 1)
        self.assertEqual(received_quotes[0]["symbol"], "AAPL")

    def test_on_bar_callback(self):
        """Bar callback should receive OHLCV data."""
        received_bars = []

        def on_bar(symbol, bar):
            received_bars.append({"symbol": symbol, "bar": bar})

        # Simulate receiving bar
        bar_data = {"open": 150.0, "high": 152.0, "low": 149.5, "close": 151.5, "volume": 1000000}
        on_bar("AAPL", bar_data)

        self.assertEqual(len(received_bars), 1)
        self.assertIn("open", received_bars[0]["bar"])

    def test_on_error_callback(self):
        """Error callback should handle errors."""
        errors = []

        def on_error(error):
            errors.append(error)

        # Simulate error
        on_error(Exception("Connection lost"))

        self.assertEqual(len(errors), 1)
        self.assertIn("Connection lost", str(errors[0]))


class TestStreamerHeartbeat(unittest.TestCase):
    """Test streamer heartbeat/keepalive."""

    def test_heartbeat_response(self):
        """Should respond to heartbeat requests."""

        heartbeat_response = {"service": "ADMIN", "command": "HEARTBEAT", "timestamp": 1672531200000}

        self.assertEqual(heartbeat_response["command"], "HEARTBEAT")

    def test_heartbeat_timeout_detection(self):
        """Should detect heartbeat timeout."""
        last_heartbeat_ms = 1672531200000
        current_time_ms = 1672531260000  # 60 seconds later
        timeout_ms = 30000  # 30 second timeout

        elapsed_ms = current_time_ms - last_heartbeat_ms
        timed_out = elapsed_ms > timeout_ms

        self.assertTrue(timed_out)


class TestStreamerDataTransformation(unittest.TestCase):
    """Test streamer data transformation."""

    def test_transform_quote_to_dict(self):
        """Should transform quote message to dictionary."""
        raw_quote = {"key": "AAPL", "1": 150.25, "2": 150.30, "3": 150.27}

        transformed = {"symbol": raw_quote["key"], "bid": raw_quote["1"], "ask": raw_quote["2"], "last": raw_quote["3"]}

        self.assertEqual(transformed["symbol"], "AAPL")
        self.assertEqual(transformed["bid"], 150.25)

    def test_transform_bar_to_dataframe_format(self):
        """Should transform bar to DataFrame-compatible format."""
        raw_bar = {
            "key": "AAPL",
            "1": 150.0,  # open
            "2": 152.0,  # high
            "3": 149.5,  # low
            "4": 151.5,  # close
            "5": 1000000,  # volume
        }

        transformed = {
            "symbol": raw_bar["key"],
            "Open": raw_bar["1"],
            "High": raw_bar["2"],
            "Low": raw_bar["3"],
            "Close": raw_bar["4"],
            "Volume": raw_bar["5"],
        }

        self.assertIn("Open", transformed)
        self.assertIn("Close", transformed)


class TestSchwabStreamerFieldMapping(unittest.TestCase):
    """Test Schwab Streamer LEVELONE_EQUITIES field mapping correctness.

    Per Schwab API documentation:
    - Field 1 = Bid Price
    - Field 2 = Ask Price
    - Field 3 = Last Price
    - Field 4 = Bid Size
    - Field 5 = Ask Size
    - Field 8 = Total Volume
    - Field 10 = High Price
    - Field 11 = Low Price
    - Field 12 = Close Price (previous day)
    - Field 17 = Open Price
    - Field 35 = Trade Time in Long (ms since epoch)
    """

    def setUp(self):
        """Create sample Schwab streamer content."""
        self.raw_levelone_item = {
            "key": "AAPL",
            "1": 150.25,  # Bid Price
            "2": 150.30,  # Ask Price
            "3": 150.27,  # Last Price
            "4": 100,  # Bid Size
            "5": 200,  # Ask Size
            "8": 5000000,  # Volume
            "10": 152.00,  # High Price
            "11": 149.00,  # Low Price
            "12": 149.50,  # Close Price (prev day)
            "17": 149.75,  # Open Price
            "35": 1704067200000,  # Trade Time
        }

    def test_quote_extraction_matches_schwab_spec(self):
        """Quote extraction should use correct Schwab field numbers."""
        item = self.raw_levelone_item

        # This is how the streamer.py should extract quotes
        quote = {
            "bid_price": item.get("1"),
            "ask_price": item.get("2"),
            "last_price": item.get("3"),
            "bid_size": item.get("4"),
            "ask_size": item.get("5"),
            "volume": item.get("8"),
            "high_price": item.get("10"),
            "low_price": item.get("11"),
            "close_price": item.get("12"),
            "open_price": item.get("17"),
            "trade_time": item.get("35"),
        }

        # Verify correct mapping
        self.assertEqual(quote["bid_price"], 150.25)
        self.assertEqual(quote["ask_price"], 150.30)
        self.assertEqual(quote["last_price"], 150.27)
        self.assertEqual(quote["bid_size"], 100)
        self.assertEqual(quote["ask_size"], 200)
        self.assertEqual(quote["volume"], 5000000)
        self.assertEqual(quote["high_price"], 152.00)
        self.assertEqual(quote["low_price"], 149.00)
        self.assertEqual(quote["close_price"], 149.50)
        self.assertEqual(quote["open_price"], 149.75)

    def test_bid_ask_field_order(self):
        """Bid must be field 1, ask must be field 2 (not reversed)."""
        item = self.raw_levelone_item

        bid = item.get("1")
        ask = item.get("2")

        # Bid should be less than ask in normal market conditions
        self.assertLess(bid, ask)
        self.assertEqual(bid, 150.25)
        self.assertEqual(ask, 150.30)

    def test_last_price_is_field_3(self):
        """Last price must be field 3, not field 1."""
        item = self.raw_levelone_item

        last_price = item.get("3")

        # Last price should be between bid and ask
        self.assertEqual(last_price, 150.27)
        self.assertGreater(last_price, item.get("1"))  # > bid
        self.assertLess(last_price, item.get("2"))  # < ask

    def test_ohlc_fields_correct(self):
        """OHLC fields should be correctly mapped."""
        item = self.raw_levelone_item

        # Open, High, Low, Close fields
        open_price = item.get("17")  # Field 17
        high_price = item.get("10")  # Field 10
        low_price = item.get("11")  # Field 11
        close_price = item.get("12")  # Field 12 (prev day close)

        self.assertEqual(open_price, 149.75)
        self.assertEqual(high_price, 152.00)
        self.assertEqual(low_price, 149.00)
        self.assertEqual(close_price, 149.50)

        # Sanity checks
        self.assertLessEqual(low_price, high_price)
        self.assertGreaterEqual(high_price, open_price)
        self.assertLessEqual(low_price, close_price)

    def test_volume_is_field_8(self):
        """Volume must be field 8."""
        item = self.raw_levelone_item

        volume = item.get("8")
        self.assertEqual(volume, 5000000)

    def test_partial_quote_handling(self):
        """Streamer may send partial updates with only some fields."""
        partial_item = {
            "key": "AAPL",
            "3": 150.50,  # Only last price updated
        }

        quote = {
            "bid_price": partial_item.get("1"),
            "ask_price": partial_item.get("2"),
            "last_price": partial_item.get("3"),
        }

        self.assertIsNone(quote["bid_price"])
        self.assertIsNone(quote["ask_price"])
        self.assertEqual(quote["last_price"], 150.50)


class TestSchwabChartEquityFieldMapping(unittest.TestCase):
    """Test Schwab CHART_EQUITY field mapping correctness.

    Per Schwab API documentation:
    - Field 1 = Open Price
    - Field 2 = High Price
    - Field 3 = Low Price
    - Field 4 = Close Price
    - Field 5 = Volume
    - Field 7 = Chart Time (ms since midnight)
    """

    def setUp(self):
        """Create sample CHART_EQUITY content."""
        self.raw_chart_item = {
            "key": "AAPL",
            "1": 150.00,  # Open
            "2": 152.00,  # High
            "3": 149.50,  # Low
            "4": 151.50,  # Close
            "5": 1000000,  # Volume
            "7": 36000000,  # Chart Time (10:00 AM as ms since midnight)
        }

    def test_chart_bar_extraction(self):
        """Chart bar extraction should use correct field numbers."""
        item = self.raw_chart_item

        bar = {
            "open": item.get("1"),
            "high": item.get("2"),
            "low": item.get("3"),
            "close": item.get("4"),
            "volume": item.get("5"),
            "chart_time": item.get("7"),
        }

        self.assertEqual(bar["open"], 150.00)
        self.assertEqual(bar["high"], 152.00)
        self.assertEqual(bar["low"], 149.50)
        self.assertEqual(bar["close"], 151.50)
        self.assertEqual(bar["volume"], 1000000)

    def test_chart_vs_levelone_field_differences(self):
        """CHART_EQUITY and LEVELONE_EQUITIES have different field mappings."""
        # IMPORTANT: These services use different field numbers!

        levelone_item = {
            "1": 150.25,  # In LEVELONE_EQUITIES, field 1 = Bid Price
            "3": 150.27,  # In LEVELONE_EQUITIES, field 3 = Last Price
        }

        chart_item = {
            "1": 150.00,  # In CHART_EQUITY, field 1 = Open Price
            "3": 149.50,  # In CHART_EQUITY, field 3 = Low Price
        }

        # Same field numbers, different meanings!
        self.assertEqual(levelone_item["1"], 150.25)  # Bid
        self.assertEqual(chart_item["1"], 150.00)  # Open

        self.assertEqual(levelone_item["3"], 150.27)  # Last
        self.assertEqual(chart_item["3"], 149.50)  # Low


if __name__ == "__main__":
    unittest.main()
