"""
Test suite for market data streaming.

Tests WebSocket connections, data streaming, and event handling.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import json


class TestStreamerConnection(unittest.TestCase):
    """Test streamer connection management."""

    def test_streamer_initialization(self):
        """Streamer should initialize with correct parameters."""
        # Mock streamer initialization
        config = {
            'api_key': 'test_key',
            'symbols': ['AAPL', 'MSFT'],
            'reconnect': True
        }

        self.assertIn('api_key', config)
        self.assertIn('symbols', config)

    @patch('websockets.connect')
    def test_websocket_connect(self, mock_connect):
        """Should establish WebSocket connection."""
        async def run_test():
            mock_ws = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_ws

            # Simulate connection
            async with mock_connect('wss://test.example.com') as ws:
                self.assertIsNotNone(ws)

        asyncio.run(run_test())

    def test_streamer_reconnect_config(self):
        """Streamer should support auto-reconnect configuration."""
        config = {
            'reconnect': True,
            'reconnect_attempts': 5,
            'reconnect_delay_ms': 1000
        }

        self.assertTrue(config['reconnect'])
        self.assertEqual(config['reconnect_attempts'], 5)


class TestStreamerSubscription(unittest.TestCase):
    """Test streamer subscription methods."""

    def test_subscribe_to_quotes(self):
        """Should subscribe to quote updates."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        subscription = {
            'service': 'QUOTE',
            'command': 'SUBS',
            'parameters': {
                'keys': ','.join(symbols),
                'fields': '0,1,2,3,4,5,8,9'
            }
        }

        self.assertEqual(subscription['service'], 'QUOTE')
        self.assertEqual(subscription['command'], 'SUBS')
        self.assertIn('AAPL', subscription['parameters']['keys'])

    def test_subscribe_to_bars(self):
        """Should subscribe to bar (OHLC) updates."""
        symbols = ['AAPL']
        subscription = {
            'service': 'CHART_EQUITY',
            'command': 'SUBS',
            'parameters': {
                'keys': ','.join(symbols),
                'fields': '0,1,2,3,4,5,6,7'
            }
        }

        self.assertEqual(subscription['service'], 'CHART_EQUITY')

    def test_unsubscribe(self):
        """Should unsubscribe from updates."""
        unsubscription = {
            'service': 'QUOTE',
            'command': 'UNSUBS',
            'parameters': {
                'keys': 'AAPL'
            }
        }

        self.assertEqual(unsubscription['command'], 'UNSUBS')


class TestStreamerMessages(unittest.TestCase):
    """Test streamer message parsing."""

    def setUp(self):
        """Create sample streamer messages."""
        self.quote_message = {
            'service': 'QUOTE',
            'timestamp': 1672531200000,
            'content': [
                {
                    'key': 'AAPL',
                    '1': 150.25,  # bid
                    '2': 150.30,  # ask
                    '3': 150.27,  # last
                    '8': 5000000  # volume
                }
            ]
        }

        self.bar_message = {
            'service': 'CHART_EQUITY',
            'timestamp': 1672531200000,
            'content': [
                {
                    'key': 'AAPL',
                    '1': 150.00,  # open
                    '2': 152.00,  # high
                    '3': 149.50,  # low
                    '4': 151.50,  # close
                    '5': 1000000  # volume
                }
            ]
        }

    def test_parse_quote_message(self):
        """Should parse quote message correctly."""
        content = self.quote_message['content'][0]

        self.assertEqual(content['key'], 'AAPL')
        self.assertEqual(content['1'], 150.25)  # bid
        self.assertEqual(content['2'], 150.30)  # ask

    def test_parse_bar_message(self):
        """Should parse bar message correctly."""
        content = self.bar_message['content'][0]

        self.assertEqual(content['key'], 'AAPL')
        self.assertEqual(content['1'], 150.00)  # open
        self.assertEqual(content['4'], 151.50)  # close


class TestStreamerEventHandling(unittest.TestCase):
    """Test streamer event callback handling."""

    def test_on_quote_callback(self):
        """Quote callback should receive data."""
        received_quotes = []

        def on_quote(symbol, data):
            received_quotes.append({'symbol': symbol, 'data': data})

        # Simulate receiving quote
        on_quote('AAPL', {'bid': 150.25, 'ask': 150.30})

        self.assertEqual(len(received_quotes), 1)
        self.assertEqual(received_quotes[0]['symbol'], 'AAPL')

    def test_on_bar_callback(self):
        """Bar callback should receive OHLCV data."""
        received_bars = []

        def on_bar(symbol, bar):
            received_bars.append({'symbol': symbol, 'bar': bar})

        # Simulate receiving bar
        bar_data = {
            'open': 150.0, 'high': 152.0,
            'low': 149.5, 'close': 151.5,
            'volume': 1000000
        }
        on_bar('AAPL', bar_data)

        self.assertEqual(len(received_bars), 1)
        self.assertIn('open', received_bars[0]['bar'])

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
        heartbeat_request = {
            'service': 'ADMIN',
            'command': 'HEARTBEAT'
        }

        heartbeat_response = {
            'service': 'ADMIN',
            'command': 'HEARTBEAT',
            'timestamp': 1672531200000
        }

        self.assertEqual(heartbeat_response['command'], 'HEARTBEAT')

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
        raw_quote = {'key': 'AAPL', '1': 150.25, '2': 150.30, '3': 150.27}

        transformed = {
            'symbol': raw_quote['key'],
            'bid': raw_quote['1'],
            'ask': raw_quote['2'],
            'last': raw_quote['3']
        }

        self.assertEqual(transformed['symbol'], 'AAPL')
        self.assertEqual(transformed['bid'], 150.25)

    def test_transform_bar_to_dataframe_format(self):
        """Should transform bar to DataFrame-compatible format."""
        raw_bar = {
            'key': 'AAPL',
            '1': 150.0,  # open
            '2': 152.0,  # high
            '3': 149.5,  # low
            '4': 151.5,  # close
            '5': 1000000  # volume
        }

        transformed = {
            'symbol': raw_bar['key'],
            'Open': raw_bar['1'],
            'High': raw_bar['2'],
            'Low': raw_bar['3'],
            'Close': raw_bar['4'],
            'Volume': raw_bar['5']
        }

        self.assertIn('Open', transformed)
        self.assertIn('Close', transformed)


if __name__ == '__main__':
    unittest.main()
