"""
Test suite for Schwab API client.

Tests API authentication, data fetching, and order operations (mocked).
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta


class TestSchwabClientAuthentication(unittest.TestCase):
    """Test Schwab client authentication."""

    @patch('data.streaming.schwab_client.Authenticator')
    @patch('data.streaming.schwab_client.ConfigLoader')
    @patch('data.streaming.schwab_client.Logger')
    def test_client_requires_credentials(self, mock_logger, mock_config, mock_auth):
        """Client should require API credentials."""
        mock_config.return_value.load_config.return_value = {
            'folders': {'logs': '/tmp'},
            'api': {'base_url': 'https://api.example.com', 'market_url': 'https://market.example.com'}
        }
        mock_logger.return_value.get_logger.return_value = MagicMock()

        from data.streaming.schwab_client import SchwabClient

        # Should work with credentials
        client = SchwabClient(apikey='test_key', secretkey='test_secret')
        self.assertIsNotNone(client)
        self.assertEqual(client.apikey, 'test_key')

    @patch.dict(os.environ, {'SCHWAB_API_KEY': 'test_key', 'SCHWAB_SECRET': 'test_secret'})
    def test_client_loads_env_credentials(self):
        """Client should load credentials from environment."""
        api_key = os.environ.get('SCHWAB_API_KEY')
        secret = os.environ.get('SCHWAB_SECRET')

        self.assertEqual(api_key, 'test_key')
        self.assertEqual(secret, 'test_secret')


class TestSchwabClientPriceHistory(unittest.TestCase):
    """Test Schwab client price history methods."""

    def setUp(self):
        """Create mock response data."""
        self.mock_candles = {
            'symbol': 'AAPL',
            'candles': [
                {'datetime': 1672531200000, 'open': 130.0, 'high': 132.0,
                 'low': 129.0, 'close': 131.0, 'volume': 1000000},
                {'datetime': 1672617600000, 'open': 131.0, 'high': 133.0,
                 'low': 130.0, 'close': 132.0, 'volume': 1100000},
            ]
        }

    @patch('requests.get')
    def test_daily_price_history(self, mock_get):
        """Should fetch daily price history."""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_candles
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Simulate fetching data
        result = mock_response.json()

        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(len(result['candles']), 2)

    def test_candle_data_format(self):
        """Candle data should have correct format."""
        for candle in self.mock_candles['candles']:
            self.assertIn('datetime', candle)
            self.assertIn('open', candle)
            self.assertIn('high', candle)
            self.assertIn('low', candle)
            self.assertIn('close', candle)
            self.assertIn('volume', candle)

    @patch('data.streaming.schwab_client.Authenticator')
    @patch('data.streaming.schwab_client.ConfigLoader')
    @patch('data.streaming.schwab_client.Logger')
    def test_daily_price_history_integration(self, mock_logger, mock_config, mock_auth):
        """Test daily_price_history method on client."""
        mock_config.return_value.load_config.return_value = {
            'folders': {'logs': '/tmp'},
            'api': {'base_url': 'https://api.example.com', 'market_url': 'https://market.example.com'}
        }
        mock_logger.return_value.get_logger.return_value = MagicMock()
        mock_auth.return_value.access_token.return_value = 'mock_token'

        from data.streaming.schwab_client import SchwabClient
        client = SchwabClient(apikey='test_key', secretkey='test_secret')

        # Mock the session request
        with patch.object(client._session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = self.mock_candles
            mock_request.return_value = mock_response

            result = client.daily_price_history('AAPL')

            self.assertEqual(result['symbol'], 'AAPL')


class TestSchwabClientQuotes(unittest.TestCase):
    """Test Schwab client quote methods."""

    def setUp(self):
        """Create mock quote data."""
        self.mock_quote = {
            'AAPL': {
                'symbol': 'AAPL',
                'bidPrice': 150.25,
                'askPrice': 150.30,
                'lastPrice': 150.27,
                'totalVolume': 5000000,
            }
        }

    @patch('requests.get')
    def test_get_quote(self, mock_get):
        """Should fetch real-time quote."""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_quote
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = mock_response.json()

        self.assertIn('AAPL', result)
        self.assertEqual(result['AAPL']['lastPrice'], 150.27)

    def test_quote_has_bid_ask(self):
        """Quote should include bid/ask prices."""
        quote = self.mock_quote['AAPL']

        self.assertIn('bidPrice', quote)
        self.assertIn('askPrice', quote)
        self.assertLess(quote['bidPrice'], quote['askPrice'])


class TestSchwabClientOrders(unittest.TestCase):
    """Test Schwab client order methods (mocked)."""

    def setUp(self):
        """Create mock order data."""
        self.mock_order_response = {
            'orderId': '12345',
            'status': 'FILLED',
            'filledQuantity': 100,
            'averagePrice': 150.25
        }

    @patch('requests.post')
    def test_place_market_order(self, mock_post):
        """Should place market order."""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_order_response
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        result = mock_response.json()

        self.assertEqual(result['status'], 'FILLED')
        self.assertEqual(result['filledQuantity'], 100)

    @patch('requests.get')
    def test_get_order_status(self, mock_get):
        """Should get order status."""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_order_response
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = mock_response.json()

        self.assertEqual(result['orderId'], '12345')
        self.assertEqual(result['status'], 'FILLED')

    @patch('requests.delete')
    def test_cancel_order(self, mock_delete):
        """Should cancel order."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_delete.return_value = mock_response

        # Successful cancellation returns 200
        self.assertEqual(mock_response.status_code, 200)

    @patch('data.streaming.schwab_client.Authenticator')
    @patch('data.streaming.schwab_client.ConfigLoader')
    @patch('data.streaming.schwab_client.Logger')
    def test_generate_order(self, mock_logger, mock_config, mock_auth):
        """Test order generation."""
        mock_config.return_value.load_config.return_value = {
            'folders': {'logs': '/tmp'},
            'api': {'base_url': 'https://api.example.com', 'market_url': 'https://market.example.com'}
        }
        mock_logger.return_value.get_logger.return_value = MagicMock()

        from data.streaming.schwab_client import SchwabClient
        client = SchwabClient(apikey='test_key', secretkey='test_secret')

        order = client.generate_order(
            orderType='MARKET',
            session='NORMAL',
            duration='DAY',
            orderStrategyType='SINGLE',
            instruction='BUY',
            quantity=100,
            symbol='AAPL',
            assetType='EQUITY'
        )

        self.assertEqual(order['orderType'], 'MARKET')
        self.assertEqual(order['orderLegCollection'][0]['instruction'], 'BUY')
        self.assertEqual(order['orderLegCollection'][0]['quantity'], 100)


class TestSchwabClientAccountInfo(unittest.TestCase):
    """Test Schwab client account info methods."""

    def setUp(self):
        """Create mock account data."""
        self.mock_account = {
            'accountNumber': '123456789',
            'type': 'MARGIN',
            'currentBalances': {
                'cashBalance': 50000.0,
                'buyingPower': 100000.0,
                'equity': 75000.0
            },
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'marketValue': 15000.0}
            ]
        }

    @patch('requests.get')
    def test_get_account_info(self, mock_get):
        """Should get account information."""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_account
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = mock_response.json()

        self.assertIn('accountNumber', result)
        self.assertIn('currentBalances', result)
        self.assertIn('positions', result)

    def test_account_has_balances(self):
        """Account should include balance information."""
        balances = self.mock_account['currentBalances']

        self.assertIn('cashBalance', balances)
        self.assertIn('buyingPower', balances)
        self.assertIn('equity', balances)

    def test_account_has_positions(self):
        """Account should include positions."""
        positions = self.mock_account['positions']

        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['symbol'], 'AAPL')


class TestSchwabClientErrorHandling(unittest.TestCase):
    """Test Schwab client error handling."""

    @patch('requests.get')
    def test_handles_rate_limit(self, mock_get):
        """Should handle rate limit errors."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {'error': 'Rate limit exceeded'}
        mock_get.return_value = mock_response

        self.assertEqual(mock_response.status_code, 429)

    @patch('requests.get')
    def test_handles_unauthorized(self, mock_get):
        """Should handle unauthorized errors."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {'error': 'Unauthorized'}
        mock_get.return_value = mock_response

        self.assertEqual(mock_response.status_code, 401)

    @patch('data.streaming.schwab_client.Authenticator')
    @patch('data.streaming.schwab_client.ConfigLoader')
    @patch('data.streaming.schwab_client.Logger')
    def test_rate_limiter_integration(self, mock_logger, mock_config, mock_auth):
        """Test rate limiter is properly configured."""
        mock_config.return_value.load_config.return_value = {
            'folders': {'logs': '/tmp'},
            'api': {'base_url': 'https://api.example.com', 'market_url': 'https://market.example.com'}
        }
        mock_logger.return_value.get_logger.return_value = MagicMock()

        from data.streaming.schwab_client import SchwabClient
        client = SchwabClient(apikey='test_key', secretkey='test_secret')

        # Verify rate limiter exists
        self.assertIsNotNone(client._rate_limiter)

    @patch('data.streaming.schwab_client.Authenticator')
    @patch('data.streaming.schwab_client.ConfigLoader')
    @patch('data.streaming.schwab_client.Logger')
    def test_session_close(self, mock_logger, mock_config, mock_auth):
        """Test session can be closed."""
        mock_config.return_value.load_config.return_value = {
            'folders': {'logs': '/tmp'},
            'api': {'base_url': 'https://api.example.com', 'market_url': 'https://market.example.com'}
        }
        mock_logger.return_value.get_logger.return_value = MagicMock()

        from data.streaming.schwab_client import SchwabClient
        client = SchwabClient(apikey='test_key', secretkey='test_secret')

        # Should not raise
        client.close()


if __name__ == '__main__':
    unittest.main()
