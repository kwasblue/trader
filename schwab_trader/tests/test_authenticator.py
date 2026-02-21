"""
Test suite for the Authenticator class.

Tests token management, authentication, and renewal operations.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock, mock_open, AsyncMock
import time
import json
import asyncio


def reset_authenticator_singleton():
    """Reset Authenticator singleton for testing."""
    from data.streaming.authenticator import Authenticator
    Authenticator._instance = None


class TestAuthenticatorInit(unittest.TestCase):
    """Test Authenticator initialization."""

    def setUp(self):
        reset_authenticator_singleton()

    @patch('data.streaming.authenticator.ConfigLoader')
    @patch('data.streaming.authenticator.Logger')
    @patch('data.streaming.authenticator.FileWriter')
    @patch('data.streaming.authenticator.load_dotenv')
    @patch.dict(os.environ, {
        'SCHWAB_API_KEY': 'test_key',
        'SCHWAB_SECRET': 'test_secret',
        'SCHWAB_REDIRECT_URL': 'https://localhost'
    })
    def test_singleton_pattern(self, mock_dotenv, mock_writer, mock_logger, mock_config):
        """Authenticator should be a singleton."""
        mock_config.return_value.load_config.return_value = {
            'folders': {'env': '.env', 'logs': '/tmp', 'app_path': '/tmp', 'tokens': '/tmp/tokens'},
            'auth': {'authentication_url': 'https://auth.example.com', 'token_endpoint': 'https://token.example.com'}
        }
        mock_logger.return_value.get_logger.return_value = MagicMock()

        from data.streaming.authenticator import Authenticator
        auth1 = Authenticator()
        auth2 = Authenticator()

        self.assertIs(auth1, auth2)

    @patch('data.streaming.authenticator.ConfigLoader')
    @patch('data.streaming.authenticator.Logger')
    @patch('data.streaming.authenticator.FileWriter')
    @patch('data.streaming.authenticator.load_dotenv')
    @patch.dict(os.environ, {
        'SCHWAB_API_KEY': 'test_key',
        'SCHWAB_SECRET': 'test_secret',
        'SCHWAB_REDIRECT_URL': 'https://localhost'
    })
    def test_loads_credentials_from_env(self, mock_dotenv, mock_writer, mock_logger, mock_config):
        """Should load API credentials from environment."""
        mock_config.return_value.load_config.return_value = {
            'folders': {'env': '.env', 'logs': '/tmp', 'app_path': '/tmp', 'tokens': '/tmp/tokens'},
            'auth': {'authentication_url': 'https://auth.example.com', 'token_endpoint': 'https://token.example.com'}
        }
        mock_logger.return_value.get_logger.return_value = MagicMock()

        from data.streaming.authenticator import Authenticator
        auth = Authenticator()

        self.assertEqual(auth.apikey, 'test_key')
        self.assertEqual(auth.secret, 'test_secret')


class TestTokenOperations(unittest.TestCase):
    """Test token read/write operations."""

    def setUp(self):
        reset_authenticator_singleton()

    @patch('data.streaming.authenticator.ConfigLoader')
    @patch('data.streaming.authenticator.Logger')
    @patch('data.streaming.authenticator.FileWriter')
    @patch('data.streaming.authenticator.load_dotenv')
    @patch.dict(os.environ, {
        'SCHWAB_API_KEY': 'test_key',
        'SCHWAB_SECRET': 'test_secret',
        'SCHWAB_REDIRECT_URL': 'https://localhost'
    })
    def _create_authenticator(self, mock_dotenv, mock_writer, mock_logger, mock_config):
        mock_config.return_value.load_config.return_value = {
            'folders': {'env': '.env', 'logs': '/tmp', 'app_path': '/tmp', 'tokens': '/tmp/tokens'},
            'auth': {'authentication_url': 'https://auth.example.com', 'token_endpoint': 'https://token.example.com'}
        }
        mock_logger.return_value.get_logger.return_value = MagicMock()

        from data.streaming.authenticator import Authenticator
        return Authenticator()

    def test_read_token_file_success(self):
        """Should read token file successfully."""
        auth = self._create_authenticator()

        token_data = {
            'access_token': 'test_access_token',
            'refresh_token': 'test_refresh_token',
            'access_time': time.time(),
            'expires_in': 1800
        }

        auth.writer.find.return_value = '/tmp/token_file.json'
        with patch('builtins.open', mock_open(read_data=json.dumps(token_data))):
            result = auth._read_token_file()

        self.assertEqual(result['access_token'], 'test_access_token')

    def test_read_token_file_not_found(self):
        """Should return empty dict if token file not found."""
        auth = self._create_authenticator()
        auth.writer.find.return_value = '/nonexistent/path'

        with patch('builtins.open', side_effect=FileNotFoundError):
            result = auth._read_token_file()

        self.assertEqual(result, {})

    def test_access_token_retrieval(self):
        """Should retrieve access token from file."""
        auth = self._create_authenticator()

        token_data = {'access_token': 'my_access_token', 'refresh_token': 'my_refresh'}
        auth.writer.find.return_value = '/tmp/token_file.json'

        with patch('builtins.open', mock_open(read_data=json.dumps(token_data))):
            result = auth.access_token()

        self.assertEqual(result, 'my_access_token')

    def test_refresh_token_retrieval(self):
        """Should retrieve refresh token from file."""
        auth = self._create_authenticator()

        token_data = {'access_token': 'my_access', 'refresh_token': 'my_refresh_token'}
        auth.writer.find.return_value = '/tmp/token_file.json'

        with patch('builtins.open', mock_open(read_data=json.dumps(token_data))):
            result = auth.refresh_token()

        self.assertEqual(result, 'my_refresh_token')


class TestTokenExpiration(unittest.TestCase):
    """Test token expiration detection."""

    def setUp(self):
        reset_authenticator_singleton()

    @patch('data.streaming.authenticator.ConfigLoader')
    @patch('data.streaming.authenticator.Logger')
    @patch('data.streaming.authenticator.FileWriter')
    @patch('data.streaming.authenticator.load_dotenv')
    @patch.dict(os.environ, {
        'SCHWAB_API_KEY': 'test_key',
        'SCHWAB_SECRET': 'test_secret',
        'SCHWAB_REDIRECT_URL': 'https://localhost'
    })
    def _create_authenticator(self, mock_dotenv, mock_writer, mock_logger, mock_config):
        mock_config.return_value.load_config.return_value = {
            'folders': {'env': '.env', 'logs': '/tmp', 'app_path': '/tmp', 'tokens': '/tmp/tokens'},
            'auth': {'authentication_url': 'https://auth.example.com', 'token_endpoint': 'https://token.example.com'}
        }
        mock_logger.return_value.get_logger.return_value = MagicMock()

        from data.streaming.authenticator import Authenticator
        return Authenticator()

    def test_access_token_expired(self):
        """Should detect expired access token."""
        auth = self._create_authenticator()

        # Token that expired 1 hour ago
        token_data = {
            'access_time': time.time() - 7200,  # 2 hours ago
            'expires_in': 1800  # 30 minutes
        }

        result = auth._is_access_token_expired(token_data)
        self.assertTrue(result)

    def test_access_token_valid(self):
        """Should detect valid access token."""
        auth = self._create_authenticator()

        # Token issued 10 minutes ago, expires in 30 minutes
        token_data = {
            'access_time': time.time() - 600,
            'expires_in': 1800
        }

        result = auth._is_access_token_expired(token_data)
        self.assertFalse(result)

    def test_refresh_token_expired(self):
        """Should detect expired refresh token."""
        auth = self._create_authenticator()

        # Refresh token from 7 days ago (expires in 6 days by default)
        token_data = {
            'refresh_time': time.time() - (7 * 24 * 60 * 60)
        }

        result = auth._is_refresh_token_expired(token_data)
        self.assertTrue(result)

    def test_refresh_token_valid(self):
        """Should detect valid refresh token."""
        auth = self._create_authenticator()

        # Refresh token from 1 day ago
        token_data = {
            'refresh_time': time.time() - (24 * 60 * 60)
        }

        result = auth._is_refresh_token_expired(token_data)
        self.assertFalse(result)

    def test_contains_error(self):
        """Should detect error in token data."""
        auth = self._create_authenticator()

        token_data = {'error': 'invalid_grant'}
        result = auth._contains_error(token_data)
        self.assertTrue(result)

    def test_no_error(self):
        """Should return False when no error."""
        auth = self._create_authenticator()

        token_data = {'access_token': 'valid_token'}
        result = auth._contains_error(token_data)
        self.assertFalse(result)


class TestTokenRenewal(unittest.TestCase):
    """Test token renewal operations."""

    def setUp(self):
        reset_authenticator_singleton()

    @patch('data.streaming.authenticator.ConfigLoader')
    @patch('data.streaming.authenticator.Logger')
    @patch('data.streaming.authenticator.FileWriter')
    @patch('data.streaming.authenticator.load_dotenv')
    @patch.dict(os.environ, {
        'SCHWAB_API_KEY': 'test_key',
        'SCHWAB_SECRET': 'test_secret',
        'SCHWAB_REDIRECT_URL': 'https://localhost'
    })
    def _create_authenticator(self, mock_dotenv, mock_writer, mock_logger, mock_config):
        mock_config.return_value.load_config.return_value = {
            'folders': {'env': '.env', 'logs': '/tmp', 'app_path': '/tmp', 'tokens': '/tmp/tokens'},
            'auth': {'authentication_url': 'https://auth.example.com', 'token_endpoint': 'https://token.example.com'}
        }
        mock_logger.return_value.get_logger.return_value = MagicMock()

        from data.streaming.authenticator import Authenticator
        return Authenticator()

    def test_renew_access_success(self):
        """Should renew access token successfully."""
        auth = self._create_authenticator()

        # Mock the refresh token
        auth.writer.find.return_value = '/tmp/token_file.json'
        token_data = {'refresh_token': 'valid_refresh_token'}

        async def run_test():
            with patch('builtins.open', mock_open(read_data=json.dumps(token_data))):
                # Mock the _get method
                auth._get = AsyncMock(return_value={
                    'access_token': 'new_access_token',
                    'refresh_token': 'valid_refresh_token',
                    'expires_in': 1800
                })

                result = await auth.renew_access()
                return result

        result = asyncio.run(run_test())
        self.assertTrue(result)

    def test_set_headers_params(self):
        """Should set correct headers and params."""
        auth = self._create_authenticator()

        endpoint, headers, params = auth._set_headers_params(
            endpoint='https://api.example.com/token',
            apikey='test_key',
            secret='test_secret',
            grant_type='refresh_token',
            refresh_token='my_refresh_token'
        )

        self.assertEqual(endpoint, 'https://api.example.com/token')
        self.assertIn('Authorization', headers)
        self.assertIn('Basic', headers['Authorization'])
        self.assertEqual(params['grant_type'], 'refresh_token')
        self.assertEqual(params['refresh_token'], 'my_refresh_token')


if __name__ == '__main__':
    unittest.main()
