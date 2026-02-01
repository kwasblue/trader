"""
Tests for CredentialValidator

Coverage:
- Schwab token validation
- Alpaca API key validation
- Token expiry detection
- Best source recommendations
- Error handling
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time
import json
import os

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.credential_validator import (
    CredentialValidator,
    CredentialStatus,
    ValidationResult,
    check_credentials,
    can_use_schwab,
    can_use_alpaca
)


class TestCredentialStatus:
    """Tests for CredentialStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert CredentialStatus.VALID.value == "valid"
        assert CredentialStatus.EXPIRED.value == "expired"
        assert CredentialStatus.EXPIRING_SOON.value == "expiring_soon"
        assert CredentialStatus.MISSING.value == "missing"
        assert CredentialStatus.INVALID.value == "invalid"
        assert CredentialStatus.ERROR.value == "error"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            broker="alpaca",
            status=CredentialStatus.VALID,
            message="Credentials valid",
            expires_in=3600,
            can_trade=True,
            can_fetch_data=True,
            details={"account_status": "ACTIVE"}
        )
        assert result.broker == "alpaca"
        assert result.status == CredentialStatus.VALID
        assert result.can_trade is True
        assert result.details["account_status"] == "ACTIVE"

    def test_defaults(self):
        """Test ValidationResult default values."""
        result = ValidationResult(
            broker="test",
            status=CredentialStatus.MISSING,
            message="Test"
        )
        assert result.expires_in is None
        assert result.can_trade is False
        assert result.can_fetch_data is False
        assert result.details == {}


class TestCredentialValidator:
    """Tests for CredentialValidator."""

    @pytest.fixture
    def validator(self):
        return CredentialValidator()

    @pytest.mark.asyncio
    async def test_validate_alpaca_missing_credentials(self, validator):
        """Test Alpaca validation with missing credentials."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure env vars are not set
            os.environ.pop('ALPACA_API_KEY', None)
            os.environ.pop('ALPACA_KEY_ID', None)
            os.environ.pop('ALPACA_SECRET_KEY', None)
            os.environ.pop('ALPACA_SECRET', None)

            result = await validator.validate_alpaca()
            assert result.status == CredentialStatus.MISSING
            assert result.can_trade is False
            assert result.can_fetch_data is False

    @pytest.mark.asyncio
    async def test_validate_alpaca_valid_credentials(self, validator):
        """Test Alpaca validation with valid credentials."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            with patch('alpaca.trading.client.TradingClient') as MockClient:
                mock_client = MagicMock()
                mock_account = MagicMock()
                mock_account.status = 'ACTIVE'
                mock_account.buying_power = 10000.0
                mock_client.get_account.return_value = mock_account
                MockClient.return_value = mock_client

                with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_thread:
                    mock_thread.return_value = mock_account

                    result = await validator.validate_alpaca()
                    assert result.status == CredentialStatus.VALID
                    assert result.can_trade is True
                    assert result.can_fetch_data is True

    @pytest.mark.asyncio
    async def test_validate_alpaca_invalid_credentials(self, validator):
        """Test Alpaca validation with invalid credentials."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'bad_key',
            'ALPACA_SECRET_KEY': 'bad_secret'
        }):
            with patch('alpaca.trading.client.TradingClient') as MockClient:
                MockClient.side_effect = Exception("forbidden")

                with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_thread:
                    mock_thread.side_effect = Exception("forbidden")

                    result = await validator.validate_alpaca()
                    assert result.status == CredentialStatus.INVALID
                    assert result.can_trade is False

    @pytest.mark.asyncio
    async def test_validate_schwab_missing_env_vars(self, validator):
        """Test Schwab validation with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('SCHWAB_API_KEY', None)
            os.environ.pop('SCHWAB_SECRET', None)

            result = await validator.validate_schwab()
            assert result.status == CredentialStatus.MISSING
            assert "not set" in result.message.lower()

    @pytest.mark.asyncio
    async def test_validate_schwab_missing_token_file(self, validator):
        """Test Schwab validation with missing token file."""
        with patch.dict(os.environ, {
            'SCHWAB_API_KEY': 'test_key',
            'SCHWAB_SECRET': 'test_secret'
        }):
            with patch.object(validator, '_find_schwab_token_file', return_value=None):
                result = await validator.validate_schwab()
                assert result.status == CredentialStatus.MISSING
                assert "token file" in result.message.lower()

    @pytest.mark.asyncio
    async def test_validate_schwab_valid_tokens(self, validator, tmp_path):
        """Test Schwab validation with valid tokens."""
        # Create a valid token file
        token_file = tmp_path / "token_file.json"
        token_data = {
            "access_token": "valid_access_token",
            "refresh_token": "valid_refresh_token",
            "access_time": time.time(),  # Just created
            "refresh_time": time.time(),  # Just created
            "expires_in": 1800  # 30 minutes
        }
        token_file.write_text(json.dumps(token_data))

        with patch.dict(os.environ, {
            'SCHWAB_API_KEY': 'test_key',
            'SCHWAB_SECRET': 'test_secret'
        }):
            with patch.object(validator, '_find_schwab_token_file', return_value=token_file):
                result = await validator.validate_schwab()
                assert result.status == CredentialStatus.VALID
                assert result.can_trade is True
                assert result.can_fetch_data is True

    @pytest.mark.asyncio
    async def test_validate_schwab_expired_access_token(self, validator, tmp_path):
        """Test Schwab validation with expired access token but valid refresh."""
        token_file = tmp_path / "token_file.json"
        token_data = {
            "access_token": "expired_access_token",
            "refresh_token": "valid_refresh_token",
            "access_time": time.time() - 3600,  # 1 hour ago (expired)
            "refresh_time": time.time(),  # Just refreshed
            "expires_in": 1800
        }
        token_file.write_text(json.dumps(token_data))

        with patch.dict(os.environ, {
            'SCHWAB_API_KEY': 'test_key',
            'SCHWAB_SECRET': 'test_secret'
        }):
            with patch.object(validator, '_find_schwab_token_file', return_value=token_file):
                result = await validator.validate_schwab()
                # Should still be usable since refresh token is valid
                assert result.status == CredentialStatus.EXPIRING_SOON
                assert result.can_trade is True

    @pytest.mark.asyncio
    async def test_validate_schwab_expired_refresh_token(self, validator, tmp_path):
        """Test Schwab validation with expired refresh token."""
        token_file = tmp_path / "token_file.json"
        token_data = {
            "access_token": "expired_token",
            "refresh_token": "expired_refresh_token",
            "access_time": time.time() - 3600,  # Expired
            "refresh_time": time.time() - (8 * 24 * 3600),  # 8 days ago (expired)
            "expires_in": 1800
        }
        token_file.write_text(json.dumps(token_data))

        with patch.dict(os.environ, {
            'SCHWAB_API_KEY': 'test_key',
            'SCHWAB_SECRET': 'test_secret'
        }):
            with patch.object(validator, '_find_schwab_token_file', return_value=token_file):
                result = await validator.validate_schwab()
                assert result.status == CredentialStatus.EXPIRED
                assert result.can_trade is False

    @pytest.mark.asyncio
    async def test_validate_all(self, validator):
        """Test validating all brokers."""
        with patch.object(validator, 'validate_schwab', new_callable=AsyncMock) as mock_schwab:
            with patch.object(validator, 'validate_alpaca', new_callable=AsyncMock) as mock_alpaca:
                mock_schwab.return_value = ValidationResult(
                    broker="schwab",
                    status=CredentialStatus.VALID,
                    message="OK",
                    can_trade=True,
                    can_fetch_data=True
                )
                mock_alpaca.return_value = ValidationResult(
                    broker="alpaca",
                    status=CredentialStatus.VALID,
                    message="OK",
                    can_trade=True,
                    can_fetch_data=True
                )

                results = await validator.validate_all()
                assert 'schwab' in results
                assert 'alpaca' in results
                assert results['schwab'].status == CredentialStatus.VALID
                assert results['alpaca'].status == CredentialStatus.VALID

    def test_get_best_data_source_prefers_alpaca(self, validator):
        """Test that Alpaca is preferred for data."""
        results = {
            'alpaca': ValidationResult(
                broker="alpaca", status=CredentialStatus.VALID,
                message="OK", can_fetch_data=True
            ),
            'schwab': ValidationResult(
                broker="schwab", status=CredentialStatus.VALID,
                message="OK", can_fetch_data=True
            )
        }
        assert validator.get_best_data_source(results) == 'alpaca'

    def test_get_best_data_source_falls_back_to_schwab(self, validator):
        """Test fallback to Schwab when Alpaca unavailable."""
        results = {
            'alpaca': ValidationResult(
                broker="alpaca", status=CredentialStatus.MISSING,
                message="Missing", can_fetch_data=False
            ),
            'schwab': ValidationResult(
                broker="schwab", status=CredentialStatus.VALID,
                message="OK", can_fetch_data=True
            )
        }
        assert validator.get_best_data_source(results) == 'schwab'

    def test_get_best_data_source_none_available(self, validator):
        """Test when no data source is available."""
        results = {
            'alpaca': ValidationResult(
                broker="alpaca", status=CredentialStatus.MISSING,
                message="Missing", can_fetch_data=False
            ),
            'schwab': ValidationResult(
                broker="schwab", status=CredentialStatus.EXPIRED,
                message="Expired", can_fetch_data=False
            )
        }
        assert validator.get_best_data_source(results) == 'none'

    def test_get_best_trading_broker(self, validator):
        """Test best trading broker selection."""
        results = {
            'alpaca': ValidationResult(
                broker="alpaca", status=CredentialStatus.VALID,
                message="OK", can_trade=True
            ),
            'schwab': ValidationResult(
                broker="schwab", status=CredentialStatus.VALID,
                message="OK", can_trade=True
            )
        }
        # Should prefer Alpaca
        assert validator.get_best_trading_broker(results) == 'alpaca'


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_check_credentials(self):
        """Test check_credentials function."""
        with patch('core.credential_validator.CredentialValidator') as MockValidator:
            mock_validator = MagicMock()
            mock_validator.validate_all = AsyncMock(return_value={
                'alpaca': ValidationResult(
                    broker="alpaca", status=CredentialStatus.VALID,
                    message="OK"
                )
            })
            MockValidator.return_value = mock_validator

            results = await check_credentials()
            assert 'alpaca' in results

    @pytest.mark.asyncio
    async def test_can_use_schwab(self):
        """Test can_use_schwab function."""
        with patch('core.credential_validator.CredentialValidator') as MockValidator:
            mock_validator = MagicMock()
            mock_validator.validate_schwab = AsyncMock(return_value=ValidationResult(
                broker="schwab", status=CredentialStatus.VALID,
                message="OK", can_fetch_data=True
            ))
            MockValidator.return_value = mock_validator

            result = await can_use_schwab()
            assert result is True

    @pytest.mark.asyncio
    async def test_can_use_alpaca(self):
        """Test can_use_alpaca function."""
        with patch('core.credential_validator.CredentialValidator') as MockValidator:
            mock_validator = MagicMock()
            mock_validator.validate_alpaca = AsyncMock(return_value=ValidationResult(
                broker="alpaca", status=CredentialStatus.VALID,
                message="OK", can_fetch_data=True
            ))
            MockValidator.return_value = mock_validator

            result = await can_use_alpaca()
            assert result is True
