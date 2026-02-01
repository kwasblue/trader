"""
Tests for PreFlightChecker

Coverage:
- Environment variable validation
- Credential checking
- Data freshness validation
- Configuration verification
- Summary generation
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preflight import PreFlightChecker


class TestPreFlightChecker:
    """Tests for PreFlightChecker."""

    @pytest.fixture
    def checker(self):
        with patch('loggers.logger.Logger'):
            return PreFlightChecker(verbose=False)

    @pytest.fixture
    def verbose_checker(self):
        with patch('loggers.logger.Logger'):
            return PreFlightChecker(verbose=True)

    def test_init(self, checker):
        """Test PreFlightChecker initialization."""
        assert checker.verbose is False
        assert checker.issues == []
        assert checker.warnings == []
        assert checker.passed == []

    def test_init_verbose(self, verbose_checker):
        """Test verbose mode initialization."""
        assert verbose_checker.verbose is True


class TestEnvironmentChecks:
    """Tests for environment variable checking."""

    @pytest.fixture
    def checker(self):
        with patch('loggers.logger.Logger'):
            return PreFlightChecker(verbose=False)

    def test_check_environment_all_set(self, checker):
        """Test environment check with all variables set."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_alpaca_key',
            'ALPACA_SECRET_KEY': 'test_alpaca_secret',
            'SCHWAB_API_KEY': 'test_schwab_key',
            'SCHWAB_SECRET': 'test_schwab_secret'
        }):
            with patch.object(checker, 'print_header'):
                with patch.object(checker, 'print_status'):
                    checker._check_environment()

                    # Should have passed checks for required vars
                    assert len(checker.issues) == 0

    def test_check_environment_missing_alpaca(self, checker):
        """Test environment check with missing Alpaca credentials."""
        with patch.dict(os.environ, {
            'SCHWAB_API_KEY': 'test_schwab_key',
            'SCHWAB_SECRET': 'test_schwab_secret'
        }, clear=True):
            with patch.object(checker, 'print_header'):
                with patch.object(checker, 'print_status'):
                    checker._check_environment()

                    # Should have issues for missing Alpaca creds
                    assert any('ALPACA' in issue for issue in checker.issues)

    def test_check_environment_missing_schwab_is_warning(self, checker):
        """Test that missing Schwab credentials are warnings, not errors."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }, clear=True):
            with patch.object(checker, 'print_header'):
                with patch.object(checker, 'print_status'):
                    with patch.object(checker, 'print_warning'):
                        checker._check_environment()

                        # Schwab missing should be warning, not issue
                        assert any('SCHWAB' in warn for warn in checker.warnings)


class TestCredentialChecks:
    """Tests for credential validation."""

    @pytest.fixture
    def checker(self):
        with patch('loggers.logger.Logger'):
            return PreFlightChecker(verbose=False)

    @pytest.mark.asyncio
    async def test_check_credentials_all_valid(self, checker):
        """Test credential check with all valid credentials."""
        from core.credential_validator import ValidationResult, CredentialStatus

        with patch.object(checker.validator, 'validate_all', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = {
                'alpaca': ValidationResult(
                    broker="alpaca", status=CredentialStatus.VALID,
                    message="Valid", can_trade=True, can_fetch_data=True
                ),
                'schwab': ValidationResult(
                    broker="schwab", status=CredentialStatus.VALID,
                    message="Valid", can_trade=True, can_fetch_data=True
                )
            }

            with patch.object(checker.validator, 'get_best_data_source', return_value='alpaca'):
                with patch.object(checker.validator, 'get_best_trading_broker', return_value='alpaca'):
                    with patch.object(checker, 'print_header'):
                        with patch.object(checker, 'print_status'):
                            await checker._check_credentials()

                            assert 'Alpaca credentials' in checker.passed

    @pytest.mark.asyncio
    async def test_check_credentials_alpaca_invalid(self, checker):
        """Test credential check with invalid Alpaca credentials."""
        from core.credential_validator import ValidationResult, CredentialStatus

        with patch.object(checker.validator, 'validate_all', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = {
                'alpaca': ValidationResult(
                    broker="alpaca", status=CredentialStatus.INVALID,
                    message="Invalid API key", can_trade=False, can_fetch_data=False
                ),
                'schwab': ValidationResult(
                    broker="schwab", status=CredentialStatus.MISSING,
                    message="Missing", can_trade=False, can_fetch_data=False
                )
            }

            with patch.object(checker.validator, 'get_best_data_source', return_value='none'):
                with patch.object(checker.validator, 'get_best_trading_broker', return_value='none'):
                    with patch.object(checker, 'print_header'):
                        with patch.object(checker, 'print_status'):
                            await checker._check_credentials()

                            assert any('Alpaca' in issue for issue in checker.issues)

    @pytest.mark.asyncio
    async def test_check_credentials_schwab_expiring(self, checker):
        """Test credential check with expiring Schwab token."""
        from core.credential_validator import ValidationResult, CredentialStatus

        with patch.object(checker.validator, 'validate_all', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = {
                'alpaca': ValidationResult(
                    broker="alpaca", status=CredentialStatus.VALID,
                    message="Valid", can_trade=True, can_fetch_data=True
                ),
                'schwab': ValidationResult(
                    broker="schwab", status=CredentialStatus.EXPIRING_SOON,
                    message="Token expiring in 12 hours", can_trade=True, can_fetch_data=True
                )
            }

            with patch.object(checker.validator, 'get_best_data_source', return_value='alpaca'):
                with patch.object(checker.validator, 'get_best_trading_broker', return_value='alpaca'):
                    with patch.object(checker, 'print_header'):
                        with patch.object(checker, 'print_status'):
                            with patch.object(checker, 'print_warning'):
                                await checker._check_credentials(reauth_schwab=False)

                                assert any('expiring' in warn.lower() for warn in checker.warnings)


class TestDataFreshnessChecks:
    """Tests for data freshness validation."""

    @pytest.fixture
    def checker(self):
        with patch('loggers.logger.Logger'):
            return PreFlightChecker(verbose=False)

    @pytest.mark.asyncio
    async def test_check_data_freshness_fresh(self, checker):
        """Test data freshness check with fresh data."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            with patch('preflight.HistoricalDataUpdater') as MockUpdater:
                mock_updater = MagicMock()
                mock_updater.get_data_freshness.return_value = {
                    'is_stale': False,
                    'bar_count': 1000,
                    'age_minutes': 30
                }
                MockUpdater.return_value = mock_updater

                with patch.object(checker, 'print_header'):
                    with patch.object(checker, 'print_status'):
                        await checker._check_data_freshness(['AAPL'], update_data=False)

                        assert 'Data: AAPL' in checker.passed

    @pytest.mark.asyncio
    async def test_check_data_freshness_stale(self, checker):
        """Test data freshness check with stale data."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            with patch('preflight.HistoricalDataUpdater') as MockUpdater:
                mock_updater = MagicMock()
                mock_updater.get_data_freshness.return_value = {
                    'is_stale': True,
                    'bar_count': 500,
                    'age_minutes': 120
                }
                MockUpdater.return_value = mock_updater

                with patch.object(checker, 'print_header'):
                    with patch.object(checker, 'print_warning'):
                        await checker._check_data_freshness(['AAPL'], update_data=False)

                        assert any('stale' in warn.lower() for warn in checker.warnings)

    @pytest.mark.asyncio
    async def test_check_data_freshness_missing(self, checker):
        """Test data freshness check with missing data."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            with patch('preflight.HistoricalDataUpdater') as MockUpdater:
                mock_updater = MagicMock()
                mock_updater.get_data_freshness.return_value = None
                MockUpdater.return_value = mock_updater

                with patch.object(checker, 'print_header'):
                    with patch.object(checker, 'print_status'):
                        await checker._check_data_freshness(['AAPL'], update_data=False)

                        assert any('No data' in warn for warn in checker.warnings)


class TestConfigurationChecks:
    """Tests for configuration validation."""

    @pytest.fixture
    def checker(self):
        with patch('loggers.logger.Logger'):
            return PreFlightChecker(verbose=False)

    def test_check_configuration_all_present(self, checker, tmp_path):
        """Test configuration check with all files present."""
        # Create config files
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "trading_config.json").write_text("{}")
        (config_dir / "strategy_routing.json").write_text("{}")

        # Create data dir
        data_dir = tmp_path / "data" / "data_storage" / "proc_data"
        data_dir.mkdir(parents=True)
        (data_dir / "proc_AAPL_file.json").write_text("{}")

        with patch('preflight.ROOT', tmp_path):
            with patch.object(checker, 'print_header'):
                with patch.object(checker, 'print_status'):
                    checker._check_configuration()

                    assert 'Config file' in checker.passed

    def test_check_configuration_missing_config(self, checker, tmp_path):
        """Test configuration check with missing config file."""
        with patch('preflight.ROOT', tmp_path):
            with patch.object(checker, 'print_header'):
                with patch.object(checker, 'print_warning'):
                    with patch.object(checker, 'print_status'):
                        checker._check_configuration()

                        assert any('Config' in warn or 'config' in warn for warn in checker.warnings)


class TestSummary:
    """Tests for summary generation."""

    @pytest.fixture
    def checker(self):
        with patch('loggers.logger.Logger'):
            return PreFlightChecker(verbose=False)

    def test_print_summary_all_passed(self, checker):
        """Test summary with all checks passed."""
        checker.passed = ['Check 1', 'Check 2', 'Check 3']
        checker.warnings = []
        checker.issues = []

        with patch.object(checker, 'print_header'):
            result = checker._print_summary()
            assert result is True

    def test_print_summary_with_issues(self, checker):
        """Test summary with critical issues."""
        checker.passed = ['Check 1']
        checker.warnings = ['Warning 1']
        checker.issues = ['Critical Issue']

        with patch.object(checker, 'print_header'):
            result = checker._print_summary()
            assert result is False

    def test_print_summary_warnings_only(self, checker):
        """Test summary with warnings but no issues."""
        checker.passed = ['Check 1']
        checker.warnings = ['Warning 1', 'Warning 2']
        checker.issues = []

        with patch.object(checker, 'print_header'):
            result = checker._print_summary()
            # Should still pass - warnings don't block trading
            assert result is True


class TestRunAllChecks:
    """Tests for the complete preflight check flow."""

    @pytest.fixture
    def checker(self):
        with patch('loggers.logger.Logger'):
            return PreFlightChecker(verbose=False)

    @pytest.mark.asyncio
    async def test_run_all_checks_success(self, checker):
        """Test complete preflight check with all passing."""
        with patch.object(checker, '_check_environment'):
            with patch.object(checker, '_check_credentials', new_callable=AsyncMock):
                with patch.object(checker, '_check_data_freshness', new_callable=AsyncMock):
                    with patch.object(checker, '_check_configuration'):
                        with patch.object(checker, '_print_summary', return_value=True):
                            with patch.object(checker, 'print_header'):
                                result = await checker.run_all_checks(
                                    symbols=['AAPL'],
                                    update_data=False,
                                    reauth_schwab=False
                                )
                                assert result is True

    @pytest.mark.asyncio
    async def test_run_all_checks_failure(self, checker):
        """Test complete preflight check with failures."""
        checker.issues = ['Critical failure']

        with patch.object(checker, '_check_environment'):
            with patch.object(checker, '_check_credentials', new_callable=AsyncMock):
                with patch.object(checker, '_check_data_freshness', new_callable=AsyncMock):
                    with patch.object(checker, '_check_configuration'):
                        with patch.object(checker, '_print_summary', return_value=False):
                            with patch.object(checker, 'print_header'):
                                result = await checker.run_all_checks(
                                    symbols=['AAPL'],
                                    update_data=False,
                                    reauth_schwab=False
                                )
                                assert result is False
