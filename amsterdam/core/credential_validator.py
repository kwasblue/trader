# core/credential_validator.py
"""
Credential Validator - Validates broker credentials and token status

Features:
- Schwab OAuth token validation and expiry checking
- Alpaca API key validation
- Pre-flight checks before trading
- Automatic fallback recommendations

Usage:
    validator = CredentialValidator()

    # Check all credentials
    status = await validator.validate_all()

    # Check specific broker
    schwab_ok = await validator.validate_schwab()
    alpaca_ok = await validator.validate_alpaca()
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loggers.logger import Logger


class CredentialStatus(Enum):
    """Status of credential validation."""

    VALID = "valid"
    EXPIRED = "expired"
    EXPIRING_SOON = "expiring_soon"
    MISSING = "missing"
    INVALID = "invalid"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result of credential validation."""

    broker: str
    status: CredentialStatus
    message: str
    expires_in: int | None = None  # seconds until expiry
    can_trade: bool = False
    can_fetch_data: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class CredentialValidator:
    """
    Validates broker credentials and provides status information.

    Supports:
    - Schwab OAuth tokens (access + refresh)
    - Alpaca API keys

    Logs to: logs/credential_validator.log
    """

    def __init__(self, config_path: str | None = None):
        """
        Initialize credential validator.

        Args:
            config_path: Path to config. If None, uses default.
        """
        self.logger = Logger(
            "credential_validator.log",
            "CredentialValidator",
            propagate=True,  # Also logs to app.log
        ).get_logger()

        # Find project root
        self.root = Path(__file__).resolve().parents[1]

        # Load config for paths
        try:
            from core.config_loader import get_app_path

            self.program_path = str(get_app_path())
            self.logger.debug(f"Config loaded. Program path: {self.program_path}")
        except Exception as e:
            self.program_path = str(self.root)
            self.logger.warning(f"Config load failed, using defaults: {e}")

        self.logger.info(f"CredentialValidator initialized. Root: {self.root}")

    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================

    async def validate_all(self) -> dict[str, ValidationResult]:
        """
        Validate all broker credentials.

        Returns:
            Dict mapping broker name to ValidationResult
        """
        self.logger.info("Starting credential validation for all brokers")
        results = {}

        # Validate Schwab
        self.logger.debug("Validating Schwab credentials...")
        results["schwab"] = await self.validate_schwab()
        self.logger.info(f"Schwab validation: {results['schwab'].status.value} - {results['schwab'].message}")

        # Validate Alpaca
        self.logger.debug("Validating Alpaca credentials...")
        results["alpaca"] = await self.validate_alpaca()
        self.logger.info(f"Alpaca validation: {results['alpaca'].status.value} - {results['alpaca'].message}")

        # Log summary
        schwab_ok = results["schwab"].can_trade
        alpaca_ok = results["alpaca"].can_trade
        self.logger.info(
            f"Validation complete. Schwab: {'OK' if schwab_ok else 'FAIL'}, Alpaca: {'OK' if alpaca_ok else 'FAIL'}"
        )

        return results

    async def validate_schwab(self) -> ValidationResult:
        """
        Validate Schwab OAuth tokens.

        Checks:
        - Token file exists
        - Access token not expired
        - Refresh token not expired
        """
        self.logger.debug("=== Schwab Credential Validation ===")
        try:
            # Check environment variables
            api_key = os.getenv("SCHWAB_API_KEY")
            secret = os.getenv("SCHWAB_SECRET")

            self.logger.debug(f"SCHWAB_API_KEY set: {bool(api_key)}")
            self.logger.debug(f"SCHWAB_SECRET set: {bool(secret)}")

            if not api_key or not secret:
                self.logger.warning("Schwab environment variables not set")
                return ValidationResult(
                    broker="schwab",
                    status=CredentialStatus.MISSING,
                    message="SCHWAB_API_KEY or SCHWAB_SECRET not set in environment",
                    can_trade=False,
                    can_fetch_data=False,
                )

            # Find token file
            token_path = self._find_schwab_token_file()

            if not token_path or not token_path.exists():
                return ValidationResult(
                    broker="schwab",
                    status=CredentialStatus.MISSING,
                    message="Token file not found. Run manual authentication first.",
                    can_trade=False,
                    can_fetch_data=False,
                    details={"searched_paths": self._get_schwab_token_paths()},
                )

            # Read token file
            with open(token_path) as f:
                token_data = json.load(f)

            # Check for errors in token file
            if "error" in token_data:
                return ValidationResult(
                    broker="schwab",
                    status=CredentialStatus.INVALID,
                    message=f"Token file contains error: {token_data['error']}",
                    can_trade=False,
                    can_fetch_data=False,
                )

            # Check access token
            access_status, access_expires = self._check_access_token(token_data)

            # Check refresh token
            refresh_status, refresh_expires = self._check_refresh_token(token_data)

            # Determine overall status
            if refresh_status == CredentialStatus.EXPIRED:
                return ValidationResult(
                    broker="schwab",
                    status=CredentialStatus.EXPIRED,
                    message="Refresh token expired. Manual re-authentication required.",
                    expires_in=0,
                    can_trade=False,
                    can_fetch_data=False,
                    details={
                        "access_token_status": access_status.value,
                        "refresh_token_status": refresh_status.value,
                        "token_path": str(token_path),
                    },
                )

            if access_status == CredentialStatus.EXPIRED:
                # Access expired but refresh is valid - can auto-renew
                return ValidationResult(
                    broker="schwab",
                    status=CredentialStatus.EXPIRING_SOON,
                    message="Access token expired but refresh token valid. Will auto-renew.",
                    expires_in=refresh_expires,
                    can_trade=True,
                    can_fetch_data=True,
                    details={
                        "access_token_status": access_status.value,
                        "refresh_token_status": refresh_status.value,
                        "refresh_expires_in_hours": refresh_expires // 3600 if refresh_expires else None,
                    },
                )

            if refresh_status == CredentialStatus.EXPIRING_SOON:
                return ValidationResult(
                    broker="schwab",
                    status=CredentialStatus.EXPIRING_SOON,
                    message=f"Refresh token expiring in {refresh_expires // 3600} hours. Consider re-authenticating.",
                    expires_in=refresh_expires,
                    can_trade=True,
                    can_fetch_data=True,
                    details={"access_token_status": access_status.value, "refresh_token_status": refresh_status.value},
                )

            return ValidationResult(
                broker="schwab",
                status=CredentialStatus.VALID,
                message="Schwab credentials valid",
                expires_in=refresh_expires,
                can_trade=True,
                can_fetch_data=True,
                details={
                    "access_expires_in_minutes": access_expires // 60 if access_expires else None,
                    "refresh_expires_in_days": refresh_expires // 86400 if refresh_expires else None,
                },
            )

        except json.JSONDecodeError as e:
            return ValidationResult(
                broker="schwab",
                status=CredentialStatus.INVALID,
                message=f"Token file is corrupted: {e}",
                can_trade=False,
                can_fetch_data=False,
            )
        except Exception as e:
            self.logger.exception(f"Error validating Schwab credentials: {e}")
            return ValidationResult(
                broker="schwab",
                status=CredentialStatus.ERROR,
                message=f"Validation error: {e}",
                can_trade=False,
                can_fetch_data=False,
            )

    async def validate_alpaca(self) -> ValidationResult:
        """
        Validate Alpaca API credentials.

        Checks:
        - API key and secret set
        - Can connect to Alpaca API
        """
        try:
            api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID")
            api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET")

            if not api_key or not api_secret:
                return ValidationResult(
                    broker="alpaca",
                    status=CredentialStatus.MISSING,
                    message="ALPACA_API_KEY or ALPACA_SECRET_KEY not set in environment",
                    can_trade=False,
                    can_fetch_data=False,
                )

            # Try to connect to Alpaca
            try:
                from alpaca.data.historical.stock import StockHistoricalDataClient
                from alpaca.trading.client import TradingClient

                # Test trading client
                trading_client = TradingClient(api_key, api_secret, paper=True)
                account = await asyncio.to_thread(trading_client.get_account)

                # Test data client
                StockHistoricalDataClient(api_key, api_secret)

                return ValidationResult(
                    broker="alpaca",
                    status=CredentialStatus.VALID,
                    message="Alpaca credentials valid",
                    can_trade=True,
                    can_fetch_data=True,
                    details={
                        "account_status": getattr(account, "status", "unknown"),
                        "buying_power": float(getattr(account, "buying_power", 0)),
                        "paper_trading": True,
                    },
                )

            except Exception as e:
                error_msg = str(e).lower()
                if "forbidden" in error_msg or "unauthorized" in error_msg:
                    return ValidationResult(
                        broker="alpaca",
                        status=CredentialStatus.INVALID,
                        message="Alpaca API keys are invalid or unauthorized",
                        can_trade=False,
                        can_fetch_data=False,
                    )
                else:
                    return ValidationResult(
                        broker="alpaca",
                        status=CredentialStatus.ERROR,
                        message=f"Failed to connect to Alpaca: {e}",
                        can_trade=False,
                        can_fetch_data=False,
                    )

        except Exception as e:
            self.logger.exception(f"Error validating Alpaca credentials: {e}")
            return ValidationResult(
                broker="alpaca",
                status=CredentialStatus.ERROR,
                message=f"Validation error: {e}",
                can_trade=False,
                can_fetch_data=False,
            )

    def get_best_data_source(self, results: dict[str, ValidationResult]) -> str:
        """
        Determine the best data source based on validation results.

        Prefers Schwab (more historical data) over Alpaca.

        Args:
            results: Validation results from validate_all()

        Returns:
            'alpaca', 'schwab', or 'none'
        """
        alpaca = results.get("alpaca")
        schwab = results.get("schwab")

        # Prefer Schwab (provides 20 years of data vs Alpaca's 9 months)
        if schwab and schwab.can_fetch_data:
            return "schwab"

        if alpaca and alpaca.can_fetch_data:
            return "alpaca"

        return "none"

    def get_best_trading_broker(self, results: dict[str, ValidationResult]) -> str:
        """
        Determine the best broker for trading based on validation results.

        Args:
            results: Validation results from validate_all()

        Returns:
            'alpaca', 'schwab', or 'none'
        """
        alpaca = results.get("alpaca")
        schwab = results.get("schwab")

        # Both can trade - let user choose (prefer Alpaca for simplicity)
        if alpaca and alpaca.can_trade:
            return "alpaca"

        if schwab and schwab.can_trade:
            return "schwab"

        return "none"

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _find_schwab_token_file(self) -> Path | None:
        """Find the Schwab token file."""
        paths = self._get_schwab_token_paths()

        for path in paths:
            if path.exists():
                return path

        return None

    def _get_schwab_token_paths(self) -> list:
        """Get possible paths for Schwab token file."""
        from core.config_loader import get_tokens_path

        paths = []

        # First, check config-specified path (e.g., "tokens/token_file")
        try:
            config_token_path = get_tokens_path()
            paths.append(config_token_path)
            # Also check with .json extension
            paths.append(Path(str(config_token_path) + ".json"))
        except Exception:
            pass

        # Standard locations (with and without .json extension)
        for base in [Path(self.program_path), self.root]:
            paths.append(base / "tokens" / "token_file")
            paths.append(base / "tokens" / "token_file.json")

        # Home directory fallback
        paths.append(Path.home() / ".schwab" / "token_file")
        paths.append(Path.home() / ".schwab" / "token_file.json")

        return paths

    def _check_access_token(self, token_data: dict) -> tuple[CredentialStatus, int | None]:
        """
        Check access token status.

        Returns:
            Tuple of (status, seconds_until_expiry)
        """
        access_time = token_data.get("access_time")
        expires_in = token_data.get("expires_in", 1800)  # Default 30 min

        if not access_time:
            return CredentialStatus.EXPIRED, 0

        expiry_time = access_time + expires_in
        now = time.time()
        remaining = int(expiry_time - now)

        if remaining <= 0:
            return CredentialStatus.EXPIRED, 0
        elif remaining < 300:  # Less than 5 minutes
            return CredentialStatus.EXPIRING_SOON, remaining
        else:
            return CredentialStatus.VALID, remaining

    def _check_refresh_token(self, token_data: dict) -> tuple[CredentialStatus, int | None]:
        """
        Check refresh token status.

        Returns:
            Tuple of (status, seconds_until_expiry)
        """
        refresh_time = token_data.get("refresh_time")
        refresh_interval_days = 7  # Schwab refresh tokens last ~7 days

        if not refresh_time:
            return CredentialStatus.EXPIRED, 0

        expiry_time = refresh_time + (refresh_interval_days * 24 * 60 * 60)
        now = time.time()
        remaining = int(expiry_time - now)

        if remaining <= 0:
            return CredentialStatus.EXPIRED, 0
        elif remaining < (24 * 60 * 60):  # Less than 1 day
            return CredentialStatus.EXPIRING_SOON, remaining
        else:
            return CredentialStatus.VALID, remaining


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def check_credentials() -> dict[str, ValidationResult]:
    """Quick credential check."""
    validator = CredentialValidator()
    return await validator.validate_all()


async def can_use_schwab() -> bool:
    """Check if Schwab is usable."""
    validator = CredentialValidator()
    result = await validator.validate_schwab()
    return result.can_fetch_data


async def can_use_alpaca() -> bool:
    """Check if Alpaca is usable."""
    validator = CredentialValidator()
    result = await validator.validate_alpaca()
    return result.can_fetch_data
