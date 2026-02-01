#!/usr/bin/env python
# preflight.py
"""
Pre-Flight Check - Validates system readiness before trading

Checks:
1. Broker credentials (Alpaca, Schwab)
2. Token expiry status
3. Historical data freshness
4. System configuration
5. Network connectivity

Usage:
    # Quick check
    python preflight.py

    # Update data if needed
    python preflight.py --update-data

    # Force Schwab re-authentication
    python preflight.py --reauth-schwab

    # Full check with verbose output
    python preflight.py -v --update-data
"""

from __future__ import annotations

import os
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".venv" / ".env")
load_dotenv()

from core.credential_validator import CredentialValidator, CredentialStatus, ValidationResult
from core.unified_data_pipeline import UnifiedDataPipeline
from core.historical_data_updater import HistoricalDataUpdater


class PreFlightChecker:
    """
    Comprehensive pre-flight system checker.

    Logs to: logs/preflight.log
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.validator = CredentialValidator()
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []

        # Setup file logging
        from loggers.logger import Logger
        self.logger = Logger(
            "preflight.log",
            "PreFlightChecker",
            propagate=True,  # Also logs to app.log
        ).get_logger()

        self.logger.info("=" * 60)
        self.logger.info("PRE-FLIGHT CHECKER INITIALIZED")
        self.logger.info("=" * 60)

    def print_header(self, text: str) -> None:
        """Print a section header."""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}")

    def print_status(self, name: str, passed: bool, message: str) -> None:
        """Print a status line and log it."""
        icon = "✓" if passed else "✗"
        color_start = "\033[92m" if passed else "\033[91m"
        color_end = "\033[0m"
        print(f"  {color_start}{icon}{color_end} {name}: {message}")

        # Log to file
        if passed:
            self.logger.info(f"[PASS] {name}: {message}")
        else:
            self.logger.error(f"[FAIL] {name}: {message}")

    def print_warning(self, name: str, message: str) -> None:
        """Print a warning line and log it."""
        print(f"  \033[93m⚠\033[0m {name}: {message}")
        self.logger.warning(f"[WARN] {name}: {message}")

    async def run_all_checks(
        self,
        symbols: List[str],
        update_data: bool = False,
        reauth_schwab: bool = False,
    ) -> bool:
        """
        Run all pre-flight checks.

        Returns:
            True if all critical checks pass
        """
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

        self.logger.info(f"Starting pre-flight checks at {timestamp}")
        self.logger.info(f"Symbols: {symbols}")
        self.logger.info(f"Options: update_data={update_data}, reauth_schwab={reauth_schwab}")

        self.print_header("PRE-FLIGHT SYSTEM CHECK")
        print(f"  Timestamp: {timestamp}")
        print(f"  Symbols: {', '.join(symbols)}")

        # 1. Check environment
        self._check_environment()

        # 2. Check credentials
        await self._check_credentials(reauth_schwab)

        # 3. Check data freshness
        await self._check_data_freshness(symbols, update_data)

        # 4. Check configuration
        self._check_configuration()

        # 5. Print summary
        return self._print_summary()

    def _check_environment(self) -> None:
        """Check environment variables."""
        self.print_header("ENVIRONMENT")

        required_vars = {
            'ALPACA_API_KEY': ['ALPACA_API_KEY', 'ALPACA_KEY_ID'],
            'ALPACA_SECRET_KEY': ['ALPACA_SECRET_KEY', 'ALPACA_SECRET'],
            'SCHWAB_API_KEY': ['SCHWAB_API_KEY'],
            'SCHWAB_SECRET': ['SCHWAB_SECRET'],
        }

        for name, variants in required_vars.items():
            value = None
            for var in variants:
                value = os.getenv(var)
                if value:
                    break

            if value:
                masked = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '****'
                self.print_status(name, True, f"Set ({masked})")
                self.passed.append(f"ENV: {name}")
            else:
                if 'SCHWAB' in name:
                    self.print_warning(name, "Not set (Schwab features disabled)")
                    self.warnings.append(f"ENV: {name} not set")
                else:
                    self.print_status(name, False, "Not set")
                    self.issues.append(f"ENV: {name} not set")

    async def _check_credentials(self, reauth_schwab: bool = False) -> None:
        """Check broker credentials."""
        self.print_header("BROKER CREDENTIALS")

        results = await self.validator.validate_all()

        # Alpaca
        alpaca = results['alpaca']
        if alpaca.status == CredentialStatus.VALID:
            self.print_status("Alpaca", True, alpaca.message)
            if self.verbose and alpaca.details:
                print(f"      Account: {alpaca.details.get('account_status', 'N/A')}")
                print(f"      Buying Power: ${alpaca.details.get('buying_power', 0):,.2f}")
            self.passed.append("Alpaca credentials")
        elif alpaca.status == CredentialStatus.MISSING:
            self.print_status("Alpaca", False, alpaca.message)
            self.issues.append("Alpaca credentials missing")
        else:
            self.print_status("Alpaca", False, alpaca.message)
            self.issues.append(f"Alpaca: {alpaca.message}")

        # Schwab
        schwab = results['schwab']
        if schwab.status == CredentialStatus.VALID:
            self.print_status("Schwab", True, schwab.message)
            if self.verbose and schwab.details:
                print(f"      Access expires in: {schwab.details.get('access_expires_in_minutes', '?')} min")
                print(f"      Refresh expires in: {schwab.details.get('refresh_expires_in_days', '?')} days")
            self.passed.append("Schwab credentials")

        elif schwab.status == CredentialStatus.EXPIRING_SOON:
            self.print_warning("Schwab", schwab.message)
            self.warnings.append(f"Schwab token expiring soon")

            if reauth_schwab:
                print("\n  Initiating Schwab re-authentication...")
                await self._reauth_schwab()

        elif schwab.status == CredentialStatus.EXPIRED:
            self.print_status("Schwab", False, schwab.message)
            self.warnings.append("Schwab token expired (use --reauth-schwab)")

            if reauth_schwab:
                print("\n  Initiating Schwab re-authentication...")
                await self._reauth_schwab()

        elif schwab.status == CredentialStatus.MISSING:
            self.print_warning("Schwab", schwab.message)
            self.warnings.append("Schwab credentials not configured")
        else:
            self.print_status("Schwab", False, schwab.message)
            self.warnings.append(f"Schwab: {schwab.message}")

        # Recommendation
        best_data = self.validator.get_best_data_source(results)
        best_trading = self.validator.get_best_trading_broker(results)
        print(f"\n  Recommended data source: {best_data.upper()}")
        print(f"  Recommended trading broker: {best_trading.upper()}")

    async def _check_data_freshness(
        self,
        symbols: List[str],
        update_data: bool = False
    ) -> None:
        """Check historical data freshness."""
        self.print_header("HISTORICAL DATA")

        api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_KEY_ID')
        api_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_SECRET')

        if not api_key or not api_secret:
            self.print_warning("Data Check", "Cannot check - Alpaca credentials missing")
            return

        updater = HistoricalDataUpdater(api_key, api_secret)
        stale_symbols = []

        for symbol in symbols:
            freshness = updater.get_data_freshness(symbol)

            if freshness is None:
                self.print_status(symbol, False, "No data file found")
                stale_symbols.append(symbol)
                self.warnings.append(f"No data for {symbol}")
            elif freshness['is_stale']:
                age_hours = freshness['age_minutes'] // 60
                self.print_warning(
                    symbol,
                    f"Stale ({freshness['bar_count']} bars, {age_hours}h old)"
                )
                stale_symbols.append(symbol)
                self.warnings.append(f"{symbol} data is stale")
            else:
                self.print_status(
                    symbol,
                    True,
                    f"Fresh ({freshness['bar_count']} bars, {freshness['age_minutes']} min old)"
                )
                self.passed.append(f"Data: {symbol}")

        if stale_symbols and update_data:
            print(f"\n  Updating data for: {', '.join(stale_symbols)}")
            pipeline = UnifiedDataPipeline()
            results = await pipeline.update_symbols(stale_symbols, days=30)

            for symbol, count in results.items():
                if count > 0:
                    print(f"    ✓ {symbol}: {count} bars fetched")
                else:
                    print(f"    ✗ {symbol}: Update failed")

    def _check_configuration(self) -> None:
        """Check system configuration."""
        self.print_header("CONFIGURATION")

        # Check config file
        config_path = ROOT / "config" / "trading_config.json"
        if config_path.exists():
            self.print_status("trading_config.json", True, "Found")
            self.passed.append("Config file")
        else:
            self.print_warning("trading_config.json", "Not found (using defaults)")
            self.warnings.append("Config file missing")

        # Check strategy routing
        routing_path = ROOT / "config" / "strategy_routing.json"
        if routing_path.exists():
            self.print_status("strategy_routing.json", True, "Found")
            self.passed.append("Strategy routing")
        else:
            self.print_warning("strategy_routing.json", "Not found")
            self.warnings.append("Strategy routing missing")

        # Check data directories
        proc_data = ROOT / "data" / "data_storage" / "proc_data"
        if proc_data.exists():
            file_count = len(list(proc_data.glob("proc_*_file.json")))
            self.print_status("Processed data dir", True, f"Found ({file_count} files)")
            self.passed.append("Data directory")
        else:
            self.print_status("Processed data dir", False, "Not found")
            self.issues.append("Data directory missing")

    async def _reauth_schwab(self) -> None:
        """Trigger Schwab re-authentication."""
        try:
            from data.streaming.authenticator import Authenticator
            auth = Authenticator()
            result = await auth.manual_refresh_token(use_gui=False)

            if result is True:
                print("    ✓ Schwab re-authentication successful")
            else:
                print(f"    ✗ Schwab re-authentication failed: {result}")

        except Exception as e:
            print(f"    ✗ Schwab re-authentication error: {e}")

    def _print_summary(self) -> bool:
        """Print summary and return success status."""
        self.print_header("SUMMARY")

        # Counts
        print(f"  Passed:   {len(self.passed)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Issues:   {len(self.issues)}")

        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("PRE-FLIGHT SUMMARY")
        self.logger.info(f"Passed: {len(self.passed)}, Warnings: {len(self.warnings)}, Issues: {len(self.issues)}")

        # Details
        if self.verbose and self.passed:
            print("\n  Passed checks:")
            for item in self.passed:
                print(f"    ✓ {item}")

        if self.warnings:
            print("\n  Warnings:")
            for item in self.warnings:
                print(f"    ⚠ {item}")
                self.logger.warning(f"Summary warning: {item}")

        if self.issues:
            print("\n  Critical Issues:")
            for item in self.issues:
                print(f"    ✗ {item}")
                self.logger.error(f"Summary issue: {item}")

        # Final status
        print()
        if not self.issues:
            print("  \033[92m✓ SYSTEM READY FOR TRADING\033[0m")
            self.logger.info("RESULT: SYSTEM READY FOR TRADING")
            self.logger.info("=" * 60)
            return True
        else:
            print("  \033[91m✗ SYSTEM NOT READY - FIX ISSUES ABOVE\033[0m")
            self.logger.error("RESULT: SYSTEM NOT READY - FIX ISSUES")
            self.logger.info("=" * 60)
            return False


async def main():
    parser = argparse.ArgumentParser(
        description='Pre-flight system check for trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preflight.py                        # Quick check
  python preflight.py --symbols AAPL TSLA    # Check specific symbols
  python preflight.py --update-data          # Update stale data
  python preflight.py --reauth-schwab        # Re-authenticate Schwab
  python preflight.py -v --update-data       # Verbose with data update
        """
    )

    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=['AAPL', 'MSFT'],
        help='Symbols to check (default: AAPL MSFT)'
    )
    parser.add_argument(
        '--update-data', '-u',
        action='store_true',
        help='Update stale historical data'
    )
    parser.add_argument(
        '--reauth-schwab',
        action='store_true',
        help='Trigger Schwab re-authentication'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    checker = PreFlightChecker(verbose=args.verbose)
    success = await checker.run_all_checks(
        symbols=args.symbols,
        update_data=args.update_data,
        reauth_schwab=args.reauth_schwab,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())
