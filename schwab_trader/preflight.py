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
load_dotenv(ROOT / ".env")

from core.credential_validator import CredentialValidator, CredentialStatus, ValidationResult
from core.unified_data_pipeline import UnifiedDataPipeline
from core.config_loader import get_config
import subprocess


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
            # Start token keeper to maintain token freshness
            self._start_token_keeper()

        elif schwab.status == CredentialStatus.EXPIRING_SOON:
            self.print_warning("Schwab", schwab.message)
            self.warnings.append(f"Schwab token expiring soon")
            # Still start token keeper - it will handle renewal
            self._start_token_keeper()

            if reauth_schwab:
                print("\n  Initiating Schwab re-authentication...")
                await self._reauth_schwab()

        elif schwab.status == CredentialStatus.EXPIRED:
            self.print_status("Schwab", False, schwab.message)
            self.warnings.append("Schwab token expired (use --reauth-schwab)")

            if reauth_schwab:
                print("\n  Initiating Schwab re-authentication...")
                await self._reauth_schwab()
                # Start token keeper after successful re-auth
                self._start_token_keeper()

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
        """Check historical data freshness using UnifiedDataPipeline."""
        self.print_header("HISTORICAL DATA")

        pipeline = UnifiedDataPipeline()
        stale_symbols = []

        for symbol in symbols:
            # Check both raw and processed data status
            raw_ts = pipeline._get_last_raw_timestamp(symbol)
            proc_ts = pipeline._get_last_processed_timestamp(symbol)
            is_current = pipeline._is_processed_data_current(symbol)

            if raw_ts is None and proc_ts is None:
                self.print_status(symbol, False, "No data files found")
                stale_symbols.append(symbol)
                self.warnings.append(f"No data for {symbol}")
            elif not is_current:
                # Processed data is behind raw data
                self.print_warning(symbol, "Processed data needs update")
                stale_symbols.append(symbol)
                self.warnings.append(f"{symbol} needs reprocessing")
            else:
                # Check age of data
                from datetime import datetime, timezone
                if proc_ts:
                    age_hours = (datetime.now(timezone.utc).timestamp() * 1000 - proc_ts) / 3600000
                    # Get bar count from file
                    df = pipeline.get_data_from_file(symbol)
                    bar_count = len(df) if not df.empty else 0

                    if age_hours > 24:  # Consider stale if > 24 hours old
                        self.print_warning(
                            symbol,
                            f"Data may be stale ({bar_count} bars, {age_hours:.0f}h old)"
                        )
                        stale_symbols.append(symbol)
                        self.warnings.append(f"{symbol} data may be stale")
                    else:
                        self.print_status(
                            symbol,
                            True,
                            f"Fresh ({bar_count} bars, {age_hours:.1f}h old)"
                        )
                        self.passed.append(f"Data: {symbol}")
                else:
                    self.print_status(symbol, True, "Raw data available")
                    self.passed.append(f"Data: {symbol}")

        if stale_symbols and update_data:
            print(f"\n  Updating data for: {', '.join(stale_symbols)}")
            results = await pipeline.update_symbols(stale_symbols, days=30)

            for symbol, count in results.items():
                if count > 0:
                    print(f"    ✓ {symbol}: {count} bars fetched/processed")
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

    def _is_token_keeper_running(self) -> bool:
        """Check if token keeper is already running."""
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'token_keeper.py'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def _start_token_keeper(self) -> bool:
        """Start the token keeper as a background daemon."""
        if self._is_token_keeper_running():
            self.print_status("Token Keeper", True, "Already running")
            self.logger.info("Token keeper already running")
            return True

        try:
            # Start token keeper as background daemon
            subprocess.Popen(
                [sys.executable, str(ROOT / "token_keeper.py"), "--daemon", "--interval", "60"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            self.print_status("Token Keeper", True, "Started background daemon")
            self.logger.info("Token keeper daemon started")
            self.passed.append("Token keeper started")
            return True
        except Exception as e:
            self.print_warning("Token Keeper", f"Failed to start: {e}")
            self.logger.error(f"Failed to start token keeper: {e}")
            self.warnings.append("Token keeper failed to start")
            return False

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
    # Load config to get default symbols
    try:
        config = get_config()
        default_symbols = config.general.default_symbols
    except Exception:
        default_symbols = ['AAPL', 'MSFT']

    parser = argparse.ArgumentParser(
        description='Pre-flight system check for trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python preflight.py                        # Check symbols from config: {default_symbols}
  python preflight.py --symbols AAPL TSLA    # Check specific symbols
  python preflight.py --update-data          # Update stale data
  python preflight.py --reauth-schwab        # Re-authenticate Schwab
  python preflight.py -v --update-data       # Verbose with data update
        """
    )

    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=default_symbols,
        help=f'Symbols to check (default from config: {default_symbols})'
    )
    parser.add_argument(
        '--all-data',
        action='store_true',
        help='Check all symbols that have data files (ignores --symbols)'
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

    # If --all-data, get symbols from existing data files
    symbols = args.symbols
    if args.all_data:
        pipeline = UnifiedDataPipeline()
        symbols = pipeline.list_available_symbols('file')
        if not symbols:
            print("No data files found in proc_data/")
            sys.exit(1)
        print(f"Checking all {len(symbols)} symbols with data: {', '.join(symbols)}")

    checker = PreFlightChecker(verbose=args.verbose)
    success = await checker.run_all_checks(
        symbols=symbols,
        update_data=args.update_data,
        reauth_schwab=args.reauth_schwab,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())
