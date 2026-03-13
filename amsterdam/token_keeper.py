#!/usr/bin/env python3
"""
Schwab Token Keeper - Background Service

Keeps Schwab tokens fresh by periodically checking and renewing them.
Run this as a background process to prevent token expiration.

Usage:
    python token_keeper.py                  # Run in foreground
    python token_keeper.py --daemon         # Run as background daemon
    python token_keeper.py --interval 300   # Check every 5 minutes (default: 60s)

To run as a launchd service on macOS, see the plist file.
"""

import asyncio
import argparse
import os
import sys
import signal
import time
from datetime import datetime

# Load .env from project root before any other imports
from dotenv import load_dotenv
project_root = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(project_root, ".env"))


def setup_logging():
    """Setup basic logging to console and file."""
    from loggers.logger import Logger
    return Logger(
        log_file="token_keeper.log",
        logger_name="TokenKeeper",
        propagate=True
    ).get_logger()


async def token_keeper(interval: int = 60):
    """
    Main token keeper loop.

    Args:
        interval: Seconds between token checks (default: 60)
    """
    from data.streaming.authenticator import Authenticator

    logger = setup_logging()
    auth = Authenticator()

    logger.info(f"Token keeper started. Checking every {interval} seconds.")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Token keeper started (interval: {interval}s)")

    while True:
        try:
            # Read token data
            token_data = auth._read_token_file()

            if not token_data:
                logger.warning("No token file found. Waiting for manual authentication...")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No token file - run refresh_schwab_token.py first")
                await asyncio.sleep(interval)
                continue

            # Check access token
            if auth._is_access_token_expired(token_data):
                logger.info("Access token expired. Renewing...")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Access token expired - renewing...")

                result = await auth.renew_access()

                if result is True:
                    logger.info("Access token renewed successfully")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Access token renewed successfully")
                else:
                    logger.error(f"Access token renewal failed: {result}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Renewal failed - run refresh_schwab_token.py manually")

            # Check refresh token (warn if getting close to expiration)
            if auth._is_refresh_token_expired(token_data):
                logger.warning("Refresh token expired! Manual re-authentication required.")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] REFRESH TOKEN EXPIRED - run refresh_schwab_token.py --force")
            elif auth._is_refresh_token_expired(token_data, refresh_interval_days=5):
                # Warn 2 days before expiration (6-day default minus 5-day check)
                logger.warning("Refresh token expiring soon. Consider manual refresh.")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Refresh token expiring soon - consider running refresh_schwab_token.py --force")

            # Check for errors in token file
            if auth._contains_error(token_data):
                logger.error("Token file contains errors")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Token file has errors - run refresh_schwab_token.py --force")

            await asyncio.sleep(interval)

        except Exception as e:
            logger.error(f"Error in token keeper: {e}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")
            await asyncio.sleep(interval)


def daemonize():
    """Fork process to run as daemon."""
    # First fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)

    # Second fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    with open('/dev/null', 'r') as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())

    # Keep stdout/stderr for logging
    log_path = os.path.join(os.path.dirname(__file__), "logs", "token_keeper_daemon.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'a') as log:
        os.dup2(log.fileno(), sys.stdout.fileno())
        os.dup2(log.fileno(), sys.stderr.fileno())


def main():
    parser = argparse.ArgumentParser(description="Schwab Token Keeper")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds (default: 60)")
    args = parser.parse_args()

    # Change to script directory for imports
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.daemon:
        print(f"Starting token keeper daemon (interval: {args.interval}s)...")
        daemonize()

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Token keeper stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the keeper
    asyncio.run(token_keeper(interval=args.interval))


if __name__ == "__main__":
    main()
