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
import requests
from datetime import datetime
from pathlib import Path

# Load .env from project root before any other imports
from dotenv import load_dotenv
project_root = Path(__file__).resolve().parents[2]  # Go up to amsterdam root
load_dotenv(project_root / ".env")


def setup_logging():
    """Setup basic logging to console and file."""
    from loggers.logger import Logger
    return Logger(
        log_file="token_keeper.log",
        logger_name="TokenKeeper",
        propagate=True
    ).get_logger()


def load_slack_config():
    """Load Slack bot configuration from .env"""
    return {
        'token': os.getenv('SLACK_BOT_TOKEN'),
        'channel': os.getenv('SLACK_CHANNEL_ID')
    }


def send_slack_alert(slack_config, title, message, auth_url=None):
    """Send alert to Slack using bot token"""
    if not slack_config or not slack_config.get('token') or not slack_config.get('channel'):
        return False

    # Build message text
    text = f"🔐 *{title}*\n\n{message}"
    if auth_url:
        text += f"\n\n*Re-authentication Required:*\n<{auth_url}|Click here to authenticate>"

    try:
        response = requests.post(
            'https://slack.com/api/chat.postMessage',
            headers={'Authorization': f'Bearer {slack_config["token"]}'},
            json={
                'channel': slack_config['channel'],
                'text': text,
                'mrkdwn': True
            },
            timeout=10
        )
        result = response.json()
        return result.get('ok', False)
    except Exception as e:
        print(f"Slack notification error: {e}")
        return False


# Track if we've already sent alerts (avoid spam)
_alert_state = {
    'refresh_expiring_sent': False,
    'refresh_expired_sent': False,
    'last_alert_day': None
}


async def token_keeper(interval: int = 60):
    """
    Main token keeper loop.

    Args:
        interval: Seconds between token checks (default: 60)
    """
    from data.streaming.authenticator import Authenticator

    logger = setup_logging()
    auth = Authenticator()
    slack_config = load_slack_config()

    logger.info(f"Token keeper started. Checking every {interval} seconds.")
    if slack_config.get('token') and slack_config.get('channel'):
        logger.info("Slack notifications enabled")
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
            today = datetime.now().date()

            if auth._is_refresh_token_expired(token_data):
                logger.warning("Refresh token expired! Manual re-authentication required.")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] REFRESH TOKEN EXPIRED - run refresh_schwab_token.py --force")

                # Send Slack alert (once per day)
                if slack_config.get('token') and (_alert_state['last_alert_day'] != today or not _alert_state['refresh_expired_sent']):
                    # Generate proper OAuth URL
                    auth_url = "https://developer.schwab.com"  # Fallback URL
                    try:
                        from urllib.parse import urlencode
                        params = {
                            'client_id': auth.apikey,
                            'redirect_uri': auth.redirect_url
                        }
                        auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?{urlencode(params)}"
                    except:
                        pass

                    message = (
                        "*Your Schwab refresh token has expired!*\n\n"
                        "This happens every 7 days for security.\n\n"
                        "*Action Required:*\n"
                        "1. Click the authentication link below\n"
                        "2. Log into your Schwab account\n"
                        "3. Approve the application\n"
                        "4. Copy the redirect URL from your browser\n"
                        "5. *Paste the URL back in this Slack channel*\n"
                        "6. System will auto-process and confirm\n\n"
                        "_The redirect URL will look like: https://127.0.0.1/?code=..._"
                    )

                    if send_slack_alert(slack_config, "Schwab Token Expired - Action Required", message, auth_url):
                        logger.info("Slack alert sent for expired refresh token")
                        _alert_state['refresh_expired_sent'] = True
                        _alert_state['last_alert_day'] = today

                        # Start Slack auth handler in background to monitor for response
                        try:
                            import subprocess
                            handler_script = project_root / "monitoring" / "scripts" / "slack_auth_handler.py"
                            venv_python = project_root / "venv" / "bin" / "python"

                            if handler_script.exists() and venv_python.exists():
                                # Start in background, output to log
                                log_file = project_root / "logs" / "slack_auth.log"
                                # Open file for subprocess (don't use 'with' - it closes before subprocess writes)
                                log_f = open(log_file, 'a')
                                subprocess.Popen(
                                    [str(venv_python), str(handler_script)],
                                    stdout=log_f,
                                    stderr=log_f,
                                    start_new_session=True,  # Detach from parent
                                    cwd=str(project_root)  # Ensure correct working directory
                                )
                                # Don't close log_f - let subprocess inherit it
                                logger.info("Started Slack auth handler to monitor for response")
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Started Slack auth monitor")
                        except Exception as e:
                            logger.warning(f"Could not start Slack auth handler: {e}")

            elif auth._is_refresh_token_expired(token_data, refresh_interval_days=5):
                # Warn 2 days before expiration (6-day default minus 5-day check)
                logger.warning("Refresh token expiring soon. Consider manual refresh.")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Refresh token expiring soon - consider running refresh_schwab_token.py --force")

                # Send Slack warning (once per day)
                if slack_config.get('token') and (_alert_state['last_alert_day'] != today or not _alert_state['refresh_expiring_sent']):
                    message = (
                        "*Schwab refresh token expires in ~2 days*\n\n"
                        "You'll receive another alert when it expires.\n"
                        "Or you can refresh it now using:\n"
                        "`ssh raspi 'cd ~/trader/amsterdam && source venv/bin/activate && python refresh_schwab_token.py'`"
                    )

                    if send_slack_alert(slack_config, "Schwab Token Expiring Soon", message):
                        logger.info("Slack warning sent for expiring refresh token")
                        _alert_state['refresh_expiring_sent'] = True
                        _alert_state['last_alert_day'] = today
            else:
                # Reset alert state when token is fresh
                if _alert_state['last_alert_day'] != today:
                    _alert_state['refresh_expiring_sent'] = False
                    _alert_state['refresh_expired_sent'] = False

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
