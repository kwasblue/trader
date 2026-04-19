#!/usr/bin/env python3
"""
Schwab Token Refresh with Slack Alerts

Automatically refreshes Schwab tokens and sends Slack notification
when manual re-authentication is required.

Usage:
    python tools/refresh_schwab_token_with_alert.py

Cron setup (daily at 2 AM):
    0 2 * * * cd /home/kwasi/trader/amsterdam && /home/kwasi/trader/amsterdam/venv/bin/python tools/refresh_schwab_token_with_alert.py >> logs/token_refresh.log 2>&1
"""

import sys
from datetime import datetime
from pathlib import Path

import requests

# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.streaming.authenticator import Authenticator


def load_slack_webhook():
    """Load Slack webhook URL from .env"""
    env_file = ROOT / ".env"
    if not env_file.exists():
        print(f"Warning: .env file not found at {env_file}")
        return None

    with open(env_file) as f:
        for line in f:
            if line.startswith("SLACK_WEBHOOK_URL="):
                url = line.split("=", 1)[1].strip()
                # Remove quotes if present
                return url.strip('"').strip("'")
    return None


def send_slack_alert(webhook_url, title, message, auth_url=None):
    """Send alert to Slack"""
    if not webhook_url:
        print("Warning: No Slack webhook URL configured")
        return False

    payload = {
        "text": f"🔐 *{title}*",
        "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": f"🔐 *{title}*\n\n{message}"}}],
    }

    if auth_url:
        payload["blocks"].append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Re-authentication Required:*\n<{auth_url}|Click here to authenticate>",
                },
            }
        )

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✓ Slack notification sent")
            return True
        else:
            print(f"✗ Slack notification failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Slack notification error: {e}")
        return False


def get_auth_url():
    """Generate Schwab authentication URL"""
    try:
        auth = Authenticator()
        # The auth URL format - adjust based on your Schwab app setup
        redirect_uri = getattr(auth, "redirect_url", "https://127.0.0.1")
        app_key = getattr(auth, "apikey", "YOUR_APP_KEY")

        auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={app_key}&redirect_uri={redirect_uri}"
        return auth_url
    except:
        return "https://developer.schwab.com"


def main():
    print("=" * 70)
    print(f"  SCHWAB TOKEN REFRESH - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    webhook_url = load_slack_webhook()
    if webhook_url:
        print("✓ Slack webhook configured")
    else:
        print("⚠ Slack webhook not configured - notifications disabled")

    try:
        auth = Authenticator()

        # Check if refresh token is expired
        token_file = auth._read_token_file()
        is_expired = auth._is_refresh_token_expired(token_file)

        if is_expired:
            print("⚠ Refresh token is expired - manual re-authentication required")
            print()

            auth_url = get_auth_url()

            message = (
                "Your Schwab refresh token has expired and requires manual re-authentication.\n\n"
                "This happens every 7 days for security.\n\n"
                "Steps:\n"
                "1. Click the authentication link\n"
                "2. Log into your Schwab account\n"
                "3. Approve the application\n"
                "4. The system will automatically resume trading"
            )

            print(message)
            print()
            print(f"Auth URL: {auth_url}")

            # Send Slack notification
            if webhook_url:
                send_slack_alert(webhook_url, "Schwab Token Expired - Action Required", message, auth_url)

            return 1  # Exit code 1 = manual action needed

        # Attempt automatic refresh
        print("Attempting to refresh access token...")

        # The renew_access method refreshes the access token
        import asyncio

        success = asyncio.run(auth.renew_access())

        if success:
            print("✓ Token refreshed successfully!")
            print()

            # Send success notification (optional, maybe only on first success after failure)
            if webhook_url:
                token_file = auth._read_token_file()
                expires_in = token_file.get("expires_in", 1800) // 60  # Convert to minutes

                send_slack_alert(
                    webhook_url,
                    "Schwab Token Refreshed",
                    f"Access token refreshed successfully.\n"
                    f"Valid for: {expires_in} minutes\n"
                    f"Auto-refreshes daily at 2 AM",
                )

            return 0  # Success
        else:
            print("✗ Token refresh failed")
            print()

            auth_url = get_auth_url()
            message = "Automatic token refresh failed.\n\nManual re-authentication may be required."

            if webhook_url:
                send_slack_alert(webhook_url, "Schwab Token Refresh Failed", message, auth_url)

            return 1

    except Exception as e:
        print(f"✗ Error: {e}")
        print()

        if webhook_url:
            send_slack_alert(
                webhook_url, "Schwab Token Refresh Error", f"An error occurred during token refresh:\n\n```{str(e)}```"
            )

        return 2  # Exit code 2 = error


if __name__ == "__main__":
    sys.exit(main())
