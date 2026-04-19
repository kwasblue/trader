#!/usr/bin/env python3
"""
Slack Auth Handler - Processes Schwab auth redirects from Slack

Monitors Slack channel for auth redirect URLs and automatically
processes them to refresh Schwab tokens.

Usage:
    Started automatically by token_keeper when refresh token expires.
    Monitors Slack every 10 seconds until auth URL is received (or timeout).

User flow:
    1. Token expires → token_keeper sends Slack alert + starts this script
    2. User clicks link, logs into Schwab
    3. User copies redirect URL and pastes in Slack
    4. This script detects it, processes, and exits
    5. Sends confirmation back to Slack
"""

import os
import sys
import re
import json
import time
import asyncio
import requests
from pathlib import Path
from datetime import datetime, timedelta

# Project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from data.streaming.authenticator import Authenticator


# State file to track processed messages
STATE_FILE = ROOT / "logs" / "slack_auth_state.json"


def load_state():
    """Load processing state"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except:
            pass
    return {
        'last_checked_ts': 0,
        'processed_urls': []
    }


def save_state(state):
    """Save processing state"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def get_slack_token():
    """Get Slack bot token from environment"""
    return os.getenv('SLACK_BOT_TOKEN')


def get_slack_channel():
    """Get Slack channel ID from environment"""
    return os.getenv('SLACK_CHANNEL_ID') or os.getenv('SLACK_DEFAULT_CHANNEL')


def send_slack_message(token, channel, message):
    """Send message to Slack channel"""
    if not token or not channel:
        print("⚠️  Slack token or channel not configured")
        return False

    try:
        response = requests.post(
            'https://slack.com/api/chat.postMessage',
            headers={'Authorization': f'Bearer {token}'},
            json={
                'channel': channel,
                'text': message,
                'mrkdwn': True
            },
            timeout=10
        )

        result = response.json()
        return result.get('ok', False)

    except Exception as e:
        print(f"Error sending Slack message: {e}")
        return False


def get_recent_messages(token, channel, since_ts=0):
    """Get recent messages from Slack channel"""
    if not token or not channel:
        return []

    try:
        response = requests.get(
            'https://slack.com/api/conversations.history',
            headers={'Authorization': f'Bearer {token}'},
            params={
                'channel': channel,
                'oldest': str(since_ts),
                'limit': 100
            },
            timeout=10
        )

        result = response.json()
        if result.get('ok'):
            return result.get('messages', [])
        else:
            print(f"Slack API error: {result.get('error')}")
            return []

    except Exception as e:
        print(f"Error fetching messages: {e}")
        return []


def extract_auth_code(url):
    """Extract authorization code from Schwab redirect URL"""
    # Pattern: code=XXXXX (may have %40 for @ symbol)
    match = re.search(r'code=([^&\s]+)', url)
    if match:
        code = match.group(1)
        # Decode URL-encoded @ symbol
        code = code.replace('%40', '@')
        return code
    return None


async def process_auth_url(url):
    """Process Schwab auth redirect URL and refresh tokens"""
    print(f"Processing auth URL: {url[:50]}...")

    try:
        # Extract authorization code
        code = extract_auth_code(url)
        if not code:
            return False, "Could not extract authorization code from URL"

        print(f"Extracted code: {code[:20]}...")

        # Initialize authenticator and process the code
        auth = Authenticator()

        # Process authorization code to get tokens
        endpoint, headers, params = auth._set_headers_params(
            endpoint=auth.token_endpoint,
            apikey=auth.apikey,
            secret=auth.secret,
            grant_type='authorization_code',
            code=code,
            redrirect_url=auth.redirect_url
        )

        token_dict = await auth._get(endpoint, headers, params)

        if not token_dict or 'access_token' not in token_dict:
            error_msg = token_dict.get('error_description', 'Unknown error')
            return False, f"Token exchange failed: {error_msg}"

        # Save tokens
        token_dict['access_time'] = time.time()
        token_dict['refresh_time'] = time.time()

        auth.writer.modify_json(
            f'{auth.program_path}/tokens/',
            'token_file',
            token_dict
        )

        print("✓ Tokens saved successfully!")
        return True, "Tokens refreshed successfully!"

    except Exception as e:
        print(f"Error processing auth URL: {e}")
        return False, f"Error: {str(e)}"


async def main():
    """Main handler loop - monitors until auth URL found or timeout"""
    print("=" * 70)
    print(f"Slack Auth Handler - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Get Slack credentials
    token = get_slack_token()
    channel = get_slack_channel()

    if not token or not channel:
        print("❌ SLACK_BOT_TOKEN or SLACK_CHANNEL_ID not set in .env")
        print("\nRequired environment variables:")
        print("  SLACK_BOT_TOKEN=xoxb-...")
        print("  SLACK_CHANNEL_ID=C...")
        print("\nSee docs/SLACK_AUTH_SETUP.md for instructions")
        return 1

    print(f"✓ Slack configured (channel: {channel})")
    print("\n👀 Monitoring Slack for auth redirect URL...")
    print("⏱️  Will check every 10 seconds for up to 2 hours")
    print("💡 Paste the redirect URL in Slack after authenticating\n")

    # Monitor for up to 2 hours
    timeout = time.time() + (2 * 60 * 60)  # 2 hours
    check_interval = 10  # Check every 10 seconds

    # Load state
    state = load_state()
    since_ts = time.time() - 30  # Start checking from 30 seconds ago

    # Look for Schwab redirect URLs
    auth_pattern = re.compile(r'https?://[^\s]*(?:127\.0\.0\.1|localhost)[^\s]*code=[^\s]+')

    while time.time() < timeout:
        # Fetch recent messages
        messages = get_recent_messages(token, channel, since_ts)

        # Check each message
        for msg in reversed(messages):  # Process oldest first
            text = msg.get('text', '')
            ts = msg.get('ts', '')

            # Skip if already processed
            if ts in state.get('processed_urls', []):
                continue

            # Look for auth redirect URL
            match = auth_pattern.search(text)
            if match:
                url = match.group(0)
                print(f"\n🔐 Found auth URL!")

                # Process the URL
                success, message = await process_auth_url(url)

                # Mark as processed
                if 'processed_urls' not in state:
                    state['processed_urls'] = []
                state['processed_urls'].append(ts)
                state['processed_urls'] = state['processed_urls'][-100:]
                save_state(state)

                # Send result back to Slack
                if success:
                    response_msg = (
                        "✅ *Schwab Authentication Successful!*\n\n"
                        f"{message}\n\n"
                        "Your trading system is now authenticated and ready to trade."
                    )
                    send_slack_message(token, channel, response_msg)
                    print("\n✅ Authentication successful! Exiting.")
                    return 0  # Success - exit
                else:
                    response_msg = (
                        "❌ *Schwab Authentication Failed*\n\n"
                        f"{message}\n\n"
                        "Please try getting a new auth URL and paste it again."
                    )
                    send_slack_message(token, channel, response_msg)
                    print(f"\n❌ Authentication failed: {message}")
                    # Continue monitoring in case user tries again

        # Update since timestamp for next check
        if messages:
            since_ts = max(float(msg.get('ts', since_ts)) for msg in messages)

        # Wait before next check
        elapsed = int(time.time() - (timeout - 2 * 60 * 60))
        remaining = int(timeout - time.time())
        print(f"[{elapsed}s elapsed, {remaining}s remaining] Waiting...", end='\r')
        await asyncio.sleep(check_interval)

    # Timeout reached
    print("\n\n⏱️  Timeout reached (2 hours). Exiting.")
    print("If you still need to authenticate, the system will alert again later.")
    return 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
