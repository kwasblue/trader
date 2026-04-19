#!/usr/bin/env python3
"""
Amsterdam Daily Summary
Sends end-of-day trading summary to Slack
"""

import os
from datetime import date, datetime
from pathlib import Path

import requests

# Paths
BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "logs" / "autotrader.log"
ENV_FILE = BASE_DIR / ".env"


def load_slack_webhook():
    """Load Slack webhook URL from .env"""
    if not ENV_FILE.exists():
        return None

    with open(ENV_FILE) as f:
        for line in f:
            if line.startswith("SLACK_WEBHOOK_URL="):
                return line.split("=", 1)[1].strip()
    return None


def parse_todays_logs():
    """Parse today's trading activity from logs"""
    if not LOG_FILE.exists():
        return None

    today = date.today().strftime("%Y-%m-%d")
    trades = []
    errors = []
    state_changes = []

    try:
        with open(LOG_FILE) as f:
            for line in f:
                if today not in line:
                    continue

                # Capture state changes
                if "State change:" in line:
                    state_changes.append(line.strip())

                # Capture errors
                if " ERROR " in line or " CRITICAL " in line:
                    errors.append(line.strip())

                # Capture trade-related logs (customize based on your logging)
                if any(keyword in line.lower() for keyword in ["filled", "order", "buy", "sell", "trade"]):
                    trades.append(line.strip())

    except Exception as e:
        return {"error": str(e)}

    return {
        "date": today,
        "trades": trades,
        "errors": errors,
        "state_changes": state_changes,
        "total_trades": len(trades),
        "total_errors": len(errors),
    }


def get_current_status():
    """Get current Amsterdam status"""
    pid_file = BASE_DIR / "logs" / "autotrader.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return "✅ Running"
        except:
            return "❌ Stopped"
    return "⚠️ Unknown"


def format_slack_message(data):
    """Format data for Slack message"""
    if not data or "error" in data:
        return {
            "text": "❌ Amsterdam Daily Summary - Error",
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "❌ Amsterdam Daily Summary - Error"}},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Could not generate summary: {data.get('error', 'Unknown error')}",
                    },
                },
            ],
        }

    status = get_current_status()

    # Build summary text
    summary_lines = [
        f"*Date:* {data['date']}",
        f"*Status:* {status}",
        f"*Total Trades:* {data['total_trades']}",
        f"*Errors:* {data['total_errors']}",
    ]

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": "📊 Amsterdam Daily Summary"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(summary_lines)}},
    ]

    # Add recent trades if any
    if data["trades"]:
        trade_text = "```\n" + "\n".join(data["trades"][-5:]) + "\n```"
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Recent Activity (last 5):*\n{trade_text}"}}
        )
    else:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "_No trades recorded today_"}})

    # Add errors if any
    if data["errors"]:
        error_text = "```\n" + "\n".join(data["errors"][:3]) + "\n```"
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"⚠️ *Errors ({data['total_errors']}):*\n{error_text}"},
            }
        )

    # Add footer
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Generated at {datetime.now().strftime('%I:%M %p ET')} • <http://100.101.141.79:8080|View Dashboard>",
                }
            ],
        }
    )

    return {"blocks": blocks}


def send_to_slack(webhook_url, message):
    """Send message to Slack"""
    try:
        response = requests.post(webhook_url, json=message, headers={"Content-Type": "application/json"}, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending to Slack: {e}")
        return False


def main():
    # Load webhook URL
    webhook_url = load_slack_webhook()
    if not webhook_url:
        print("ERROR: SLACK_WEBHOOK_URL not found in .env file")
        return 1

    # Parse logs
    print(f"Parsing logs for {date.today()}...")
    data = parse_todays_logs()

    # Format message
    message = format_slack_message(data)

    # Send to Slack
    print("Sending to Slack...")
    if send_to_slack(webhook_url, message):
        print("✅ Daily summary sent successfully")
        return 0
    else:
        print("❌ Failed to send summary")
        return 1


if __name__ == "__main__":
    exit(main())
