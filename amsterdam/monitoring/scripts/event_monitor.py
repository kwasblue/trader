#!/usr/bin/env python3
"""
Amsterdam Event Monitor - Critical Alerts Only
Monitors for critical events and sends immediate Slack alerts
"""

import os
import json
import requests
from datetime import datetime, date
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "logs" / "autotrader.log"
PID_FILE = BASE_DIR / "logs" / "autotrader.pid"
ENV_FILE = BASE_DIR / ".env"
STATE_FILE = BASE_DIR / "logs" / "alert_state.json"

def load_slack_webhook():
    """Load Slack webhook URL from .env"""
    if not ENV_FILE.exists():
        return None
    
    with open(ENV_FILE) as f:
        for line in f:
            if line.startswith('SLACK_WEBHOOK_URL='):
                return line.split('=', 1)[1].strip()
    return None

def load_state():
    """Load previous alert state to avoid duplicates"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except:
            pass
    return {
        'last_check': None,
        'alerted_today': [],
        'last_log_position': 0,
        'process_was_running': True
    }

def save_state(state):
    """Save alert state"""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error saving state: {e}")

def send_alert(webhook_url, title, message, severity="critical"):
    """Send alert to Slack"""
    emoji_map = {
        "critical": "🚨",
        "warning": "⚠️",
        "success": "✅"
    }
    
    color_map = {
        "critical": "#ef4444",
        "warning": "#f59e0b",
        "success": "#10b981"
    }
    
    emoji = emoji_map.get(severity, "🚨")
    color = color_map.get(severity, "#ef4444")
    
    payload = {
        "attachments": [{
            "color": color,
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"{emoji} {title}"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": message}
                },
                {
                    "type": "context",
                    "elements": [{
                        "type": "mrkdwn",
                        "text": f"{datetime.now().strftime('%I:%M:%S %p ET')} • <http://100.101.141.79:8080|View Dashboard>"
                    }]
                }
            ]
        }]
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending alert: {e}")
        return False

def check_process_status(state, webhook_url):
    """Check if Amsterdam is running"""
    is_running = False
    
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            is_running = True
        except:
            pass
    
    # Alert if crashed
    if state['process_was_running'] and not is_running:
        send_alert(
            webhook_url,
            "Amsterdam Stopped",
            "🚨 *Amsterdam trading bot has stopped unexpectedly*\n\nThe process is no longer running. Check logs immediately.",
            "critical"
        )
        state['alerted_today'].append('process_stopped')
    elif not state['process_was_running'] and is_running:
        send_alert(
            webhook_url,
            "Amsterdam Restarted",
            "✅ *Amsterdam trading bot is now running again*",
            "success"
        )
    
    state['process_was_running'] = is_running
    return is_running

def parse_new_log_entries(state):
    """Get new log entries since last check"""
    if not LOG_FILE.exists():
        return []
    
    try:
        with open(LOG_FILE, 'r') as f:
            f.seek(state['last_log_position'])
            new_lines = f.readlines()
            state['last_log_position'] = f.tell()
            return new_lines
    except Exception as e:
        print(f"Error reading logs: {e}")
        return []

def check_critical_events(new_logs, state, webhook_url):
    """Check for critical events only"""
    today = date.today().strftime('%Y-%m-%d')
    
    for line in new_logs:
        if today not in line:
            continue
        
        # CRITICAL: Errors and exceptions
        if (" ERROR " in line or " CRITICAL " in line or "Exception" in line):
            if "error" not in state['alerted_today']:
                error_msg = line.split(' - ')[-1].strip() if ' - ' in line else line.strip()
                send_alert(
                    webhook_url,
                    "Critical Error",
                    f"🚨 *Amsterdam encountered an error:*\n```{error_msg[:500]}```\n\nCheck logs immediately.",
                    "critical"
                )
                state['alerted_today'].append('error')
        
        # CRITICAL: Pre-flight check failure
        if "pre-flight" in line.lower() and ("fail" in line.lower() or "error" in line.lower()):
            if "preflight_failed" not in state['alerted_today']:
                send_alert(
                    webhook_url,
                    "Pre-flight Failed",
                    "🚨 *Pre-flight checks failed before market open*\n\nTrading may not start. Check dashboard now.",
                    "critical"
                )
                state['alerted_today'].append('preflight_failed')
        
        # CRITICAL: Connection/API issues
        if any(keyword in line.lower() for keyword in ['connection failed', 'timeout', 'unreachable', 'api error']):
            if "connection_issue" not in state['alerted_today']:
                send_alert(
                    webhook_url,
                    "Connection Issue",
                    f"⚠️ *Network or API connection problem detected*\n\nAmsterdam may be unable to trade.",
                    "warning"
                )
                state['alerted_today'].append('connection_issue')

def main():
    webhook_url = load_slack_webhook()
    if not webhook_url:
        print("ERROR: SLACK_WEBHOOK_URL not found")
        return 1
    
    state = load_state()
    
    # Reset daily alerts if new day
    today = date.today().strftime('%Y-%m-%d')
    if not state.get('last_check') or state['last_check'].split(' ')[0] != today:
        state['alerted_today'] = []
    
    state['last_check'] = datetime.now().isoformat()
    
    # Check for issues
    check_process_status(state, webhook_url)
    new_logs = parse_new_log_entries(state)
    if new_logs:
        check_critical_events(new_logs, state, webhook_url)
    
    save_state(state)
    return 0

if __name__ == '__main__':
    exit(main())
