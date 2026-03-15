#!/usr/bin/env python3
"""
Amsterdam Trading Dashboard
Simple web interface to monitor the trading bot
"""

from flask import Flask, render_template_string, jsonify
import os
import json
from datetime import datetime
from pathlib import Path
import re

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "logs" / "autotrader.log"
PID_FILE = BASE_DIR / "logs" / "autotrader.pid"
CACHE_FILE = BASE_DIR / "cache" / "system_cache.json"

def get_status():
    """Get Amsterdam running status"""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if process is running
            os.kill(pid, 0)
            return "Running"
        except (ProcessLookupError, ValueError):
            return "Stopped"
    return "Unknown"

def get_current_state():
    """Parse current state from logs"""
    if not LOG_FILE.exists():
        return "Unknown"
    
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines[-50:]):  # Check last 50 lines
                if "State change:" in line or "waiting_for_market" in line:
                    if "waiting_for_market" in line:
                        return "Waiting for Market"
                    elif "TRADING" in line or "trading" in line:
                        return "Trading"
                    elif "PRE_FLIGHT" in line:
                        return "Pre-flight Checks"
                    elif "POST_MARKET" in line:
                        return "Post-Market"
                    elif "UPDATING_DATA" in line:
                        return "Updating Data"
    except Exception:
        pass
    
    return "Unknown"

def get_next_market_open():
    """Get next market open time from logs"""
    if not LOG_FILE.exists():
        return "Unknown"
    
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines[-20:]):
                if "Market opens at" in line:
                    match = re.search(r'Market opens at ([\d-]+ [\d:]+ \w+)', line)
                    if match:
                        return match.group(1)
                if "Waiting (\d+)h (\d+)m" in line:
                    match = re.search(r'Waiting (\d+h \d+m)', line)
                    if match:
                        return f"in {match.group(1)}"
    except Exception:
        pass
    
    return "Unknown"

def get_recent_logs(n=10):
    """Get recent log entries"""
    if not LOG_FILE.exists():
        return []
    
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-n:]]
    except Exception:
        return []

def parse_log_line(line):
    """Parse log line into structured data"""
    try:
        # Format: 2026-03-14 19:21:31 - AutoTrader - INFO - Message
        parts = line.split(' - ', 3)
        if len(parts) >= 4:
            return {
                'timestamp': parts[0],
                'level': parts[2],
                'message': parts[3]
            }
    except Exception:
        pass
    return {'timestamp': '', 'level': '', 'message': line}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Amsterdam Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 {
            font-size: 28px;
            margin-bottom: 8px;
            color: #fff;
        }
        .subtitle {
            color: #888;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 20px;
        }
        .card h2 {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        .status-value {
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .status-running { color: #10b981; }
        .status-stopped { color: #ef4444; }
        .status-unknown { color: #888; }
        .state-value {
            font-size: 24px;
            color: #3b82f6;
        }
        .next-market {
            font-size: 18px;
            color: #fbbf24;
        }
        .log-container {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 20px;
        }
        .log-entry {
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            padding: 8px;
            border-bottom: 1px solid #2a2a2a;
            line-height: 1.5;
        }
        .log-entry:last-child { border-bottom: none; }
        .log-timestamp { color: #666; }
        .log-info { color: #3b82f6; }
        .log-warning { color: #fbbf24; }
        .log-error { color: #ef4444; }
        .log-message { color: #e0e0e0; }
        .updated {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 20px;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
    <script>
        // Auto-refresh every 5 seconds
        setTimeout(() => location.reload(), 5000);
    </script>
</head>
<body>
    <div class="container">
        <h1>Amsterdam Trading Dashboard</h1>
        <div class="subtitle">Live monitoring • Auto-refresh every 5s</div>
        
        <div class="grid">
            <div class="card">
                <h2>Status</h2>
                <div class="status-value status-{{ status_class }}">{{ status }}</div>
            </div>
            
            <div class="card">
                <h2>Current State</h2>
                <div class="state-value">{{ state }}</div>
            </div>
            
            <div class="card">
                <h2>Next Market Open</h2>
                <div class="next-market">{{ next_open }}</div>
            </div>
        </div>
        
        <div class="log-container">
            <h2 style="margin-bottom: 15px;">Recent Activity</h2>
            {% for log in logs %}
            <div class="log-entry">
                <span class="log-timestamp">{{ log.timestamp }}</span>
                <span class="log-{{ log.level|lower }}">{{ log.level }}</span>
                <span class="log-message">{{ log.message }}</span>
            </div>
            {% endfor %}
        </div>
        
        <div class="updated">Last updated: {{ now }}</div>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    status = get_status()
    state = get_current_state()
    next_open = get_next_market_open()
    recent_logs = get_recent_logs(15)
    
    status_class = {
        'Running': 'running',
        'Stopped': 'stopped',
        'Unknown': 'unknown'
    }.get(status, 'unknown')
    
    logs = [parse_log_line(line) for line in recent_logs]
    
    return render_template_string(
        HTML_TEMPLATE,
        status=status,
        status_class=status_class,
        state=state,
        next_open=next_open,
        logs=logs,
        now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

@app.route('/api/status')
def api_status():
    """JSON API endpoint"""
    return jsonify({
        'status': get_status(),
        'state': get_current_state(),
        'next_market_open': get_next_market_open(),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
