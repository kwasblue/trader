# Slack Auth Handler Setup

## Overview

This allows you to refresh Schwab tokens directly from Slack - no SSH needed!

**User Flow:**
1. 🔐 Get Slack alert: "Token expired - click to auth"
2. 🌐 Click link, log into Schwab
3. 📋 Copy redirect URL, paste in Slack
4. ✅ System auto-processes and confirms
5. 🚀 Trading resumes

---

## Setup (One-time)

### Step 1: Create Slack App

1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From scratch"
3. Name: "Amsterdam Trading Auth"
4. Select your workspace
5. Click "Create App"

### Step 2: Add Bot Token Scopes

1. In your app, go to "OAuth & Permissions"
2. Scroll to "Scopes" → "Bot Token Scopes"
3. Add these scopes:
   - `chat:write` - Send messages
   - `channels:history` - Read channel messages
   - `channels:read` - View channel info

### Step 3: Install App to Workspace

1. Scroll up to "OAuth Tokens"
2. Click "Install to Workspace"
3. Click "Allow"
4. Copy the **Bot User OAuth Token** (starts with `xoxb-`)

### Step 4: Get Channel ID

1. Open Slack
2. Go to the channel where you want auth alerts
3. Click channel name → scroll down
4. Copy the **Channel ID** (starts with `C`)

### Step 5: Invite Bot to Channel

In Slack channel, type:
```
/invite @Amsterdam Trading Auth
```

### Step 6: Add to Pi .env

SSH into Pi and edit .env:

```bash
ssh raspi
cd ~/trader/amsterdam
nano .env
```

Add these lines:
```bash
# Slack Bot (for auth handling)
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_CHANNEL_ID=C01234567890
```

Save and exit (Ctrl+X, Y, Enter)

### Step 7: Set Up Cron Job

Add cron job to check Slack every minute:

```bash
crontab -e
```

Add this line:
```bash
# Check Slack for Schwab auth URLs every minute
* * * * * cd /home/kwasi/trader/amsterdam && /home/kwasi/trader/amsterdam/venv/bin/python monitoring/scripts/slack_auth_handler.py >> logs/slack_auth.log 2>&1
```

Save and exit.

### Step 8: Test It

```bash
cd ~/trader/amsterdam
source venv/bin/activate
python monitoring/scripts/slack_auth_handler.py
```

You should see:
```
✓ Slack configured (channel: C01234567890)
Found 0 messages since last check
No auth URLs found
```

---

## Usage

### When Token Expires (Every 7 Days)

**You receive Slack message:**
```
🔐 Schwab Token Expired - Action Required

Your Schwab refresh token has expired!

Action Required:
1. Click the authentication link below
2. Log into your Schwab account
3. Approve the application
4. Trading will resume automatically

[Click here to authenticate]
```

**You do:**
1. Click the link
2. Log into Schwab
3. Approve the app
4. Browser redirects to `https://127.0.0.1/?code=ABC123...`
5. **Copy the entire URL from browser address bar**
6. **Paste it in the same Slack channel**

**System responds:**
```
✅ Schwab Authentication Successful!

Tokens refreshed successfully!

Your trading system is now authenticated and ready to trade.
```

**That's it!** No SSH, no terminal - just paste in Slack.

---

## Example

### What You Paste in Slack:
```
https://127.0.0.1/?code=ABC123XYZ456DEF789%40
```

### System Responds:
```
✅ Schwab Authentication Successful!
Tokens refreshed successfully!
Your trading system is now authenticated and ready to trade.
```

---

## Troubleshooting

### "SLACK_BOT_TOKEN not set"

Check .env file:
```bash
ssh raspi "cat ~/trader/amsterdam/.env | grep SLACK_BOT"
```

Should show:
```
SLACK_BOT_TOKEN=xoxb-...
```

### "Could not extract authorization code"

Make sure you paste the **full redirect URL**, including:
- `https://127.0.0.1/` (or whatever your redirect URI is)
- `?code=...` (the authorization code)

### "Token exchange failed"

The code may have expired (they're short-lived). Just:
1. Click the auth link again
2. Log in again
3. Paste the new redirect URL

### Not Detecting Messages

Check cron is running:
```bash
ssh raspi "grep slack_auth /var/log/syslog | tail -5"
```

Check logs:
```bash
ssh raspi "tail ~/trader/amsterdam/logs/slack_auth.log"
```

### Bot Not in Channel

Make sure you invited the bot:
```
/invite @Amsterdam Trading Auth
```

---

## Security Notes

- ✅ Bot token stays on your Pi (not exposed)
- ✅ Only works in the specified channel
- ✅ Authorization codes are single-use and expire quickly
- ✅ Bot only reads messages, doesn't post unless responding to auth
- ✅ All communication over HTTPS

---

## Files

- `monitoring/scripts/slack_auth_handler.py` - Main handler script
- `logs/slack_auth.log` - Processing log
- `logs/slack_auth_state.json` - Tracking state (auto-generated)

---

## Disable Manual SSH Method

Once this is working, you can disable the manual SSH instructions in token_keeper Slack alerts by setting:

```bash
# In .env
SLACK_AUTH_HANDLER_ENABLED=true
```

The alerts will then say: "Paste the redirect URL in Slack" instead of "SSH in and run script"
