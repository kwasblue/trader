# Schwab Token Auto-Refresh with Slack Notifications

## Overview

Schwab tokens expire regularly:
- **Access token**: 30 minutes (refreshed automatically by the system)
- **Refresh token**: 7 days (requires this setup)

This system automatically refreshes tokens daily and sends you a Slack message when manual re-authentication is needed.

---

## Setup

### Step 1: Get Slack Webhook URL

1. Go to https://api.slack.com/apps
2. Create a new app or select existing
3. Enable "Incoming Webhooks"
4. Add webhook to your workspace
5. Copy the webhook URL (looks like: `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX`)

### Step 2: Configure Webhook on Pi

SSH into your Pi and add the webhook to `.env`:

```bash
ssh raspi
cd ~/trader/amsterdam

# Create or edit .env file
nano .env
```

Add this line:
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

Save and exit (Ctrl+X, Y, Enter)

### Step 3: Set Up Daily Auto-Refresh

Add a cron job to run the refresh script daily at 2 AM:

```bash
# Edit crontab
crontab -e

# Add this line:
0 2 * * * cd /home/kwasi/trader/amsterdam && /home/kwasi/trader/amsterdam/venv/bin/python tools/refresh_schwab_token_with_alert.py >> logs/token_refresh.log 2>&1
```

Save and exit.

### Step 4: Test It

Test the notification system:

```bash
cd ~/trader/amsterdam
source venv/bin/activate
python tools/refresh_schwab_token_with_alert.py
```

You should see:
```
✓ Slack webhook configured
Attempting to refresh access token...
✓ Token refreshed successfully!
✓ Slack notification sent
```

And receive a Slack message!

---

## How It Works

### Daily Refresh (Automatic)

Every day at 2 AM, the cron job runs and:

1. ✅ Checks if refresh token is still valid
2. ✅ Refreshes the access token
3. ✅ Logs the result
4. ✅ Sends Slack notification (success or failure)

### When Re-Auth Needed (Every 7 Days)

When the refresh token expires:

1. 🔐 Script detects expiration
2. 📱 Sends you a Slack message with:
   - Alert that action is needed
   - Direct link to re-authenticate
   - Instructions
3. ⏸️ Trading pauses until you re-authenticate
4. ✅ After re-auth, trading resumes automatically

---

## Slack Notifications

### Success Message (Daily)
```
🔐 Schwab Token Refreshed

Access token refreshed successfully.
Valid for: 30 minutes
Auto-refreshes daily at 2 AM
```

### Action Required (Every 7 Days)
```
🔐 Schwab Token Expired - Action Required

Your Schwab refresh token has expired and requires manual re-authentication.

This happens every 7 days for security.

Steps:
1. Click the authentication link
2. Log into your Schwab account
3. Approve the application
4. The system will automatically resume trading

[Click here to authenticate]
```

### Failure Alert
```
🔐 Schwab Token Refresh Failed

Automatic token refresh failed.

Manual re-authentication may be required.

[Click here to authenticate]
```

---

## Manual Refresh

If you need to refresh manually:

```bash
ssh raspi
cd ~/trader/amsterdam
source venv/bin/activate
python tools/refresh_schwab_token_with_alert.py
```

---

## Monitoring

### Check Refresh Logs

```bash
ssh raspi
tail -f ~/trader/amsterdam/logs/token_refresh.log
```

### Check Last Refresh

```bash
ssh raspi
crontab -l  # View cron jobs
grep "Token refresh" ~/trader/amsterdam/logs/token_refresh.log | tail -5
```

### Verify Cron is Running

```bash
# Check if cron job executed today
grep "refresh_schwab_token" /var/log/syslog | tail -5
```

---

## Troubleshooting

### Not Receiving Slack Messages

1. Check webhook URL is correct:
   ```bash
   cat ~/trader/amsterdam/.env | grep SLACK
   ```

2. Test webhook manually:
   ```bash
   curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"Test message"}' \
     YOUR_WEBHOOK_URL
   ```

3. Check logs:
   ```bash
   tail ~/trader/amsterdam/logs/token_refresh.log
   ```

### Cron Job Not Running

1. Verify cron service is running:
   ```bash
   sudo systemctl status cron
   ```

2. Check cron logs:
   ```bash
   grep CRON /var/log/syslog | tail -20
   ```

3. Verify crontab entry:
   ```bash
   crontab -l | grep schwab
   ```

### Token Still Expired

If you keep getting expiration notifications:

1. Click the re-auth link in Slack
2. Complete the authentication flow
3. Verify token file updated:
   ```bash
   ls -la ~/trader/amsterdam/tokens/token_file.json
   cat ~/trader/amsterdam/tokens/token_file.json | jq '.refresh_token'
   ```

---

## Security Notes

⚠️ **Important:**

1. Never commit `.env` file to git (already in .gitignore)
2. Keep Slack webhook URL private
3. Tokens are stored locally on Pi only
4. Re-authentication link is time-limited

---

## Files

- `tools/refresh_schwab_token_with_alert.py` - Main refresh script with Slack alerts
- `.env` - Webhook URL configuration
- `logs/token_refresh.log` - Refresh history
- `tokens/token_file.json` - Current tokens (auto-generated)

---

## Quick Reference

```bash
# Setup
crontab -e  # Add: 0 2 * * * cd /home/kwasi/trader/amsterdam && ...

# Test
python tools/refresh_schwab_token_with_alert.py

# Monitor
tail -f logs/token_refresh.log

# Check token status
cat tokens/token_file.json | jq '.expires_in'
```
