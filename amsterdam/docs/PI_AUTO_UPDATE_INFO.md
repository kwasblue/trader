# Raspberry Pi Auto-Update Configuration

## Already Configured ✅

Your Raspberry Pi already has a complete auto-update system running.

### Auto-Update Schedule

**Runs:** Every day at 2:00 AM ET
**Script:** `/home/kwasi/bin/auto-update-amsterdam`
**Log:** `/home/kwasi/amsterdam/logs/auto-update.log`

### What It Does

1. ✅ Fetches latest code from `origin/efficiency_improvements`
2. ✅ Checks if updates are available
3. ✅ Stops the amsterdam systemd service
4. ✅ Pulls changes from GitHub
5. ✅ Reinstalls the package (`pip install -e .`)
6. ✅ Restarts the amsterdam service
7. ✅ Verifies service started successfully
8. ✅ Logs everything with timestamps

### Safety Features

- **Trading hours protection:** Won't update between 9:30 AM - 4:00 PM ET on weekdays
- **Lock file:** Prevents concurrent updates
- **Error handling:** If pull fails, restarts service anyway (no downtime)
- **Status verification:** Checks service is running after restart

### Systemd Service

**Service file:** `/etc/systemd/system/amsterdam.service`
**Status:** Active and running
**Auto-restart:** Enabled (restarts if crashes)
**Logs:**
- stdout: `/home/kwasi/amsterdam/logs/amsterdam.log`
- stderr: `/home/kwasi/amsterdam/logs/amsterdam-error.log`

### Other Scheduled Tasks

```cron
# Weekly optimization - Sunday 11 PM
0 23 * * 0 /home/kwasi/trader/amsterdam/bin/daily-optimize

# Daily summary at 4:05 PM ET weekdays
5 16 * * 1-5 cd /home/kwasi/trader/amsterdam && venv/bin/python monitoring/scripts/daily_summary.py

# Monitor for critical events every minute
* * * * * cd /home/kwasi/trader/amsterdam && venv/bin/python monitoring/scripts/event_monitor.py
```

## How to Use

### Push Updates (Automatic Deployment)

```bash
# On your local machine
git add .
git commit -m "Your changes"
git push origin efficiency_improvements

# Pi will automatically pull and restart at 2:00 AM
```

### Manual Update (Immediate)

```bash
# SSH to Pi and run update script
ssh raspi
sudo /home/kwasi/bin/auto-update-amsterdam

# Or restart service with latest code
ssh raspi "cd /home/kwasi/amsterdam && git pull && sudo systemctl restart amsterdam"
```

### Check Logs

```bash
# Auto-update log
ssh raspi "tail -50 /home/kwasi/amsterdam/logs/auto-update.log"

# Service log
ssh raspi "tail -50 /home/kwasi/amsterdam/logs/amsterdam.log"

# Service status
ssh raspi "systemctl status amsterdam"
```

### Check Cron Jobs

```bash
ssh raspi "crontab -l"
```

## File Locations on Pi

```
/home/kwasi/
├── amsterdam/                      # Main installation
│   ├── autoamsterdam.py            # Running daemon
│   ├── venv/                       # Python environment
│   ├── logs/
│   │   ├── auto-update.log         # Update history
│   │   ├── amsterdam.log           # Service stdout
│   │   └── amsterdam-error.log     # Service stderr
│   └── ... (all your code)
│
├── bin/
│   └── auto-update-amsterdam       # The update script
│
└── trader/amsterdam/               # Seems to be duplicate?
```

## Monitoring

### Verify Updates Applied

After pushing to GitHub, check next morning:

```bash
# Check last update
ssh raspi "tail -20 /home/kwasi/amsterdam/logs/auto-update.log"

# Verify latest commit
ssh raspi "cd /home/kwasi/amsterdam && git log --oneline -5"

# Check service is running
ssh raspi "systemctl status amsterdam"
```

### Example Log Output

Successful update:
```
[2026-03-15 02:00:01] ===== Auto-update check started =====
[2026-03-15 02:00:02] Updates available! Updating...
[2026-03-15 02:00:02] Local:  003de1f
[2026-03-15 02:00:02] Remote: abc1234
[2026-03-15 02:00:03] Stopping Amsterdam service...
[2026-03-15 02:00:08] Pulling changes...
[2026-03-15 02:00:09] Pull successful
[2026-03-15 02:00:10] Reinstalling package...
[2026-03-15 02:00:12] Package reinstalled successfully
[2026-03-15 02:00:13] Starting Amsterdam service...
[2026-03-15 02:00:16] ✅ Amsterdam updated and running successfully
[2026-03-15 02:00:16] ===== Auto-update completed =====
```

## Troubleshooting

### Service won't start after update

```bash
# Check service status
ssh raspi "sudo systemctl status amsterdam"

# Check error logs
ssh raspi "tail -50 /home/kwasi/amsterdam/logs/amsterdam-error.log"

# Manually restart
ssh raspi "sudo systemctl restart amsterdam"
```

### Update not happening

```bash
# Check cron is running
ssh raspi "sudo systemctl status cron"

# Check cron logs
ssh raspi "grep CRON /var/log/syslog | tail -20"

# Manually trigger update
ssh raspi "sudo /home/kwasi/bin/auto-update-amsterdam"
```

### Git conflicts

```bash
# SSH to Pi
ssh raspi
cd /home/kwasi/amsterdam

# Check status
git status

# Stash local changes if needed
git stash

# Pull again
git pull origin efficiency_improvements
```

---

**Summary:** Your Pi is fully automated! Just push to `efficiency_improvements` branch and it updates at 2 AM automatically. 🚀
