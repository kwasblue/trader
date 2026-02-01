# Pre-Flight Checks

The Pre-Flight Checker validates system readiness before trading sessions begin. It ensures credentials are valid, data is fresh, and configurations are correct.

## Overview

Pre-flight checks prevent trading with:
- Invalid or expired credentials
- Stale historical data
- Missing configuration files
- Network connectivity issues

## Usage

### Command Line

```bash
# Quick check (minimal output)
python preflight.py

# Verbose mode (detailed output)
python preflight.py -v

# Update stale data automatically
python preflight.py --update-data

# Force Schwab re-authentication
python preflight.py --reauth-schwab

# Full check with data update
python preflight.py -v --update-data
```

### From Code

```python
from preflight import PreFlightChecker

async def check_before_trading():
    checker = PreFlightChecker(verbose=True)

    # Run all checks
    passed = await checker.run_all_checks()

    if passed:
        print("All checks passed!")
    else:
        print("Pre-flight failed:")
        for issue in checker.issues:
            print(f"  - {issue}")
```

## Checks Performed

### 1. Environment Variables

Validates required environment variables are set:

| Variable | Broker | Required |
|----------|--------|----------|
| `ALPACA_API_KEY` | Alpaca | Yes (for Alpaca) |
| `ALPACA_SECRET_KEY` | Alpaca | Yes (for Alpaca) |
| `SCHWAB_API_KEY` | Schwab | Yes (for Schwab) |
| `SCHWAB_SECRET` | Schwab | Yes (for Schwab) |

### 2. Credential Validation

#### Alpaca
- Verifies API keys are valid
- Checks account status (ACTIVE)
- Confirms trading permissions

#### Schwab
- Validates OAuth tokens exist
- Checks access token expiry
- Checks refresh token expiry
- Warns if tokens expire soon (< 24 hours)

### 3. Data Freshness

For each configured symbol:
- Checks if data file exists
- Validates last data timestamp
- Flags as stale if > 1 hour old (during market hours)

### 4. Configuration Files

Verifies required configuration files exist:
- `config/trading_config.json`
- `config/symbol_configuration.json`
- `.env` file

### 5. Network Connectivity

- Tests connection to broker APIs
- Verifies WebSocket endpoints are reachable

## Output

### Summary View

```
============================================================
  PRE-FLIGHT CHECK SUMMARY
============================================================

  [PASS] Environment variables configured
  [PASS] Alpaca credentials valid
  [WARN] Schwab token expires in 18 hours
  [PASS] Historical data is fresh
  [PASS] Configuration files present

============================================================
  RESULT: READY FOR TRADING
============================================================
```

### Detailed View (-v)

```
============================================================
  ENVIRONMENT CHECK
============================================================

  Checking environment variables...
  [OK] ALPACA_API_KEY is set
  [OK] ALPACA_SECRET_KEY is set
  [OK] SCHWAB_API_KEY is set
  [OK] SCHWAB_SECRET is set

============================================================
  CREDENTIAL CHECK
============================================================

  Validating Alpaca credentials...
  [OK] Account status: ACTIVE
  [OK] Buying power: $25,432.50

  Validating Schwab credentials...
  [OK] Access token valid (expires in 28 minutes)
  [WARN] Refresh token expires in 18 hours
  [INFO] Consider refreshing Schwab tokens

============================================================
  DATA FRESHNESS CHECK
============================================================

  Checking AAPL...
  [OK] Last update: 2024-01-15 16:00:00 (15 minutes ago)
  [OK] Bar count: 1,234

  Checking MSFT...
  [OK] Last update: 2024-01-15 16:00:00 (15 minutes ago)
  [OK] Bar count: 1,234
```

## Issue Classification

### Critical Issues (Blocks Trading)

- Missing API credentials
- Invalid/expired credentials
- Network connectivity failure
- Missing required config files

### Warnings (Allows Trading)

- Token expires soon (< 24 hours)
- Data slightly stale (< 2 hours)
- Optional config missing

### Info (Informational Only)

- Credential refresh recommended
- Data update available
- System status notes

## Data Update

When `--update-data` is specified:

```bash
python preflight.py --update-data
```

The checker will:
1. Identify stale symbols
2. Fetch latest data from best available source
3. Process through ML pipeline
4. Save to storage (JSON + SQLite)
5. Re-validate freshness

## Schwab Re-Authentication

When `--reauth-schwab` is specified:

```bash
python preflight.py --reauth-schwab
```

This will:
1. Open browser for Schwab OAuth flow
2. Capture new tokens
3. Save to token file
4. Validate new tokens

## Integration with AutoTrader

AutoTrader automatically runs pre-flight checks before each trading session:

```python
# In autotrader.py
async def _run_preflight(self) -> bool:
    from preflight import PreFlightChecker

    checker = PreFlightChecker(verbose=self.verbose)
    passed = await checker.run_all_checks()

    if not passed:
        self.logger.error("Pre-flight checks failed")
        for issue in checker.issues:
            self.logger.error(f"  - {issue}")

    return passed
```

## Programmatic Usage

### Full Check

```python
from preflight import PreFlightChecker

async def main():
    checker = PreFlightChecker(verbose=True)

    # Run all checks
    passed = await checker.run_all_checks()

    # Access results
    print(f"Passed: {passed}")
    print(f"Issues: {checker.issues}")
    print(f"Warnings: {checker.warnings}")
    print(f"Passed checks: {checker.passed}")
```

### Individual Checks

```python
checker = PreFlightChecker()

# Check environment only
checker._check_environment()

# Check credentials only
await checker._check_credentials()

# Check data freshness only
checker._check_data_freshness(['AAPL', 'MSFT'])

# Check configuration only
checker._check_configuration()
```

## Credential Validator

The pre-flight system uses `CredentialValidator` internally:

```python
from core.credential_validator import (
    CredentialValidator,
    CredentialStatus,
    ValidationResult,
    check_credentials,
    can_use_alpaca,
    can_use_schwab
)

# Quick check
alpaca_ok = await can_use_alpaca()
schwab_ok = await can_use_schwab()

# Detailed validation
validator = CredentialValidator()
result = await validator.validate_alpaca()

print(f"Status: {result.status}")  # CredentialStatus.VALID
print(f"Can trade: {result.can_trade}")
print(f"Can fetch data: {result.can_fetch_data}")
print(f"Expires in: {result.expires_in} seconds")
```

### CredentialStatus Values

| Status | Meaning |
|--------|---------|
| `VALID` | Credentials are valid and ready |
| `EXPIRED` | Credentials have expired |
| `EXPIRING_SOON` | Valid but expires within 24 hours |
| `MISSING` | Credentials not configured |
| `INVALID` | Credentials rejected by API |
| `ERROR` | Error during validation |

## Configuration

### trading_config.json

```json
{
  "preflight": {
    "data_staleness_hours": 1,
    "token_warning_hours": 24,
    "required_symbols": ["AAPL", "MSFT"]
  }
}
```

## Logging

Pre-flight logs to `logs/preflight.log`:

```
2024-01-15 09:15:00 INFO [PreFlightChecker] Starting pre-flight checks
2024-01-15 09:15:01 INFO [PreFlightChecker] Environment check: PASSED
2024-01-15 09:15:02 INFO [PreFlightChecker] Credential check: PASSED
2024-01-15 09:15:03 WARN [PreFlightChecker] Schwab token expires in 18h
2024-01-15 09:15:04 INFO [PreFlightChecker] Data freshness check: PASSED
2024-01-15 09:15:05 INFO [PreFlightChecker] All checks passed
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All checks passed |
| 1 | Critical issues found |
| 2 | Warnings only (still passes) |

## Best Practices

1. **Run before trading sessions**
   ```bash
   python preflight.py && python run_trading.py
   ```

2. **Schedule regular credential checks**
   ```bash
   # crontab entry for daily check at 9 AM ET
   0 9 * * 1-5 cd /path/to/schwab_trader && python preflight.py
   ```

3. **Set up alerts for failures**
   - Monitor exit codes
   - Watch log files for ERROR level

4. **Keep tokens refreshed**
   - Run `--reauth-schwab` before tokens expire
   - Schwab refresh tokens last 7 days

## Troubleshooting

### "Missing environment variables"

```bash
# Check your .env file
cat .env

# Ensure it's loaded
source .venv/bin/activate
```

### "Alpaca credentials invalid"

```bash
# Verify keys at https://app.alpaca.markets
# Check paper vs live mode
```

### "Schwab token expired"

```bash
# Re-authenticate
python preflight.py --reauth-schwab
```

### "Data is stale"

```bash
# Update data
python preflight.py --update-data

# Or run data pipeline directly
python -m core.unified_data_pipeline --symbols AAPL MSFT
```

## Related Documentation

- [AutoTrader](autotrader.md)
- [Configuration Guide](configuration.md)
- [Credential Management](architecture.md#credentials)
