# Configuration Guide

This guide covers all configuration options for Schwab Trader.

## Configuration Precedence

Configuration values are resolved in the following order (highest to lowest priority):

1. **Environment variables** (`.env` file or system environment)
2. **Runtime overrides** (passed to Settings constructor)
3. **Centralized config** (`config/trading_config.json`) - Primary config source
4. **Individual config files** (`config/*.json`)
5. **Default values** (in dataclasses)

```
Environment Variables
        ↓
Runtime Overrides
        ↓
trading_config.json (Centralized)
        ↓
Individual Config Files
        ↓
Dataclass Defaults
```

---

## Centralized Configuration System

The primary configuration is in `config/trading_config.json`. Access it via:

```python
from core.config_loader import get_config

cfg = get_config()
print(cfg.trade_logic.cooldown_bars)  # 5
print(cfg.position_sizer.risk_percentage)  # 0.05
```

### Factory Functions

Use factory functions to create pre-configured component instances:

```python
from core.config_loader import (
    get_config,
    create_position_sizer,
    create_drawdown_monitor,
    create_trade_approver,
    create_position_manager,
)

cfg = get_config()

# Create components from config
sizer = create_position_sizer(cfg)       # KellyPositionSizer
ddm = create_drawdown_monitor(cfg)       # DrawdownMonitor (or None if disabled)
approver = create_trade_approver(cfg)    # StandardTradeApprover
pm = create_position_manager(cfg)        # PositionManager
```

### Configuration Sections

#### General

```json
{
  "general": {
    "default_symbols": [
      "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
      "AVGO", "JPM", "V", "UNH", "XOM", "MA", "COST", "HD",
      "PG", "JNJ", "ABBV", "WMT"
    ],
    "default_mode": "simulation",
    "log_level": "INFO"
  }
}
```

#### Simulation

```json
{
  "simulation": {
    "enabled": true,
    "steps": 2000,
    "bar_sleep": 0.5,
    "warmup_bars": 200,
    "starting_cash": 100000.0,
    "gbm_mu": 0.05,
    "gbm_sigma": 0.2
  }
}
```

#### Risk Management

```json
{
  "risk": {
    "risk_per_trade": 0.05,
    "max_trade_pct": 0.10,
    "max_holding_pct": 0.25,
    "max_pyramid_layers": 2,
    "min_bars_between_layers": 2,
    "daily_loss_limit": 1000.0
  }
}
```

#### Position Sizer

```json
{
  "position_sizer": {
    "type": "dynamic",
    "risk_percentage": 0.05,
    "max_trade_pct": 0.10,
    "max_holding_pct": 0.25,
    "min_position_size": 1,
    "max_position_size": 1000,
    "fee_rate": 0.001,
    "allow_fractional": false,
    "lot_size": 1
  }
}
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `risk_percentage` | Fraction of equity risked per trade | 0.05 (5%) |
| `max_trade_pct` | Max notional as % of equity per trade | 0.10 (10%) |
| `max_holding_pct` | Max notional in single position | 0.25 (25%) |
| `fee_rate` | Transaction fee rate | 0.001 (0.1%) |
| `allow_fractional` | Allow fractional shares | false |
| `lot_size` | Minimum lot size for rounding | 1 |

#### Trade Logic

```json
{
  "trade_logic": {
    "cooldown_mode": "bars",
    "cooldown_bars": 5,
    "cooldown_seconds": 300,
    "tp_mult_low": 1.5,
    "tp_mult_normal": 2.0,
    "tp_mult_high": 3.0,
    "sl_mult_low": 1.0,
    "sl_mult_normal": 1.5,
    "sl_mult_high": 2.0,
    "exit_fraction": 0.25,
    "trailing_stop": true,
    "max_positions": 10,
    "min_bars_to_hold": 3,
    "swing_mode": true,
    "min_hold_days": 1,
    "allow_after_hours": false
  }
}
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cooldown_mode` | "bars", "time", or "both" | "bars" |
| `cooldown_bars` | Bars between trades per symbol | 5 |
| `cooldown_seconds` | Seconds between trades (if time mode) | 300 |
| `tp_mult_low/normal/high` | Take profit ATR multipliers by regime | 1.5/2.0/3.0 |
| `sl_mult_low/normal/high` | Stop loss ATR multipliers by regime | 1.0/1.5/2.0 |
| `exit_fraction` | Fraction for partial exits | 0.25 |
| `trailing_stop` | Enable trailing stops | true |
| `max_positions` | Maximum concurrent positions | 10 |
| `min_bars_to_hold` | Min bars before TP/reversal exits | 3 |
| `swing_mode` | Prevent same-day exits (except SL) | true |
| `min_hold_days` | Days to hold in swing mode | 1 |
| `allow_after_hours` | Trade outside market hours | false |

#### Drawdown Monitor

```json
{
  "drawdown_monitor": {
    "enabled": false,
    "max_symbol_drawdown": 0.30,
    "max_symbol_daily_drawdown": 0.15,
    "symbol_cooldown_seconds": 5,
    "max_portfolio_drawdown": 0.25,
    "max_portfolio_daily_drawdown": 0.10,
    "portfolio_cooldown_seconds": 5
  }
}
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enabled` | Enable drawdown monitoring | false |
| `max_symbol_drawdown` | Max intraday drawdown per symbol | 0.30 (30%) |
| `max_symbol_daily_drawdown` | Max daily drawdown per symbol | 0.15 (15%) |
| `max_portfolio_drawdown` | Max intraday portfolio drawdown | 0.25 (25%) |
| `max_portfolio_daily_drawdown` | Max daily portfolio drawdown | 0.10 (10%) |
| `*_cooldown_seconds` | Cooldown after unlock | 5 |

#### Indicators

```json
{
  "indicators": {
    "atr_period": 14,
    "sma_short": 20,
    "sma_long": 50,
    "rsi_period": 14,
    "bollinger_period": 20,
    "bollinger_std": 2.0
  }
}
```

#### Error Recovery

```json
{
  "error_recovery": {
    "retry_max_attempts": 3,
    "retry_base_delay": 1.0,
    "circuit_breaker_failure_threshold": 5,
    "circuit_breaker_timeout": 60.0
  }
}
```

### Config Validation

```python
from core.config_loader import get_config, validate_config

cfg = get_config()
is_valid, errors = validate_config(cfg)

if not is_valid:
    print(f"Config errors: {errors}")
```

### Day Trade Mode

To enable day trading (disable swing mode):

```python
from core.config_loader import enable_day_trade_mode

cfg = enable_day_trade_mode()  # Sets swing_mode=False, min_hold_days=0
```

---

## Symbol Management

The system uses two symbol sources for different purposes:

### 1. Config Default Symbols

Located in `config/trading_config.json` under `general.default_symbols`.

**Used by:**
- Pre-flight checks (default symbols to validate)
- Live runners (symbols to trade if not overridden)
- Simulation mode
- Historical data updates

```python
from core.config_loader import get_config

cfg = get_config()
symbols = cfg.general.default_symbols
# ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', ...]
```

### 2. Trade List Database

Located in `data/trading_state.db` (SQLite).

**Used by:**
- AutoTrader (autonomous trading daemon)
- Determines which symbols are actively traded

```python
from core.symbol_list_manager import get_trade_list, add_symbol_to_list

# Get current trade list
symbols = get_trade_list()

# Add a symbol
add_symbol_to_list('AAPL', 'trade')

# Remove a symbol
remove_symbol_from_list('TSLA', 'trade')
```

### Migration Between Sources

On first run, if the trade list database is empty, it migrates symbols from config:

```python
# Automatic migration happens in symbol_list_manager.py
# Config default_symbols → trade list database
```

### Which to Modify?

| Goal | Modify |
|------|--------|
| Change symbols for preflight/simulation | `config/trading_config.json` |
| Change symbols for autotrader | Use `add_symbol_to_list()` / `remove_symbol_from_list()` |
| Change both | Update config AND database |

---

## Unified Symbol Configuration

The recommended approach is to use `config/symbol_configuration.json` which consolidates:
- Strategy routing
- Strategy parameters
- Trade logic configuration
- Risk overrides per symbol

### Structure

```json
{
  "default_settings": {
    "risk_per_trade": 0.01,
    "max_position_pct": 0.10,
    "sl_mults": {
      "low_volatility": 2.0,
      "normal": 1.5,
      "high_volatility": 1.0
    },
    "tp_mults": {
      "low_volatility": 3.0,
      "normal": 2.0,
      "high_volatility": 1.5
    }
  },
  "symbols": {
    "AAPL": {
      "enabled": true,
      "regimes": {
        "low_volatility": {
          "strategy": "sma_strategy",
          "strategy_params": {"fast": 10, "slow": 30},
          "trade_logic": "default",
          "trade_logic_params": {}
        },
        "normal": {
          "strategy": "momentum_strategy",
          "strategy_params": {"lookback": 20},
          "trade_logic": "default"
        },
        "high_volatility": {
          "strategy": "mean_reversion_strategy",
          "strategy_params": {"window": 14},
          "trade_logic": "default"
        }
      },
      "overrides": {
        "risk_per_trade": 0.01
      }
    }
  }
}
```

### Adding a New Symbol

1. Add the symbol to `config/symbol_configuration.json`:

```json
{
  "symbols": {
    "NEW_SYMBOL": {
      "enabled": true,
      "regimes": {
        "low_volatility": {
          "strategy": "sma_strategy",
          "strategy_params": {"fast": 10, "slow": 30},
          "trade_logic": "default"
        },
        "normal": {
          "strategy": "momentum_strategy",
          "strategy_params": {"lookback": 20},
          "trade_logic": "default"
        },
        "high_volatility": {
          "strategy": "mean_reversion_strategy",
          "strategy_params": {"window": 14},
          "trade_logic": "default"
        }
      }
    }
  }
}
```

2. Enable the symbol by setting `"enabled": true`

3. (Optional) Add symbol-specific risk overrides

### Adding a New Strategy

1. Create the strategy in `strategies/`:

```python
from strategies.strategy_registry import register_strategy

@register_strategy("my_new_strategy")
class MyNewStrategy:
    def __init__(self, params=None):
        self.params = params or {}
        self.my_param = self.params.get("my_param", 14)

    def generate_signal(self, data):
        # Your signal logic here
        # Return: 1 (buy), 0 (hold), -1 (sell)
        return 0
```

2. Reference it in the unified config:

```json
{
  "symbols": {
    "AAPL": {
      "regimes": {
        "normal": {
          "strategy": "my_new_strategy",
          "strategy_params": {"my_param": 20}
        }
      }
    }
  }
}
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# ============================================
# BROKER CREDENTIALS
# ============================================

# Schwab API
SCHWAB_CLIENT_ID=your_client_id
SCHWAB_CLIENT_SECRET=your_client_secret
SCHWAB_REDIRECT_URI=https://localhost:8080/callback
SCHWAB_ACCOUNT_ID=your_account_id

# Alpaca API (optional)
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Coinbase API (optional)
COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_api_secret

# ============================================
# TRADING SETTINGS
# ============================================

# Risk Management
DEFAULT_RISK_PER_TRADE=0.02    # 2% risk per trade
MAX_POSITION_PCT=0.20          # Max 20% in single position
MAX_DAILY_DRAWDOWN=0.05        # 5% daily loss limit
MAX_TOTAL_DRAWDOWN=0.15        # 15% total drawdown limit

# Position Sizing
INITIAL_CAPITAL=10000
POSITION_SIZE_METHOD=risk_based  # risk_based, fixed, volatility_scaled

# Execution
SLIPPAGE_MODEL=volume_based      # fixed, random, volume_based, volatility
DEFAULT_SLIPPAGE_PCT=0.001       # 0.1%
TRANSACTION_COST=0.001           # 0.1%

# ============================================
# DATA SETTINGS
# ============================================

# Database
DATABASE_PATH=data/trading.db

# Historical Data
DATA_STORAGE_PATH=data/data_storage/proc_data
CACHE_PATH=cache/

# ============================================
# LOGGING
# ============================================

LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR
LOG_DIR=logs/
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

---

## Configuration Files

### Strategy Routing (`config/strategy_routing.json`)

Maps symbols to strategies:

```json
{
    "default_strategy": "sma",
    "default_params": {
        "fast": 10,
        "slow": 30
    },
    "symbol_routing": {
        "AAPL": {
            "strategy": "ema",
            "params": {
                "short_window": 12,
                "long_window": 26
            }
        },
        "GOOGL": {
            "strategy": "macd",
            "params": {
                "fast_window": 12,
                "slow_window": 26,
                "signal_window": 9
            }
        },
        "TSLA": {
            "strategy": "momentum",
            "params": {
                "lookback": 20
            }
        }
    },
    "sector_routing": {
        "TECH": "ema",
        "FINANCE": "mean_reversion",
        "ENERGY": "breakout"
    }
}
```

### Trade Logic Routing (`config/trade_logic_routing.json`)

Configures trade execution logic:

```json
{
    "default_logic": "default",
    "symbol_overrides": {
        "TSLA": {
            "logic": "aggressive",
            "risk_multiplier": 1.5
        }
    },
    "logic_definitions": {
        "default": {
            "entry_rules": {
                "require_trend_confirmation": true,
                "min_signal_strength": 0.6,
                "volume_filter": true
            },
            "exit_rules": {
                "trailing_stop_atr": 2.0,
                "take_profit_atr": 3.0,
                "time_stop_bars": 20
            },
            "position_rules": {
                "max_positions": 5,
                "max_per_symbol": 1,
                "min_holding_period": 1
            }
        },
        "aggressive": {
            "entry_rules": {
                "require_trend_confirmation": false,
                "min_signal_strength": 0.5
            },
            "exit_rules": {
                "trailing_stop_atr": 1.5,
                "take_profit_atr": 4.0
            }
        },
        "conservative": {
            "entry_rules": {
                "require_trend_confirmation": true,
                "min_signal_strength": 0.8,
                "adx_filter": 25
            },
            "exit_rules": {
                "trailing_stop_atr": 3.0,
                "take_profit_atr": 2.0
            }
        }
    }
}
```

### ML Configuration (`config/ml_config.json`)

Machine learning model settings:

```json
{
    "model_path": "models/logistic_model.joblib",
    "feature_config": {
        "technical_features": [
            "sma_20", "sma_50", "ema_12", "ema_26",
            "rsi_14", "macd", "macd_signal",
            "bb_upper", "bb_lower", "atr_14"
        ],
        "price_features": [
            "returns_1d", "returns_5d", "returns_20d",
            "volatility_20d"
        ],
        "volume_features": [
            "volume_ratio", "volume_sma_20"
        ]
    },
    "prediction": {
        "buy_threshold": 0.52,
        "sell_threshold": 0.48,
        "confidence_filter": 0.6
    },
    "training": {
        "test_size": 0.2,
        "cv_folds": 5,
        "max_iter": 1000
    }
}
```

---

## Runtime Configuration

### Backtester Configuration

```python
from core.backtester import Backtester

bt = Backtester(
    data=data,
    initial_capital=10000,      # Starting capital
    transaction_cost=0.001,     # 0.1% per trade
    risk_free_rate=0.02         # 2% annual for Sharpe
)
```

### Position Sizer Configuration

```python
from core.position_sizer import DynamicPositionSizer

sizer = DynamicPositionSizer(
    risk_per_trade=0.02,        # Risk 2% per trade
    max_position_pct=0.20,      # Max 20% in one position
    capital=10000,              # Current capital

    # Optional advanced settings
    min_position_size=100,      # Minimum shares
    round_to_lot=True,          # Round to 100-share lots
    volatility_adjust=True      # Adjust for volatility
)
```

### Drawdown Monitor Configuration

```python
from core.drawdown_monitor import DrawdownMonitor

monitor = DrawdownMonitor(
    max_drawdown=0.15,          # 15% max drawdown
    daily_drawdown=0.05,        # 5% daily limit
    cooldown_period=300,        # 5 min cooldown after unlock
    per_symbol_limit=0.03       # 3% per-symbol limit
)
```

---

## Strategy Parameters

### SMA Strategy

```python
params = {
    'fast': 10,     # Fast SMA period (default: 10)
    'slow': 30      # Slow SMA period (default: 30)
}
```

### EMA Strategy

```python
params = {
    'short_window': 12,    # Short EMA period (default: 20)
    'long_window': 26      # Long EMA period (default: 50)
}
```

### MACD Strategy

```python
params = {
    'fast_window': 12,     # Fast EMA (default: 12)
    'slow_window': 26,     # Slow EMA (default: 26)
    'signal_window': 9     # Signal line (default: 9)
}
```

### RSI Strategy

```python
params = {
    'window': 14,          # RSI period (default: 14)
    'oversold': 30,        # Buy threshold (default: 30)
    'overbought': 70       # Sell threshold (default: 70)
}
```

### Bollinger Strategy

```python
params = {
    'window': 20,          # MA period (default: 20)
    'num_std': 2           # Standard deviations (default: 2)
}
```

### Momentum Strategy

```python
params = {
    'lookback': 20         # Comparison period (default: 20)
}
```

### Mean Reversion Strategy

```python
params = {
    'window': 14,          # Lookback period (default: 14)
    'threshold': 1.0       # Z-score threshold (default: 1.0)
}
```

### ADX Strategy

```python
params = {
    'window': 14,          # ADX period (default: 14)
    'threshold': 25        # Min ADX for signal (default: 25)
}
```

### Stochastic Strategy

```python
params = {
    'k_window': 14,        # %K period (default: 14)
    'd_window': 3,         # %D smoothing (default: 3)
    'oversold': 20,        # Buy threshold (default: 20)
    'overbought': 80       # Sell threshold (default: 80)
}
```

---

## Logging Configuration

### Logger Setup

```python
from loggers.logger import Logger

logger = Logger(
    log_file="trading.log",
    logger_name="Trader",
    log_dir="logs/",
    level="INFO"
).get_logger()
```

### Log Levels

| Level | Use Case |
|-------|----------|
| DEBUG | Detailed debugging info |
| INFO | General operational info |
| WARNING | Something unexpected |
| ERROR | Error occurred |
| CRITICAL | System failure |

### Log Rotation

Logs are automatically rotated:
- Max size: 10 MB
- Keep: 5 backup files

---

## Database Configuration

### DataStore Setup

```python
from data.datastorage import DataStore

# With context manager (recommended)
with DataStore(db_path="data/trading.db") as ds:
    ds.fill_database("AAPL", df)
    data = ds.read_data("AAPL")

# Manual connection
ds = DataStore(db_path="data/trading.db")
ds.open_db()
# ... operations ...
ds.close_db()
```

### Database Location

Default: `data/trading.db`

Override with environment variable:
```bash
DATABASE_PATH=/custom/path/trading.db
```

---

## Broker Configuration

### Schwab

```python
from data.streaming.schwab_client import SchwabClient

client = SchwabClient(
    client_id=os.getenv('SCHWAB_CLIENT_ID'),
    client_secret=os.getenv('SCHWAB_CLIENT_SECRET'),
    redirect_uri=os.getenv('SCHWAB_REDIRECT_URI')
)
```

### Alpaca

```python
from core.broker.alpaca_broker import AlpacaBroker

broker = AlpacaBroker(
    api_key=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL')
)
```

### Mock Broker (Paper Trading)

```python
from core.broker.mock_broker import MockBroker

broker = MockBroker(
    initial_cash=10000,
    slippage=0.001
)
```

---

## Environment-Specific Config

### Development

```bash
# .env.development
LOG_LEVEL=DEBUG
SLIPPAGE_MODEL=fixed
DEFAULT_SLIPPAGE_PCT=0
TRANSACTION_COST=0
```

### Paper Trading

```bash
# .env.paper
LOG_LEVEL=INFO
ALPACA_BASE_URL=https://paper-api.alpaca.markets
SLIPPAGE_MODEL=volume_based
```

### Production

```bash
# .env.production
LOG_LEVEL=WARNING
ALPACA_BASE_URL=https://api.alpaca.markets
SLIPPAGE_MODEL=volatility
MAX_DAILY_DRAWDOWN=0.03
MAX_TOTAL_DRAWDOWN=0.10
```

---

## Validation

### Validate Configuration

```python
def validate_config():
    """Validate all configuration settings."""
    import os
    from pathlib import Path

    errors = []

    # Check required env vars
    required = ['SCHWAB_CLIENT_ID', 'SCHWAB_CLIENT_SECRET']
    for var in required:
        if not os.getenv(var):
            errors.append(f"Missing {var}")

    # Check config files exist
    config_files = [
        'config/strategy_routing.json',
        'config/trade_logic_routing.json'
    ]
    for f in config_files:
        if not Path(f).exists():
            errors.append(f"Missing config file: {f}")

    # Validate numeric ranges
    risk = float(os.getenv('DEFAULT_RISK_PER_TRADE', 0.02))
    if not 0 < risk < 0.1:
        errors.append(f"Invalid risk per trade: {risk}")

    if errors:
        raise ValueError(f"Configuration errors: {errors}")

    print("Configuration valid!")
```
