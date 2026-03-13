# ML Training Data Workflow

This document describes the workflow for generating training data for the meta-model (trade quality predictor).

## Overview

```
Historical Data → Simulation → Meta-Trade Logs → Parquet → ML Training
     (JSON)         (Python)       (JSONL)        (Parquet)   (Model)
```

## Step 1: Run Historical Simulation

Run simulations using real historical market data to generate realistic trading data.

### Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with default settings (AAPL, MSFT, NVDA, 500 bars)
python tools/run_historical_sim.py

# Run with custom symbols and bars
python tools/run_historical_sim.py --symbols AAPL GOOGL AMD --bars 2000

# Run with higher slippage for conservative estimates
python tools/run_historical_sim.py --slippage 0.002
```

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `--symbols` | AAPL MSFT NVDA | Symbols to simulate |
| `--bars` | 500 | Number of bars to simulate |
| `--data-path` | data/data_storage/proc_data | Path to historical data (use proc_data for indicators) |
| `--start-index` | 50 | Starting bar index (skip warmup) |
| `--slippage` | 0.001 | Slippage (0.001 = 0.1% = 10 bps) |
| `--commission` | 0.0 | Commission per trade ($) |
| `--cash` | 100000 | Starting cash |
| `--output` | meta_trades_historical.jsonl | Output filename |

### Output

Simulation generates `logs/<output>.jsonl` containing:
- Entry events with 17 features
- Exit events with outcome metrics

## Step 2: Convert to Parquet

Convert JSONL logs to Parquet format for efficient ML training.

```bash
# Basic conversion
python tools/convert_meta_trades.py \
    --input logs/meta_trades_historical.jsonl \
    --output data/trades_for_training.parquet \
    --stats

# Filter to complete trades only
python tools/convert_meta_trades.py \
    --input logs/meta_trades_historical.jsonl \
    --output data/trades_complete.parquet \
    --complete-only \
    --stats
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `trade_id` | str | Unique trade identifier |
| `entry_timestamp` | datetime | Entry time |
| `exit_timestamp` | datetime | Exit time |
| `symbol` | category | Ticker symbol |
| `side` | category | buy/sell |
| `qty` | int | Position size |
| `entry_price` | float | Entry fill price |
| `exit_price` | float | Exit fill price |
| `feat_strategy` | category | Strategy name |
| `feat_regime` | category | Market regime |
| `feat_atr` | float | ATR value |
| `feat_atr_percentile` | float | ATR percentile (0-1) |
| `feat_drawdown_portfolio_pct` | float | Portfolio drawdown |
| `feat_drawdown_symbol_pct` | float | Symbol drawdown |
| `feat_position_size_pct` | float | Position size % |
| `feat_hour_of_day` | int | Hour (0-23) |
| `feat_day_of_week` | int | Day (0=Mon, 6=Sun) |
| `feat_minutes_since_open` | int | Minutes since market open |
| `feat_bars_in_regime` | int | Bars in current regime |
| `feat_hours_since_last_trade` | float | Time since last trade |
| `feat_signal_strength` | int | Signal strength |
| `out_pnl_dollars` | float | P&L in dollars |
| `out_pnl_percent` | float | P&L percentage |
| `out_hold_bars` | int | Holding period in bars |
| `out_mae_percent` | float | Max adverse excursion |
| `out_mfe_percent` | float | Max favorable excursion |
| `out_exit_reason` | category | Exit reason |

## Step 3: Train Model

Load the Parquet file in pandas/sklearn for model training.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load training data
df = pd.read_parquet('data/trades_for_training.parquet')

# Filter to complete trades
df = df[df['is_complete']]

# Create target variable (profitable trade)
df['is_profitable'] = df['out_pnl_dollars'] > 0

# Select features
feature_cols = [
    'feat_atr_percentile',
    'feat_drawdown_portfolio_pct',
    'feat_position_size_pct',
    'feat_hour_of_day',
    'feat_day_of_week',
    'feat_bars_in_regime',
]

X = df[feature_cols]
y = df['is_profitable']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

## Step 4: Validate with Paper Trading

1. Enable meta-logging in live trading config
2. Run paper trading for 1-2 weeks
3. Compare real trade outcomes to simulation predictions
4. Retrain model with real data

## Data Sources

### GBM Simulation (Synthetic)
- Fast, unlimited data
- May not capture real market dynamics
- Good for testing pipeline

### Historical Data (Real)
- Realistic price movements
- Limited to available data
- Better for training

Available historical data:
- `data/data_storage/proc_data/`: **Use this** - Processed data with indicators (SMA, EMA, RSI, ATR, etc.)
- `data/data_storage/raw_data/`: Raw candle data (468 symbols, no indicators)

### Data Pipeline Behavior

The `UnifiedDataPipeline` automatically processes ALL raw data when updating:
- Fetches new bars and appends to raw_data file
- Checks if processed data is current (compares timestamps)
- If out-of-sync: reprocesses the COMPLETE raw_data file (not just new bars)
- If current: skips reprocessing to save time
- This ensures indicators like SMA_200 are computed correctly

```bash
# Update data (smart reprocessing - skips if current)
python -m core.unified_data_pipeline --symbols AAPL MSFT --days 5

# Force reprocessing even if current
python -m core.unified_data_pipeline --symbols AAPL MSFT --days 5 --force-reprocess
```

### Manual Reprocessing (if needed)

If proc_data files need to be rebuilt manually:

```bash
# Reprocess specific symbols
python tools/reprocess_raw_data.py --symbols AAPL MSFT NVDA

# Reprocess all symbols with raw data
python tools/reprocess_raw_data.py --all
```

## Realism Settings

The simulation includes realism parameters:

| Setting | Default | Description |
|---------|---------|-------------|
| `slippage` | 0.001 | 0.1% slippage (10 bps) |
| `commission` | 0.0 | Commission per trade |

### Slippage Guidelines

| Scenario | Slippage |
|----------|----------|
| Large cap, small orders | 0.0005 (5 bps) |
| Normal trading | 0.001 (10 bps) |
| Volatile markets | 0.002 (20 bps) |
| Illiquid stocks | 0.003+ |

## Example Workflow

```bash
# 1. Generate training data (2000 bars across 5 symbols)
python tools/run_historical_sim.py \
    --symbols AAPL MSFT NVDA AMD GOOGL \
    --bars 2000 \
    --slippage 0.001 \
    --output meta_trades_train.jsonl

# 2. Convert to Parquet
python tools/convert_meta_trades.py \
    --input logs/meta_trades_train.jsonl \
    --output data/trades_train.parquet \
    --complete-only \
    --stats

# 3. Verify data
python -c "
import pandas as pd
df = pd.read_parquet('data/trades_train.parquet')
print(f'Trades: {len(df)}')
print(f'Win Rate: {(df.out_pnl_dollars > 0).mean():.1%}')
print(f'Profit Factor: {df[df.out_pnl_dollars > 0].out_pnl_dollars.sum() / abs(df[df.out_pnl_dollars < 0].out_pnl_dollars.sum()):.2f}')
"
```

## Troubleshooting

### No trades generated
- Check that historical data exists for symbols
- Increase `--bars` parameter
- Check strategy signals are being generated

### Zero P&L values
- Verify entry_price is being set on trade entry
- Check that exit events are matching entry events

### Missing features
- Some features require warmup period
- Increase `--start-index` to skip initial bars
