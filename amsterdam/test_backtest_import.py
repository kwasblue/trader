#!/usr/bin/env python3
"""Test that AdaptiveBacktester works correctly after import."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from tools.backtest_adaptive_features import AdaptiveBacktester
from core.unified_data_pipeline import UnifiedDataPipeline

# Load data
pipeline = UnifiedDataPipeline()
bars = pipeline.get_data('AAPL', timeframe='day')

print(f'Loaded {len(bars)} bars')
print(f'Columns: {list(bars.columns)[:10]}...')

# Create backtester
backtester = AdaptiveBacktester('AAPL', bars.tail(800))

print(f'\nbars_with_caps shape: {backtester.bars_with_caps.shape}')
print(f'Has Close: {"Close" in backtester.bars_with_caps.columns}')
print(f'Has close: {"close" in backtester.bars_with_caps.columns}')

if "close" not in backtester.bars_with_caps.columns:
    print('\n❌ ERROR: lowercase columns not added!')
    sys.exit(1)

print('\n✓ Lowercase columns present, running backtest...')

# Run backtest
metrics = backtester.run_backtest(
    regime_method='unstable',
    sizing_method='fixed'
)

print(f'\nTrades: {metrics.num_trades}')
print(f'Sharpe: {metrics.sharpe_ratio:.2f}')
print(f'Return: {metrics.total_return:.2f}%')

if metrics.num_trades > 0:
    print('\n✓ SUCCESS: Backtest generated trades!')
else:
    print('\n❌ ERROR: Backtest generated 0 trades!')
    sys.exit(1)
