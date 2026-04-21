#!/usr/bin/env python3
"""
Test ensemble voting functionality.

Usage:
    python tools/test_ensemble_voting.py
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def create_test_data(n_bars: int = 300, trend: str = "up") -> pd.DataFrame:
    """Create synthetic price data for testing."""
    np.random.seed(42)

    if trend == "up":
        close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5 + 0.1)
    elif trend == "down":
        close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5 - 0.1)
    else:
        close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)

    return pd.DataFrame({
        "Open": close + np.random.randn(n_bars) * 0.2,
        "High": close + np.abs(np.random.randn(n_bars) * 0.3),
        "Low": close - np.abs(np.random.randn(n_bars) * 0.3),
        "Close": close,
        "Volume": np.random.randint(10000, 100000, n_bars),
    })


def test_ensemble_voter():
    """Test the EnsembleVoter class."""
    from core.logic.ensemble_voter import EnsembleVoter, VoteResult

    print("=" * 60)
    print("  ENSEMBLE VOTING TEST")
    print("=" * 60)
    print()

    # Test each voting mode
    modes = ["majority", "unanimous", "weighted", "any"]

    for mode in modes:
        print(f"\n--- Testing {mode.upper()} mode ---")

        voter = EnsembleVoter(
            strategies=["sma", "rsi", "macd", "momentum"],
            mode=mode,
            threshold=0.5,
        )

        print(f"Loaded strategies: {list(voter._strategies.keys())}")

        # Test with uptrending data
        df_up = create_test_data(300, "up")
        result_up = voter.vote(df_up)
        print(f"\nUptrend data:")
        print(f"  Signal: {result_up.signal} ({['SELL', 'HOLD', 'BUY'][result_up.signal + 1]})")
        print(f"  Confidence: {result_up.confidence:.2f}")
        print(f"  Details: {result_up.details}")
        print(f"  Individual signals: {result_up.strategy_signals}")

        # Test with downtrending data
        df_down = create_test_data(300, "down")
        result_down = voter.vote(df_down)
        print(f"\nDowntrend data:")
        print(f"  Signal: {result_down.signal} ({['SELL', 'HOLD', 'BUY'][result_down.signal + 1]})")
        print(f"  Confidence: {result_down.confidence:.2f}")
        print(f"  Details: {result_down.details}")
        print(f"  Individual signals: {result_down.strategy_signals}")


def test_config_loading():
    """Test that streaming config loads correctly."""
    from core.config_loader import get_config

    print("\n" + "=" * 60)
    print("  CONFIG LOADING TEST")
    print("=" * 60)

    config = get_config()

    print(f"\nStreaming config:")
    print(f"  bar_interval_ms: {config.streaming.bar_interval_ms}")
    print(f"  ensemble_enabled: {config.streaming.ensemble_enabled}")
    print(f"  ensemble_strategies: {config.streaming.ensemble_strategies}")
    print(f"  ensemble_mode: {config.streaming.ensemble_mode}")
    print(f"  ensemble_weights: {config.streaming.ensemble_weights}")
    print(f"  ensemble_threshold: {config.streaming.ensemble_threshold}")


def test_bar_interval_config():
    """Test that bar interval is properly configurable."""
    print("\n" + "=" * 60)
    print("  BAR INTERVAL TEST")
    print("=" * 60)

    from datetime import datetime, timezone
    from core.config_loader import get_config

    config = get_config()

    # Test different intervals
    intervals_ms = [100, 250, 500, 1000, 60000]

    print(f"\nCurrent config interval: {config.streaming.bar_interval_ms}ms")
    print(f"Equivalent to: {config.streaming.bar_interval_ms / 1000}s bars")
    print()

    # Simulate bar bucketing
    now = datetime.now(timezone.utc)
    print("Bar bucket IDs for different intervals:")
    for interval_ms in intervals_ms:
        interval_sec = interval_ms / 1000
        bucket_id = int(now.timestamp() // interval_sec)
        bars_per_minute = 60000 / interval_ms
        print(f"  {interval_ms}ms ({interval_sec}s): bucket={bucket_id}, bars/min={bars_per_minute:.1f}")


def test_ensemble_from_config():
    """Test creating ensemble voter from config."""
    from core.logic.ensemble_voter import create_ensemble_voter_from_config
    from core.config_loader import get_config

    print("\n" + "=" * 60)
    print("  ENSEMBLE FROM CONFIG TEST")
    print("=" * 60)

    config = get_config()

    voter = create_ensemble_voter_from_config(config)

    if voter is None:
        print("\nEnsemble voting is DISABLED in config")
        print("To enable, set streaming.ensemble_enabled = true")
    else:
        print(f"\nEnsemble voter created from config:")
        print(f"  Mode: {voter.mode}")
        print(f"  Strategies: {voter.strategy_names}")
        print(f"  Threshold: {voter.threshold}")
        print(f"  Weights: {voter.weights}")


def main():
    print("Testing Ensemble Voting System")
    print()

    try:
        test_config_loading()
    except Exception as e:
        print(f"Config loading test failed: {e}")

    try:
        test_bar_interval_config()
    except Exception as e:
        print(f"Bar interval test failed: {e}")

    try:
        test_ensemble_voter()
    except Exception as e:
        print(f"Ensemble voter test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_ensemble_from_config()
    except Exception as e:
        print(f"Ensemble from config test failed: {e}")

    print("\n" + "=" * 60)
    print("  TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
