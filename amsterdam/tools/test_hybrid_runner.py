#!/usr/bin/env python3
"""
Test the AlpacaSchwabHybridRunner.

Usage:
    python tools/test_hybrid_runner.py [SYMBOLS...]

    # Default: AAPL, SPY
    python tools/test_hybrid_runner.py

    # Custom symbols
    python tools/test_hybrid_runner.py TSLA NVDA
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


async def main():
    from core.runner_factory import RunnerFactory
    from core.config_loader import get_config

    # Get symbols from args or use defaults
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "SPY"]
    symbols = [s.upper() for s in symbols]

    print("=" * 60)
    print("  HYBRID RUNNER TEST")
    print("  Schwab Data + Alpaca Execution")
    print("=" * 60)
    print(f"  Symbols: {symbols}")
    print(f"  Time: {datetime.now()}")
    print("=" * 60)
    print()

    # Check credentials
    alpaca_key = os.getenv("ALPACA_API_KEY")
    schwab_key = os.getenv("SCHWAB_API_KEY")

    if not alpaca_key:
        print("ERROR: ALPACA_API_KEY not set")
        sys.exit(1)
    if not schwab_key:
        print("ERROR: SCHWAB_API_KEY not set")
        sys.exit(1)

    print("✓ Alpaca credentials found")
    print("✓ Schwab credentials found")
    print()

    # Get config
    config = get_config()

    # Show available brokers
    print(f"Available brokers: {RunnerFactory.available_brokers()}")
    print()

    # Create hybrid runner
    print("Creating hybrid runner...")
    try:
        runner = RunnerFactory.create("hybrid", symbols=symbols, config=config)
        print(f"✓ Created: {runner.__class__.__name__}")
    except Exception as e:
        print(f"✗ Failed to create runner: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("Starting runner (will run for 60 seconds)...")
    print("Press Ctrl+C to stop early")
    print("-" * 60)

    try:
        # Run with timeout
        await asyncio.wait_for(runner.run(), timeout=60)
    except asyncio.TimeoutError:
        print("\n[Timeout reached - stopping]")
    except KeyboardInterrupt:
        print("\n[Interrupted - stopping]")
    except Exception as e:
        print(f"\n[Error: {e}]")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
