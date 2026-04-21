#!/usr/bin/env python3
"""
Quick test script for Schwab WebSocket streaming.

Usage:
    python tools/test_schwab_stream.py [SYMBOLS...]

    # Default: AAPL, MSFT, SPY
    python tools/test_schwab_stream.py

    # Custom symbols
    python tools/test_schwab_stream.py TSLA NVDA AMD
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
    from data.streaming.streamer import SchwabStreamingClient

    # Get API credentials
    api_key = os.getenv("SCHWAB_API_KEY")
    secret_key = os.getenv("SCHWAB_SECRET")

    if not api_key or not secret_key:
        print("Error: SCHWAB_API_KEY and SCHWAB_SECRET must be set in .env")
        sys.exit(1)

    # Get symbols from args or use defaults
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "MSFT", "SPY"]
    symbols = [s.upper() for s in symbols]

    print(f"Testing Schwab WebSocket stream")
    print(f"Symbols: {symbols}")
    print(f"Time: {datetime.now()}")
    print("-" * 60)

    # Create client
    client = SchwabStreamingClient(api_key, secret_key)

    # Track quotes received
    quote_count = 0
    start_time = datetime.now()

    async def on_quote(symbol: str, quote: dict):
        nonlocal quote_count
        quote_count += 1

        last = quote.get("last_price", "N/A")
        bid = quote.get("bid_price", "N/A")
        ask = quote.get("ask_price", "N/A")
        vol = quote.get("volume", "N/A")

        # Format volume with commas
        if isinstance(vol, (int, float)):
            vol = f"{int(vol):,}"

        print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol:6} | "
              f"Last: ${last:>8} | Bid: ${bid:>8} | Ask: ${ask:>8} | Vol: {vol:>12}")

    # Set callback
    client.set_quote_callback(on_quote)

    # Run for 30 seconds
    print("\nConnecting to Schwab WebSocket...")
    print("Will run for 30 seconds (Ctrl+C to stop early)\n")

    try:
        # Create task for streaming
        stream_task = asyncio.create_task(client.run(symbols))

        # Wait 30 seconds
        await asyncio.sleep(30)

        # Stop client
        await client.stop()
        stream_task.cancel()

    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("\n\nStopping...")
        await client.stop()

    # Print summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "-" * 60)
    print(f"Summary:")
    print(f"  Duration: {elapsed:.1f} seconds")
    print(f"  Quotes received: {quote_count}")
    print(f"  Rate: {quote_count / elapsed:.1f} quotes/second" if elapsed > 0 else "")


if __name__ == "__main__":
    asyncio.run(main())
