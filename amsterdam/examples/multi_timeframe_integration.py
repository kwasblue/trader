# examples/multi_timeframe_integration.py
"""
Multi-Timeframe Integration Example

Demonstrates how to integrate the multi-timeframe system into the Amsterdam
trading platform for live trading.

This is a reference implementation showing:
1. Configuring BarAggregator from strategy routing
2. Connecting to Schwab websocket stream
3. Handling aggregated bars with strategies
4. Managing regime changes with timeframe switches
5. Market close handling
"""

import asyncio
import json
from datetime import datetime

from core.bar_aggregator import Bar, BarAggregator
from core.logic.strategy_routing_manager import StrategyRoutingManager
from loggers.logger import Logger


class MultiTimeframeTrader:
    """
    Multi-timeframe trading system integration.

    Manages:
    - Bar aggregation at multiple timeframes
    - Strategy routing with timeframe configuration
    - Regime-based timeframe switching
    - Live trading execution
    """

    def __init__(self, symbols: list[str], routing_config_path: str = "config/strategy_routing.json"):
        """
        Initialize multi-timeframe trader.

        Args:
            symbols: List of symbols to trade
            routing_config_path: Path to strategy routing configuration
        """
        self.symbols = symbols
        self.logger = Logger("multi_timeframe_trader.log", "MultiTimeframeTrader", propagate=True).get_logger()

        # Initialize components
        self.bar_aggregator = BarAggregator()
        self.routing_manager = StrategyRoutingManager(routing_config_path)

        # Track current regimes
        self.current_regimes = {}

        # Statistics
        self.stats = {
            "bars_processed": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "regime_changes": 0,
        }

        self.logger.info(f"Initialized MultiTimeframeTrader for {len(symbols)} symbols")

    def configure_timeframes(self, regime_detector=None):
        """
        Configure BarAggregator timeframes from strategy routing config.

        Args:
            regime_detector: Optional RegimeDetector instance
                           If None, uses 'normal' regime for all symbols
        """
        self.logger.info("Configuring timeframes from strategy routing...")

        for symbol in self.symbols:
            # Determine regime
            if regime_detector:
                regime = regime_detector.get_current_regime(symbol)
            else:
                regime = "normal"  # Default regime

            # Get routing decision
            routing = self.routing_manager.get_routing(symbol, regime)

            # Configure aggregator
            self.bar_aggregator.set_timeframe(symbol, routing["timeframe"])

            # Track regime
            self.current_regimes[symbol] = regime

            self.logger.info(
                f"[{symbol}] Configured: regime={regime}, "
                f"strategy={routing['strategy']}, timeframe={routing['timeframe']}"
            )

    def on_aggregated_bar(self, bar: Bar):
        """
        Handle completed aggregated bars.

        This is the main callback that receives bars at the configured
        timeframe and generates trading signals.

        Args:
            bar: Completed aggregated bar
        """
        try:
            # Get current regime for this symbol
            regime = self.current_regimes.get(bar.symbol, "normal")

            # Get strategy instance
            strategy = self.routing_manager.get_strategy(bar.symbol, regime)

            self.logger.debug(f"[{bar.symbol}] Processing {bar.timeframe} bar at {bar.timestamp.strftime('%H:%M:%S')}")

            # Generate signal
            # Note: Strategy needs to be adapted to work with Bar objects
            # or we convert Bar to DataFrame format
            signal = self._generate_signal(strategy, bar)

            if signal:
                self.stats["signals_generated"] += 1
                self.logger.info(f"[{bar.symbol}] Signal: {signal['action']} @ {bar.close}")

                # Execute trade
                success = self._execute_trade(bar.symbol, signal)
                if success:
                    self.stats["trades_executed"] += 1

        except Exception as e:
            self.logger.exception(f"Error processing bar for {bar.symbol}: {e}")

    def _generate_signal(self, strategy, bar: Bar):
        """
        Generate trading signal from strategy.

        Adapts Bar object to strategy's expected format.

        Args:
            strategy: Strategy instance
            bar: Aggregated bar

        Returns:
            Signal dict or None
        """
        # Convert Bar to format strategy expects
        # Most strategies expect DataFrame with OHLCV columns

        # For demonstration, we'll use a simple signal logic
        # In production, integrate with actual strategy.generate_signal()

        # Example: Simple momentum signal
        if hasattr(strategy, "generate_signal_from_bar"):
            return strategy.generate_signal_from_bar(bar)
        else:
            # Fallback: convert to single-row DataFrame
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                    }
                ]
            )
            df.set_index("timestamp", inplace=True)

            # Call strategy's generate_signal with DataFrame
            # (This assumes strategy can work with single-row data)
            return strategy.generate_signal(df)

    def _execute_trade(self, symbol: str, signal: dict) -> bool:
        """
        Execute trade based on signal.

        Args:
            symbol: Stock symbol
            signal: Signal dict with action, size, etc.

        Returns:
            True if successful
        """
        # TODO: Integrate with actual broker (Alpaca)
        # For now, just log

        self.logger.info(f"[{symbol}] EXECUTE: {signal.get('action')} {signal.get('size', 'N/A')} shares")

        return True  # Placeholder

    def check_regime_changes(self, regime_detector):
        """
        Check for regime changes and update timeframes accordingly.

        Args:
            regime_detector: RegimeDetector instance
        """
        for symbol in self.symbols:
            old_regime = self.current_regimes.get(symbol)
            new_regime = regime_detector.detect_regime(symbol)

            if new_regime != old_regime:
                self._handle_regime_change(symbol, old_regime, new_regime)

    def _handle_regime_change(self, symbol: str, old_regime: str, new_regime: str):
        """
        Handle regime change for a symbol.

        Updates timeframe and strategy routing.

        Args:
            symbol: Stock symbol
            old_regime: Previous regime
            new_regime: New regime
        """
        self.logger.info(f"[{symbol}] Regime change: {old_regime} → {new_regime}")

        # Get new routing
        new_routing = self.routing_manager.get_routing(symbol, new_regime)

        # Get old timeframe for comparison
        old_timeframe = self.bar_aggregator.get_timeframe(symbol)

        # Update aggregator (will complete partial window if timeframe changes)
        self.bar_aggregator.set_timeframe(symbol, new_routing["timeframe"])

        if old_timeframe != new_routing["timeframe"]:
            self.logger.info(f"[{symbol}] Timeframe changed: {old_timeframe} → {new_routing['timeframe']}")

        self.logger.info(f"[{symbol}] New strategy: {new_routing['strategy']}")

        # Update tracking
        self.current_regimes[symbol] = new_regime
        self.stats["regime_changes"] += 1

    def on_market_close(self):
        """
        Handle market close event.

        Completes all partial windows and prepares for next trading day.
        """
        self.logger.info("Market close - completing all windows")

        # Force complete all partial windows
        self.bar_aggregator.force_complete_all()

        # Log statistics
        self.logger.info(
            f"Daily stats: bars_processed={self.stats['bars_processed']}, "
            f"signals={self.stats['signals_generated']}, "
            f"trades={self.stats['trades_executed']}, "
            f"regime_changes={self.stats['regime_changes']}"
        )

    def get_statistics(self) -> dict:
        """
        Get trading statistics.

        Returns:
            Dict with statistics
        """
        aggregator_stats = self.bar_aggregator.get_stats()

        return {
            **self.stats,
            "aggregator": aggregator_stats,
            "active_symbols": len(self.symbols),
            "current_regimes": self.current_regimes.copy(),
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


async def main():
    """Example integration with Schwab websocket stream."""
    # Trading symbols
    symbols = ["AAPL", "TSLA", "MSFT", "NVDA", "AMD"]

    # Initialize trader
    trader = MultiTimeframeTrader(symbols=symbols, routing_config_path="config/strategy_routing.json")

    # Configure initial timeframes
    # Note: In production, pass actual RegimeDetector
    trader.configure_timeframes(regime_detector=None)

    # Register bar callback
    trader.bar_aggregator.register_callback(trader.on_aggregated_bar)

    print("=" * 60)
    print("Multi-Timeframe Trading System")
    print("=" * 60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframes: {trader.bar_aggregator.get_stats()['timeframes']}")
    print("=" * 60)

    # Simulate receiving bars from Schwab websocket
    # In production, this would be schwab_client.set_quote_callback()

    print("\nSimulating market data stream...\n")

    base_time = datetime(2026, 3, 14, 9, 30, 0)

    # Send 30 minutes of 1-minute bars
    for i in range(30):
        for symbol in symbols:
            # Create mock bar
            bar = Bar(
                timestamp=base_time + timedelta(minutes=i),
                open=150.0 + i * 0.1,
                high=151.0 + i * 0.1,
                low=149.0 + i * 0.1,
                close=150.5 + i * 0.1,
                volume=10000,
                symbol=symbol,
                timeframe="1min",  # Raw incoming data
            )

            # Process through aggregator
            trader.bar_aggregator.process_bar(bar)
            trader.stats["bars_processed"] += 1

        # Simulate regime check every 5 minutes
        if i % 5 == 0 and i > 0:
            print(f"\n[{base_time + timedelta(minutes=i)}] Checking regimes...")
            # trader.check_regime_changes(regime_detector)

    # Market close
    print("\n" + "=" * 60)
    print("Market Close")
    print("=" * 60)
    trader.on_market_close()

    # Show final stats
    print("\nFinal Statistics:")
    stats = trader.get_statistics()
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    from datetime import timedelta

    # Run example
    asyncio.run(main())
