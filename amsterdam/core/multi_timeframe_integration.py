"""
Multi-Timeframe Trading Integration Module

This module provides a drop-in integration layer for the multi-timeframe
bar aggregation system with the existing Amsterdam trading system.

Usage in autoamsterdam.py or runner:
    from core.multi_timeframe_integration import MultiTimeframeManager

    # Initialize
    mtf_manager = MultiTimeframeManager(
        symbols=['AAPL', 'TSLA', 'MSFT'],
        routing_config_path='config/strategy_routing.json'
    )

    # In websocket callback
    def on_websocket_data(raw_data):
        bar = mtf_manager.process_websocket_data(raw_data)
        if bar:
            # Aggregated bar ready - pass to strategy
            signal = strategy.process_bar(bar)

    # On regime change
    mtf_manager.update_regime('AAPL', new_regime)

    # At market close
    mtf_manager.force_complete_all()
"""

from __future__ import annotations

from typing import Dict, Optional, List, Callable, Any
from datetime import datetime
from pathlib import Path

from core.bar_aggregator import BarAggregator, Bar
from core.logic.strategy_routing_manager import StrategyRoutingManager
from loggers.logger import Logger


class MultiTimeframeManager:
    """
    Integration layer between existing trading system and multi-timeframe aggregation.

    Responsibilities:
    - Configure BarAggregator from strategy routing
    - Convert websocket data to Bar objects
    - Handle regime changes with timeframe updates
    - Provide statistics and monitoring
    """

    def __init__(
        self,
        symbols: List[str],
        routing_config_path: str = "config/strategy_routing.json",
        initial_regime: str = "normal"
    ):
        """
        Initialize multi-timeframe manager.

        Args:
            symbols: List of symbols to trade
            routing_config_path: Path to strategy routing config
            initial_regime: Initial regime for all symbols (default: 'normal')
        """
        self.symbols = symbols
        self.initial_regime = initial_regime

        # Initialize logger
        self.logger = Logger(
            "multi_timeframe_manager.log",
            "MultiTimeframeManager",
            propagate=True,
            level=10  # DEBUG
        ).get_logger()

        # Initialize components
        self.aggregator = BarAggregator()
        self.routing_manager = StrategyRoutingManager(routing_config_path)

        # Track current regimes
        self.current_regimes: Dict[str, str] = {}

        # Callbacks for aggregated bars
        self.bar_callbacks: List[Callable[[Bar], None]] = []

        # Initialize timeframes
        self._configure_initial_timeframes()

        self.logger.info(
            f"MultiTimeframeManager initialized for {len(symbols)} symbols"
        )

    def _configure_initial_timeframes(self):
        """Configure aggregator timeframes for all symbols at startup."""
        self.logger.info("Configuring initial timeframes...")

        for symbol in self.symbols:
            regime = self.initial_regime
            routing = self.routing_manager.get_routing(symbol, regime)

            # Configure aggregator
            self.aggregator.set_timeframe(symbol, routing['timeframe'])

            # Track regime
            self.current_regimes[symbol] = regime

            self.logger.info(
                f"  [{symbol}] regime={regime}, strategy={routing['strategy']}, "
                f"timeframe={routing['timeframe']}"
            )

    def register_bar_callback(self, callback: Callable[[Bar], None]):
        """
        Register callback to receive completed aggregated bars.

        Args:
            callback: Function to call with completed bars
                     Signature: callback(bar: Bar) -> None

        Example:
            def on_bar(bar: Bar):
                strategy = get_strategy(bar.symbol)
                signal = strategy.process_bar(bar)
                if signal:
                    execute_trade(signal)

            manager.register_bar_callback(on_bar)
        """
        # Register with aggregator
        self.aggregator.register_callback(callback)
        self.bar_callbacks.append(callback)

        self.logger.info(f"Registered bar callback: {callback.__name__}")

    def process_websocket_data(self, raw_data: dict) -> Optional[List[Bar]]:
        """
        Process raw websocket data and return any completed aggregated bars.

        Args:
            raw_data: Raw data from websocket with keys:
                     - symbol: str
                     - timestamp: int (epoch ms) or datetime
                     - open, high, low, close: float
                     - volume: int

        Returns:
            List of completed aggregated bars (empty if no bars completed)

        Example:
            completed_bars = manager.process_websocket_data({
                'symbol': 'AAPL',
                'timestamp': 1710422400000,
                'open': 150.0,
                'high': 151.0,
                'low': 149.0,
                'close': 150.5,
                'volume': 10000
            })

            for bar in completed_bars:
                # Bar is aggregated to configured timeframe
                process_bar(bar)
        """
        try:
            # Convert raw data to Bar object
            bar = self._convert_to_bar(raw_data)

            # Process through aggregator
            completed_bars = self.aggregator.process_bar(bar)

            return completed_bars

        except Exception as e:
            self.logger.exception(f"Error processing websocket data: {e}")
            return []

    def _convert_to_bar(self, raw_data: dict) -> Bar:
        """
        Convert raw websocket data to Bar object.

        Args:
            raw_data: Raw websocket data dict

        Returns:
            Bar object
        """
        # Extract timestamp (handle both epoch ms and datetime)
        timestamp = raw_data.get('timestamp')
        if isinstance(timestamp, int):
            # Epoch milliseconds
            timestamp = datetime.fromtimestamp(timestamp / 1000)
        elif isinstance(timestamp, datetime):
            # Already datetime
            pass
        else:
            # Fallback to current time
            timestamp = datetime.now()

        return Bar(
            timestamp=timestamp,
            open=float(raw_data.get('open', 0)),
            high=float(raw_data.get('high', 0)),
            low=float(raw_data.get('low', 0)),
            close=float(raw_data.get('close', 0)),
            volume=int(raw_data.get('volume', 0)),
            symbol=raw_data.get('symbol', ''),
            timeframe='raw'  # Input timeframe (before aggregation)
        )

    def update_regime(self, symbol: str, new_regime: str):
        """
        Update regime for a symbol and switch timeframe if needed.

        This should be called when the regime detector detects a regime change.
        If the new regime requires a different timeframe, the aggregator will
        automatically complete the partial window before switching.

        Args:
            symbol: Stock symbol
            new_regime: New regime name

        Example:
            # In regime monitoring loop
            old_regime = current_regimes['AAPL']
            new_regime = regime_detector.detect_regime('AAPL')

            if new_regime != old_regime:
                manager.update_regime('AAPL', new_regime)
        """
        old_regime = self.current_regimes.get(symbol)

        if old_regime == new_regime:
            return  # No change

        self.logger.info(f"[{symbol}] Regime change: {old_regime} → {new_regime}")

        # Get new routing
        new_routing = self.routing_manager.get_routing(symbol, new_regime)

        # Get old timeframe for comparison
        old_timeframe = self.aggregator.get_timeframe(symbol)

        # Update aggregator (completes partial window if timeframe changes)
        self.aggregator.set_timeframe(symbol, new_routing['timeframe'])

        if old_timeframe != new_routing['timeframe']:
            self.logger.info(
                f"[{symbol}] Timeframe changed: {old_timeframe} → {new_routing['timeframe']}"
            )

        self.logger.info(
            f"[{symbol}] New strategy: {new_routing['strategy']}"
        )

        # Update tracking
        self.current_regimes[symbol] = new_regime

    def force_complete_all(self):
        """
        Force completion of all partial windows.

        Call this at market close to ensure all partial windows are emitted.

        Example:
            # At 4:00 PM ET
            manager.force_complete_all()
        """
        self.logger.info("Force completing all aggregation windows")
        self.aggregator.force_complete_all()

    def get_statistics(self) -> dict:
        """
        Get statistics about the multi-timeframe system.

        Returns:
            Dict with statistics including:
            - aggregator_stats: BarAggregator statistics
            - active_symbols: Number of active symbols
            - current_regimes: Current regime for each symbol
            - current_timeframes: Current timeframe for each symbol

        Example:
            stats = manager.get_statistics()
            print(f"Bars received: {stats['aggregator_stats']['bars_received']}")
            print(f"Bars emitted: {stats['aggregator_stats']['bars_emitted']}")
        """
        aggregator_stats = self.aggregator.get_stats()

        return {
            'aggregator_stats': aggregator_stats,
            'active_symbols': len(self.symbols),
            'current_regimes': self.current_regimes.copy(),
            'current_timeframes': aggregator_stats.get('timeframes', {}),
        }

    def get_timeframe(self, symbol: str) -> Optional[str]:
        """
        Get current timeframe for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Timeframe string or None if symbol not configured
        """
        return self.aggregator.get_timeframe(symbol)

    def get_regime(self, symbol: str) -> Optional[str]:
        """
        Get current regime for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Regime string or None if symbol not configured
        """
        return self.current_regimes.get(symbol)

    def get_routing(self, symbol: str) -> Optional[dict]:
        """
        Get current routing decision for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with strategy, timeframe, use_hybrid or None
        """
        regime = self.get_regime(symbol)
        if regime:
            return self.routing_manager.get_routing(symbol, regime)
        return None


# ============================================================================
# STANDALONE DEMO
# ============================================================================

if __name__ == '__main__':
    """Demonstration of MultiTimeframeManager usage."""
    import time
    from datetime import timedelta

    print("=" * 70)
    print("MultiTimeframeManager - Integration Demo")
    print("=" * 70)

    # Initialize manager
    manager = MultiTimeframeManager(
        symbols=['AAPL', 'TSLA', 'MSFT'],
        routing_config_path='config/strategy_routing.json'
    )

    # Register bar callback
    def on_aggregated_bar(bar: Bar):
        print(f"✓ [{bar.symbol:5s}] {bar.timestamp.strftime('%H:%M:%S')} "
              f"[{bar.timeframe:5s}] OHLCV({bar.open:.2f}, {bar.high:.2f}, "
              f"{bar.low:.2f}, {bar.close:.2f}, {bar.volume})")

    manager.register_bar_callback(on_aggregated_bar)

    print("\n=== Initial Configuration ===")
    for symbol in ['AAPL', 'TSLA', 'MSFT']:
        routing = manager.get_routing(symbol)
        print(f"  {symbol}: {routing}")

    print("\n=== Simulating Websocket Stream (10 minutes) ===")

    base_time = datetime(2026, 3, 14, 9, 30, 0)

    # Simulate 10 minutes of 1-minute bars
    for i in range(10):
        for symbol in ['AAPL', 'TSLA', 'MSFT']:
            # Simulate websocket data
            raw_data = {
                'symbol': symbol,
                'timestamp': int((base_time + timedelta(minutes=i)).timestamp() * 1000),
                'open': 150.0 + i * 0.1,
                'high': 151.0 + i * 0.1,
                'low': 149.0 + i * 0.1,
                'close': 150.5 + i * 0.1,
                'volume': 10000
            }

            # Process through manager
            manager.process_websocket_data(raw_data)

    print("\n=== Simulating Regime Change ===")
    manager.update_regime('AAPL', 'high_volatility')

    print("\n=== Force Complete (Market Close) ===")
    manager.force_complete_all()

    print("\n=== Final Statistics ===")
    stats = manager.get_statistics()
    import json
    print(json.dumps(stats, indent=2, default=str))

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)
