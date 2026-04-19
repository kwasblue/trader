# core/schwab_runner_multitf.py
"""
SchwabLiveRunner with Multi-Timeframe Support

This is an enhanced version of SchwabLiveRunner that uses the MultiTimeframeManager
for bar aggregation instead of the built-in simple aggregation.

Key differences from standard SchwabLiveRunner:
- Uses MultiTimeframeManager for timeframe-aware bar aggregation
- Supports different timeframes per symbol based on regime
- Automatically switches timeframes on regime changes

Usage:
    # In autoamsterdam.py, replace:
    # from core.schwab_runner import SchwabLiveRunner
    # with:
    from core.schwab_runner_multitf import SchwabLiveRunnerMultiTF as SchwabLiveRunner

    # Everything else stays the same!
"""

from __future__ import annotations

import asyncio

from core.bar_aggregator import Bar
from core.config_loader import TradingConfig
from core.multi_timeframe_integration import MultiTimeframeManager
from core.schwab_runner import SchwabLiveRunner
from data.streaming.schwab_client import SchwabClient


class SchwabLiveRunnerMultiTF(SchwabLiveRunner):
    """
    Multi-timeframe enhanced Schwab live trading runner.

    Extends SchwabLiveRunner with multi-timeframe bar aggregation.
    All bars are aggregated through MultiTimeframeManager which supports:
    - Different timeframes per symbol
    - Regime-based timeframe switching
    - Proper OHLCV aggregation with window alignment
    """

    def __init__(
        self,
        symbols: list[str],
        client: SchwabClient | None = None,
        config: TradingConfig | None = None,
        routing_config_path: str = "config/strategy_routing.json",
    ):
        """
        Initialize multi-timeframe Schwab runner.

        Args:
            symbols: List of symbols to trade
            client: Optional pre-configured SchwabClient
            config: Optional TradingConfig
            routing_config_path: Path to strategy routing config
        """
        # Initialize multi-timeframe manager
        self.mtf_manager = MultiTimeframeManager(
            symbols=symbols, routing_config_path=routing_config_path, initial_regime="normal"
        )

        # Register callback for aggregated bars
        self.mtf_manager.register_bar_callback(self._on_aggregated_bar)

        # Track regime for each symbol (for regime change detection)
        self._current_regimes: dict[str, str] = {symbol: "normal" for symbol in symbols}

        # Call parent constructor
        super().__init__(symbols, client, config)

        self.logger.info("=" * 60)
        self.logger.info("Multi-Timeframe Mode ENABLED")
        self.logger.info("=" * 60)
        for symbol in symbols:
            routing = self.mtf_manager.get_routing(symbol)
            self.logger.info(f"  {symbol}: {routing['strategy']} @ {routing['timeframe']}")
        self.logger.info("=" * 60)

    async def _on_quote(self, symbol: str, quote: dict) -> None:
        """
        Handle incoming Schwab quote data with multi-timeframe aggregation.

        Overrides parent method to use MultiTimeframeManager instead of
        built-in bar aggregation.

        Args:
            symbol: The symbol for this quote
            quote: Raw quote data from Schwab streaming
        """
        # Canonicalize quote to standard format
        raw_bar = self._canonicalize_schwab_quote(quote, symbol)

        # Convert to Bar object format for MultiTimeframeManager
        bar_data = {
            "symbol": symbol,
            "timestamp": raw_bar.get("timestamp"),
            "open": float(raw_bar.get("Open", 0)),
            "high": float(raw_bar.get("High", 0)),
            "low": float(raw_bar.get("Low", 0)),
            "close": float(raw_bar.get("Close", 0)),
            "volume": int(raw_bar.get("Volume", 0)),
        }

        # Process through multi-timeframe manager
        # This will aggregate to the configured timeframe and emit when window completes
        self.mtf_manager.process_websocket_data(bar_data)

        # Completed bars are handled via callback (_on_aggregated_bar)
        # We also update current price for MTM
        price = float(raw_bar["Close"])
        self.portfolio.update_price(symbol, price)

    async def _on_aggregated_bar(self, bar: Bar):
        """
        Handle completed aggregated bars from MultiTimeframeManager.

        This is called when a timeframe window completes (e.g., end of 5-minute period).
        The bar is already aggregated to the correct timeframe.

        Args:
            bar: Aggregated Bar object at configured timeframe
        """
        symbol = bar.symbol

        # Convert Bar to canonical format expected by strategy engine
        canonical_bar = {
            "symbol": symbol,
            "timestamp": bar.timestamp,
            "Open": bar.open,
            "High": bar.high,
            "Low": bar.low,
            "Close": bar.close,
            "Volume": bar.volume,
            "bar_closed": True,  # Always true for aggregated bars
        }

        self.logger.info(
            f"[{symbol}] Aggregated {bar.timeframe} bar: "
            f"{bar.timestamp.strftime('%H:%M')} "
            f"OHLCV({bar.open:.2f}, {bar.high:.2f}, {bar.low:.2f}, "
            f"{bar.close:.2f}, {bar.volume})"
        )

        # Process bar through strategy engine
        await self._process_completed_bar(canonical_bar)

        # Update price tracker
        self.portfolio.update_price(symbol, bar.close)

    async def _check_regime_changes(self):
        """
        Check for regime changes and update timeframes accordingly.

        Call this periodically (e.g., every 5 minutes) to detect regime changes
        and automatically switch timeframes.
        """
        if not hasattr(self, "regime_detector"):
            # Regime detector not available - skip check
            return

        for symbol in self.symbols:
            try:
                # Detect current regime
                new_regime = self.regime_detector.detect_regime(symbol)
                old_regime = self._current_regimes.get(symbol, "normal")

                if new_regime != old_regime:
                    self.logger.info(f"[{symbol}] Regime change detected: {old_regime} → {new_regime}")

                    # Update timeframe through manager
                    self.mtf_manager.update_regime(symbol, new_regime)

                    # Track new regime
                    self._current_regimes[symbol] = new_regime

                    # Log new configuration
                    routing = self.mtf_manager.get_routing(symbol)
                    self.logger.info(f"[{symbol}] New config: {routing['strategy']} @ {routing['timeframe']}")

            except Exception as e:
                self.logger.exception(f"Error checking regime for {symbol}: {e}")

    async def stop(self):
        """
        Stop the runner and force complete all partial windows.

        Overrides parent to ensure all partial bars are emitted before shutdown.
        """
        self.logger.info("Stopping multi-timeframe runner...")

        # Force complete all partial windows (market close)
        self.mtf_manager.force_complete_all()

        # Log statistics
        stats = self.mtf_manager.get_statistics()
        self.logger.info("=" * 60)
        self.logger.info("Multi-Timeframe Statistics")
        self.logger.info("=" * 60)
        self.logger.info(f"Bars received: {stats['aggregator_stats']['bars_received']}")
        self.logger.info(f"Bars emitted: {stats['aggregator_stats']['bars_emitted']}")
        self.logger.info(f"Timeframe changes: {stats['aggregator_stats']['timeframe_changes']}")
        self.logger.info("=" * 60)

        # Call parent stop
        await super().stop()

    # Optional: Override run() to add periodic regime checking
    async def run(self):
        """
        Run the trading session with periodic regime checking.

        Extends parent run() to add periodic regime change detection.
        """
        # Start regime checking task (every 5 minutes)
        regime_check_task = None
        if hasattr(self, "regime_detector"):

            async def regime_check_loop():
                while self.running:
                    await asyncio.sleep(300)  # 5 minutes
                    await self._check_regime_changes()

            regime_check_task = asyncio.create_task(regime_check_loop())

        try:
            # Run parent (main trading loop)
            await super().run()
        finally:
            # Cancel regime checking
            if regime_check_task:
                regime_check_task.cancel()
                try:
                    await regime_check_task
                except asyncio.CancelledError:
                    pass


# ============================================================================
# FACTORY REGISTRATION
# ============================================================================


def register_with_factory():
    """
    Register this runner with RunnerFactory.

    Call this to make MultiTF runner available via:
        RunnerFactory.create(broker='schwab_multitf', ...)

    Usage:
        from core.schwab_runner_multitf import register_with_factory
        register_with_factory()
    """
    try:
        from core.runner_factory import RunnerFactory

        RunnerFactory.register("schwab_multitf", SchwabLiveRunnerMultiTF)
        print("✓ Registered schwab_multitf runner with factory")
    except Exception as e:
        print(f"✗ Failed to register with factory: {e}")


if __name__ == "__main__":
    """Quick test of multi-timeframe runner."""
    print("=" * 70)
    print("SchwabLiveRunnerMultiTF - Quick Test")
    print("=" * 70)

    # Test initialization
    try:
        runner = SchwabLiveRunnerMultiTF(
            symbols=["AAPL", "TSLA", "MSFT"], routing_config_path="config/strategy_routing.json"
        )
        print("✓ Runner initialized successfully")

        # Check configuration
        for symbol in ["AAPL", "TSLA", "MSFT"]:
            routing = runner.mtf_manager.get_routing(symbol)
            print(f"  {symbol}: {routing}")

        print("\n✓ Multi-timeframe runner ready!")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
