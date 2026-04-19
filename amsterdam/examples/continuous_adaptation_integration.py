#!/usr/bin/env python3
"""
Example: Integrating Continuous Adaptation with Existing Trading System

Shows how to integrate continuous position sizing with:
1. Existing regime-based strategy routing
2. Current position sizing logic
3. Trade execution

This is a reference implementation showing the integration points.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.continuous_metrics import ContinuousMetrics
from core.continuous_position_sizer import ContinuousPositionSizer
from core.unified_data_pipeline import UnifiedDataPipeline


class AdaptiveTrader:
    """
    Example trader that combines:
    - Discrete regime detection (for strategy selection)
    - Continuous metrics (for position sizing)
    """

    def __init__(self, config_path: Path | None = None):
        """Initialize adaptive trader."""
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config_path is None:
            config_path = ROOT / "config" / "continuous_adaptation.json"

        with open(config_path) as f:
            self.config = json.load(f)

        # Initialize components
        self.data_pipeline = UnifiedDataPipeline()
        self.continuous_sizer = ContinuousPositionSizer(
            config=self.config.get("position_sizing", {}), logger=self.logger
        )

        # Load strategy routing (your existing config)
        routing_path = ROOT / "config" / "strategy_routing.json"
        with open(routing_path) as f:
            self.strategy_routing = json.load(f)

        # Trade history (in real system, load from database)
        self.trade_history: dict[str, list[dict]] = {}

        self.logger.info("AdaptiveTrader initialized")

    def detect_regime(self, symbol: str, bars) -> str:
        """
        Detect current market regime (discrete buckets).

        This is your EXISTING regime detection logic.
        """
        # Simplified example - in real system use your RegimeDetector
        metrics_calc = ContinuousMetrics()
        atr_percentile = metrics_calc.calculate_atr_percentile(symbol, bars)

        if atr_percentile < 33:
            return "low_volatility"
        elif atr_percentile < 67:
            return "normal"
        else:
            return "high_volatility"

    def get_strategy_for_regime(self, symbol: str, regime: str) -> dict:
        """
        Get strategy configuration for (symbol, regime).

        This uses your EXISTING strategy routing.
        """
        routing = self.strategy_routing.get(symbol, {})
        regime_config = routing.get(regime, routing.get("default", {}))

        return {"strategy": regime_config.get("strategy", "sma"), "timeframe": regime_config.get("timeframe", "day")}

    def calculate_adaptive_position_size(
        self, symbol: str, strategy: str, base_capital: float = 100000, max_position_pct: float = 0.10
    ) -> dict:
        """
        Calculate position size with continuous adaptation.

        This is the NEW functionality being added.
        """
        # Get recent bars for volatility calculation
        bars = self.data_pipeline.get_data(symbol, timeframe="day")

        if bars is None or bars.empty:
            self.logger.error(f"No data for {symbol}")
            return {"size": 0, "multiplier": 0, "skip": True, "reason": "No data available"}

        # Get recent trades for performance calculation
        recent_trades = self.trade_history.get(f"{symbol}_{strategy}", [])

        # Calculate position size with continuous adaptation
        result = self.continuous_sizer.calculate_position_size(
            symbol=symbol,
            strategy=strategy,
            base_capital=base_capital,
            max_position_pct=max_position_pct,
            bars=bars,
            recent_trades=recent_trades,
        )

        # Check if should skip
        should_skip = self.continuous_sizer.should_skip_trade(result)

        return {
            "size": result.final_size,
            "base_size": result.base_size,
            "multiplier": result.multiplier,
            "skip": should_skip,
            "metrics": {
                "atr_percentile": result.metrics.atr_percentile,
                "sharpe": result.metrics.rolling_sharpe,
                "combined_score": result.metrics.combined_score,
            },
            "rationale": result.rationale,
        }

    def generate_trading_signal(self, symbol: str, bars) -> str:
        """
        Generate trading signal (simplified example).

        In real system, this would use your actual strategy classes.
        """
        # Placeholder - use your actual strategy logic
        # For example: RSIStrategy, BollingerStrategy, etc.
        return "BUY"  # Simplified for example

    def execute_trade_decision(self, symbol: str, base_capital: float = 100000):
        """
        Complete trading decision flow combining discrete and continuous adaptation.

        This shows how BOTH systems work together:
        1. Discrete regime → selects strategy
        2. Continuous metrics → adjusts position size
        """
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Trading Decision: {symbol}")
        self.logger.info(f"{'=' * 60}")

        # Get price data
        bars = self.data_pipeline.get_data(symbol, timeframe="day")

        if bars is None or bars.empty:
            self.logger.error(f"No data for {symbol}")
            return None

        # === STEP 1: Discrete Regime Detection ===
        regime = self.detect_regime(symbol, bars)
        self.logger.info(f"1. Regime Detection: {regime}")

        # === STEP 2: Strategy Selection (from routing config) ===
        config = self.get_strategy_for_regime(symbol, regime)
        strategy = config["strategy"]
        timeframe = config["timeframe"]
        self.logger.info(f"2. Strategy Selection: {strategy} @ {timeframe}")

        # === STEP 3: Generate Signal (from strategy) ===
        signal = self.generate_trading_signal(symbol, bars)
        self.logger.info(f"3. Signal Generated: {signal}")

        if signal not in ["BUY", "SELL"]:
            self.logger.info(f"   → No action (signal={signal})")
            return None

        # === STEP 4: Continuous Position Sizing ===
        position_info = self.calculate_adaptive_position_size(
            symbol=symbol, strategy=strategy, base_capital=base_capital, max_position_pct=0.10
        )

        self.logger.info("4. Position Sizing:")
        self.logger.info(f"   Base Size: ${position_info['base_size']:,.0f}")
        self.logger.info(f"   Multiplier: {position_info['multiplier']:.2%}")
        self.logger.info(f"   Final Size: ${position_info['size']:,.0f}")
        self.logger.info(f"   Metrics: {position_info['metrics']}")
        self.logger.info(f"   Rationale: {position_info['rationale']}")

        # === STEP 5: Trade Decision ===
        if position_info["skip"]:
            self.logger.warning(f"5. SKIP TRADE - {position_info.get('reason', 'Low confidence')}")
            return None

        # Calculate shares
        current_price = bars.iloc[-1]["close"]
        shares = int(position_info["size"] / current_price)

        if shares == 0:
            self.logger.warning(
                f"5. SKIP TRADE - Position too small (${position_info['size']:.0f} / ${current_price:.2f} = 0 shares)"
            )
            return None

        # === STEP 6: Execute Order ===
        order = {
            "symbol": symbol,
            "action": signal,
            "shares": shares,
            "price": current_price,
            "value": shares * current_price,
            "strategy": strategy,
            "regime": regime,
            "timeframe": timeframe,
            "multiplier": position_info["multiplier"],
            "timestamp": datetime.now(),
        }

        self.logger.info(f"5. ORDER: {signal} {shares} shares {symbol} @ ${current_price:.2f} = ${order['value']:,.2f}")
        self.logger.info(f"   Strategy: {strategy} (regime={regime}, timeframe={timeframe})")

        # In real system: execute_order_via_broker(order)

        return order


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Initialize trader
    trader = AdaptiveTrader()

    # Mock some trade history for Sharpe calculation
    mock_trades = [
        {
            "symbol": "AAPL",
            "strategy": "psar",
            "pnl": 150,
            "entry_value": 10000,
            "return_pct": 1.5,
            "timestamp": datetime.now() - timedelta(days=i),
        }
        for i in range(20)
    ]
    trader.trade_history["AAPL_psar"] = mock_trades

    # Test symbols
    symbols = ["AAPL", "GOOGL", "TSLA"]

    for symbol in symbols:
        try:
            order = trader.execute_trade_decision(symbol, base_capital=100000)

            if order:
                print(f"\n✅ Order placed for {symbol}")
            else:
                print(f"\n⏭️  No order for {symbol}")

        except Exception as e:
            print(f"\n❌ Error processing {symbol}: {e}")
            import traceback

            traceback.print_exc()

        print()  # Blank line between symbols


if __name__ == "__main__":
    main()
