"""
Sharpe Ratio Filter - Blocks trades from strategies with poor backtested performance

Prevents the system from taking signals from strategy/regime combinations
that have demonstrated negative or low risk-adjusted returns.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SharpeFilter:
    """
    Filters trading signals based on backtested Sharpe ratios.

    Reads optimized Sharpe ratios from strategy_params.json and blocks
    signals from strategies that don't meet minimum performance thresholds.

    Usage:
        filter = SharpeFilter(min_sharpe=0.5)

        if filter.should_trade(symbol="AAPL", regime="normal"):
            # Execute trade
        else:
            # Skip - poor historical performance
    """

    def __init__(self, min_sharpe: float = 0.5, config_path: str | None = None):
        """
        Initialize Sharpe filter.

        Args:
            min_sharpe: Minimum Sharpe ratio to allow trading (default: 0.5)
                       - 0.5: Conservative (skip marginal strategies)
                       - 0.0: Only skip losing strategies
                       - 1.0: Aggressive (only trade excellent strategies)
            config_path: Path to strategy_params.json (auto-detected if None)
        """
        self.min_sharpe = min_sharpe

        # Auto-detect config path
        if config_path is None:
            root = Path(__file__).resolve().parents[2]
            config_path = root / "config" / "strategy_params.json"

        self.config_path = Path(config_path)
        self.sharpe_map: dict[str, dict[str, float]] = {}

        self._load_sharpe_data()

        logger.info(f"SharpeFilter initialized with min_sharpe={min_sharpe} ({len(self.sharpe_map)} symbols loaded)")

    def _load_sharpe_data(self) -> None:
        """Load Sharpe ratios from strategy_params.json."""
        if not self.config_path.exists():
            logger.warning(f"Strategy params not found at {self.config_path}, filter will allow all trades")
            return

        try:
            with open(self.config_path) as f:
                params = json.load(f)

            # Extract Sharpe ratios per symbol/regime
            for symbol, regimes in params.items():
                if not isinstance(regimes, dict):
                    continue

                self.sharpe_map[symbol] = {}

                for regime, config in regimes.items():
                    if isinstance(config, dict) and "_optimized_sharpe" in config:
                        sharpe = config["_optimized_sharpe"]
                        self.sharpe_map[symbol][regime] = sharpe

            logger.info(f"Loaded Sharpe data for {len(self.sharpe_map)} symbols")

        except Exception as e:
            logger.error(f"Failed to load Sharpe data: {e}")

    def should_trade(self, symbol: str, regime: str) -> bool:
        """
        Check if trading should be allowed for this symbol/regime.

        Args:
            symbol: Trading symbol
            regime: Market regime

        Returns:
            True if Sharpe >= min_sharpe, False otherwise
        """
        # Allow if no data (fail open)
        if symbol not in self.sharpe_map:
            logger.debug(f"No Sharpe data for {symbol}, allowing trade")
            return True

        if regime not in self.sharpe_map[symbol]:
            logger.debug(f"No Sharpe data for {symbol}/{regime}, allowing trade")
            return True

        sharpe = self.sharpe_map[symbol][regime]

        if sharpe < self.min_sharpe:
            logger.info(f"Blocking trade for {symbol}/{regime}: Sharpe {sharpe:.2f} < {self.min_sharpe:.2f}")
            return False

        return True

    def get_sharpe(self, symbol: str, regime: str) -> float | None:
        """
        Get backtested Sharpe ratio for symbol/regime.

        Args:
            symbol: Trading symbol
            regime: Market regime

        Returns:
            Sharpe ratio or None if not available
        """
        return self.sharpe_map.get(symbol, {}).get(regime)

    def get_blocked_regimes(self) -> dict[str, list]:
        """
        Get all symbol/regime combinations that are blocked.

        Returns:
            Dict mapping symbol -> list of blocked regimes
        """
        blocked = {}

        for symbol, regimes in self.sharpe_map.items():
            blocked_regimes = [regime for regime, sharpe in regimes.items() if sharpe < self.min_sharpe]

            if blocked_regimes:
                blocked[symbol] = blocked_regimes

        return blocked

    def print_summary(self) -> None:
        """Print summary of blocked regimes."""
        blocked = self.get_blocked_regimes()

        if not blocked:
            print(f"\n✅ No regimes blocked (min_sharpe={self.min_sharpe})")
            return

        print(f"\n🔴 Blocked Regimes (Sharpe < {self.min_sharpe}):")
        print("=" * 80)

        for symbol, regimes in sorted(blocked.items()):
            sharpe_info = [f"{regime} (Sharpe: {self.sharpe_map[symbol][regime]:.2f})" for regime in regimes]
            print(f"{symbol:6} : {', '.join(sharpe_info)}")

        total_blocked = sum(len(r) for r in blocked.values())
        total_combos = sum(len(r) for r in self.sharpe_map.values())

        print("=" * 80)
        print(f"Total blocked: {total_blocked}/{total_combos} ({100 * total_blocked / total_combos:.1f}%)")
