import json
import os
from typing import Dict, Optional
from loggers.logger import Logger


class StrategyRoutingManager:
    """
    Manages strategy routing for each symbol and market regime.

    Expected JSON structure:
    {
        "AAPL": {
            "uptrend_low_vol": "momentum_strategy",
            "downtrend_high_vol": "mean_reversion_strategy"
        },
        "TSLA": {
            "sideways_normal_vol": "breakout_strategy"
        }
    }
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.logger = Logger('strategy_routing.log', self.__class__.__name__).get_logger()
        self.routing_map: Dict[str, Dict[str, str]] = {}

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Strategy routing config file not found at {config_path}")

        self._load_config()

    def _load_config(self) -> None:
        """
        Load the routing configuration from the JSON file.
        """
        try:
            with open(self.config_path, 'r') as f:
                self.routing_map = json.load(f)
                self.logger.info(f"Routing config loaded from {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load strategy routing config: {e}")
            raise

    def refresh(self) -> None:
        """
        Reload the routing config from disk.
        """
        self.logger.info("Refreshing routing config...")
        self._load_config()

    def get_strategy(self, symbol: str, regime: str) -> str:
        """
        Retrieve the strategy assigned to a symbol and regime.
        Falls back to 'default' if not found.
        """
        strategy = self.routing_map.get(symbol, {}).get(regime, "default")
        self.logger.debug(f"Strategy for {symbol} in regime '{regime}': {strategy}")
        return strategy

    def set_strategy(self, symbol: str, regime: str, strategy: str) -> None:
        """
        Dynamically assign a strategy to a symbol-regime pair.
        Does not persist to disk unless explicitly saved.
        """
        if symbol not in self.routing_map:
            self.routing_map[symbol] = {}

        self.routing_map[symbol][regime] = strategy
        self.logger.info(f"Set strategy for {symbol} in regime '{regime}' to '{strategy}'")

    def save(self) -> None:
        """
        Persist the current routing map to disk.
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.routing_map, f, indent=4)
            self.logger.info("Routing config saved.")
        except Exception as e:
            self.logger.error(f"Failed to save routing config: {e}")
