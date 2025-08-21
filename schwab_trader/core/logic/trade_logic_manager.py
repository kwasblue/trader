import json
import os
from typing import Dict, Any
from core.base.trade_logic_manager_base import TradeLogicManagerBase
from core.logic.default_trade_logic import DefaultTradeLogic
from core.base.trade_logic_base import TradeLogic

# Add more logic class imports here as needed
# from core.logic.pyramiding_logic import PyramidingTradeLogic
# from core.logic.scalping_logic import ScalpTradeLogic


TRADE_LOGIC_CLASS_REGISTRY = {
    "default": DefaultTradeLogic,
    # "pyramiding": PyramidingTradeLogic,
    # "scalp": ScalpTradeLogic,
    # Add additional mappings here
}


class DynamicTradeLogicManager(TradeLogicManagerBase):
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Routing config file not found at {config_path}")

        with open(config_path, 'r') as f:
            self.routing_config: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)

        self.logic_instances: Dict[str, Dict[str, TradeLogic]] = {}

    def _resolve_config(self, symbol: str, regime: str) -> Dict[str, Any]:
        """
        Resolution order:
          1) exact:   routing_config[symbol][regime]
          2) symbol default: routing_config[symbol]["default"]
          3) global default for regime: routing_config["default"][regime]
          4) global catch-all: routing_config["default"]["default"]
        """
        rc = self.routing_config
        if symbol in rc and regime in rc[symbol]:
            return rc[symbol][regime]
        if symbol in rc and "default" in rc[symbol]:
            return rc[symbol]["default"]
        if "default" in rc and regime in rc["default"]:
            return rc["default"][regime]
        if "default" in rc and "default" in rc["default"]:
            return rc["default"]["default"]
        # last-resort: built-in default
        return {"trade_logic_class": "default", "params": {}}

    def get(self, symbol: str, regime: str) -> TradeLogic:
        config = self._resolve_config(symbol, regime)
        trade_logic_key = config.get("trade_logic_class", "default")

        if symbol not in self.logic_instances:
            self.logic_instances[symbol] = {}
        cache_key = regime  # cache per regime

        if cache_key not in self.logic_instances[symbol]:
            logic_class = TRADE_LOGIC_CLASS_REGISTRY.get(trade_logic_key)
            if logic_class is None:
                raise ValueError(f"Unknown trade logic class: {trade_logic_key}")
            instance = logic_class(**config.get("params", {}))
            self.logic_instances[symbol][cache_key] = instance

        return self.logic_instances[symbol][cache_key]