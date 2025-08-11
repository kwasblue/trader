import json
import os
from typing import Dict, Type, Any
from core.base.base_strategy import BaseStrategy
from core.logic.strategy_routing_manager import StrategyRoutingManager
from loggers.logger import Logger

# === Registry of strategy classes ===
from strategies.strategy_registry.momentum_strategy import MomentumStrategy
from strategies.strategy_registry.mean_reversion_strategy import MeanReversionStrategy
from strategies.strategy_registry.sma_strategy import SMAStrategy

# ✨ Import the dynamic registry
from strategies.strategy_registry.strategy_registry import STRATEGY_CLASS_REGISTRY


# core/simulator/strategy_router.py  (or wherever your StrategyRouter lives)
import inspect
from typing import Dict, Type, Any
from loggers.logger import Logger
from core.base.base_strategy import BaseStrategy
from strategies.strategy_registry.strategy_registry import STRATEGY_CLASS_REGISTRY

class StrategyRouter:
    def __init__(self, routing_manager: StrategyRoutingManager, config_path: str):
        self.routing_manager = routing_manager
        self.logger = Logger("strategy_router.log", self.__class__.__name__).get_logger()

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Strategy config file not found at {config_path}")

        with open(config_path, "r") as f:
            self.strategy_config: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)

        self.strategy_instances: Dict[str, Dict[str, BaseStrategy]] = {}
        self.logger.info(f"Loaded strategies: {sorted(STRATEGY_CLASS_REGISTRY.keys())}")

    def _filter_kwargs_for_init(self, cls: Type[BaseStrategy], params: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only kwargs that __init__ of the class actually accepts (besides self)."""
        if not params:
            return {}
        sig = inspect.signature(cls.__init__)
        allowed = {
            k: v for k, v in params.items()
            if k in sig.parameters and k not in ("self",)
        }
        return allowed

    def get_strategy(self, symbol: str, regime: str) -> BaseStrategy:
        strategy_name = self.routing_manager.get_strategy(symbol, regime)
        
        if symbol not in self.strategy_instances:
            self.strategy_instances[symbol] = {}

        if regime not in self.strategy_instances[symbol]:
            # default to SMA if missing
            cls = STRATEGY_CLASS_REGISTRY.get(
                strategy_name,
                STRATEGY_CLASS_REGISTRY.get("sma_strategy")
            )
            if cls is None:
                raise RuntimeError("No default 'sma_strategy' found in registry.")

            raw_params = (self.strategy_config.get(symbol, {})
                          .get(regime, {})
                          .get("params", {}))

            # Filter for explicit __init__ signatures
            clean_params = self._filter_kwargs_for_init(cls, raw_params)

            # If the class didn’t declare those params, BaseStrategy will still swallow them via **kwargs
            # (but we prefer clean instantiation to avoid typos)
            ignored = set(raw_params) - set(clean_params)
            if ignored:
                self.logger.warning(
                    f"[{symbol}][{regime}] Ignoring unknown strategy params for {cls.__name__}: {sorted(ignored)}"
                )

            instance = cls(**clean_params)
            self.strategy_instances[symbol][regime] = instance

            self.logger.info(
                f"[{symbol}][{regime}] Initialized strategy '{cls.__name__}' "
                f"with params: {clean_params}"
            )

        return self.strategy_instances[symbol][regime]

# class StrategyRouter:
#     """
#     Routes strategies dynamically based on symbol and regime,
#     supporting instantiation with dynamic parameters.
#     """

#     def __init__(self, routing_manager: StrategyRoutingManager, config_path: str):
#         self.routing_manager = routing_manager
#         self.logger = Logger("strategy_router.log", self.__class__.__name__).get_logger()

#         if not os.path.exists(config_path):
#             raise FileNotFoundError(f"Strategy config file not found at {config_path}")

#         with open(config_path, "r") as f:
#             self.strategy_config: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)

#         self.strategy_instances: Dict[str, Dict[str, BaseStrategy]] = {}

#     def get_strategy(self, symbol: str, regime: str) -> BaseStrategy:
#         """
#         Return a strategy instance for a given symbol and regime.
#         Caches instances and injects parameters from config.
#         """
#         strategy_name = self.routing_manager.get_strategy(symbol, regime)

#         if symbol not in self.strategy_instances:
#             self.strategy_instances[symbol] = {}

#         if regime not in self.strategy_instances[symbol]:
#             strategy_class = STRATEGY_CLASS_REGISTRY.get(strategy_name, STRATEGY_CLASS_REGISTRY.get('default'))

#             params = self.strategy_config.get(symbol, {}).get(regime, {}).get("params", {})
#             instance = strategy_class(**params)

#             self.strategy_instances[symbol][regime] = instance

#             self.logger.info(
#                 f"[{symbol}][{regime}] Initialized strategy '{strategy_name}' "
#                 f"with params: {params}"
#             )

#         return self.strategy_instances[symbol][regime]
