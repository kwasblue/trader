"""
Strategy Routing Manager - Routes symbols to appropriate strategies

Maps (symbol, regime) pairs to strategy instances with:
- JSON-based configuration
- Multi-timeframe support
- Strategy instance caching
- Dynamic routing updates
- Fallback strategies
- Hot reloading

Configuration Format (JSON):

Simple format (backward compatible):
{
    "AAPL": {
        "trending": "momentum_strategy",
        "ranging": "mean_reversion_strategy"
    }
}

Extended format (with timeframes):
{
    "AAPL": {
        "trending": {
            "strategy": "momentum_strategy",
            "timeframe": "15min"
        },
        "ranging": {
            "strategy": "mean_reversion_strategy",
            "timeframe": "5min"
        },
        "default": {
            "strategy": "sma_strategy",
            "timeframe": "5min"
        },
        "use_hybrid": false
    }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.base.base_strategy import BaseStrategy
from core.logic.sharpe_filter import SharpeFilter
from loggers.logger import Logger


class StrategyRoutingManager:
    """
    Routes trading signals to appropriate strategies.

    Manages strategy selection based on:
    - Symbol-specific strategies
    - Regime-specific strategies
    - Fallback/default strategies

    Configuration Format (JSON):
    Simple format (backward compatible):
    {
        "AAPL": {
            "trending": "momentum_strategy",
            "ranging": "mean_reversion_strategy",
            "default": "momentum_strategy"
        }
    }

    Extended format (with timeframes):
    {
        "AAPL": {
            "trending": {
                "strategy": "momentum_strategy",
                "timeframe": "15min"
            },
            "ranging": {
                "strategy": "mean_reversion_strategy",
                "timeframe": "5min"
            },
            "use_hybrid": false
        }
    }

    Example:
        router = StrategyRoutingManager("config/strategy_routing.json")

        # Get strategy for symbol/regime
        strategy = router.get_strategy("AAPL", "trending")
        signal = strategy.generate_signal(df)

        # Get full routing decision (strategy + timeframe)
        routing = router.get_routing("AAPL", "trending")
        # Returns: {'strategy': 'momentum_strategy', 'timeframe': '15min', 'use_hybrid': False}

        # Update routing with timeframe
        router.set_strategy("AAPL", "ranging", "scalping_strategy", timeframe="5min", persist=True)

        # Hot reload config
        router.refresh()
    """

    def __init__(self, config_path: str, auto_create: bool = True, min_sharpe: float | None = None):
        """
        Initialize strategy router.

        Args:
            config_path: Path to JSON routing configuration
            auto_create: Create default config if not found
            min_sharpe: Minimum Sharpe ratio to allow trading (None = disabled)
                       - 0.0: Only block losing strategies
                       - 0.5: Block marginal strategies (recommended)
                       - 1.0: Only trade excellent strategies

        Raises:
            FileNotFoundError: If config not found and auto_create=False
        """
        self.config_path = Path(config_path)
        self.min_sharpe = min_sharpe
        # Logger - own file with propagation to app.log
        self.logger = Logger(
            log_file="strategy_routing.log", logger_name="StrategyRoutingManager", propagate=True
        ).get_logger()

        # Routing configuration
        self.routing_map: dict[str, dict[str, str]] = {}

        # Strategy parameters configuration
        self.params_map: dict[str, dict[str, dict[str, Any]]] = {}

        # Strategy instance cache: (symbol, regime, strategy_name) -> instance
        self._strategy_cache: dict[tuple[str, str, str], BaseStrategy] = {}

        # Strategy class registry cache
        self._registry_cache: dict[str, Any] | None = None

        # Sharpe filter (optional)
        self.sharpe_filter: SharpeFilter | None = None
        if min_sharpe is not None:
            self.sharpe_filter = SharpeFilter(min_sharpe=min_sharpe)
            self.logger.info(f"Sharpe filter enabled with min_sharpe={min_sharpe}")

        # Load configuration
        if not self.config_path.exists():
            if auto_create:
                self._create_default_config()
            else:
                raise FileNotFoundError(f"Strategy routing config not found: {config_path}")

        self._load_config()
        self._load_params_config()

    # ========================================================================
    # CORE ROUTING METHODS
    # ========================================================================

    def get_strategy(self, symbol: str, regime: str) -> BaseStrategy:
        """
        Get strategy instance for symbol and regime.

        Resolution order:
        1. Check Sharpe filter (if enabled)
        2. Check symbol-specific regime mapping
        3. Check symbol default
        4. Check global regime mapping
        5. Use global default

        Args:
            symbol: Trading symbol
            regime: Market regime

        Returns:
            Strategy instance (cached), or NoOp if blocked by filter

        Example:
            strategy = router.get_strategy("AAPL", "trending")
            signal = strategy.generate_signal(df)
        """
        # Ensure all cache key components are strings (defensive)
        regime_str = str(regime) if not isinstance(regime, str) else regime

        # Check Sharpe filter first
        if self.sharpe_filter is not None:
            if not self.sharpe_filter.should_trade(symbol, regime_str):
                sharpe = self.sharpe_filter.get_sharpe(symbol, regime_str)
                self.logger.info(
                    f"Sharpe filter blocked {symbol}/{regime_str} (Sharpe: {sharpe:.2f} < {self.min_sharpe:.2f})"
                )
                # Return NoOp strategy that always signals 0 (hold)
                return self._create_noop_strategy()

        # Resolve strategy name
        strategy_name = self._resolve_strategy_name(symbol, regime)
        strategy_str = str(strategy_name) if not isinstance(strategy_name, str) else strategy_name

        self.logger.debug(f"Resolved strategy for {symbol} in '{regime_str}': {strategy_str}")

        # Check cache
        cache_key = (symbol, regime_str, strategy_str)
        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]

        # Instantiate and cache
        strategy = self._instantiate_strategy(strategy_str, symbol=symbol, regime=regime_str)
        self._strategy_cache[cache_key] = strategy

        return strategy

    def get_strategy_name(self, symbol: str, regime: str) -> str:
        """
        Get strategy name without instantiating.

        Useful for logging or validation.

        Args:
            symbol: Trading symbol
            regime: Market regime

        Returns:
            Strategy name
        """
        return self._resolve_strategy_name(symbol, regime)

    def get_routing(self, symbol: str, regime: str) -> dict[str, Any]:
        """
        Get full routing decision for symbol-regime pair.

        Returns strategy name, timeframe, and other routing parameters.

        Args:
            symbol: Trading symbol
            regime: Market regime

        Returns:
            Dict containing:
            - strategy: Strategy name
            - timeframe: Target timeframe (default: '5min')
            - use_hybrid: Whether to use hybrid sizing (default: False)

        Example:
            routing = router.get_routing("AAPL", "trending")
            # Returns: {'strategy': 'momentum', 'timeframe': '5min', 'use_hybrid': False}

            # Use in bar aggregator
            aggregator.set_timeframe(symbol, routing['timeframe'])
        """
        # Resolve strategy name
        strategy_name = self._resolve_strategy_name(symbol, regime)

        # Get timeframe and other parameters
        timeframe = self._resolve_timeframe(symbol, regime)
        use_hybrid = self._resolve_use_hybrid(symbol)

        return {
            "strategy": strategy_name,
            "timeframe": timeframe,
            "use_hybrid": use_hybrid,
        }

    def set_strategy(
        self, symbol: str, regime: str, strategy_name: str, timeframe: str | None = None, persist: bool = False
    ) -> None:
        """
        Set strategy for symbol-regime pair with optional timeframe.

        Args:
            symbol: Trading symbol
            regime: Market regime
            strategy_name: Strategy to use
            timeframe: Optional timeframe (e.g., '5min', '15min')
            persist: Whether to save to disk immediately

        Example:
            # Simple format (backward compatible)
            router.set_strategy("AAPL", "ranging", "scalping", persist=True)

            # With timeframe (new format)
            router.set_strategy("AAPL", "trending", "momentum", timeframe="15min", persist=True)
        """
        # Ensure symbol exists in map
        if symbol not in self.routing_map:
            self.routing_map[symbol] = {}

        # Create routing config (dict format if timeframe specified, else string)
        if timeframe:
            routing_config = {
                "strategy": strategy_name,
                "timeframe": timeframe,
            }
        else:
            # Backward compatible: just store strategy name as string
            routing_config = strategy_name

        # Update mapping
        self.routing_map[symbol][regime] = routing_config

        if timeframe:
            self.logger.info(f"Set strategy for {symbol} in '{regime}' to '{strategy_name}' @ {timeframe}")
        else:
            self.logger.info(f"Set strategy for {symbol} in '{regime}' to '{strategy_name}'")

        # Clear cache for this symbol-regime
        cache_keys_to_remove = [key for key in self._strategy_cache.keys() if key[0] == symbol and key[1] == regime]
        for key in cache_keys_to_remove:
            del self._strategy_cache[key]

        # Persist if requested
        if persist:
            self.save()

    def remove_strategy(self, symbol: str, regime: str | None = None, persist: bool = False) -> None:
        """
        Remove strategy mapping.

        Args:
            symbol: Trading symbol
            regime: Specific regime (None = remove all for symbol)
            persist: Whether to save to disk
        """
        if symbol not in self.routing_map:
            self.logger.warning(f"Symbol {symbol} not in routing map")
            return

        if regime is None:
            # Remove entire symbol
            del self.routing_map[symbol]
            self.logger.info(f"Removed all strategies for {symbol}")
        else:
            # Remove specific regime
            if regime in self.routing_map[symbol]:
                del self.routing_map[symbol][regime]
                self.logger.info(f"Removed strategy for {symbol} in '{regime}'")

        # Clear relevant cache entries
        cache_keys_to_remove = [
            key for key in self._strategy_cache.keys() if key[0] == symbol and (regime is None or key[1] == regime)
        ]
        for key in cache_keys_to_remove:
            del self._strategy_cache[key]

        if persist:
            self.save()

    # ========================================================================
    # CONFIGURATION MANAGEMENT
    # ========================================================================

    def _load_config(self) -> None:
        """Load routing configuration from JSON file."""
        try:
            with open(self.config_path) as f:
                self.routing_map = json.load(f)

            self.logger.info(f"Loaded routing config from {self.config_path} ({len(self.routing_map)} symbols)")

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            self.routing_map = {}
            raise
        except Exception as e:
            self.logger.error(f"Failed to load routing config: {e}")
            raise

    def _load_params_config(self) -> None:
        """Load strategy parameters configuration from JSON file."""
        params_path = self.config_path.parent / "strategy_params.json"

        if not params_path.exists():
            self.logger.info(
                f"No strategy params config found at {params_path}, strategies will use default parameters"
            )
            self.params_map = {}
            return

        try:
            with open(params_path) as f:
                self.params_map = json.load(f)

            self.logger.info(f"Loaded strategy params from {params_path} ({len(self.params_map)} symbols)")

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in params file: {e}")
            self.params_map = {}
        except Exception as e:
            self.logger.error(f"Failed to load params config: {e}")
            self.params_map = {}

    def _create_default_config(self) -> None:
        """Create default routing configuration."""
        default_config = {
            "default": {
                "trending": "momentum_strategy",
                "ranging": "mean_reversion_strategy",
                "volatile": "defensive_strategy",
                "default": "momentum_strategy",
            }
        }

        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write default config
        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=4)

        self.logger.info(f"Created default routing config at {self.config_path}")

    def refresh(self) -> None:
        """
        Reload configuration from disk.

        Useful for hot-reloading changes without restarting.
        Clears strategy cache to pick up new mappings.
        """
        self.logger.info("Refreshing routing configuration...")

        # Clear cache
        self._strategy_cache.clear()

        # Reload configs
        self._load_config()
        self._load_params_config()

        self.logger.info("Routing configuration refreshed")

    def save(self) -> None:
        """
        Persist current routing map to disk.

        Example:
            router.set_strategy("AAPL", "trending", "new_strategy")
            router.save()  # Persist to file
        """
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(self.config_path, "w") as f:
                json.dump(self.routing_map, f, indent=4)

            self.logger.info(f"Routing config saved to {self.config_path}")

        except Exception as e:
            self.logger.error(f"Failed to save routing config: {e}")
            raise

    # ========================================================================
    # STRATEGY RESOLUTION & INSTANTIATION
    # ========================================================================

    def _resolve_timeframe(self, symbol: str, regime: str) -> str:
        """
        Resolve timeframe for symbol-regime pair.

        Resolution order (checks for 'timeframe' key):
        1. symbol -> regime -> timeframe
        2. symbol -> default -> timeframe
        3. default -> regime -> timeframe
        4. default -> default -> timeframe
        5. '5min' (hardcoded fallback)

        Args:
            symbol: Trading symbol
            regime: Market regime

        Returns:
            Timeframe string (e.g., '5min', '15min')
        """
        # Ensure regime is a string
        if not isinstance(regime, str):
            regime = "normal"

        result = None

        # Try symbol-specific regime timeframe
        if symbol in self.routing_map:
            symbol_map = self.routing_map[symbol]
            if isinstance(symbol_map, dict):
                # Check regime-specific timeframe
                if regime in symbol_map and isinstance(symbol_map[regime], dict):
                    result = symbol_map[regime].get("timeframe")
                # Check symbol default timeframe
                elif "default" in symbol_map and isinstance(symbol_map["default"], dict):
                    result = symbol_map["default"].get("timeframe")

        # Try global regime timeframe
        if result is None and "default" in self.routing_map:
            default_map = self.routing_map["default"]
            if isinstance(default_map, dict):
                if regime in default_map and isinstance(default_map[regime], dict):
                    result = default_map[regime].get("timeframe")
                elif "default" in default_map and isinstance(default_map["default"], dict):
                    result = default_map["default"].get("timeframe")

        # Final fallback
        if result is None:
            result = "5min"

        return result

    def _resolve_use_hybrid(self, symbol: str) -> bool:
        """
        Resolve use_hybrid flag for a symbol.

        Checks symbol-level use_hybrid setting.

        Args:
            symbol: Trading symbol

        Returns:
            Boolean indicating whether to use hybrid sizing
        """
        if symbol in self.routing_map:
            symbol_map = self.routing_map[symbol]
            if isinstance(symbol_map, dict):
                return symbol_map.get("use_hybrid", False)

        # Check global default
        if "default" in self.routing_map:
            default_map = self.routing_map["default"]
            if isinstance(default_map, dict):
                return default_map.get("use_hybrid", False)

        return False

    def _resolve_strategy_name(self, symbol: str, regime: str) -> str:
        """
        Resolve strategy name for symbol-regime pair.

        Resolution order:
        1. symbol -> regime -> strategy (or direct string)
        2. symbol -> default -> strategy (or direct string)
        3. default -> regime -> strategy (or direct string)
        4. default -> default -> strategy (or direct string)
        5. "momentum_strategy" (hardcoded fallback)

        Note: Config values can be either strings (strategy names) or dicts
        containing 'strategy', 'timeframe', etc.
        """
        # Ensure regime is a string
        if not isinstance(regime, str):
            self.logger.warning(f"regime is not a string: {type(regime).__name__}, using 'normal'")
            regime = "normal"

        result = None

        # Try symbol-specific regime
        if symbol in self.routing_map:
            symbol_map = self.routing_map[symbol]
            if isinstance(symbol_map, dict):
                if regime in symbol_map:
                    result = symbol_map[regime]
                elif "default" in symbol_map:
                    result = symbol_map["default"]

        # Try global regime
        if result is None and "default" in self.routing_map:
            default_map = self.routing_map["default"]
            if isinstance(default_map, dict):
                if regime in default_map:
                    result = default_map[regime]
                elif "default" in default_map:
                    result = default_map["default"]

        # Final fallback
        if result is None:
            result = "momentum_strategy"

        # Extract strategy name if result is a dict
        if isinstance(result, dict):
            # New format: {'strategy': 'name', 'timeframe': '5min', ...}
            result = result.get("strategy", "momentum_strategy")
        elif not isinstance(result, str):
            # Unexpected type
            self.logger.warning(f"Strategy resolved to unexpected type: {type(result).__name__}, using default")
            result = "momentum_strategy"

        return result

    def _instantiate_strategy(
        self, strategy_name: str, symbol: str | None = None, regime: str | None = None
    ) -> BaseStrategy:
        """
        Instantiate strategy from name with optimized parameters.

        Args:
            strategy_name: Strategy name from config
            symbol: Trading symbol (for parameter lookup)
            regime: Market regime (for parameter lookup)

        Returns:
            Strategy instance
        """
        # Get strategy registry
        registry = self._get_strategy_registry()

        # Normalize name
        key = strategy_name.lower().strip()

        # Look up strategy class
        strategy_class = registry.get(key)

        if strategy_class is None:
            self.logger.warning(f"Strategy '{strategy_name}' not found in registry, trying 'default'")
            strategy_class = registry.get("default")

        # Instantiate strategy
        if strategy_class is None:
            self.logger.error(f"No strategy class found for '{strategy_name}' and no 'default' registered; using no-op")
            return self._create_noop_strategy()

        # Look up optimized parameters
        params = {}
        if symbol and regime and symbol in self.params_map:
            regime_params = self.params_map[symbol].get(regime, {})
            if regime_params and "params" in regime_params:
                params = regime_params["params"]
                self.logger.debug(f"Using optimized params for {symbol}/{regime}: {params}")

        try:
            # Instantiate with parameters
            instance = strategy_class(**params)
            self.logger.debug(f"Instantiated strategy: {strategy_name} with params: {params}")
            return instance

        except Exception as e:
            self.logger.error(f"Failed to instantiate strategy '{strategy_name}' with params {params}: {e}")
            # Try without parameters as fallback
            try:
                instance = strategy_class()
                self.logger.warning(f"Instantiated {strategy_name} without parameters as fallback")
                return instance
            except Exception as e2:
                self.logger.error(f"Complete failure to instantiate {strategy_name}: {e2}")
                return self._create_noop_strategy()

    def _get_strategy_registry(self) -> dict[str, Any]:
        """
        Get strategy class registry.

        Caches registry for performance.

        Returns:
            Dictionary mapping strategy names to classes
        """
        if self._registry_cache is not None:
            return self._registry_cache

        try:
            from strategies.strategy_registry.strategy_registry import STRATEGY_CLASS_REGISTRY

            self._registry_cache = STRATEGY_CLASS_REGISTRY
            self.logger.debug(f"Loaded strategy registry with {len(self._registry_cache)} strategies")
            return self._registry_cache

        except ImportError as e:
            self.logger.error(f"Failed to import strategy registry: {e}")
            self._registry_cache = {}
            return self._registry_cache

    def _create_noop_strategy(self) -> BaseStrategy:
        """
        Create no-op strategy as fallback.

        Returns:
            Strategy that always returns hold signal
        """

        class NoOpStrategy(BaseStrategy):
            """Fallback strategy that does nothing."""

            def __init__(self):
                super().__init__(name="noop")

            def generate_signal(self, data):
                return 0  # Always hold

        return NoOpStrategy()

    # ========================================================================
    # INSPECTION & VALIDATION
    # ========================================================================

    def list_symbols(self) -> list[str]:
        """Get list of symbols with routing configured."""
        return [s for s in self.routing_map.keys() if s != "default"]

    def list_regimes(self, symbol: str) -> list[str]:
        """
        Get list of regimes configured for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of regime names
        """
        if symbol not in self.routing_map:
            return []
        return list(self.routing_map[symbol].keys())

    def get_routing_map(self) -> dict[str, dict[str, str]]:
        """Get copy of routing map for inspection."""
        return dict(self.routing_map)

    def clear_cache(self) -> None:
        """Clear strategy instance cache."""
        count = len(self._strategy_cache)
        self._strategy_cache.clear()
        self.logger.info(f"Cleared strategy cache ({count} entries)")

    def validate_config(self) -> bool:
        """
        Validate current configuration.

        Checks:
        - All strategy names exist in registry
        - Configuration structure is valid

        Returns:
            True if valid
        """
        registry = self._get_strategy_registry()
        errors = []

        for symbol, regimes in self.routing_map.items():
            if not isinstance(regimes, dict):
                errors.append(f"Invalid regime mapping for {symbol}")
                continue

            for regime, strategy_name in regimes.items():
                key = strategy_name.lower().strip()
                if key not in registry and key != "default":
                    errors.append(f"Unknown strategy '{strategy_name}' for {symbol}/{regime}")

        if errors:
            for error in errors:
                self.logger.error(f"Validation error: {error}")
            return False

        self.logger.info("Configuration validation passed")
        return True

    def __repr__(self) -> str:
        return f"StrategyRoutingManager(symbols={len(self.list_symbols())}, cached={len(self._strategy_cache)})"
