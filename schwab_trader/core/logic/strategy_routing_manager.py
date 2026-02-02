"""
Strategy Routing Manager - Routes symbols to appropriate strategies

Maps (symbol, regime) pairs to strategy instances with:
- JSON-based configuration
- Strategy instance caching
- Dynamic routing updates
- Fallback strategies
- Hot reloading
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import logging

from core.base.base_strategy import BaseStrategy
from loggers.logger import Logger



class StrategyRoutingManager:
    """
    Routes trading signals to appropriate strategies.
    
    Manages strategy selection based on:
    - Symbol-specific strategies
    - Regime-specific strategies
    - Fallback/default strategies
    
    Configuration Format (JSON):
    {
        "AAPL": {
            "trending": "momentum_strategy",
            "ranging": "mean_reversion_strategy",
            "volatile": "defensive_strategy",
            "default": "momentum_strategy"
        },
        "TSLA": {
            "trending": "breakout_strategy",
            "default": "momentum_strategy"
        },
        "default": {
            "default": "momentum_strategy"
        }
    }
    
    Example:
        router = StrategyRoutingManager("config/strategy_routing.json")
        
        # Get strategy for symbol/regime
        strategy = router.get_strategy("AAPL", "trending")
        signal = strategy.generate_signal(df)
        
        # Update routing
        router.set_strategy("AAPL", "ranging", "scalping_strategy")
        router.save()
        
        # Hot reload config
        router.refresh()
    """
    
    def __init__(
        self,
        config_path: str,
        auto_create: bool = True
    ):
        """
        Initialize strategy router.
        
        Args:
            config_path: Path to JSON routing configuration
            auto_create: Create default config if not found
            
        Raises:
            FileNotFoundError: If config not found and auto_create=False
        """
        self.config_path = Path(config_path)
        # Logger - own file with propagation to app.log
        self.logger = Logger(
            log_file="strategy_routing.log",
            logger_name="StrategyRoutingManager",
            propagate=True
        ).get_logger()
        
        # Routing configuration
        self.routing_map: Dict[str, Dict[str, str]] = {}
        
        # Strategy instance cache: (symbol, regime, strategy_name) -> instance
        self._strategy_cache: Dict[Tuple[str, str, str], BaseStrategy] = {}
        
        # Strategy class registry cache
        self._registry_cache: Optional[Dict[str, Any]] = None
        
        # Load configuration
        if not self.config_path.exists():
            if auto_create:
                self._create_default_config()
            else:
                raise FileNotFoundError(
                    f"Strategy routing config not found: {config_path}"
                )
        
        self._load_config()
    
    # ========================================================================
    # CORE ROUTING METHODS
    # ========================================================================
    
    def get_strategy(
        self,
        symbol: str,
        regime: str
    ) -> BaseStrategy:
        """
        Get strategy instance for symbol and regime.
        
        Resolution order:
        1. Check symbol-specific regime mapping
        2. Check symbol default
        3. Check global regime mapping
        4. Use global default
        
        Args:
            symbol: Trading symbol
            regime: Market regime
            
        Returns:
            Strategy instance (cached)
            
        Example:
            strategy = router.get_strategy("AAPL", "trending")
            signal = strategy.generate_signal(df)
        """
        # Resolve strategy name
        strategy_name = self._resolve_strategy_name(symbol, regime)

        # Ensure all cache key components are strings (defensive)
        regime_str = str(regime) if not isinstance(regime, str) else regime
        strategy_str = str(strategy_name) if not isinstance(strategy_name, str) else strategy_name

        self.logger.debug(
            f"Resolved strategy for {symbol} in '{regime_str}': {strategy_str}"
        )

        # Check cache
        cache_key = (symbol, regime_str, strategy_str)
        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]
        
        # Instantiate and cache
        strategy = self._instantiate_strategy(strategy_str)
        self._strategy_cache[cache_key] = strategy
        
        return strategy
    
    def get_strategy_name(
        self,
        symbol: str,
        regime: str
    ) -> str:
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
    
    def set_strategy(
        self,
        symbol: str,
        regime: str,
        strategy_name: str,
        persist: bool = False
    ) -> None:
        """
        Set strategy for symbol-regime pair.
        
        Args:
            symbol: Trading symbol
            regime: Market regime
            strategy_name: Strategy to use
            persist: Whether to save to disk immediately
            
        Example:
            router.set_strategy("AAPL", "ranging", "scalping", persist=True)
        """
        # Ensure symbol exists in map
        if symbol not in self.routing_map:
            self.routing_map[symbol] = {}
        
        # Update mapping
        self.routing_map[symbol][regime] = strategy_name
        
        self.logger.info(
            f"Set strategy for {symbol} in '{regime}' to '{strategy_name}'"
        )
        
        # Clear cache for this symbol-regime
        cache_keys_to_remove = [
            key for key in self._strategy_cache.keys()
            if key[0] == symbol and key[1] == regime
        ]
        for key in cache_keys_to_remove:
            del self._strategy_cache[key]
        
        # Persist if requested
        if persist:
            self.save()
    
    def remove_strategy(
        self,
        symbol: str,
        regime: Optional[str] = None,
        persist: bool = False
    ) -> None:
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
            key for key in self._strategy_cache.keys()
            if key[0] == symbol and (regime is None or key[1] == regime)
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
            with open(self.config_path, 'r') as f:
                self.routing_map = json.load(f)
            
            self.logger.info(
                f"Loaded routing config from {self.config_path} "
                f"({len(self.routing_map)} symbols)"
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            self.routing_map = {}
            raise
        except Exception as e:
            self.logger.error(f"Failed to load routing config: {e}")
            raise
    
    def _create_default_config(self) -> None:
        """Create default routing configuration."""
        default_config = {
            "default": {
                "trending": "momentum_strategy",
                "ranging": "mean_reversion_strategy",
                "volatile": "defensive_strategy",
                "default": "momentum_strategy"
            }
        }
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write default config
        with open(self.config_path, 'w') as f:
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
        
        # Reload config
        self._load_config()
        
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
            with open(self.config_path, 'w') as f:
                json.dump(self.routing_map, f, indent=4)
            
            self.logger.info(f"Routing config saved to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save routing config: {e}")
            raise
    
    # ========================================================================
    # STRATEGY RESOLUTION & INSTANTIATION
    # ========================================================================
    
    def _resolve_strategy_name(
        self,
        symbol: str,
        regime: str
    ) -> str:
        """
        Resolve strategy name for symbol-regime pair.

        Resolution order:
        1. symbol -> regime
        2. symbol -> default
        3. default -> regime
        4. default -> default
        5. "default" (hardcoded fallback)
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

        # Ensure result is a string
        if not isinstance(result, str):
            self.logger.warning(f"Strategy resolved to non-string: {type(result).__name__}, using default")
            result = "momentum_strategy"

        return result
    
    def _instantiate_strategy(self, strategy_name: str) -> BaseStrategy:
        """
        Instantiate strategy from name.
        
        Args:
            strategy_name: Strategy name from config
            
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
            self.logger.warning(
                f"Strategy '{strategy_name}' not found in registry, "
                f"trying 'default'"
            )
            strategy_class = registry.get("default")
        
        # Instantiate strategy
        if strategy_class is None:
            self.logger.error(
                f"No strategy class found for '{strategy_name}' "
                f"and no 'default' registered; using no-op"
            )
            return self._create_noop_strategy()
        
        try:
            instance = strategy_class()
            self.logger.debug(f"Instantiated strategy: {strategy_name}")
            return instance
            
        except Exception as e:
            self.logger.error(
                f"Failed to instantiate strategy '{strategy_name}': {e}"
            )
            return self._create_noop_strategy()
    
    def _get_strategy_registry(self) -> Dict[str, Any]:
        """
        Get strategy class registry.
        
        Caches registry for performance.
        
        Returns:
            Dictionary mapping strategy names to classes
        """
        if self._registry_cache is not None:
            return self._registry_cache
        
        try:
            from strategies.strategy_registry.strategy_registry import (
                STRATEGY_CLASS_REGISTRY
            )
            self._registry_cache = STRATEGY_CLASS_REGISTRY
            self.logger.debug(
                f"Loaded strategy registry with {len(self._registry_cache)} strategies"
            )
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
    
    def get_routing_map(self) -> Dict[str, Dict[str, str]]:
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
                    errors.append(
                        f"Unknown strategy '{strategy_name}' for "
                        f"{symbol}/{regime}"
                    )
        
        if errors:
            for error in errors:
                self.logger.error(f"Validation error: {error}")
            return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    def __repr__(self) -> str:
        return (
            f"StrategyRoutingManager("
            f"symbols={len(self.list_symbols())}, "
            f"cached={len(self._strategy_cache)})"
        )