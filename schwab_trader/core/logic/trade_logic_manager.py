"""
Dynamic Trade Logic Manager - Route to different trade approvers per symbol/regime

Provides dynamic routing of trade approval logic based on configuration:
- JSON-based configuration
- Per-symbol approvers
- Per-regime approvers
- Fallback logic
- Instance caching
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from core.base.trade_logic_manager_base import TradeApprover
from core.logic.default_trade_logic import StandardTradeApprover
from loggers.logger import Logger


# Trade Approver Class Registry
# Add your custom trade approvers here
TRADE_APPROVER_REGISTRY: Dict[str, type[TradeApprover]] = {
    "default": StandardTradeApprover,
    # "aggressive": AggressiveTradeApprover,
    # "conservative": ConservativeTradeApprover,
    # "scalping": ScalpingTradeApprover,
    # "pyramiding": PyramidingTradeApprover,
}

# DEPRECATED: Old name kept for backwards compatibility
TRADE_LOGIC_CLASS_REGISTRY = TRADE_APPROVER_REGISTRY


class DynamicTradeLogicManager:
    """
    Dynamic router for trade logic managers.
    
    Routes to different TradeLogicManager instances based on:
    - Symbol-specific logic
    - Regime-specific logic
    - Configurable parameters per instance
    
    Configuration Format (JSON):
    {
        "AAPL": {
            "trending": {
                "trade_logic_class": "aggressive",
                "params": {
                    "tp_mult_normal": 3.0,
                    "exit_fraction": 0.5
                }
            },
            "ranging": {
                "trade_logic_class": "conservative",
                "params": {
                    "tp_mult_normal": 1.5,
                    "sl_mult_normal": 2.0
                }
            },
            "default": {
                "trade_logic_class": "default",
                "params": {}
            }
        },
        "default": {
            "trending": {
                "trade_logic_class": "default",
                "params": {
                    "tp_mult_normal": 2.0
                }
            },
            "default": {
                "trade_logic_class": "default",
                "params": {}
            }
        }
    }
    
    Example:
        router = DynamicTradeLogicManager("config/trade_logic_routing.json")
        
        # Get logic for symbol/regime
        logic = router.get("AAPL", "trending")
        
        # Use logic for decision
        should_trade, reason = logic.should_trade(...)
    """
    
    def __init__(
        self,
        config_path: str,
        auto_create: bool = True,
        event_handler=None
    ):
        """
        Initialize dynamic trade logic router.

        Args:
            config_path: Path to JSON routing configuration
            auto_create: Create default config if not found
            event_handler: Event handler for signal emission

        Raises:
            FileNotFoundError: If config not found and auto_create=False
        """
        self.config_path = Path(config_path)
        self.event_handler = event_handler
        # Logger - own file with propagation to app.log
        self.logger = Logger(
            log_file="trade_logic_manager.log",
            logger_name="DynamicTradeLogicManager",
            propagate=True
        ).get_logger()

        # Configuration
        self.routing_config: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Instance cache: symbol -> regime -> approver instance
        self.approver_instances: Dict[str, Dict[str, TradeApprover]] = {}
        # DEPRECATED: Old name kept for backwards compatibility
        self.logic_instances = self.approver_instances

        # Global settings from trading_config.json (swing_mode, etc.)
        self.global_trade_logic_params = self._load_global_trade_logic_params()

        # Load configuration
        if not self.config_path.exists():
            if auto_create:
                self._create_default_config()
            else:
                raise FileNotFoundError(
                    f"Trade logic routing config not found: {config_path}"
                )

        self._load_config()

    def _load_global_trade_logic_params(self) -> Dict[str, Any]:
        """Load global trade logic params from config loader (respects runtime overrides)."""
        try:
            # Use config loader to get runtime config (includes day_trade mode overrides)
            from core.config_loader import get_config
            config = get_config()
            trade_logic = config.trade_logic

            # Extract relevant params from dataclass
            params = {
                "swing_mode": trade_logic.swing_mode,
                "min_hold_days": trade_logic.min_hold_days,
                "cooldown_mode": trade_logic.cooldown_mode,
                "cooldown_bars": trade_logic.cooldown_bars,
                "cooldown_seconds": trade_logic.cooldown_seconds,
                "min_bars_to_hold": trade_logic.min_bars_to_hold,
            }

            if params.get("swing_mode"):
                self.logger.info(f"SWING MODE ENABLED: min_hold_days={params['min_hold_days']}")
            else:
                self.logger.info(f"DAY TRADE MODE: swing_mode=False, min_hold_days={params['min_hold_days']}")

            return params
        except Exception as e:
            self.logger.warning(f"Failed to load global trade logic params: {e}")

        return {}
    
    # ========================================================================
    # CORE ROUTING
    # ========================================================================
    
    def get(
        self,
        symbol: str,
        regime: str
    ) -> TradeApprover:
        """
        Get trade approver for symbol and regime.

        Resolution order:
        1. symbol -> regime (exact match)
        2. symbol -> default
        3. default -> regime
        4. default -> default
        5. Built-in default

        Args:
            symbol: Trading symbol
            regime: Market regime

        Returns:
            TradeApprover instance (cached)

        Example:
            approver = router.get("AAPL", "trending")
            should_trade, reason = approver.should_trade(...)
        """
        # Get configuration for this symbol/regime
        config = self._resolve_config(symbol, regime)

        # Check cache
        if symbol in self.approver_instances:
            if regime in self.approver_instances[symbol]:
                return self.approver_instances[symbol][regime]

        # Create and cache new instance
        approver = self._instantiate_approver(symbol, regime, config)

        # Cache instance
        if symbol not in self.approver_instances:
            self.approver_instances[symbol] = {}
        self.approver_instances[symbol][regime] = approver

        return approver
    
    def _resolve_config(
        self,
        symbol: str,
        regime: str
    ) -> Dict[str, Any]:
        """
        Resolve configuration for symbol/regime.
        
        Args:
            symbol: Trading symbol
            regime: Market regime
            
        Returns:
            Configuration dict with 'trade_logic_class' and 'params'
        """
        rc = self.routing_config
        
        # 1. Exact match: symbol + regime
        if symbol in rc and regime in rc[symbol]:
            config = rc[symbol][regime]
            self.logger.debug(
                f"Resolved {symbol}/{regime}: exact match"
            )
            return config
        
        # 2. Symbol default
        if symbol in rc and "default" in rc[symbol]:
            config = rc[symbol]["default"]
            self.logger.debug(
                f"Resolved {symbol}/{regime}: symbol default"
            )
            return config
        
        # 3. Global regime
        if "default" in rc and regime in rc["default"]:
            config = rc["default"][regime]
            self.logger.debug(
                f"Resolved {symbol}/{regime}: global regime"
            )
            return config
        
        # 4. Global default
        if "default" in rc and "default" in rc["default"]:
            config = rc["default"]["default"]
            self.logger.debug(
                f"Resolved {symbol}/{regime}: global default"
            )
            return config
        
        # 5. Built-in fallback
        self.logger.warning(
            f"No config found for {symbol}/{regime}, using built-in default"
        )
        return {
            "trade_logic_class": "default",
            "params": {}
        }
    
    def _instantiate_approver(
        self,
        symbol: str,
        regime: str,
        config: Dict[str, Any]
    ) -> TradeApprover:
        """
        Instantiate trade approver from config.

        Args:
            symbol: Trading symbol
            regime: Market regime
            config: Configuration dict

        Returns:
            TradeApprover instance
        """
        approver_key = config.get("trade_logic_class", "default")
        params = config.get("params", {})

        # Get approver class from registry
        approver_class = TRADE_APPROVER_REGISTRY.get(approver_key)

        if approver_class is None:
            self.logger.error(
                f"Unknown trade approver class '{approver_key}', using default"
            )
            approver_class = TRADE_APPROVER_REGISTRY["default"]

        # Instantiate with params, including event_handler and global settings
        try:
            # Merge global params first, then route-specific params override
            merged_params = {**self.global_trade_logic_params, **params}

            # Add event_handler to params for signal emission
            if self.event_handler is not None:
                merged_params['event_handler'] = self.event_handler

            instance = approver_class(**merged_params)
            self.logger.info(
                f"Created {approver_key} approver for {symbol}/{regime}"
            )
            return instance

        except Exception as e:
            self.logger.error(
                f"Failed to instantiate {approver_key}: {e}, using default"
            )
            # Also include global params in fallback
            fallback_params = {**self.global_trade_logic_params, 'event_handler': self.event_handler}
            return TRADE_APPROVER_REGISTRY["default"](**fallback_params)

    # DEPRECATED: Old name kept for backwards compatibility
    _instantiate_logic = _instantiate_approver
    
    # ========================================================================
    # CONFIGURATION MANAGEMENT
    # ========================================================================
    
    def _load_config(self) -> None:
        """Load routing configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self.routing_config = json.load(f)
            
            self.logger.info(
                f"Loaded trade logic routing config from {self.config_path}"
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            self.routing_config = {}
            raise
        except Exception as e:
            self.logger.error(f"Failed to load routing config: {e}")
            raise
    
    def _create_default_config(self) -> None:
        """Create default routing configuration."""
        default_config = {
            "default": {
                "trending": {
                    "trade_logic_class": "default",
                    "params": {
                        "tp_mult_normal": 2.0,
                        "sl_mult_normal": 1.5,
                        "exit_fraction": 0.25,
                        "trailing_stop": True,
                        "cooldown_seconds": 300,
                        "max_positions": 10
                    }
                },
                "ranging": {
                    "trade_logic_class": "default",
                    "params": {
                        "tp_mult_normal": 1.5,
                        "sl_mult_normal": 2.0,
                        "exit_fraction": 0.5,
                        "trailing_stop": False,
                        "cooldown_seconds": 600,
                        "max_positions": 5
                    }
                },
                "volatile": {
                    "trade_logic_class": "default",
                    "params": {
                        "tp_mult_normal": 3.0,
                        "sl_mult_normal": 2.5,
                        "exit_fraction": 0.25,
                        "trailing_stop": True,
                        "cooldown_seconds": 300,
                        "max_positions": 3
                    }
                },
                "default": {
                    "trade_logic_class": "default",
                    "params": {}
                }
            }
        }
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write default config
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        self.logger.info(
            f"Created default trade logic routing config at {self.config_path}"
        )
    
    def refresh(self) -> None:
        """
        Reload configuration from disk.
        
        Clears instance cache to pick up new configurations.
        """
        self.logger.info("Refreshing trade logic routing configuration...")
        
        # Clear cache
        self.logic_instances.clear()
        
        # Reload config
        self._load_config()
        
        self.logger.info("Trade logic routing configuration refreshed")
    
    def save(self) -> None:
        """Persist current routing configuration to disk."""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(self.config_path, 'w') as f:
                json.dump(self.routing_config, f, indent=4)
            
            self.logger.info(
                f"Trade logic routing config saved to {self.config_path}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save routing config: {e}")
            raise
    
    # ========================================================================
    # DYNAMIC UPDATES
    # ========================================================================
    
    def set_logic(
        self,
        symbol: str,
        regime: str,
        logic_class: str,
        params: Optional[Dict[str, Any]] = None,
        persist: bool = False
    ) -> None:
        """
        Set trade logic for symbol-regime pair.
        
        Args:
            symbol: Trading symbol
            regime: Market regime
            logic_class: Logic class name from registry
            params: Logic parameters
            persist: Whether to save to disk
            
        Example:
            router.set_logic(
                "AAPL",
                "trending",
                "aggressive",
                params={'tp_mult_normal': 3.0},
                persist=True
            )
        """
        if symbol not in self.routing_config:
            self.routing_config[symbol] = {}
        
        self.routing_config[symbol][regime] = {
            "trade_logic_class": logic_class,
            "params": params or {}
        }
        
        self.logger.info(
            f"Set logic for {symbol}/{regime} to '{logic_class}'"
        )
        
        # Clear cache for this symbol-regime
        if symbol in self.logic_instances:
            if regime in self.logic_instances[symbol]:
                del self.logic_instances[symbol][regime]
        
        if persist:
            self.save()
    
    def remove_logic(
        self,
        symbol: str,
        regime: Optional[str] = None,
        persist: bool = False
    ) -> None:
        """
        Remove logic configuration.
        
        Args:
            symbol: Trading symbol
            regime: Specific regime (None = remove all for symbol)
            persist: Whether to save to disk
        """
        if symbol not in self.routing_config:
            return
        
        if regime is None:
            # Remove entire symbol
            del self.routing_config[symbol]
            if symbol in self.logic_instances:
                del self.logic_instances[symbol]
            self.logger.info(f"Removed all logic for {symbol}")
        else:
            # Remove specific regime
            if regime in self.routing_config[symbol]:
                del self.routing_config[symbol][regime]
                if symbol in self.logic_instances:
                    if regime in self.logic_instances[symbol]:
                        del self.logic_instances[symbol][regime]
                self.logger.info(f"Removed logic for {symbol}/{regime}")
        
        if persist:
            self.save()
    
    # ========================================================================
    # INSPECTION
    # ========================================================================
    
    def list_symbols(self) -> list[str]:
        """Get list of symbols with custom logic configured."""
        return [s for s in self.routing_config.keys() if s != "default"]
    
    def list_regimes(self, symbol: str) -> list[str]:
        """Get list of regimes configured for symbol."""
        if symbol not in self.routing_config:
            return []
        return list(self.routing_config[symbol].keys())
    
    def get_config(
        self,
        symbol: str,
        regime: str
    ) -> Dict[str, Any]:
        """Get resolved configuration for symbol/regime."""
        return self._resolve_config(symbol, regime)
    
    def clear_cache(self) -> None:
        """Clear all cached logic instances."""
        count = sum(len(regimes) for regimes in self.logic_instances.values())
        self.logic_instances.clear()
        self.logger.info(f"Cleared logic instance cache ({count} entries)")
    
    def validate_config(self) -> bool:
        """
        Validate current configuration.
        
        Checks:
        - All logic classes exist in registry
        - Configuration structure is valid
        
        Returns:
            True if valid
        """
        errors = []
        
        for symbol, regimes in self.routing_config.items():
            if not isinstance(regimes, dict):
                errors.append(f"Invalid regime mapping for {symbol}")
                continue
            
            for regime, config in regimes.items():
                if not isinstance(config, dict):
                    errors.append(f"Invalid config for {symbol}/{regime}")
                    continue
                
                logic_class = config.get("trade_logic_class", "default")
                if logic_class not in TRADE_LOGIC_CLASS_REGISTRY:
                    errors.append(
                        f"Unknown logic class '{logic_class}' for {symbol}/{regime}"
                    )
        
        if errors:
            for error in errors:
                self.logger.error(f"Validation error: {error}")
            return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    def __repr__(self) -> str:
        cached = sum(len(regimes) for regimes in self.logic_instances.values())
        return (
            f"DynamicTradeLogicManager("
            f"symbols={len(self.list_symbols())}, "
            f"cached={cached})"
        )