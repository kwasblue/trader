"""
Centralized Configuration Loader

Loads and provides access to all system configuration from trading_config.json.
Use this instead of hardcoding values throughout the codebase.

Usage:
    from core.config_loader import get_config, config

    # Get entire config
    cfg = get_config()

    # Access sections
    if cfg.drawdown_monitor.enabled:
        ddm = DrawdownMonitor(...)

    # Or use the global instance
    from core.config_loader import config
    print(config.simulation.bar_sleep)
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class GeneralConfig:
    default_symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT"])
    default_mode: str = "simulation"
    log_level: str = "INFO"


@dataclass
class SimulationConfig:
    enabled: bool = True
    steps: int = 999999
    bar_sleep: float = 0.1
    warmup_bars: int = 200
    starting_cash: float = 100000.0
    gbm_mu: float = 0.05
    gbm_sigma: float = 0.2


@dataclass
class AlpacaConfig:
    enabled: bool = True
    paper: bool = True
    data_feed: str = "iex"


@dataclass
class SchwabConfig:
    enabled: bool = True
    session: str = "NORMAL"


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01
    max_trade_pct: float = 0.10
    max_holding_pct: float = 0.25
    max_pyramid_layers: int = 2
    min_bars_between_layers: int = 2
    daily_loss_limit: float = 1000.0


@dataclass
class DrawdownMonitorConfig:
    enabled: bool = False
    max_symbol_drawdown: float = 0.30
    max_symbol_daily_drawdown: float = 0.15
    symbol_cooldown_seconds: int = 5
    max_portfolio_drawdown: float = 0.25
    max_portfolio_daily_drawdown: float = 0.10
    portfolio_cooldown_seconds: int = 5


@dataclass
class PositionSizerConfig:
    type: str = "dynamic"
    risk_percentage: float = 0.01
    max_trade_pct: float = 0.10
    max_holding_pct: float = 0.25
    min_position_size: int = 1
    max_position_size: int = 1000


@dataclass
class TradeLogicConfig:
    cooldown_mode: str = "bars"  # "bars", "time", or "both"
    cooldown_bars: int = 5
    cooldown_seconds: int = 300
    tp_mult_normal: float = 2.0
    tp_mult_trending: float = 3.0
    sl_mult_normal: float = 1.5
    sl_mult_trending: float = 1.0
    exit_fraction: float = 0.25
    trailing_stop: bool = True
    max_positions: int = 10
    min_bars_to_hold: int = 3  # Minimum bars before allowing TP/reversal exits
    swing_mode: bool = True  # If True, prevent same-day exits (except stop loss)
    min_hold_days: int = 1  # Minimum days to hold before allowing exit


@dataclass
class IndicatorsConfig:
    atr_period: int = 14
    sma_short: int = 20
    sma_long: int = 50
    rsi_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0


@dataclass
class DataConfig:
    historical_data_path: str = "data/data_storage/proc_data"
    raw_data_path: str = "data/data_storage/raw_data"
    logs_path: str = "logs"
    max_stale_minutes: int = 60  # Max age before data is considered stale
    historical_update_interval_minutes: int = 60  # How often to update historical data
    seed_bars: int = 200  # Number of bars to seed for warmup
    historical_days_to_fetch: int = 5  # Days of history to fetch when updating


@dataclass
class GUIConfig:
    update_interval_ms: int = 1000
    equity_maxlen: int = 5000
    price_maxlen: int = 2000
    auto_refresh_history: bool = True
    auto_refresh_performance: bool = True


@dataclass
class LoggingConfig:
    console_enabled: bool = True
    file_enabled: bool = True
    propagate_to_app_log: bool = True
    debug_print_enabled: bool = True


@dataclass
class AutoTraderConfig:
    enabled: bool = True
    default_broker: str = "alpaca"
    pre_market_buffer_minutes: int = 15
    post_market_delay_minutes: int = 5
    data_update_days: int = 5
    dry_run: bool = False


@dataclass
class TradingConfig:
    """Master configuration container."""
    general: GeneralConfig = field(default_factory=GeneralConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    schwab: SchwabConfig = field(default_factory=SchwabConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    drawdown_monitor: DrawdownMonitorConfig = field(default_factory=DrawdownMonitorConfig)
    position_sizer: PositionSizerConfig = field(default_factory=PositionSizerConfig)
    trade_logic: TradeLogicConfig = field(default_factory=TradeLogicConfig)
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    autotrader: AutoTraderConfig = field(default_factory=AutoTraderConfig)

    # Raw dict for any custom/unknown fields
    _raw: Dict[str, Any] = field(default_factory=dict)


def _dict_to_dataclass(cls, data: dict):
    """Convert a dict to a dataclass, ignoring unknown fields."""
    if data is None:
        return cls()

    # Get field names from the dataclass
    field_names = {f.name for f in cls.__dataclass_fields__.values()}

    # Filter to only known fields
    filtered = {k: v for k, v in data.items() if k in field_names}

    return cls(**filtered)


def load_config(config_path: Optional[str] = None) -> TradingConfig:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        TradingConfig instance with all settings
    """
    if config_path is None:
        # Try to find config relative to this file or current directory
        possible_paths = [
            Path(__file__).parent.parent / "config" / "trading_config.json",
            Path("config/trading_config.json"),
            Path("trading_config.json"),
        ]

        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break
        else:
            print("[CONFIG] No config file found, using defaults")
            return TradingConfig()

    try:
        with open(config_path, 'r') as f:
            raw = json.load(f)

        config = TradingConfig(
            general=_dict_to_dataclass(GeneralConfig, raw.get("general")),
            simulation=_dict_to_dataclass(SimulationConfig, raw.get("simulation")),
            alpaca=_dict_to_dataclass(AlpacaConfig, raw.get("alpaca")),
            schwab=_dict_to_dataclass(SchwabConfig, raw.get("schwab")),
            risk=_dict_to_dataclass(RiskConfig, raw.get("risk")),
            drawdown_monitor=_dict_to_dataclass(DrawdownMonitorConfig, raw.get("drawdown_monitor")),
            position_sizer=_dict_to_dataclass(PositionSizerConfig, raw.get("position_sizer")),
            trade_logic=_dict_to_dataclass(TradeLogicConfig, raw.get("trade_logic")),
            indicators=_dict_to_dataclass(IndicatorsConfig, raw.get("indicators")),
            data=_dict_to_dataclass(DataConfig, raw.get("data")),
            gui=_dict_to_dataclass(GUIConfig, raw.get("gui")),
            logging=_dict_to_dataclass(LoggingConfig, raw.get("logging")),
            autotrader=_dict_to_dataclass(AutoTraderConfig, raw.get("autotrader")),
            _raw=raw
        )

        print(f"[CONFIG] Loaded from {config_path}")
        print(f"[CONFIG] Drawdown monitor: {'ENABLED' if config.drawdown_monitor.enabled else 'DISABLED'}")

        return config

    except FileNotFoundError:
        print(f"[CONFIG] File not found: {config_path}, using defaults")
        return TradingConfig()
    except json.JSONDecodeError as e:
        print(f"[CONFIG] Invalid JSON in {config_path}: {e}, using defaults")
        return TradingConfig()
    except Exception as e:
        print(f"[CONFIG] Error loading config: {e}, using defaults")
        return TradingConfig()


# Global config instance (lazy loaded)
_config: Optional[TradingConfig] = None


def get_config(reload: bool = False) -> TradingConfig:
    """
    Get the global config instance.

    Args:
        reload: If True, reload from disk even if already loaded

    Returns:
        TradingConfig instance
    """
    global _config
    if _config is None or reload:
        _config = load_config()
    return _config


def set_config(config: TradingConfig) -> None:
    """
    Set the global config instance.

    Use this to apply runtime overrides (e.g., from command line flags).

    Args:
        config: TradingConfig instance to use globally
    """
    global _config
    _config = config


def enable_day_trade_mode() -> TradingConfig:
    """
    Enable day trading mode globally.

    Disables swing mode and sets min_hold_days to 0, allowing same-day exits.

    Returns:
        Updated TradingConfig instance
    """
    from dataclasses import replace
    config = get_config()
    updated = replace(
        config,
        trade_logic=replace(
            config.trade_logic,
            swing_mode=False,
            min_hold_days=0
        )
    )
    set_config(updated)
    return updated


def reload_config() -> TradingConfig:
    """Reload config from disk."""
    return get_config(reload=True)


# Convenience alias
config = property(lambda self: get_config())


# For direct import: from core.config_loader import config
class _ConfigProxy:
    """Proxy that lazily loads config on first access."""
    def __getattr__(self, name):
        return getattr(get_config(), name)

config = _ConfigProxy()
