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
import os
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Type, TypeVar

if TYPE_CHECKING:
    from core.position_sizer import KellyPositionSizer
    from core.drawdown_monitor import DrawdownMonitor
    from core.logic.default_trade_logic import StandardTradeApprover
    from core.logic.position_manager import PositionManager

T = TypeVar("T")


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
    poll_timeout_seconds: int = 30  # Timeout for REST API poll requests


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
class ExecutionConfig:
    """Configuration for execution friction (slippage, commission)."""
    slippage_pct: float = 0.001  # 0.1% slippage
    commission_per_trade: float = 0.0  # Commission per trade in dollars
    apply_friction_in_simulation: bool = True  # Apply friction in simulation mode


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
    fee_rate: float = 0.001  # Transaction fee rate
    allow_fractional: bool = False  # Allow fractional shares
    lot_size: int = 1  # Minimum lot size for rounding


@dataclass
class TradeLogicConfig:
    cooldown_mode: str = "bars"  # "bars", "time", or "both"
    cooldown_bars: int = 5
    cooldown_seconds: int = 300
    min_regime_persist_bars: int = 3  # Bars regime must persist before strategy switch allowed
    # Take profit multipliers by regime
    tp_mult_low: float = 1.5  # Low volatility regime
    tp_mult_normal: float = 2.0  # Normal regime
    tp_mult_high: float = 3.0  # High volatility regime
    # Stop loss multipliers by regime
    sl_mult_low: float = 1.0  # Low volatility regime
    sl_mult_normal: float = 1.5  # Normal regime
    sl_mult_high: float = 2.0  # High volatility regime
    exit_fraction: float = 0.25
    trailing_stop: bool = True
    max_positions: int = 10
    min_bars_to_hold: int = 5  # Minimum bars before allowing TP/reversal exits
    swing_mode: bool = True  # If True, prevent same-day exits (except stop loss)
    min_hold_days: int = 1  # Minimum days to hold before allowing exit
    allow_after_hours: bool = False  # Allow trading outside market hours
    trading_cutoff_hour_et: int = 15  # Stop new entries after this hour ET (15 = 3pm)
    trading_cutoff_minute_et: int = 45  # Stop new entries after this minute (combined with hour)
    close_positions_eod: bool = True  # Close all positions at end of day
    signal_based_exits: bool = True  # If False, only exit on SL/TP (no signal reversal exits)


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
    trace_enabled: bool = False  # Enable execution tracing (writes to trace_file)
    trace_file: str = "logs/trace.log"  # Path to trace log


@dataclass
class AutoTraderConfig:
    enabled: bool = True
    default_broker: str = "alpaca"
    pre_market_buffer_minutes: int = 15
    post_market_delay_minutes: int = 5
    data_update_days: int = 5
    dry_run: bool = False
    close_positions_on_market_close: bool = True  # Safety net: close positions before market closes
    market_close_minutes_before: int = 5  # Minutes before close to trigger position closing
    preflight_max_retries: int = 3  # Number of preflight retry attempts
    preflight_retry_delay: int = 60  # Seconds between preflight retries


@dataclass
class ErrorRecoveryConfig:
    """Configuration for error recovery and retry behavior."""
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: float = 60.0


@dataclass
class MLTrainingConfig:
    """Configuration for ML training and trade quality prediction."""
    meta_logging_enabled: bool = True
    meta_log_file: str = "meta_trades_live.jsonl"
    training_data_path: str = "data/ml_training"
    model_path: str = "models/trade_quality_model.joblib"
    min_trades_for_training: int = 100
    retrain_interval_days: int = 7
    features: List[str] = field(default_factory=lambda: [
        "feat_atr_percentile",
        "feat_drawdown_portfolio_pct",
        "feat_position_size_pct",
        "feat_hour_of_day",
        "feat_day_of_week",
        "feat_bars_in_regime",
        "feat_signal_strength",
    ])


@dataclass
class ScannerConfig:
    """Configuration for the stock scanner/screener."""
    enabled: bool = True
    universe_source: str = "sp500"
    universe: List[str] = field(default_factory=list)
    providers: Dict[str, Any] = field(default_factory=lambda: {
        "technical_enabled": True,
        "fundamental_enabled": False,
        "sentiment_enabled": False,
        "insider_enabled": False,
        "lookback_bars": 200,
    })
    scoring: Dict[str, Any] = field(default_factory=lambda: {
        "weights": {
            "momentum_breakout": 0.30,
            "relative_strength": 0.25,
            "volume_surge": 0.20,
            "trend_following": 0.25,
        },
        "trade_threshold": 0.7,
        "watch_threshold": 0.4,
        "max_trade_symbols": 10,
        "max_watch_symbols": 20,
    })
    schedule: Dict[str, Any] = field(default_factory=lambda: {
        "pre_market_scan_enabled": True,
        "scan_time_minutes_before_open": 10,
        "auto_update_lists": False,
    })
    criteria: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "momentum_breakout", "params": {}},
        {"name": "relative_strength", "params": {"benchmark": "SPY", "period": 20}},
        {"name": "volume_surge", "params": {"surge_threshold": 1.5}},
        {"name": "trend_following", "params": {}},
    ])

    def to_engine_config(self) -> Dict[str, Any]:
        """Convert to the dict format expected by ScannerEngine."""
        return {
            "universe_source": self.universe_source,
            "universe": self.universe,
            "providers": self.providers,
            "scoring": self.scoring,
            "criteria": self.criteria,
        }


@dataclass
class HybridSizingConfig:
    """
    Configuration for hybrid position sizing.

    Hybrid sizing uses daily indicators for trend context while executing
    on minute bars. Position sizes are adjusted based on:
    - Signal confidence (high/medium/low)
    - Trend alignment (with/against daily trend)

    Position Sizing Matrix:
                           Against Trend    With Trend
                         +----------------+----------------+
         High Confidence |    Medium      |     Full       |
              (>0.7)     |     (50%)      |    (100%)      |
                         +----------------+----------------+
         Med Confidence  |    Small       |    Medium      |
            (0.5-0.7)    |     (25%)      |     (60%)      |
                         +----------------+----------------+
         Low Confidence  |   NO TRADE     |    Small       |
              (<0.5)     |      (0%)      |     (30%)      |
                         +----------------+----------------+

    Attributes:
        enabled: If False, hybrid sizing is disabled (pass-through mode)
        high_confidence_threshold: Confidence >= this is "high" tier
        low_confidence_threshold: Confidence < this is "low" tier
        trend_indicators: Which indicators to use for trend detection
    """
    enabled: bool = False  # Off by default, opt-in
    high_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.5
    trend_indicators: List[str] = field(default_factory=lambda: ["RSI", "MACD", "SMA_200"])


@dataclass
class TradingConfig:
    """Master configuration container."""
    general: GeneralConfig = field(default_factory=GeneralConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    schwab: SchwabConfig = field(default_factory=SchwabConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    drawdown_monitor: DrawdownMonitorConfig = field(default_factory=DrawdownMonitorConfig)
    position_sizer: PositionSizerConfig = field(default_factory=PositionSizerConfig)
    trade_logic: TradeLogicConfig = field(default_factory=TradeLogicConfig)
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    autotrader: AutoTraderConfig = field(default_factory=AutoTraderConfig)
    error_recovery: ErrorRecoveryConfig = field(default_factory=ErrorRecoveryConfig)
    ml_training: MLTrainingConfig = field(default_factory=MLTrainingConfig)
    hybrid_sizing: HybridSizingConfig = field(default_factory=HybridSizingConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)

    # Raw dict for any custom/unknown fields
    _raw: Dict[str, Any] = field(default_factory=dict)


def _dict_to_dataclass(cls: Type[T], data: dict) -> T:
    """Convert a dict to a dataclass, ignoring unknown fields."""
    if data is None:
        return cls()

    # Get field names from the dataclass
    field_names = {f.name for f in cls.__dataclass_fields__.values()}

    # Filter to only known fields
    filtered = {k: v for k, v in data.items() if k in field_names}

    return cls(**filtered)


def _apply_env_overrides(config: TradingConfig) -> TradingConfig:
    """
    Apply environment variable overrides to config.

    Environment variables follow the pattern:
        TRADING__SECTION__FIELD=value

    Examples:
        TRADING__RISK__RISK_PER_TRADE=0.02 → config.risk.risk_per_trade = 0.02
        TRADING__SIMULATION__ENABLED=true → config.simulation.enabled = True
        TRADING__GENERAL__LOG_LEVEL=DEBUG → config.general.log_level = "DEBUG"

    Args:
        config: TradingConfig instance to apply overrides to

    Returns:
        Updated TradingConfig instance (dataclasses are replaced, not mutated)
    """
    from dataclasses import replace

    prefix = "TRADING__"

    # Map section names to config attributes
    section_map = {
        "GENERAL": "general",
        "SIMULATION": "simulation",
        "ALPACA": "alpaca",
        "SCHWAB": "schwab",
        "RISK": "risk",
        "DRAWDOWN_MONITOR": "drawdown_monitor",
        "POSITION_SIZER": "position_sizer",
        "TRADE_LOGIC": "trade_logic",
        "INDICATORS": "indicators",
        "DATA": "data",
        "GUI": "gui",
        "LOGGING": "logging",
        "AUTOTRADER": "autotrader",
        "ERROR_RECOVERY": "error_recovery",
        "HYBRID_SIZING": "hybrid_sizing",
    }

    # Collect overrides by section
    section_overrides: Dict[str, Dict[str, Any]] = {}

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        # Parse: TRADING__SECTION__FIELD → ["SECTION", "FIELD"]
        parts = env_key[len(prefix):].split("__")
        if len(parts) != 2:
            continue

        section_upper, field_upper = parts
        section_name = section_map.get(section_upper)
        if section_name is None:
            continue

        field_name = field_upper.lower()

        # Parse value (handle booleans, numbers, etc.)
        parsed_value = _parse_env_value(env_value)

        if section_name not in section_overrides:
            section_overrides[section_name] = {}
        section_overrides[section_name][field_name] = parsed_value

    # Apply overrides using dataclass replace
    for section_name, field_overrides in section_overrides.items():
        section_obj = getattr(config, section_name, None)
        if section_obj is None:
            continue

        # Get valid field names for this section
        valid_fields = {f.name for f in fields(section_obj)}

        # Filter to only valid fields
        valid_overrides = {k: v for k, v in field_overrides.items() if k in valid_fields}

        if valid_overrides:
            # Replace section with updated values
            updated_section = replace(section_obj, **valid_overrides)
            config = replace(config, **{section_name: updated_section})

            for field_name, value in valid_overrides.items():
                print(f"[CONFIG] ENV override: {section_name}.{field_name} = {value}")

    return config


def _parse_env_value(value: str) -> Any:
    """
    Parse environment variable value to appropriate Python type.

    Handles:
    - Booleans: "true", "false", "1", "0", "yes", "no"
    - Integers: "42"
    - Floats: "3.14"
    - Strings: everything else
    """
    # Boolean detection
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


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
            execution=_dict_to_dataclass(ExecutionConfig, raw.get("execution")),
            drawdown_monitor=_dict_to_dataclass(DrawdownMonitorConfig, raw.get("drawdown_monitor")),
            position_sizer=_dict_to_dataclass(PositionSizerConfig, raw.get("position_sizer")),
            trade_logic=_dict_to_dataclass(TradeLogicConfig, raw.get("trade_logic")),
            indicators=_dict_to_dataclass(IndicatorsConfig, raw.get("indicators")),
            data=_dict_to_dataclass(DataConfig, raw.get("data")),
            gui=_dict_to_dataclass(GUIConfig, raw.get("gui")),
            logging=_dict_to_dataclass(LoggingConfig, raw.get("logging")),
            autotrader=_dict_to_dataclass(AutoTraderConfig, raw.get("autotrader")),
            error_recovery=_dict_to_dataclass(ErrorRecoveryConfig, raw.get("error_recovery")),
            ml_training=_dict_to_dataclass(MLTrainingConfig, raw.get("ml_training")),
            hybrid_sizing=_dict_to_dataclass(HybridSizingConfig, raw.get("hybrid_sizing")),
            scanner=_dict_to_dataclass(ScannerConfig, raw.get("scanner")),
            _raw=raw
        )

        print(f"[CONFIG] Loaded from {config_path}")
        print(f"[CONFIG] Drawdown monitor: {'ENABLED' if config.drawdown_monitor.enabled else 'DISABLED'}")

        # Apply environment variable overrides
        config = _apply_env_overrides(config)

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


# =============================================================================
# Path Helpers - Provide resolved paths for legacy compatibility
# =============================================================================

def get_app_path() -> Path:
    """
    Get the absolute path to the application root directory.

    Returns:
        Path to the schwab_trader directory
    """
    return Path(__file__).resolve().parent.parent


def get_logs_path() -> Path:
    """
    Get the absolute path to the logs directory.

    Returns:
        Resolved path to logs directory
    """
    cfg = get_config()
    return get_app_path() / cfg.data.logs_path


def get_data_path() -> Path:
    """
    Get the absolute path to processed data directory.

    Returns:
        Resolved path to processed data directory
    """
    cfg = get_config()
    return get_app_path() / cfg.data.historical_data_path


def get_raw_data_path() -> Path:
    """
    Get the absolute path to raw data directory.

    Returns:
        Resolved path to raw data directory
    """
    cfg = get_config()
    return get_app_path() / cfg.data.raw_data_path


def get_tokens_path() -> Path:
    """
    Get the absolute path to tokens directory.

    Returns:
        Resolved path to tokens directory
    """
    return get_app_path() / "tokens" / "token_file"


def get_env_path() -> Path:
    """
    Get the absolute path to the .env file.

    Returns:
        Resolved path to .env file
    """
    return get_app_path() / ".venv" / "env" / ".env"


# Schwab API constants
SCHWAB_TOKEN_ENDPOINT = "https://api.schwabapi.com/v1/oauth/token"
SCHWAB_AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize?"
SCHWAB_BASE_URL = "https://api.schwabapi.com/trader/v1"
SCHWAB_MARKET_URL = "https://api.schwabapi.com/marketdata/v1"


def validate_config(config: TradingConfig) -> tuple[bool, list[str]]:
    """
    Validate configuration values for safety and consistency.

    Checks:
    - Risk parameters are within safe bounds
    - Holding limits are greater than trade limits
    - Retry/circuit breaker values are sensible

    Args:
        config: TradingConfig to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Risk config validation
    if not (0.001 <= config.risk.risk_per_trade <= 0.10):
        errors.append(
            f"risk_per_trade ({config.risk.risk_per_trade}) must be between 0.1% and 10%"
        )

    if config.risk.max_holding_pct < config.risk.max_trade_pct:
        errors.append(
            f"max_holding_pct ({config.risk.max_holding_pct}) must be >= "
            f"max_trade_pct ({config.risk.max_trade_pct})"
        )

    if config.risk.max_trade_pct > 0.25:
        errors.append(
            f"max_trade_pct ({config.risk.max_trade_pct}) exceeds 25% safety limit"
        )

    # Position sizer validation
    if config.position_sizer.max_holding_pct < config.position_sizer.max_trade_pct:
        errors.append(
            f"position_sizer.max_holding_pct ({config.position_sizer.max_holding_pct}) "
            f"must be >= max_trade_pct ({config.position_sizer.max_trade_pct})"
        )

    # Drawdown monitor validation
    if config.drawdown_monitor.enabled:
        if config.drawdown_monitor.max_portfolio_drawdown > 0.50:
            errors.append(
                f"max_portfolio_drawdown ({config.drawdown_monitor.max_portfolio_drawdown}) "
                f"exceeds 50% safety limit"
            )
        if config.drawdown_monitor.max_symbol_drawdown > 0.50:
            errors.append(
                f"max_symbol_drawdown ({config.drawdown_monitor.max_symbol_drawdown}) "
                f"exceeds 50% safety limit"
            )

    # Error recovery validation
    if config.error_recovery.retry_max_attempts < 1:
        errors.append("retry_max_attempts must be at least 1")
    if config.error_recovery.retry_max_attempts > 10:
        errors.append("retry_max_attempts exceeds recommended maximum of 10")
    if config.error_recovery.circuit_breaker_failure_threshold < 2:
        errors.append("circuit_breaker_failure_threshold must be at least 2")
    if config.error_recovery.circuit_breaker_timeout < 10.0:
        errors.append("circuit_breaker_timeout must be at least 10 seconds")

    # Trade logic validation
    if config.trade_logic.min_bars_to_hold < 0:
        errors.append("min_bars_to_hold cannot be negative")

    return len(errors) == 0, errors


# Convenience alias
config = property(lambda self: get_config())


# For direct import: from core.config_loader import config
class _ConfigProxy:
    """Proxy that lazily loads config on first access."""
    def __getattr__(self, name):
        return getattr(get_config(), name)

config = _ConfigProxy()


# ============================================================================
# Factory Functions - Create component instances from config
# ============================================================================

def create_position_sizer(config: Optional[TradingConfig] = None) -> "KellyPositionSizer":
    """
    Create a position sizer instance from config.

    Uses PositionSizerFactory to create the appropriate sizer type
    based on config.position_sizer.type setting.

    Args:
        config: TradingConfig instance. Uses global config if not provided.

    Returns:
        PositionSizer instance configured from config.
    """
    from core.position_sizer_factory import PositionSizerFactory

    cfg = config or get_config()
    return PositionSizerFactory.create_from_config(cfg)


def create_drawdown_monitor(config: Optional[TradingConfig] = None) -> Optional["DrawdownMonitor"]:
    """
    Create a DrawdownMonitor instance from config.

    Args:
        config: TradingConfig instance. Uses global config if not provided.

    Returns:
        DrawdownMonitor instance configured from config, or None if disabled.
    """
    from core.drawdown_monitor import DrawdownMonitor

    cfg = config or get_config()
    ddm_cfg = cfg.drawdown_monitor

    if not ddm_cfg.enabled:
        return None

    return DrawdownMonitor(
        max_symbol_drawdown=ddm_cfg.max_symbol_drawdown,
        max_symbol_daily_drawdown=ddm_cfg.max_symbol_daily_drawdown,
        symbol_cooldown_seconds=ddm_cfg.symbol_cooldown_seconds,
        max_portfolio_drawdown=ddm_cfg.max_portfolio_drawdown,
        max_portfolio_daily_drawdown=ddm_cfg.max_portfolio_daily_drawdown,
        portfolio_cooldown_seconds=ddm_cfg.portfolio_cooldown_seconds,
    )


def create_trade_approver(
    config: Optional[TradingConfig] = None,
    event_handler=None
) -> "StandardTradeApprover":
    """
    Create a StandardTradeApprover instance from config.

    Args:
        config: TradingConfig instance. Uses global config if not provided.
        event_handler: Optional event handler for signal emission.

    Returns:
        StandardTradeApprover instance configured from config.
    """
    from core.logic.default_trade_logic import StandardTradeApprover

    cfg = config or get_config()
    tl = cfg.trade_logic

    return StandardTradeApprover(
        # Multipliers
        tp_mult_low=tl.tp_mult_low,
        tp_mult_normal=tl.tp_mult_normal,
        tp_mult_high=tl.tp_mult_high,
        sl_mult_low=tl.sl_mult_low,
        sl_mult_normal=tl.sl_mult_normal,
        sl_mult_high=tl.sl_mult_high,
        # Position management
        exit_fraction=tl.exit_fraction,
        trailing_stop=tl.trailing_stop,
        min_bars_to_hold=tl.min_bars_to_hold,
        swing_mode=tl.swing_mode,
        min_hold_days=tl.min_hold_days,
        # Gating
        cooldown_seconds=tl.cooldown_seconds,
        cooldown_bars=tl.cooldown_bars,
        cooldown_mode=tl.cooldown_mode,
        max_positions=tl.max_positions,
        event_handler=event_handler,
        # Market hours
        allow_after_hours=tl.allow_after_hours,
    )


def create_position_manager(config: Optional[TradingConfig] = None) -> "PositionManager":
    """
    Create a PositionManager instance from config.

    Args:
        config: TradingConfig instance. Uses global config if not provided.

    Returns:
        PositionManager instance configured from config.
    """
    from core.logic.position_manager import PositionManager

    cfg = config or get_config()
    tl = cfg.trade_logic

    return PositionManager(
        tp_mults={
            "low_volatility": tl.tp_mult_low,
            "normal": tl.tp_mult_normal,
            "high_volatility": tl.tp_mult_high,
        },
        sl_mults={
            "low_volatility": tl.sl_mult_low,
            "normal": tl.sl_mult_normal,
            "high_volatility": tl.sl_mult_high,
        },
        exit_fraction=tl.exit_fraction,
        min_bars_to_hold=tl.min_bars_to_hold,
        swing_mode=tl.swing_mode,
        min_hold_days=tl.min_hold_days,
        signal_based_exits=tl.signal_based_exits,
    )
