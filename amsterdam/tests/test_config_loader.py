"""
Tests for ConfigLoader

Coverage:
- Config file loading
- Default value handling
- Dataclass conversion
- Global config access
- Config reload
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config_loader import (
    AlpacaConfig,
    AutoTraderConfig,
    DataConfig,
    DrawdownMonitorConfig,
    GeneralConfig,
    GUIConfig,
    IndicatorsConfig,
    LoggingConfig,
    PositionSizerConfig,
    RiskConfig,
    SchwabConfig,
    SimulationConfig,
    StreamingConfig,
    TradeLogicConfig,
    TradingConfig,
    _dict_to_dataclass,
    get_config,
    load_config,
    reload_config,
)


class TestDataclasses:
    """Tests for configuration dataclasses."""

    def test_general_config_defaults(self):
        """Test GeneralConfig default values."""
        config = GeneralConfig()
        assert config.default_symbols == ["AAPL", "MSFT"]
        assert config.default_mode == "simulation"
        assert config.log_level == "INFO"

    def test_simulation_config_defaults(self):
        """Test SimulationConfig default values."""
        config = SimulationConfig()
        assert config.enabled is True
        assert config.steps == 999999
        assert config.bar_sleep == 0.1
        assert config.warmup_bars == 200
        assert config.starting_cash == 100000.0

    def test_alpaca_config_defaults(self):
        """Test AlpacaConfig default values."""
        config = AlpacaConfig()
        assert config.enabled is True
        assert config.paper is True
        assert config.data_feed == "iex"

    def test_schwab_config_defaults(self):
        """Test SchwabConfig default values."""
        config = SchwabConfig()
        assert config.enabled is True
        assert config.session == "NORMAL"

    def test_risk_config_defaults(self):
        """Test RiskConfig default values."""
        config = RiskConfig()
        assert config.risk_per_trade == 0.01
        assert config.max_trade_pct == 0.10
        assert config.max_holding_pct == 0.25
        assert config.max_pyramid_layers == 2
        assert config.daily_loss_limit == 1000.0

    def test_drawdown_monitor_config_defaults(self):
        """Test DrawdownMonitorConfig default values."""
        config = DrawdownMonitorConfig()
        assert config.enabled is False
        assert config.max_symbol_drawdown == 0.30
        assert config.max_portfolio_drawdown == 0.25

    def test_position_sizer_config_defaults(self):
        """Test PositionSizerConfig default values."""
        config = PositionSizerConfig()
        assert config.type == "dynamic"
        assert config.risk_percentage == 0.01
        assert config.min_position_size == 1
        assert config.max_position_size == 1000

    def test_trade_logic_config_defaults(self):
        """Test TradeLogicConfig default values."""
        config = TradeLogicConfig()
        assert config.cooldown_mode == "bars"
        assert config.cooldown_bars == 5
        assert config.tp_mult_normal == 2.0
        assert config.sl_mult_normal == 1.5
        assert config.trailing_stop is True
        assert config.min_bars_to_hold == 5

    def test_indicators_config_defaults(self):
        """Test IndicatorsConfig default values."""
        config = IndicatorsConfig()
        assert config.atr_period == 14
        assert config.sma_short == 20
        assert config.sma_long == 50
        assert config.rsi_period == 14

    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        assert config.max_stale_minutes == 60
        assert config.seed_bars == 200
        assert config.historical_days_to_fetch == 5

    def test_gui_config_defaults(self):
        """Test GUIConfig default values."""
        config = GUIConfig()
        assert config.update_interval_ms == 1000
        assert config.equity_maxlen == 5000

    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        assert config.console_enabled is True
        assert config.file_enabled is True
        assert config.propagate_to_app_log is True

    def test_autotrader_config_defaults(self):
        """Test AutoTraderConfig default values."""
        config = AutoTraderConfig()
        assert config.enabled is True
        assert config.default_broker == "alpaca"
        assert config.pre_market_buffer_minutes == 15
        assert config.post_market_delay_minutes == 5
        assert config.dry_run is False

    def test_streaming_config_defaults(self):
        """Test StreamingConfig default values."""
        config = StreamingConfig()
        assert config.bar_interval_ms == 60000
        assert config.ensemble_enabled is False
        assert config.ensemble_strategies == ["sma", "rsi", "macd", "momentum"]
        assert config.ensemble_mode == "majority"
        assert config.ensemble_weights == {
            "sma": 0.25,
            "rsi": 0.25,
            "macd": 0.25,
            "momentum": 0.25,
        }
        assert config.ensemble_threshold == 0.6
        assert config.max_quotes_per_second == 1000
        assert config.quote_buffer_size == 10000


class TestDictToDataclass:
    """Tests for _dict_to_dataclass helper."""

    def test_converts_valid_dict(self):
        """Test conversion of valid dictionary."""
        data = {"default_symbols": ["TSLA", "GOOGL"], "default_mode": "live", "log_level": "DEBUG"}
        config = _dict_to_dataclass(GeneralConfig, data)
        assert config.default_symbols == ["TSLA", "GOOGL"]
        assert config.default_mode == "live"
        assert config.log_level == "DEBUG"

    def test_ignores_unknown_fields(self):
        """Test that unknown fields are ignored."""
        data = {"default_symbols": ["AAPL"], "unknown_field": "should be ignored", "another_unknown": 123}
        config = _dict_to_dataclass(GeneralConfig, data)
        assert config.default_symbols == ["AAPL"]
        # Should not raise, just ignore unknown fields

    def test_handles_none_input(self):
        """Test handling of None input."""
        config = _dict_to_dataclass(GeneralConfig, None)
        # Should return default config
        assert config.default_symbols == ["AAPL", "MSFT"]

    def test_handles_empty_dict(self):
        """Test handling of empty dictionary."""
        config = _dict_to_dataclass(GeneralConfig, {})
        # Should return default config
        assert config.default_symbols == ["AAPL", "MSFT"]

    def test_partial_override(self):
        """Test partial field override."""
        data = {"risk_per_trade": 0.02}
        config = _dict_to_dataclass(RiskConfig, data)
        assert config.risk_per_trade == 0.02
        # Other fields should be defaults
        assert config.max_trade_pct == 0.10


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_from_file(self, tmp_path):
        """Test loading config from a file."""
        config_file = tmp_path / "trading_config.json"
        config_data = {
            "general": {"default_symbols": ["NVDA"], "log_level": "DEBUG"},
            "simulation": {"starting_cash": 50000.0},
        }
        config_file.write_text(json.dumps(config_data))

        config = load_config(str(config_file))

        assert config.general.default_symbols == ["NVDA"]
        assert config.general.log_level == "DEBUG"
        assert config.simulation.starting_cash == 50000.0

    def test_load_config_missing_file(self, tmp_path):
        """Test loading config when file doesn't exist."""
        config = load_config(str(tmp_path / "nonexistent.json"))

        # Should return default config
        assert config.general.default_symbols == ["AAPL", "MSFT"]

    def test_load_config_invalid_json(self, tmp_path):
        """Test loading config with invalid JSON."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text("{ invalid json }")

        config = load_config(str(config_file))

        # Should return default config
        assert config.general.default_symbols == ["AAPL", "MSFT"]

    def test_load_config_preserves_raw(self, tmp_path):
        """Test that raw dict is preserved."""
        config_file = tmp_path / "trading_config.json"
        config_data = {"general": {"default_symbols": ["AAPL"]}, "custom_section": {"custom_value": 123}}
        config_file.write_text(json.dumps(config_data))

        config = load_config(str(config_file))

        assert "custom_section" in config._raw
        assert config._raw["custom_section"]["custom_value"] == 123

    def test_load_all_sections(self, tmp_path):
        """Test loading all configuration sections."""
        config_file = tmp_path / "trading_config.json"
        config_data = {
            "general": {"default_mode": "live"},
            "simulation": {"steps": 100},
            "alpaca": {"paper": False},
            "schwab": {"session": "EXTENDED"},
            "risk": {"risk_per_trade": 0.02},
            "drawdown_monitor": {"enabled": True},
            "position_sizer": {"type": "fixed"},
            "trade_logic": {"cooldown_bars": 10},
            "indicators": {"atr_period": 20},
            "data": {"max_stale_minutes": 30},
            "gui": {"update_interval_ms": 500},
            "logging": {"console_enabled": False},
            "autotrader": {"dry_run": True},
            "streaming": {
                "bar_interval_ms": 500,
                "ensemble_enabled": True,
                "ensemble_mode": "weighted",
            },
        }
        config_file.write_text(json.dumps(config_data))

        config = load_config(str(config_file))

        assert config.general.default_mode == "live"
        assert config.simulation.steps == 100
        assert config.alpaca.paper is False
        assert config.schwab.session == "EXTENDED"
        assert config.risk.risk_per_trade == 0.02
        assert config.drawdown_monitor.enabled is True
        assert config.position_sizer.type == "fixed"
        assert config.trade_logic.cooldown_bars == 10
        assert config.indicators.atr_period == 20
        assert config.data.max_stale_minutes == 30
        assert config.gui.update_interval_ms == 500
        assert config.logging.console_enabled is False
        assert config.autotrader.dry_run is True
        assert config.streaming.bar_interval_ms == 500
        assert config.streaming.ensemble_enabled is True
        assert config.streaming.ensemble_mode == "weighted"


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_caches(self):
        """Test that config is cached."""
        # Reset global config
        import core.config_loader as cl

        cl._config = None

        with patch("core.config_loader.load_config") as mock_load:
            mock_load.return_value = TradingConfig()

            config1 = get_config()
            config2 = get_config()

            # Should only load once
            assert mock_load.call_count == 1
            assert config1 is config2

    def test_get_config_reload(self):
        """Test config reload."""
        import core.config_loader as cl

        cl._config = None

        with patch("core.config_loader.load_config") as mock_load:
            mock_load.return_value = TradingConfig()

            get_config()
            get_config(reload=True)

            # Should load twice
            assert mock_load.call_count == 2


class TestReloadConfig:
    """Tests for reload_config function."""

    def test_reload_config(self):
        """Test reload_config function."""
        import core.config_loader as cl

        cl._config = None

        with patch("core.config_loader.load_config") as mock_load:
            mock_load.return_value = TradingConfig()

            # First load
            get_config()

            # Reload
            reload_config()

            assert mock_load.call_count == 2


class TestConfigProxy:
    """Tests for the config proxy object."""

    def test_proxy_access(self):
        """Test accessing config through proxy."""
        import core.config_loader as cl

        with patch.object(cl, "get_config") as mock_get:
            mock_config = TradingConfig()
            mock_get.return_value = mock_config

            # Access through proxy
            result = cl.config.general

            assert result is not None
            mock_get.assert_called()
