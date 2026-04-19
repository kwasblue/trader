"""
Pytest configuration for schwab_trader tests.

Provides:
- Async test support
- Common fixtures
- Test utilities
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Environment Fixtures
# ============================================================================


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "ALPACA_API_KEY": "test_alpaca_key",
        "ALPACA_SECRET_KEY": "test_alpaca_secret",
        "SCHWAB_API_KEY": "test_schwab_key",
        "SCHWAB_SECRET": "test_schwab_secret",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def clean_env():
    """Provide clean environment without trading credentials."""
    vars_to_remove = [
        "ALPACA_API_KEY",
        "ALPACA_KEY_ID",
        "ALPACA_SECRET_KEY",
        "ALPACA_SECRET",
        "SCHWAB_API_KEY",
        "SCHWAB_SECRET",
    ]
    original = {k: os.environ.get(k) for k in vars_to_remove}

    for var in vars_to_remove:
        os.environ.pop(var, None)

    yield

    # Restore
    for var, value in original.items():
        if value is not None:
            os.environ[var] = value


# ============================================================================
# Broker Fixtures
# ============================================================================


@pytest.fixture
def mock_broker():
    """Create a mock broker for testing."""
    from core.app_types import OrderResult, PositionView

    broker = MagicMock()
    broker.place_order = AsyncMock(
        return_value=OrderResult(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            qty=10,
            status="filled",
            filled_qty=10,
            avg_fill_price=150.0,
            raw={},
        )
    )
    broker.cancel_order = AsyncMock(return_value=OrderResult(order_id="ORD123", status="cancelled", raw={}))
    broker.get_position = AsyncMock(
        return_value=PositionView(symbol="AAPL", qty=100, avg_entry_price=145.0, market_price=150.0, side="long")
    )
    broker.get_all_positions = AsyncMock(return_value=[])
    broker.is_market_open = AsyncMock(return_value=True)
    broker.get_account = AsyncMock(return_value=MagicMock(equity=100000.0, buying_power=50000.0, cash=25000.0))
    broker.connect = MagicMock()
    broker.subscribe_bars = MagicMock()
    broker.start_stream = AsyncMock()

    return broker


@pytest.fixture
def mock_alpaca_broker(mock_broker):
    """Create a mock Alpaca broker."""
    mock_broker.name = "alpaca"
    mock_broker.paper = True
    return mock_broker


@pytest.fixture
def mock_schwab_broker(mock_broker):
    """Create a mock Schwab broker."""
    mock_broker.name = "schwab"
    return mock_broker


# ============================================================================
# Trading Component Fixtures
# ============================================================================


@pytest.fixture
def mock_sizer():
    """Create a mock position sizer."""
    sizer = MagicMock()
    sizer.calculate_position_size.return_value = 10
    sizer.calculate_risk_amount.return_value = 100.0
    sizer.calculate_stop_distance.return_value = 3.0
    return sizer


@pytest.fixture
def mock_trade_logger():
    """Create a mock trade logger."""
    logger = MagicMock()
    logger.log_trade = MagicMock()
    logger.log_signal = MagicMock()
    logger.flush = MagicMock()
    return logger


@pytest.fixture
def mock_trade_logic_manager():
    """Create a mock trade logic manager."""
    manager = MagicMock()
    mock_logic = MagicMock()
    mock_logic.should_enter.return_value = True
    mock_logic.should_exit.return_value = False
    mock_logic.get_stop_loss.return_value = 145.0
    mock_logic.get_take_profit.return_value = 160.0
    manager.get_logic.return_value = mock_logic
    return manager


@pytest.fixture
def mock_strategy_router():
    """Create a mock strategy router."""
    router = MagicMock()
    mock_strategy = MagicMock()
    mock_strategy.generate_signal.return_value = {"signal": "buy", "strength": 0.8}
    router.get_strategy.return_value = mock_strategy
    return router


@pytest.fixture
def mock_drawdown_monitor():
    """Create a mock drawdown monitor."""
    monitor = MagicMock()
    monitor.can_trade_symbol.return_value = True
    monitor.can_trade_portfolio.return_value = True
    monitor.update_symbol = MagicMock()
    monitor.update_portfolio = MagicMock()
    monitor.get_symbol_drawdown.return_value = 0.05
    monitor.get_portfolio_drawdown.return_value = 0.03
    return monitor


@pytest.fixture
def mock_trade_gate():
    """Create a mock trade gate."""
    gate = MagicMock()
    gate.can_enter.return_value = True
    gate.record_entry = MagicMock()
    gate.record_exit = MagicMock()
    return gate


# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def sample_bar_data():
    """Provide sample bar data for testing."""
    from datetime import datetime, timezone

    return [
        {
            "timestamp": datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc),
            "Open": 150.0,
            "High": 151.0,
            "Low": 149.0,
            "Close": 150.5,
            "Volume": 1000,
        },
        {
            "timestamp": datetime(2024, 1, 8, 10, 1, tzinfo=timezone.utc),
            "Open": 150.5,
            "High": 152.0,
            "Low": 150.0,
            "Close": 151.0,
            "Volume": 1100,
        },
        {
            "timestamp": datetime(2024, 1, 8, 10, 2, tzinfo=timezone.utc),
            "Open": 151.0,
            "High": 153.0,
            "Low": 150.5,
            "Close": 152.5,
            "Volume": 1200,
        },
    ]


@pytest.fixture
def sample_ohlcv_df(sample_bar_data):
    """Provide sample OHLCV DataFrame."""
    import pandas as pd

    return pd.DataFrame(sample_bar_data)


@pytest.fixture
def sample_position():
    """Provide sample position data."""
    from core.app_types import PositionView

    return PositionView(symbol="AAPL", qty=100, avg_entry_price=145.0, market_price=150.0, side="long")


# ============================================================================
# Config Fixtures
# ============================================================================


@pytest.fixture
def mock_config():
    """Create a mock trading config."""
    from core.config_loader import TradingConfig

    return TradingConfig()


@pytest.fixture
def sample_config_dict():
    """Provide sample configuration dictionary."""
    return {
        "general": {"default_symbols": ["AAPL", "MSFT"], "default_mode": "simulation", "log_level": "INFO"},
        "simulation": {"enabled": True, "steps": 1000, "bar_sleep": 0.1, "starting_cash": 100000.0},
        "risk": {"risk_per_trade": 0.01, "max_trade_pct": 0.10, "daily_loss_limit": 1000.0},
        "trade_logic": {"cooldown_bars": 5, "tp_mult_normal": 2.0, "sl_mult_normal": 1.5},
    }


# ============================================================================
# Event Fixtures
# ============================================================================


@pytest.fixture
def mock_event_handler():
    """Create a mock event handler."""
    handler = MagicMock()
    handler.emit = AsyncMock()
    handler.subscribe = AsyncMock()
    handler.unsubscribe = MagicMock()
    return handler


# ============================================================================
# File System Fixtures
# ============================================================================


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    proc_data = tmp_path / "data" / "data_storage" / "proc_data"
    raw_data = tmp_path / "data" / "data_storage" / "raw_data"
    logs = tmp_path / "logs"
    config = tmp_path / "config"

    for d in [proc_data, raw_data, logs, config]:
        d.mkdir(parents=True)

    return {"root": tmp_path, "proc_data": proc_data, "raw_data": raw_data, "logs": logs, "config": config}


@pytest.fixture
def sample_token_file(tmp_path):
    """Create a sample Schwab token file."""
    import json
    import time

    token_data = {
        "access_token": "test_access_token",
        "refresh_token": "test_refresh_token",
        "access_time": time.time(),
        "refresh_time": time.time(),
        "expires_in": 1800,
    }

    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    token_file = token_dir / "token_file.json"
    token_file.write_text(json.dumps(token_data))

    return token_file


# ============================================================================
# Utility Functions
# ============================================================================


def create_mock_bar(symbol, price, volume=1000):
    """Create a mock bar object."""
    from datetime import datetime, timezone

    bar = MagicMock()
    bar.symbol = symbol
    bar.timestamp = datetime.now(timezone.utc)
    bar.open = price
    bar.high = price * 1.01
    bar.low = price * 0.99
    bar.close = price
    bar.volume = volume
    return bar


def create_mock_order_result(order_id, symbol, side, qty, status="filled", fill_price=None):
    """Create a mock order result."""
    from core.app_types import OrderResult

    return OrderResult(
        order_id=order_id,
        symbol=symbol,
        side=side,
        qty=qty,
        status=status,
        filled_qty=qty if status == "filled" else 0,
        avg_fill_price=fill_price,
        raw={},
    )
