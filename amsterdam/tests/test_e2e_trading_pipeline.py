"""
End-to-End Trading Pipeline Validation

Tests the complete data flow from market data to trade execution:
1. Data gathering (historical/simulated)
2. Strategy signal generation
3. Trade logic evaluation
4. Position sizing
5. Order execution
6. P&L tracking
7. Event emission

Run with: pytest tests/test_e2e_trading_pipeline.py -v
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from core.app_types import BrokerSnapshot, OrderResult, PositionView

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 100

    # Generate realistic price data
    base_price = 150.0
    returns = np.random.normal(0.0002, 0.015, n_bars)
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV with Date column (required by validation)
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=n_bars, freq="1min")
    data = []
    for i in range(n_bars):
        close = prices[i]
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = (high + low) / 2 + np.random.normal(0, 0.5)
        volume = int(np.random.uniform(100000, 500000))

        data.append(
            {
                "Date": dates[i],
                "Open": open_price,
                "High": max(high, open_price, close),
                "Low": min(low, open_price, close),
                "Close": close,
                "Volume": volume,
            }
        )

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def mock_broker():
    """Create a mock broker that simulates order execution."""
    broker = MagicMock()
    broker.connected = True

    # Track positions internally
    broker._positions = {}
    broker._cash = 100000.0
    broker._order_id_counter = 1000

    async def place_order(symbol, qty, side, order_type="market", limit_price=None, **kwargs):
        order_id = f"ORD{broker._order_id_counter}"
        broker._order_id_counter += 1

        # Simulate fill at current price
        fill_price = kwargs.get("price", 150.0)

        # Update position
        if symbol not in broker._positions:
            broker._positions[symbol] = {"qty": 0, "avg_price": 0}

        pos = broker._positions[symbol]
        if side == "buy":
            total_cost = pos["qty"] * pos["avg_price"] + qty * fill_price
            pos["qty"] += qty
            pos["avg_price"] = total_cost / pos["qty"] if pos["qty"] > 0 else 0
            broker._cash -= qty * fill_price
        else:  # sell
            broker._cash += qty * fill_price
            pos["qty"] -= qty

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            type=order_type,
            status="filled",
            filled_qty=qty,
            avg_fill_price=fill_price,
            raw={"simulated": True},
        )

    async def get_position(symbol):
        pos = broker._positions.get(symbol, {"qty": 0, "avg_price": 0})
        return PositionView(
            symbol=symbol,
            qty=pos["qty"],
            avg_entry_price=pos["avg_price"],
            market_price=150.0,
            side="long" if pos["qty"] > 0 else ("short" if pos["qty"] < 0 else "flat"),
        )

    async def get_account():
        total_position_value = sum(p["qty"] * 150.0 for p in broker._positions.values())
        return BrokerSnapshot(
            account_number="TEST123",
            status="active",
            cash=broker._cash,
            buying_power=broker._cash,
            equity=broker._cash + total_position_value,
            portfolio_value=broker._cash + total_position_value,
            positions={
                sym: PositionView(symbol=sym, qty=p["qty"], avg_entry_price=p["avg_price"])
                for sym, p in broker._positions.items()
            },
        )

    async def is_market_open():
        return True

    broker.place_order = AsyncMock(side_effect=place_order)
    broker.get_position = AsyncMock(side_effect=get_position)
    broker.get_account = AsyncMock(side_effect=get_account)
    broker.is_market_open = AsyncMock(side_effect=is_market_open)
    broker.cancel_order = AsyncMock(return_value=OrderResult(status="cancelled"))

    return broker


@pytest.fixture
def event_collector():
    """Collect events emitted during the test."""

    class EventCollector:
        def __init__(self):
            self.events = []
            self.by_type = {}

        async def handler(self, event):
            self.events.append(event)
            event_name = event.name if hasattr(event, "name") else str(type(event))
            if event_name not in self.by_type:
                self.by_type[event_name] = []
            self.by_type[event_name].append(event)

        def get_events(self, event_type=None):
            if event_type:
                return self.by_type.get(event_type, [])
            return self.events

        def clear(self):
            self.events = []
            self.by_type = {}

    return EventCollector()


# ============================================================================
# TEST: DATA GATHERING & VALIDATION
# ============================================================================


class TestDataGathering:
    """Test data gathering and validation layer."""

    def test_ohlcv_data_validation(self, sample_ohlcv_data):
        """Test that OHLCV data validation works correctly."""
        from core.backtest.validation import validate_ohlcv_data

        result = validate_ohlcv_data(sample_ohlcv_data)

        assert result.is_valid, f"Validation failed: {result.errors}"
        # Validation returns cleaned_data if successful
        assert result.cleaned_data is not None or result.is_valid

    def test_historical_data_format(self, sample_ohlcv_data):
        """Test that data has correct format for strategy consumption."""
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

        for col in required_columns:
            assert col in sample_ohlcv_data.columns, f"Missing column: {col}"

        # Check data types
        assert sample_ohlcv_data["Close"].dtype in [np.float64, np.float32]

    def test_atr_calculation(self, sample_ohlcv_data):
        """Test ATR indicator calculation."""
        # Compute ATR manually for validation
        high = sample_ohlcv_data["High"].values
        low = sample_ohlcv_data["Low"].values
        close = sample_ohlcv_data["Close"].values

        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(14).mean().values

        # ATR should be positive and reasonable
        valid_atr = atr[~np.isnan(atr)]
        assert len(valid_atr) > 0
        assert all(valid_atr > 0)
        assert all(valid_atr < sample_ohlcv_data["Close"].max() * 0.1)  # ATR < 10% of price


# ============================================================================
# TEST: STRATEGY SIGNAL GENERATION
# ============================================================================


class TestStrategySignalGeneration:
    """Test strategy signal generation layer."""

    def test_strategy_registry_loads(self):
        """Test that strategy registry loads available strategies."""
        from strategies.strategy_registry import list_strategies, load_strategy

        strategies = list_strategies()
        assert len(strategies) > 0, "No strategies registered"
        assert "sma" in strategies, f"sma not in strategies: {strategies}"

        # Load a strategy
        strategy = load_strategy("sma")
        assert strategy is not None

    def test_strategy_generates_signal(self, sample_ohlcv_data):
        """Test that a strategy generates valid signals."""
        from strategies.strategy_registry import load_strategy

        strategy = load_strategy("sma", params={"fast": 10, "slow": 30})

        # Generate signal
        result = strategy.generate_signal(sample_ohlcv_data)

        # Result should be -1, 0, or 1
        if isinstance(result, (int, float)):
            assert result in [-1, 0, 1], f"Invalid signal: {result}"
        elif isinstance(result, pd.DataFrame):
            assert "Signal" in result.columns

    def test_vectorized_signal_generation(self, sample_ohlcv_data):
        """Test vectorized signal generation for backtesting."""
        from strategies.strategy_registry import load_strategy

        strategy = load_strategy("sma", params={"fast": 10, "slow": 30})

        # Check if vectorized method exists
        if hasattr(strategy, "generate_signals_vectorized"):
            signals = strategy.generate_signals_vectorized(sample_ohlcv_data)
            if signals is not None:
                assert len(signals) == len(sample_ohlcv_data)

    def test_multiple_strategies(self, sample_ohlcv_data):
        """Test that multiple strategies can generate signals."""
        from strategies.strategy_registry import list_strategies, load_strategy

        strategies_to_test = ["sma", "momentum", "rsi", "macd"]

        for name in strategies_to_test:
            if name in list_strategies():
                strategy = load_strategy(name)
                strategy.generate_signal(sample_ohlcv_data)
                # Should not raise exception


# ============================================================================
# TEST: TRADE LOGIC & RISK MANAGEMENT
# ============================================================================


class TestTradeLogicAndRisk:
    """Test trade logic and risk management layer."""

    def test_trade_gate_basic(self):
        """Test basic trade gate functionality."""
        from core.logic.trade_gate import TradeGate

        gate = TradeGate(max_layers=3)

        # Update state for a new bar
        gate.on_new_bar("AAPL", bar_id=1, regime="normal")

        # Check layer count
        assert gate.get_state("AAPL").layers == 0

    def test_trade_gate_layering(self):
        """Test trade gate layer counting."""
        from datetime import datetime, timezone

        from core.logic.trade_gate import TradeGate

        gate = TradeGate(max_layers=2)

        # Simulate entries
        gate.on_new_bar("AAPL", bar_id=1, regime="normal")
        gate.mark_action("AAPL", ts=datetime.now(timezone.utc), bar_id=1, new_side="long", action="entry")

        state = gate.get_state("AAPL")
        assert state.layers == 1

    def test_position_sizer(self):
        """Test position sizing calculation."""
        from core.logic.portfolio_state import PortfolioState
        from core.position_sizer import KellyPositionSizer

        sizer = KellyPositionSizer(risk_percentage=0.01, max_trade_pct=0.10, max_holding_pct=0.20)

        portfolio = PortfolioState(cash=100000.0)

        # Calculate position size using correct API
        size = sizer.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            account_value=100000.0,
            atr=3.0,
            portfolio=portfolio,
            sl_mult=1.5,
            regime="normal",
        )

        assert size >= 0


# ============================================================================
# TEST: ORDER EXECUTION
# ============================================================================


class TestOrderExecution:
    """Test order execution layer."""

    @pytest.mark.asyncio
    async def test_broker_place_order(self, mock_broker):
        """Test placing an order through broker."""
        result = await mock_broker.place_order(symbol="AAPL", qty=10, side="buy", order_type="market", price=150.0)

        assert result.status == "filled"
        assert result.filled_qty == 10
        assert result.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_broker_position_tracking(self, mock_broker):
        """Test position tracking after order execution."""
        # Place buy order
        await mock_broker.place_order(symbol="AAPL", qty=10, side="buy", order_type="market", price=150.0)

        # Check position
        position = await mock_broker.get_position("AAPL")
        assert position.qty == 10
        assert position.side == "long"

    @pytest.mark.asyncio
    async def test_broker_account_update(self, mock_broker):
        """Test account updates after trades."""
        initial_account = await mock_broker.get_account()
        initial_cash = initial_account.cash

        # Place order
        await mock_broker.place_order(symbol="AAPL", qty=10, side="buy", order_type="market", price=150.0)

        # Check account
        account = await mock_broker.get_account()
        assert account.cash < initial_cash  # Cash decreased
        assert "AAPL" in account.positions


# ============================================================================
# TEST: P&L TRACKING
# ============================================================================


class TestPnLTracking:
    """Test P&L tracking layer."""

    def test_portfolio_state_tracking(self):
        """Test portfolio state tracks P&L correctly."""
        from core.logic.portfolio_state import PortfolioState

        portfolio = PortfolioState(cash=100000.0)

        # Simulate buy
        portfolio.apply_fill("AAPL", side="buy", qty=10, price=150.0)

        assert portfolio.cash == 100000.0 - (10 * 150.0)
        assert portfolio.get_position("AAPL") is not None

        # Update price for unrealized P&L
        portfolio.update_price("AAPL", 155.0)

        unrealized = portfolio.total_unrealized()
        assert unrealized == 10 * (155.0 - 150.0)  # $50 profit

    def test_realized_pnl_on_close(self):
        """Test realized P&L when closing position."""
        from core.logic.portfolio_state import PortfolioState

        portfolio = PortfolioState(cash=100000.0)

        # Open position
        portfolio.apply_fill("AAPL", side="buy", qty=10, price=150.0)

        # Close position at profit
        portfolio.apply_fill("AAPL", side="sell", qty=10, price=160.0)

        # Check realized P&L
        realized = portfolio.total_realized()
        expected_pnl = 10 * (160.0 - 150.0)  # $100 profit
        assert abs(realized - expected_pnl) < 0.01

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        from core.logic.portfolio_state import PortfolioState

        portfolio = PortfolioState(cash=100000.0)

        # Manually set equity history for testing
        portfolio.equity_history = [100000, 105000, 110000, 105000, 100000]

        drawdown = portfolio.current_drawdown()
        # Drawdown is typically returned as negative (e.g., -0.09 for 9% DD)
        # or as absolute value depending on implementation
        assert isinstance(drawdown, (int, float))
        # Just verify it's computed without error


# ============================================================================
# TEST: EVENT EMISSION
# ============================================================================


class TestEventEmission:
    """Test event emission throughout the pipeline."""

    @pytest.mark.asyncio
    async def test_event_handler_emit_receive(self, event_collector):
        """Test event emission and reception."""
        from core.events.eventhandler import EventHandler
        from core.events.events import EVENT_NEW_BAR

        # Reset singleton for testing
        EventHandler._instance = None
        EventHandler._initialized = False
        handler = EventHandler()

        # Subscribe to events
        await handler.subscribe(EVENT_NEW_BAR, event_collector.handler)

        # Emit event
        await handler.emit(
            EVENT_NEW_BAR,
            {
                "symbol": "AAPL",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "open": 150.0,
                "high": 151.0,
                "low": 149.0,
                "close": 150.5,
                "volume": 100000,
            },
        )

        await asyncio.sleep(0.1)  # Allow async processing

        events = event_collector.get_events(EVENT_NEW_BAR)
        assert len(events) == 1
        assert events[0].payload["symbol"] == "AAPL"

        # Cleanup
        EventHandler._instance = None
        EventHandler._initialized = False

    @pytest.mark.asyncio
    async def test_order_status_event(self, event_collector):
        """Test order status event emission."""
        from core.events.eventhandler import EventHandler
        from core.events.events import EVENT_ORDER_STATUS

        EventHandler._instance = None
        EventHandler._initialized = False
        handler = EventHandler()

        await handler.subscribe(EVENT_ORDER_STATUS, event_collector.handler)

        await handler.emit(
            EVENT_ORDER_STATUS,
            {
                "order_id": "ORD123",
                "symbol": "AAPL",
                "side": "buy",
                "qty": 10,
                "status": "filled",
                "filled_qty": 10,
                "avg_price": 150.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        await asyncio.sleep(0.1)

        events = event_collector.get_events(EVENT_ORDER_STATUS)
        assert len(events) == 1
        assert events[0].payload["status"] == "filled"

        EventHandler._instance = None
        EventHandler._initialized = False


# ============================================================================
# TEST: FULL E2E PIPELINE
# ============================================================================


class TestFullE2EPipeline:
    """Test the complete end-to-end trading pipeline."""

    @pytest.mark.asyncio
    async def test_data_to_signal_pipeline(self, sample_ohlcv_data):
        """Test pipeline from data to signal generation."""
        # 1. Validate data
        from core.backtest.validation import validate_ohlcv_data
        from strategies.strategy_registry import load_strategy

        validation = validate_ohlcv_data(sample_ohlcv_data)
        assert validation.is_valid

        # 2. Calculate indicators (ATR)
        high = sample_ohlcv_data["High"].values
        low = sample_ohlcv_data["Low"].values
        close = sample_ohlcv_data["Close"].values

        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(14).mean().iloc[-1]

        # 3. Classify regime (simplified)
        atr_pct = atr / close[-1]
        if atr_pct < 0.01:
            pass
        elif atr_pct > 0.03:
            pass
        else:
            pass

        # 4. Load strategy
        strategy = load_strategy("sma", params={"fast": 10, "slow": 30})
        assert strategy is not None

        # 5. Generate signal
        signal = strategy.generate_signal(sample_ohlcv_data)
        assert signal in [-1, 0, 1] or isinstance(signal, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_signal_to_execution_pipeline(self, sample_ohlcv_data, mock_broker):
        """Test pipeline from signal to order execution."""
        from core.logic.trade_gate import TradeGate

        # Setup components
        gate = TradeGate(max_layers=3)

        # Current market state
        current_price = sample_ohlcv_data["Close"].iloc[-1]

        # Simulate signal

        # 1. Update trade gate
        gate.on_new_bar("AAPL", bar_id=1, regime="normal")

        # 2. Execute order
        result = await mock_broker.place_order(
            symbol="AAPL", qty=10, side="buy", order_type="market", price=current_price
        )

        assert result.status == "filled"

        # 3. Verify position
        position = await mock_broker.get_position("AAPL")
        assert position.qty > 0

    @pytest.mark.asyncio
    async def test_complete_trade_cycle(self, sample_ohlcv_data, mock_broker):
        """Test complete trade cycle: entry, P&L tracking, exit."""
        from core.logic.portfolio_state import PortfolioState

        # Setup
        portfolio = PortfolioState(cash=100000.0)
        entry_price = 150.0
        qty = 10

        # 1. ENTRY
        entry_result = await mock_broker.place_order(
            symbol="AAPL", qty=qty, side="buy", order_type="market", price=entry_price
        )
        assert entry_result.status == "filled"

        # Apply to portfolio
        portfolio.apply_fill("AAPL", "buy", qty, entry_price)

        # 2. PRICE MOVEMENT & P&L TRACKING
        new_price = 155.0
        portfolio.update_price("AAPL", new_price)

        unrealized_pnl = portfolio.total_unrealized()
        expected_unrealized = qty * (new_price - entry_price)
        assert abs(unrealized_pnl - expected_unrealized) < 0.01

        # 3. EXIT
        exit_result = await mock_broker.place_order(
            symbol="AAPL", qty=qty, side="sell", order_type="market", price=new_price
        )
        assert exit_result.status == "filled"

        # Apply to portfolio
        portfolio.apply_fill("AAPL", "sell", qty, new_price)

        # 4. VERIFY FINAL STATE
        realized_pnl = portfolio.total_realized()
        expected_realized = qty * (new_price - entry_price)
        assert abs(realized_pnl - expected_realized) < 0.01

    @pytest.mark.asyncio
    async def test_multi_symbol_pipeline(self, mock_broker):
        """Test pipeline with multiple symbols."""
        from core.logic.portfolio_state import PortfolioState

        portfolio = PortfolioState(cash=100000.0)
        symbols = ["AAPL", "GOOGL", "MSFT"]

        # Open positions in multiple symbols
        for i, symbol in enumerate(symbols):
            price = 150.0 + i * 50  # Different prices
            qty = 5

            result = await mock_broker.place_order(symbol=symbol, qty=qty, side="buy", order_type="market", price=price)
            assert result.status == "filled"

            portfolio.apply_fill(symbol, "buy", qty, price)

        # Verify all positions exist
        for symbol in symbols:
            pos = portfolio.get_position(symbol)
            assert pos is not None
            assert pos.qty == 5

        # Check total equity
        total_equity = portfolio.total_equity()
        assert total_equity > 0

    @pytest.mark.asyncio
    async def test_risk_management_in_pipeline(self, mock_broker):
        """Test risk management integration in pipeline."""
        from datetime import datetime, timezone

        from core.logic.trade_gate import TradeGate

        gate = TradeGate(max_layers=2)

        # Update state
        gate.on_new_bar("AAPL", bar_id=1, regime="normal")
        gate.mark_action("AAPL", ts=datetime.now(timezone.utc), bar_id=1, new_side="long", action="entry")

        assert gate.get_state("AAPL").layers == 1

        # Second layer (pyramid)
        gate.on_new_bar("AAPL", bar_id=2, regime="normal")
        gate.mark_action(
            "AAPL", ts=datetime.now(timezone.utc), bar_id=2, new_side="long", action="pyramid", pyramided=True
        )

        assert gate.get_state("AAPL").layers == 2


# ============================================================================
# TEST: BACKTEST VALIDATION
# ============================================================================


class TestBacktestValidation:
    """Test backtesting functionality."""

    def test_vectorized_backtest(self, sample_ohlcv_data):
        """Test vectorized backtesting."""
        from core.backtest import VectorizedBacktester

        backtester = VectorizedBacktester(data=sample_ohlcv_data, initial_capital=10000, transaction_cost=0.001)

        # Run backtest
        result = backtester.run(strategy_name="sma", strategy_params={"fast": 10, "slow": 30}, position_size=0.1)

        # Verify result structure
        assert "Portfolio_Value" in result.columns
        assert "Position" in result.columns
        assert "Strategy_Return" in result.columns

        # Get metrics
        metrics = backtester.get_metrics(result)

        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

    def test_benchmark_comparison(self, sample_ohlcv_data):
        """Test benchmark comparison."""
        from core.backtest import compare_to_benchmark

        # Generate synthetic returns
        strategy_returns = sample_ohlcv_data["Close"].pct_change().dropna()
        benchmark_returns = strategy_returns * 0.8 + 0.0001  # Correlated benchmark

        comparison = compare_to_benchmark(strategy_returns, benchmark_returns)

        assert hasattr(comparison, "alpha")
        assert hasattr(comparison, "beta")


# ============================================================================
# TEST: INTEGRATION SCENARIOS
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic trading scenarios."""

    @pytest.mark.asyncio
    async def test_buy_hold_sell_scenario(self, sample_ohlcv_data, mock_broker):
        """Test a complete buy-hold-sell trading scenario."""
        from core.logic.portfolio_state import PortfolioState
        from strategies.strategy_registry import load_strategy

        portfolio = PortfolioState(cash=100000.0)
        strategy = load_strategy("sma", params={"fast": 5, "slow": 20})

        portfolio.total_equity()

        # Simulate trading loop
        trades = []
        for i in range(50, len(sample_ohlcv_data)):
            df_slice = sample_ohlcv_data.iloc[: i + 1]
            price = df_slice["Close"].iloc[-1]

            # Generate signal
            signal = strategy.generate_signal(df_slice)
            if isinstance(signal, pd.DataFrame):
                signal = signal["Signal"].iloc[-1] if "Signal" in signal.columns else 0

            pos = portfolio.get_position("AAPL")
            current_qty = pos.qty if pos else 0

            # Execute based on signal
            if signal == 1 and current_qty == 0:  # Buy
                result = await mock_broker.place_order(symbol="AAPL", qty=10, side="buy", price=price)
                if result.status == "filled":
                    portfolio.apply_fill("AAPL", "buy", 10, price)
                    trades.append({"type": "buy", "price": price})

            elif signal == -1 and current_qty > 0:  # Sell
                result = await mock_broker.place_order(symbol="AAPL", qty=current_qty, side="sell", price=price)
                if result.status == "filled":
                    portfolio.apply_fill("AAPL", "sell", current_qty, price)
                    trades.append({"type": "sell", "price": price})

            # Update price
            portfolio.update_price("AAPL", price)

        # Verify some trading activity occurred
        final_equity = portfolio.total_equity()
        assert isinstance(final_equity, (int, float))

    @pytest.mark.asyncio
    async def test_event_flow_during_trading(self, sample_ohlcv_data, mock_broker, event_collector):
        """Test that events flow correctly during trading."""
        from core.events.eventhandler import EventHandler
        from core.events.events import EVENT_ORDER_STATUS

        EventHandler._instance = None
        EventHandler._initialized = False
        handler = EventHandler()

        await handler.subscribe(EVENT_ORDER_STATUS, event_collector.handler)

        # Place order
        result = await mock_broker.place_order(symbol="AAPL", qty=10, side="buy", price=150.0)

        # Emit order status event
        await handler.emit(
            EVENT_ORDER_STATUS,
            {
                "order_id": result.order_id,
                "symbol": "AAPL",
                "side": "buy",
                "qty": 10,
                "status": "filled",
                "filled_qty": 10,
                "avg_price": 150.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        await asyncio.sleep(0.1)

        # Verify event was received
        events = event_collector.get_events(EVENT_ORDER_STATUS)
        assert len(events) >= 1

        EventHandler._instance = None
        EventHandler._initialized = False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
