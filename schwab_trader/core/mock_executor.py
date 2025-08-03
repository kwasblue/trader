# core/mock_executor.py
from utils.logger import Logger
import pandas as pd
from pathlib import Path
from datetime import datetime, UTC
from collections import defaultdict
from core.position_sizer import DynamicPositionSizer

class MockExecutor:
    """
    A simulated trade execution environment for testing strategies without real trades.
    Tracks portfolio value, position sizing, drawdown, and logs all trade actions.
    """

    def __init__(self, risk_percentage=0.07):
        self.logger = Logger('app.log', 'MockExecutor', log_dir='logs').get_logger()
        self.peak_portfolio_value = defaultdict(lambda: 0.0)
        self.portfolio_history = defaultdict(list)
        self.cash = defaultdict(lambda: 100_000.0)  # Starting cash per symbol
        self.position = defaultdict(int)
        self.total_fees = defaultdict(float)
        self.risk_percentage = risk_percentage
        self.sizer = DynamicPositionSizer(risk_percentage=self.risk_percentage)

    def execute(self, symbol, df, signal, price, atr_value):
        """
        Simulate a trade given a signal and current market conditions.
        """
        if signal == 0 or pd.isna(atr_value) or atr_value <= 0:
            self.logger.debug(f"[HOLD] {symbol}: No action taken.")
            return

        # Determine volatility regime
        atr_25 = df['ATR'].quantile(0.25)
        atr_75 = df['ATR'].quantile(0.75)
        if atr_value < atr_25:
            market_conditions = "low_volatility"
        elif atr_value > atr_75:
            market_conditions = "high_volatility"
        else:
            market_conditions = "normal"

        stop_loss_price = price - (atr_value * 2) if signal == 1 else price + (atr_value * 2)

        # Calculate quantity with risk adjustment and signal direction
        quantity = self.sizer.calculate_position_size(
            stock_price=price,
            stop_loss_price=stop_loss_price,
            current_cash=self.cash[symbol],
            market_conditions=market_conditions,
            signal=signal  # <- Pass signal to influence sizing logic
        )

        # Simulated fees
        trade_fee = 0.001 * price * quantity
        max_affordable_qty = self.cash[symbol] // (price + trade_fee)
        quantity = min(quantity, max_affordable_qty)

        now = datetime.now(UTC)

        # Execute BUY
        if signal == 1 and quantity > 0 and self.position[symbol] == 0:
            self.cash[symbol] -= (price * quantity + trade_fee)
            self.position[symbol] += quantity
            self.total_fees[symbol] += trade_fee
            self.logger.info(f"[BUY] {symbol}: {quantity} @ {price:.2f} | SL: {stop_loss_price:.2f} | Cash: {self.cash[symbol]:.2f}")

        # Execute SELL
        elif signal == -1 and self.position[symbol] > 0:
            self.cash[symbol] += (price * self.position[symbol] - trade_fee)
            self.logger.info(f"[SELL] {symbol}: {self.position[symbol]} @ {price:.2f} | Cash: {self.cash[symbol]:.2f}")
            self.total_fees[symbol] += trade_fee
            self.position[symbol] = 0

        # Portfolio tracking
        portfolio_value = self.cash[symbol] + self.position[symbol] * price
        self.peak_portfolio_value[symbol] = max(self.peak_portfolio_value[symbol], portfolio_value)
        drawdown = (portfolio_value - self.peak_portfolio_value[symbol]) / self.peak_portfolio_value[symbol]

        self.portfolio_history[symbol].append({
            "Date": now,
            "Portfolio_Value": portfolio_value,
            "Cash": self.cash[symbol],
            "Position": self.position[symbol],
            "Price": price,
            "Drawdown": drawdown,
            "Fees": self.total_fees[symbol]
        })

    def save_results(self, path: str):
        """
        Save all portfolio histories to disk as individual CSVs.
        """
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        for symbol, history in self.portfolio_history.items():
            df = pd.DataFrame(history)
            df.to_csv(output_path / f"mock_results_{symbol}.csv", index=False)
            self.logger.info(f"Saved mock results for {symbol} to {output_path}")
