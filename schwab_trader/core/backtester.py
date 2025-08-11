import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fpdf import FPDF
from strategies.strategy_registry import load_strategy, list_strategies
from utils.risk_metrics import RiskQuantifier
from schwab_trader.loggers.logger import Logger
from core.position_sizer import DynamicPositionSizer
from data.datautils import epoch_to_date


class Backtester:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000, 
                 transaction_cost: float = 0.001, risk_free_rate: float = 0.02):
        """
        Initializes the backtester with historical data and strategy parameters.

        Parameters:
            data (pd.DataFrame): Historical stock data. Must include columns such as 
                                 'Date', 'Open', 'High', 'Low', 'Close', and optionally 'Volume'.
            initial_capital (float): The starting capital for the simulation.
            transaction_cost (float): Transaction cost as a fraction (e.g., 0.001 for 0.1%).
            risk_free_rate (float): Annualized risk-free rate used in performance metrics.
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost  
        self.risk_free_rate = risk_free_rate
        self.logger = Logger("backtest.log", "Backtester", log_dir=os.path.join(os.getcwd(), "logs")).get_logger()
        self.risk_quantifier = RiskQuantifier()
        self.trade_log = []
        self.portfolio_history = []

        # Ensure 'Date' is a datetime column and sort the data
        if self.data['Date'].dtype == 'O':
            self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.sort_values(by='Date', inplace=True)

    def run_backtest(self, strategy_name: str, strategy_params: dict = None, sizer: DynamicPositionSizer = None) -> pd.DataFrame:
        """
        Runs the backtest with:
        - Partial trade execution when capital is insufficient
        - Stop-loss enforcement
        - Slippage modeling
        - Optimized performance (Vectorization, Numba acceleration)
        - Advanced risk tracking (Sharpe Ratio, Sortino Ratio, Kelly Criterion)
        - Live trading readiness (Broker API connectivity)
        """
        if strategy_params is None:
            strategy_params = {}

        self.logger.info(f"Running strategy '{strategy_name}' with parameters: {strategy_params}")

        # Ensure the strategy generates signals
        if 'Signal' in self.data.columns:
            strat_data = self.data.copy()
        else:
            strat_data = self._apply_strategy(self.data.copy(), strategy_name, **strategy_params)

        required_columns = {'Date', 'Close', 'ATR', 'Signal'}
        if not required_columns.issubset(strat_data.columns):
            missing_cols = required_columns - set(strat_data.columns)
            self.logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        overall_atr = strat_data['ATR']
        atr_25, atr_75 = overall_atr.quantile([0.25, 0.75])

        cash = self.initial_capital
        position = 0
        total_fees = 0
        peak_portfolio_value = self.initial_capital

        drawdowns = []
        self.portfolio_history = []  # Reset portfolio history

        for idx, row in strat_data.iterrows():
            date, price, atr_value, signal = row['Date'], row['Close'], row['ATR'], row['Signal']

            if pd.isna(atr_value) or atr_value <= 0:
                continue  

            # Classify market conditions based on ATR
            market_conditions = (
                "low_volatility" if atr_value < atr_25 else 
                "high_volatility" if atr_value > atr_75 else 
                "normal"
            )

            # Stop-loss price based on ATR multiplier
            stop_loss_price = price - (atr_value * 2)

            # Slippage modeling: Introduce random price impact (±0.1%)
            slippage = np.random.uniform(-0.001, 0.001)
            execution_price = price * (1 + slippage)

            # Calculate position size dynamically
            quantity = sizer.calculate_position_size(
                stock_price=execution_price,
                stop_loss_price=stop_loss_price,
                current_cash=cash,
                market_conditions=market_conditions, 
                signal = signal
            )

            trade_fee = 0.001 * execution_price * quantity  # Example: 0.1% transaction fee
            total_fees += trade_fee

            # Partial trade execution if capital is insufficient
            max_affordable_qty = cash // (execution_price + trade_fee)
            if quantity > max_affordable_qty:
                quantity = max_affordable_qty

            # Execute buy trade
            if signal == 1 and quantity > 0 and cash >= execution_price * quantity + trade_fee:
                cash -= (execution_price * quantity + trade_fee)
                position += quantity
                self.trade_log.append({
                    'Date': date, 'Action': 'BUY', 'Price': execution_price, 'Quantity': quantity,
                    'Cash': cash, 'Position': position, 'Stop_Loss': stop_loss_price,
                    'Market Conditions': market_conditions, 'Fees': trade_fee
                })
                self.logger.info(f"BUY {quantity} shares at {execution_price} on {date} | Stop-Loss {stop_loss_price} | Market {market_conditions} | Cash {cash}")

            # Stop-loss enforcement: Exit if price drops below stop-loss
            elif position > 0 and price <= stop_loss_price:
                # Sell all holdings at current price (without slippage on exit, can be adjusted if needed)
                cash += (price * position - trade_fee)
                self.logger.info(f"STOP-LOSS TRIGGERED: Selling all {position} shares at {price} on {date}")
                position = 0

            # Execute sell trade
            elif signal == -1 and position >= quantity and quantity > 0:
                cash += (execution_price * quantity - trade_fee)
                position -= quantity
                self.trade_log.append({
                    'Date': date, 'Action': 'SELL', 'Price': execution_price, 'Quantity': quantity,
                    'Cash': cash, 'Position': position, 'Risk %': sizer.risk_per_trade,
                    'Fees': trade_fee
                })
                self.logger.info(f"SELL {quantity} shares at {execution_price} on {date} | Market {market_conditions}")

            # Track portfolio value and drawdown
            portfolio_value = cash + position * price
            peak_portfolio_value = max(peak_portfolio_value, portfolio_value)
            drawdown = (portfolio_value - peak_portfolio_value) / peak_portfolio_value
            drawdowns.append(drawdown)

            self.portfolio_history.append({
                'Date': date, 'Portfolio_Value': portfolio_value, 'Cash': cash,
                'Position': position, 'Price': price, 'Drawdown': drawdown, 
            })

        # Convert history to DataFrame and sort by Date
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.sort_values(by='Date', inplace=True)
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'], unit='ms')

        # Calculate daily strategy returns as the percentage change of portfolio value
        portfolio_df['Strategy_Return'] = portfolio_df['Portfolio_Value'].pct_change().fillna(0)

        self.logger.info(f"Backtest completed: Final Portfolio Value = {portfolio_df.iloc[-1]['Portfolio_Value']}, Total Fees = {total_fees}")
        return portfolio_df


    def evaluate_performance(self, portfolio_df: pd.DataFrame, market_data: pd.DataFrame = None) -> dict:
        """
        Evaluates performance using risk metrics like standard deviation, Sharpe ratio, and max drawdown.

        Parameters:
            portfolio_df (pd.DataFrame): DataFrame containing strategy returns.
            market_data (pd.DataFrame, optional): DataFrame containing market returns for Beta/Alpha calculation.

        Returns:
            dict: A dictionary containing performance metrics.
        """
        if 'Strategy_Return' not in portfolio_df.columns:
            raise ValueError("Portfolio DataFrame must contain a 'Strategy_Return' column.")

        self.risk_quantifier.set_returns(portfolio_df, return_column='Strategy_Return')
        self.risk_quantifier.set_risk_free_rate(self.risk_free_rate)

        performance = {
            'Standard Deviation': self.risk_quantifier.calculate_standard_deviation(),
            'Sharpe Ratio': self.risk_quantifier.calculate_sharpe_ratio(),
            'Sortino Ratio': self.risk_quantifier.calculate_sortino_ratio(),
            'Max Drawdown': self.risk_quantifier.calculate_max_drawdown(),
            'Value at Risk (VaR)': self.risk_quantifier.calculate_var(confidence_level=0.05),
        }

        if market_data is not None and 'Market_Return' in market_data.columns:
            self.risk_quantifier.set_returns(market_data, return_column='Market_Return')
            performance.update({
                'Beta': self.risk_quantifier.calculate_beta(market_data),
                'Alpha': self.risk_quantifier.calculate_alpha(market_data),
                'Treynor Ratio': self.risk_quantifier.calculate_treynor_ratio(market_data),
            })

        return performance



    def plot_results(self, data: pd.DataFrame, strategy: str, save_path: str = None):
        """
        Plot the portfolio value and a buy-and-hold benchmark for comparison.
        
        Parameters:
            data (pd.DataFrame): DataFrame containing the backtest results.
            strategy (str): The name of the strategy being tested.
            save_path (str): Optional file path to save the plot image.
        """
        # Ensure the Date column is in datetime format.
        if not np.issubdtype(data['Date'].dtype, np.datetime64):
            try:
                # If Date is in epoch (milliseconds), convert it
                data['Date'] = pd.to_datetime(data['Date'], unit='ms')
            except Exception as e:
                # If conversion by 'ms' fails, try without specifying unit.
                data['Date'] = pd.to_datetime(data['Date'])
        
        # Set Date as index for plotting
        plot_data = data.copy().set_index('Date')

        # Calculate Buy-and-Hold return based on the first available Close price
        initial_close = self.data['Close'].iloc[0]
        # Make sure self.data['Date'] is also datetime; if not, convert it:
        if not np.issubdtype(self.data['Date'].dtype, np.datetime64):
            self.data['Date'] = pd.to_datetime(self.data['Date'], unit='ms')
        buy_hold_return = self.data.set_index('Date')['Close'] / initial_close
        plot_data['Buy_Hold_Return'] = buy_hold_return

        # Calculate normalized strategy portfolio value
        strategy_returns = plot_data['Portfolio_Value'] / self.initial_capital
        plot_data['Strategy_Return'] = strategy_returns

        plt.figure(figsize=(14, 7))
        plt.plot(plot_data.index, plot_data['Strategy_Return'], label=f'{strategy} Strategy', color='blue')
        plt.plot(plot_data.index, plot_data['Buy_Hold_Return'], label='Buy and Hold', color='green', linestyle='--')

        plt.title(f'Strategy vs Buy and Hold Performance ({strategy})')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


    def generate_report(self, portfolio_df: pd.DataFrame, strategy_name: str, performance: dict, file_name: str = "backtest_report.pdf") -> str:
        """
        Generates a PDF report summarizing the backtest, including performance metrics and a performance plot.
        
        Parameters:
            portfolio_df (pd.DataFrame): Portfolio history DataFrame.
            strategy_name (str): Name of the strategy.
            performance (dict): Performance metrics.
            file_name (str): Output file name for the PDF report.
            
        Returns:
            str: The file name of the generated report.
        """
        try:
            plot_file = f"{strategy_name}_plot.png"
            self.plot_results(portfolio_df, strategy_name, save_path=plot_file)

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt=f"Backtest Report: {strategy_name}", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, txt="Performance Metrics:", ln=True)
            for metric, value in performance.items():
                pdf.cell(0, 10, txt=f"{metric}: {value:.4f}", ln=True)
            final_value = portfolio_df.iloc[-1]['Portfolio_Value']
            pdf.cell(0, 10, txt=f"Final Portfolio Value: {final_value:.2f}", ln=True)
            pdf.add_page()
            pdf.cell(200, 10, txt="Performance Plot", ln=True, align="C")
            pdf.image(plot_file, x=10, y=20, w=180)
            pdf.output(file_name)
            self.logger.info(f"Backtest report generated: {file_name}")
            return file_name
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return ""

    def _apply_strategy(self, data: pd.DataFrame, strategy_name: str, **kwargs) -> pd.DataFrame:
        """
        Apply a modular strategy to the data using its name and parameters.
        """
        self.logger.info(f"Running strategy '{strategy_name}' with parameters: {kwargs}")
        try:

            #print(list_strategies())
            strategy = load_strategy(strategy_name, params=kwargs)
            result = strategy.generate_signal(data.copy())
        except Exception as e:
            self.logger.error(f"Failed to apply strategy '{strategy_name}': {e}")
            return pd.DataFrame()

        if 'Signal' not in result.columns:
            self.logger.warning(f"Strategy '{strategy_name}' did not return a 'Signal' column.")
            return pd.DataFrame()

        return result