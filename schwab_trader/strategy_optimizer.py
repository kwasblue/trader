#%%
import os
import itertools
import numpy as np
import pandas as pd
from utils.logger import Logger
from alpha.strategy import Strategy
from utils.back_test import Backtester  # Your backtesting class
from utils.position_sizer import DynamicPositionSizer
from sp500 import sp500_tickers
from utils.datautils import load_stock_Data  # Function that returns a DataFrameManager or similar container
import joblib

class TrackingDict(dict):
    """A dictionary that tracks which keys were accessed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed_keys = set()

    def __getitem__(self, key):
        self.accessed_keys.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.accessed_keys.add(key)
        return super().get(key, default)
    
class CompositeStrategyOptimizer:
    def __init__(self, stocks: list, composite_param_grid: dict, initial_capital: float = 10000,
                 transaction_cost: float = 0.001, risk_free_rate: float = 0.02, sizer_params: dict = None):
        """
        Parameters:
            stocks (list): List of stock tickers.
            composite_param_grid (dict): A dictionary where keys are the composite parameters (or tuple of strategy method names)
                and values are lists of candidate values. For example:
                {
                    'strategy_methods': [
                        ( 'exponential_moving_average_strategy', 'stochastic_oscillator_strategy' ),
                        ( 'simple_moving_average_strategy', 'rsi_strategy' )
                    ],
                    'combine_method': ['vote', 'weighted'],
                    'weight_config': [None, (0.7, 0.3)]  # Only used when combine_method=='weighted'\n
                    'short_window': [20, 30],  # For EMA or SMA\n
                    'long_window': [50, 60],\n
                    'k_window': [14],\n
                    'd_window': [3],\n
                    'oversold': [20],\n
                    'overbought': [80]\n
                }\n
            initial_capital (float): Starting capital for each backtest.\n
            transaction_cost (float): Transaction cost per trade.\n
            risk_free_rate (float): Annualized risk-free rate.\n
            sizer_params (dict): Parameters for DynamicPositionSizer.\n
        """
        self.stocks = stocks
        self.composite_param_grid = composite_param_grid
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.logger = Logger("composite_optimizer.log", "CompositeStrategyOptimizer",
                               log_dir=os.path.join(os.getcwd(), "logs")).get_logger()
        if sizer_params is None:
            sizer_params = {'risk_percentage': 0.02, 'atr_multiplier': 1.5}
        self.sizer_params = sizer_params
        # Create a Strategy instance that has our composite strategy method
        self.strategy = Strategy()



    def optimize_for_stock(self, stock: str) -> dict:
        """
        Optimize the composite strategy for one stock by iterating over all candidate parameter combinations.
        
        Returns:
            dict: Best parameter combination and performance for each composite strategy configuration.
        """
        self.logger.info(f"Optimizing composite strategy for stock: {stock}")

        # Load historical data for the stock.
        frames = load_stock_Data([stock])
        data = frames.get_dataframe(stock)
        if data is None or data.empty:
            self.logger.error(f"No data available for {stock}")
            return {}

        best_results = {}
        grid_keys = list(self.composite_param_grid.keys())
        grid_values = [self.composite_param_grid[k] for k in grid_keys]

        for combo in itertools.product(*grid_values):
            params = TrackingDict(zip(grid_keys, combo))  # Use TrackingDict instead of normal dict

            self.logger.info(f"Testing composite parameters...")  # Will log used params later

            try:
                # Apply composite strategy to data.
                composite_data = self.strategy.combined_strategy(
                    data.copy(),
                    strategy_methods=list(params.get('strategy_methods')),
                    combine_method=params.get('combine_method'),
                    weights=params.get('weight_config'),
                    short_window=params.get('short_window'),
                    long_window=params.get('long_window'),
                    k_window=params.get('k_window'),
                    d_window=params.get('d_window'),
                    oversold=params.get('oversold'),
                    overbought=params.get('overbought')
                )

                # Track only the parameters that were actually accessed
                used_params = {k: params[k] for k in params.accessed_keys}
                self.logger.info(f"Used Parameters: {used_params}")

                # Initialize Backtester and Position Sizer
                backtester = Backtester(
                    data=composite_data,
                    initial_capital=self.initial_capital,
                    transaction_cost=self.transaction_cost,
                    risk_free_rate=self.risk_free_rate
                )
                sizer = DynamicPositionSizer(**self.sizer_params)

                # Run the backtest
                portfolio_df = backtester.run_backtest(
                    strategy_name='combined_strategy',
                    strategy_params={},
                    sizer=sizer
                )

                if portfolio_df.empty:
                    self.logger.warning(f"No portfolio generated with parameters: {used_params}")
                    continue

                # Evaluate performance
                performance = backtester.evaluate_performance(portfolio_df)
                sharpe = performance.get('Sharpe Ratio', -np.inf)
                self.logger.info(f"Params {used_params} produced Sharpe Ratio: {sharpe}")

                # Keep track of the best performance
                key = str(params.get('strategy_methods')) + '_' + params.get('combine_method', 'vote')
                if key not in best_results or sharpe > best_results[key]['Sharpe Ratio']:
                    best_results[key] = {
                        'parameters': used_params,
                        'Sharpe Ratio': sharpe,
                        'portfolio': portfolio_df.copy()
                    }
            except Exception as e:
                self.logger.error(f"Error testing parameters {params} for {stock}: {e}")
                continue

        return best_results

    def run_optimization(self) -> dict:
        """
        Run optimization for all stocks.
        
        Returns:
            dict: A mapping from stock tickers to their optimization results.
        """
        final_results = {}
        for stock in self.stocks:
            try:
                self.logger.info(f"Optimizing for {stock}...")
                stock_results = self.optimize_for_stock(stock)
                final_results[stock] = stock_results
            except Exception as e:
                self.logger.error(f"Optimization failed for {stock}: {e}")
        return final_results

#%%
if __name__ == "__main__":
    # For production, you might choose more stocks; here we use a subset for illustration.
    stocks_to_optimize = sp500_tickers[:10]
    model = joblib.load(r"C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader\saved_models\log_regres_modelv1_best.pkl")


    available_strategies = [
    'simple_moving_average_strategy',
    'exponential_moving_average_strategy',
    'mean_reversion_strategy',
    'breakout_strategy',
    'momentum_strategy',
    'rsi_strategy',
    'macd_strategy',
    'bollinger_bands_strategy',
    'vwap_strategy',
    'donchian_channel_strategy',
    'parabolic_sar_strategy',
    'stochastic_oscillator_strategy',
    'ichimoku_cloud_strategy',
    'adx_strategy',
    'moving_average_crossover',
    'trend_momentum_strategy'
    ]

    # Generate all pair combinations
    strategy_method_combinations = list(itertools.combinations(available_strategies, 2))
    # Define the composite parameter grid
    
    composite_param_grid = {
    'strategy_methods': [('logistic_regression_strategy', 'stochastic_oscillator_strategy')],
    'combine_method': ['weighted'],
    'weight_config': [(0.5, 0.5)],  # Must be provided if using 'weighted'
    'short_window': [20],
    'long_window': [60],
    'k_window': [14],
    'd_window': [3],
    'oversold': [20],
    'overbought': [80],
    'model' : model
    }
    

    optimizer = CompositeStrategyOptimizer(
        stocks=stocks_to_optimize,
        composite_param_grid=composite_param_grid,
        initial_capital=25000,
        transaction_cost=0.001,
        risk_free_rate=0.0425,
        sizer_params={'risk_percentage': 0.10}
    )

    optimization_results = optimizer.run_optimization()
    print("Optimization Results:")
    print(optimization_results)

# %%
