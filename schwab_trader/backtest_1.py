#%%
from utils.back_test import Backtester
from utils.aggregate import Aggregator
from utils.authenticator import Authenticator
from sp500 import sp500_tickers
from utils.position_sizer import DynamicPositionSizer
from utils.datautils import load_stock_Data
import joblib
import pandas as pd
# Load the saved model
model = joblib.load(r"C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader\saved_models\log_regres_modelv1_best.pkl")

# Custom strategy wrapper for ML model
def logistic_model_strategy(data, model, **kwargs):
    # You must preprocess the data just like during training
    features = data.drop(columns=['Target'], errors='ignore')  # 'Target' should not be used in prediction

    # Match train-time preprocessing
    X = features.copy()
    try:
        predictions = model.predict(X)
    except Exception as e:
        print("Model prediction failed:", e)
        return pd.Series(index=X.index, data=[False]*len(X))  # All false signals

    # Return boolean Series of Buy signals (1 = buy)
    return pd.Series(predictions == 1, index=data.index)

if __name__ == "__main__":
    auth = Authenticator()
    aggregator = Aggregator(apikey=auth.apikey, secret=auth.secret)

    frames_dict = load_stock_Data(sp500_tickers[:60])
#%%
    trading_days_year = 251
    years = 3
    
    ticker = 'AAPL'
    historical_data = frames_dict.get_dataframe(ticker)[int(-trading_days_year*years):]
    historical_data['Ticker'] = ticker
    
    backtester = Backtester(
        data=historical_data,
        initial_capital=25000,
        transaction_cost=0.001,
        risk_free_rate=0.0425
    )

    # Initialize Dynamic Position Sizer (No ATR multiplier, uses Stop-Loss)
    sizer = DynamicPositionSizer(risk_percentage=0.02)  # 2% risk per trade

    # set strategy here:
    strategy_name = 'logistic_regression_strategy'

    if strategy_name == 'simple_moving_average_strategy' or 'exponential_moving_average_strategy':
        strategy_params = { 'short_window':20,'long_window':50}

    if strategy_name == 'combined_strategy':
        strategy_params = {}
    if strategy_name == 'logistic_regression_strategy':
        strategy_params = {'model': model}
    
    # Run the backtest using dynamically adjusted trade sizes & stop-loss
    portfolio_df = backtester.run_backtest(
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        sizer=sizer  # Pass the position sizer
    )

    if portfolio_df is not None and not portfolio_df.empty:
        performance = backtester.evaluate_performance(portfolio_df)
        print("Performance Metrics:", performance)

        backtester.plot_results(portfolio_df, strategy=strategy_name)

        report_file = backtester.generate_report(
            portfolio_df,
            strategy_name=strategy_name,
            performance=performance
        )
        print("Report generated:", report_file)
    else:
        print("No portfolio data available from backtesting.")


# %%
