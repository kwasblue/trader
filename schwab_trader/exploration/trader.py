#%%
import os
import pandas as pd
import numpy as np
import json
import asyncio
from dotenv import load_dotenv
from utils.logger import Logger
from data.streaming.authenticator import Authenticator   
from asyncio import run
from datetime import datetime
from data.streaming.schwab_client import SchwabClient
from data.streaming.streamer import SchwabStreamingClient
from monitoring.monitor import monitor_log_directory
from strategies.signal.sma_strategy import SMAStrategy
from data.datautils import DictHandler

class Trader:
    def __init__(self, apikey: str, secretkey: str, log_file='trader.log', logger_name='TraderLogger', log_dir='logs'):
        self.cash = 0
        self.positions = {}
        self.stop_loss_levels = {}  # Store stop-loss levels for each position
        self.portfolio_value = 0
        self.trade_log = []
        self.logger = Logger(log_file, logger_name, log_dir).get_logger()
        self.logger.info("Initializing trader with real account data.")
        self.streaming_client = SchwabStreamingClient(apikey, secretkey)
        self.realtime_data = {}
        self.data = None
        self.signals = None
        self.auth = Authenticator()
        
        # Initialize with real account data
        self.initialize_account_data(apikey, secretkey)

        # Start log monitoring
        self.monitor_logs(log_dir)

        # Authenticate API session
        self.auth.token_renewal()

    def initialize_account_data(self, apikey: str, secretkey: str):
        schwab_client = SchwabClient(apikey, secretkey)

        try:
            account_info = DictHandler(schwab_client.accounts())
            self.cash = account_info.find_key(case_insensitive=True, target_key='cashAvailableForTrading')
            self.positions = account_info.find_key(case_insensitive=True, target_key='positions')
            self.portfolio_value = self.cash + sum([pos['marketValue'] for pos in self.positions.values()])
            self.logger.info(f"Initialized with cash: {self.cash}, positions: {self.positions}, portfolio value: {self.portfolio_value}")
        except Exception as e:
            self.logger.error(f"Error initializing account data: {str(e)}")

    def monitor_logs(self, log_dir):
        """
        Monitors logs for errors and system events, triggering callbacks for error handling and recovery.
        """
        monitor_log_directory(log_dir, self.error_callback, self.success_callback)

    def error_callback(self, error_message):
        self.logger.error(f"Error detected: {error_message}")
        # Additional logic for handling critical errors can go here

    def success_callback(self, success_message):
        self.logger.info(f"System status: {success_message}")

    async def fetch_real_time_data(self, tickers):
        """
        Starts the real-time data stream for the given tickers.
        """
        await self.streaming_client.run(tickers)

    def process_realtime_data(self, message):
        """
        Processes incoming real-time data and updates stop-losses or executes trades based on signals.
        """
        try:
            data = json.loads(message)
            ticker = data['symbol']
            close_price = data['lastTradePrice']
            volume = data['volume']
            timestamp = data['timestamp']

            # Update real-time data for the ticker
            self.realtime_data[ticker] = {
                'Close': close_price,
                'Volume': volume,
                'Date': datetime.fromtimestamp(timestamp)
            }

            self.logger.info(f"Processed real-time data for {ticker}: Price - {close_price}, Volume - {volume}")

            # Check for stop-loss trigger
            if ticker in self.realtime_data:
                current_price = self.realtime_data[ticker]['Close']
                self.update_trailing_stop_loss(ticker, current_price)

            # Optionally, trigger trading based on updated real-time data and strategy signals
            self.check_for_trade_signals(ticker)

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Error processing real-time data: {str(e)}")

    def check_for_trade_signals(self, ticker):
        """
        Evaluates trading signals and executes trades if necessary.
        """
        if ticker in self.signals and ticker in self.realtime_data:
            latest_price = self.realtime_data[ticker]['Close']
            date = self.realtime_data[ticker]['Date']
            signal = self.signals[self.signals['Ticker'] == ticker]['Signal'].iloc[-1]

            self.execute_trade(signal, latest_price, ticker, date)




if __name__ == "__main__":
    initial_data = pd.DataFrame({'Date': pd.date_range(start='2020-01-01', periods=100),
                                 'Ticker': ['AAPL'] * 100,
                                 'Close': np.random.randn(100).cumsum() + 100,
                                 'High': np.random.randn(100).cumsum() + 105,
                                 'Low': np.random.randn(100).cumsum() + 95})
    
    trader = Trader(apikey='your_api_key', secretkey='your_secret_key')
    trader.update_strategy(initial_data, 'simple_moving_average_strategy', short_window=20, long_window=50)
    tickers = ['AAPL', 'MSFT']
    asyncio.run(trader.trade(tickers))
    trade_summary = trader.summarize_trades()
    print(trade_summary)
    
    log_directory = f'{os.getcwd()}/logs'
    monitor_log_directory(log_directory)
