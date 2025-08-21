#%%

from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from data.streaming.authenticator import Authenticator
from data.aggregate import Aggregator
from data.processor import Processor
from loggers.logger import Logger
from utils.configloader import ConfigLoader
from cache.cache import CacheManager
import time
import os
import pandas as pd


class StockDataPipeline:
    """A class to handle the full data pipeline from fetching to storing stock data."""

    def __init__(self, watch_list=[], max_workers=os.cpu_count(), max_retries=3):
        self.watch_list = watch_list
        self.max_workers = max_workers
        self.max_retries = max_retries  # Maximum retries for failed requests
        self.skipped_stocks = []  # List to track skipped stocks

        # Load environment variables
        parent_directory = Path.cwd().parent
        env_file_path = parent_directory / '.venv' / 'env' / '.env'
        load_dotenv(env_file_path)

        # Initialize services
        self.config = ConfigLoader().load_config()
        self.auth = Authenticator()
        self.aggregator = Aggregator(apikey=self.auth.apikey, secret=self.auth.secret)
        self.processor = Processor()
        self.log_dir = self.config["folders"]["logs"]
        self.logger = Logger('app.log', 'StockDataPipeline', log_dir=self.log_dir).get_logger()
        self.cache = CacheManager()

    def fetch_data(self):
        """Fetch raw data in parallel with retries and return only updated stocks."""
        updated_stocks = {}

        def fetch(stock):
            attempts = 0
            while attempts < self.max_retries:
                try:
                    result = self.aggregator.raw_data_store(stock)
                    if result == "File Generated!":
                        updated_stocks[stock] = result
                    elif result == "No Update Needed":
                        self.logger.info(f"{stock}: No update needed.")
                    return
                except Exception as e:
                    self.logger.error(f"Error fetching data for {stock} (Attempt {attempts + 1}/{self.max_retries}): {e}")
                    attempts += 1
                    time.sleep(2)
            self.logger.warning(f"Skipping {stock} after {self.max_retries} failed attempts.")
            self.skipped_stocks.append(stock)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(fetch, self.watch_list)

        return updated_stocks


    def gather_data(self):
        """Fetch and structure stock data in parallel with retries."""
        def fetch(stock):
            attempts = 0
            while attempts < self.max_retries:
                try:
                    dict_info = self.aggregator.get_raw_data(stock)
                    if not dict_info or 'candles' not in dict_info or not dict_info['candles']:
                        raise ValueError("Empty response received.")
                    return stock, pd.DataFrame.from_dict(dict_info['candles'])
                except Exception as e:
                    self.logger.error(f"Error gathering data for {stock} (Attempt {attempts + 1}/{self.max_retries}): {e}")
                    attempts += 1
                    time.sleep(2)  # Wait before retrying

            self.logger.warning(f"Skipping {stock} after {self.max_retries} failed attempts.")
            self.skipped_stocks.append(stock)
            return stock, None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(fetch, self.watch_list)

        return {stock: frame for stock, frame in results if frame is not None}

    def process_data(self, gathered_data):
        """Process stock data in parallel."""
        def process(stock, frame):
            self.processor.update(stock, frame)
            return stock, self.processor.process(200, 50, 'standard')

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(lambda item: process(*item), gathered_data.items())

        return dict(results)

    def store_data(self, processed_data: dict):
        """Store processed data."""
        for stock, data in processed_data.items():
            self.aggregator.store_processed_data_files(stock, data)

    def run(self):
        start_time = time.time()
        self.logger.info("Checking cache for outdated stocks...")

        filtered_watchlist = []
        today = pd.Timestamp.utcnow().normalize().tz_localize(None)

        for stock in self.watch_list:
            last_date = self.cache.get_last_processed_date("stock_files", stock)
            if not last_date:
                filtered_watchlist.append(stock)
            else:
                last_date_ts = pd.to_datetime(last_date, unit='ms').normalize().tz_localize(None)
                if last_date_ts < today - pd.Timedelta(days=1):  # using yesterday logic
                    filtered_watchlist.append(stock)
                else:
                    self.logger.info(f"{stock} is already up-to-date.")

        if not filtered_watchlist:
            self.logger.info("All stocks up to date. Pipeline skipped.")
            return

        self.watch_list = filtered_watchlist
        self.logger.info(f"Running pipeline for: {', '.join(self.watch_list)}")

        # 🛠 Fetch only stocks that need updates
        fetch_results = self.fetch_data()
        updated_stocks = list(fetch_results.keys())

        if not updated_stocks:
            self.logger.info("No stocks had new data. Skipping processing.")
            return

        # ⛏ Only gather/process/store for those
        self.watch_list = updated_stocks
        gathered_data = self.gather_data()
        processed_data = self.process_data(gathered_data)
        self.store_data(processed_data)

        self.cache._save_cache()

        end_time = time.time()
        self.logger.info(f"Pipeline completed in {end_time - start_time:.2f} seconds.")

        if self.skipped_stocks:
            self.logger.warning(f"Skipped stocks: {', '.join(self.skipped_stocks)}")

                


