#%%

from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from data.streaming.authenticator import Authenticator
from data.aggregate import Aggregator
from data.processor import Processor
from utils.logger import Logger
from utils.configloader import ConfigLoader
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

    def fetch_data(self):
        """Fetch raw data in parallel with retries."""
        def fetch(stock):
            attempts = 0
            while attempts < self.max_retries:
                try:
                    self.aggregator.raw_data_store(stock)  # API Call
                    return stock, "Success"
                except Exception as e:
                    self.logger.error(f"Error fetching data for {stock} (Attempt {attempts + 1}/{self.max_retries}): {e}")
                    attempts += 1
                    time.sleep(2)  # Wait before retrying

            self.logger.warning(f"Skipping {stock} after {self.max_retries} failed attempts.")
            self.skipped_stocks.append(stock)
            return stock, "Failed"

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(fetch, self.watch_list)

        return dict(results)

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
        """Execute the full pipeline."""
        start_time = time.time()

        self.logger.info("Fetching data...")
        fetch_results = self.fetch_data()

        self.logger.info("Gathering data...")
        gathered_data = self.gather_data()

        self.logger.info("Processing data...")
        processed_data = self.process_data(gathered_data)

        self.logger.info("Storing data...")
        self.store_data(processed_data)

        end_time = time.time()
        self.logger.info(f"Pipeline completed in {end_time - start_time:.2f} seconds.")

        if self.skipped_stocks:
            self.logger.warning(f"Skipped stocks: {', '.join(self.skipped_stocks)}")
            


