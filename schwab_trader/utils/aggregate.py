import os
import json
import pandas as pd
from utils.writer import FileWriter
from utils.datastorage import DataStore
from alpha.schwab_client import SchwabClient
from utils.authenticator import Authenticator
from utils.logger import Logger
from alpha.processor import Processor
from utils.configloader import ConfigLoader
from utils.framemanager import DataFrameManager

class Aggregator:
    def __init__(self, apikey: str, secret: str, database='stock_base.db'):
        self.session = SchwabClient(apikey, secret)
        self.config = ConfigLoader().load_config()
        self.database = database
        self.log_dir = self.config["folders"]["logs"]
        self.authenticator = Authenticator()
        self.datastore = DataStore(self.database)
        self.logger = Logger('app.log', 'Aggregator', log_dir=self.log_dir).get_logger()
        self.writer = FileWriter(log_file='app.log', logger_name='FileWriter', log_dir=self.log_dir)
        self.table_name = 'stock_table'
        self.frame_manager = DataFrameManager()

    def raw_data_store(self, stock: str, start: int = '', end: int = '') -> str:
        """
        Store raw data in a JSON file.
        """
        self.logger.info(f"Fetching raw data for {stock}")
        try:
            data = self.session.daily_price_history(stock, start=start, end=end)
            filepath = f'{self.authenticator.program_path}/data/'
            self.writer.write_json(target_path=filepath, target_file=f'raw_{stock}_file.json', data=data)
            self.logger.info(f"Raw data for {stock} stored successfully")
            return "File Generated!"
        except Exception as e:
            self.logger.error(f"Error storing raw data for {stock}: {str(e)}")
            return {"error": str(e)}

    def store_processed_data_files(self, stock: str, processed_data: pd.DataFrame):
        """
        Store processed data in JSON files.
        """
        self.logger.info(f'Storing processed data file for {stock}')
        try:
            if processed_data.empty:
                self.logger.warning(f"No data to save for {stock}. The DataFrame is empty.")
                return

            processed_data = processed_data.where(processed_data.notna(), None)
            processed_data = processed_data.replace({float('inf'): None, float('-inf'): None})

            data_to_save = processed_data.to_dict(orient="records")

            filepath = f'{self.authenticator.program_path}/proc_data/'

            self.writer.write_json(target_path=filepath, target_file=f'proc_{stock}_file.json', data=data_to_save)

            self.logger.info(f"Processed data for {stock} stored successfully in {filepath}")
        except Exception as e:
            self.logger.error(f"Error storing processed data for {stock}: {str(e)}")

    def get_processed_data_files(self, stock):
        filepath = f'{self.authenticator.program_path}/proc_data/'
        file = self.writer.find(f'proc_{stock}_file.json', filepath)

        if not file or not os.path.exists(file):
            self.logger.error(f"File not found for stock {stock}")
            return None

        with open(file, 'r') as f:
            data = json.load(f)

        self.frame_manager.add_dataframe(stock, data)
        return self.frame_manager

    def store_processed_data(self, stock: str, processed_data: pd.DataFrame) -> str:
        """
        Store processed data in the database.
        """
        self.logger.info(f"Storing processed data for {stock} in the database")
        try:
            self.datastore.open_db()

            with self.datastore as store:
                processed_data['symbol'] = stock
                store.fill_database(self.table_name, processed_data)
                store.commit()

            self.logger.info(f"Processed data for {stock} stored successfully in the database")
            self.datastore.close_db()

            return "Processed data stored!"
        except Exception as e:
            self.logger.error(f"Error storing processed data for {stock}: {str(e)}")
            return {"error": str(e)}

    def store_processed_data_batch(self, stocks: list[str], processed_data_list: list[pd.DataFrame]) -> str:
        """
        Store processed data for multiple stocks in the database.
        """
        self.logger.info("Storing processed data for multiple stocks in the database")
        try:
            if len(stocks) != len(processed_data_list):
                raise ValueError("The number of stocks and processed dataframes do not match.")

            frame = pd.DataFrame()
            self.datastore.open_db()
            dataframes = []
            for stock, processed_data in zip(stocks, processed_data_list):
                processed_data['symbol'] = stock
                dataframes.append(processed_data)

            frame = pd.concat(dataframes, ignore_index=True)
            
            self.datastore.fill_database(self.table_name, frame)
            self.datastore.commit()

            self.logger.info("Processed data for all stocks stored successfully")
            self.datastore.close_db()
            return "Processed data for all stocks stored!"
        except Exception as e:
            self.logger.error(f"Error storing processed data for batch: {str(e)}")
            return {"error": str(e)}

    def get_raw_data(self, stock: str) -> dict:
        """
        Retrieve raw data from JSON file.
        """
        try:
            filepath = f"{self.config['folders']['data']}/raw_{stock}_file.json"
            return self._read_raw_data(filepath)
        except FileNotFoundError:
            self.logger.error(f"Raw data file for {stock} not found.")
            return {}
        except Exception as e:
            self.logger.error(f"Error reading raw data for {stock}: {str(e)}")
            return {}

    def _read_raw_data(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            return json.load(f)

    def aggregate_data(self, stock_list: list[str], start=None, end=None) -> str:
        """
        Aggregate and store data for multiple stocks.
        """
        self.logger.info(f"Aggregating data for {stock_list}")
        try:
            with self.datastore as store:
                for stock in stock_list:
                    self.raw_data_store(stock, start=start, end=end)
                    raw_data = self.get_raw_data(stock)
                    if not raw_data:
                        self.logger.warning(f"No raw data found for {stock}. Skipping.")
                        continue

                    frame = pd.DataFrame.from_dict(raw_data.get('candles', []))
                    processed_data = Processor(stock, frame).process(50, 50, 'standard')
                    self.store_processed_data(stock, processed_data)

                self.logger.info(f"Aggregated data for {stock_list} successfully stored.")
                return "Aggregation Complete"
        except Exception as e:
            self.logger.error(f"Error during data aggregation: {str(e)}")
            return {"error": str(e)}

    def fetch_data(self, stock: str, start_date='', end_date='') -> pd.DataFrame:
        """
        Fetch processed data from the database for analysis.
        """
        try:
            self.datastore.open_db()
            self.logger.info(f"Fetching data for {stock} from the database")

            query = f"SELECT * FROM {self.table_name} WHERE symbol = ?"
            params = [stock]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            df = self.datastore.get_data_by_symbol(self.table_name, stock)
            self.logger.info(f"Data for {stock} fetched successfully")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {stock}: {str(e)}")
            return pd.DataFrame()
        finally:
            if hasattr(self.datastore, "close_db"):
                self.datastore.close_db()

    def dataframes(self):
        return self.frame_manager
   

# class Aggregator():
#     def __init__(self, apikey: str, secret:str, database='stock_base.db'):
#         self.session = SchwabClient(apikey, secret)
#         self.config = ConfigLoader().load_config()
#         self.database = database
#         self.log_dir = self.config["folders"]["logs"]
#         self.authenticator = Authenticator()
#         self.datastore = DataStore(self.database)
#         self.logger = Logger('app.log', 'Aggregator', log_dir=self.log_dir).get_logger()
#         self.writer = FileWriter(log_file='app.log', logger_name='FileWriter', log_dir=self.log_dir)
#         self.table_name = 'stock_table'

#     def raw_data_store(self, stock:str, start=None, end=None):
#         """
#         Store raw data in a JSON file.
#         """
#         self.logger.info(f"Fetching raw data for {stock}")
#         data = self.session.daily_price_history(stock, start=start, end=end)
#         try:
#             filepath = f'{self.authenticator.program_path}/data/'
#             self.writer.write_json(target_path=filepath, target_file=f'raw_{stock}_file.json', data=data)
#             self.logger.info(f"Raw data for {stock} stored successfully in {filepath}")
#         except Exception as e:
#             self.logger.error(f"Error storing raw data for {stock}: {str(e)}")
#             return {"error": str(e)}
#         return 'File Generated!'

#     def update_raw_data(self, stock, new_data):
#         """
#         Update raw data file with new data.
#         """
#         filepath = os.path.join(f'{self.authenticator.program_path}/data/', f'raw_{stock}_file.json')
#         try:
#             with open(filepath, 'r') as f:
#                 data = json.load(f)
#             data.extend(new_data)
#             self.writer.write_json(filepath, data)
#             self.logger.info(f"Raw data for {stock} updated successfully")
#         except FileNotFoundError:
#             self.logger.warning(f"Raw data file for {stock} not found. Creating a new file.")
#             self.raw_data_store(stock, new_data)
#         except Exception as e:
#             self.logger.error(f"Error updating raw data for {stock}: {str(e)}")
#             return {"error": str(e)}
#         return 'Raw data updated!'

#     def store_processed_data(self, stock, processed_data):
#         """
#         Store processed data in the database.
#         """
#         processed_data['symbol'] = stock
#         self.logger.info(f"Storing processed data for {stock} in the database")
        
#         # Use the `with` statement for automatic opening and closing of the database connection
#         try:
#             with self.datastore as store:
#                 store.fill_database(self.table_name, processed_data)
#                 store.commit()
#             self.logger.info(f"Processed data for {stock} stored successfully in the database")
#         except Exception as e:
#             self.logger.error(f"Error storing processed data for {stock}: {str(e)}")
#             return {"error": str(e)}
#         return 'Processed data stored!'
    
#     def batch_store_data(self, stocks: list[str], processed_data: list[pd.DataFrame]):

    
#         for i, stock in enumerate(stocks):
#             try:
#                 processed_data[i]['symbol'] = stock
#                 self.logger.info(f"Updating processed data for {stock} in the database")
#                 self.datastore.fill_database(self.table_name,processed_data[i])
#             except Exception as e:
#                 self.logger.error(f"Error storing processed data for {stock}: {str(e)}")
#                 return {"error": str(e)}
#         return None

#     def update_processed_data(self, stock, new_processed_data):
#         """
#         Update the database with new processed data.
#         """
#         self.logger.info(f"Updating processed data for {stock} in the database")
#         table_name = f"{stock}_data"
        
#         # Use the `with` statement for automatic opening and closing of the database connection
#         try:
#             with self.datastore as store:
#                 store.fill_database(table_name, new_processed_data)
#                 store.commit()
#             self.logger.info(f"Processed data for {stock} updated successfully in the database")
#         except Exception as e:
#             self.logger.error(f"Error updating processed data for {stock}: {str(e)}")
#             return {"error": str(e)}
#         return 'Processed data updated!'

#     def aggregate_data(self, stock_list, start=None, end=None):
#         """
#         Aggregate data for multiple stocks.
#         """

#         self.logger.info(f"Updating processed data for {stock_list} in the database")
#         self.datastore.open_db()  # Open connection once
#         try:
#             for i , stock in enumerate(stock_list):
#                 self.raw_data_store(stock, start, end)
#                 data = self.get_raw_data(stock)
#                 frame = pd.DataFrame.from_dict(data['candles'])
#                 processed_data = Processor(stock, frame).process(50, 50, 'standard')
#                 self.store_processed_data(stock, processed_data)
        
#                 self.logger.info(f"Processed data for {stock} updated successfully in the database")
#         except Exception as e:
#             self.logger.error(f"Error aggregating processed data for {stock}: {str(e)}")
#             return {"error": str(e)}
#         finally:
#             self.datastore.close_db()
#         return None
       
#     def get_raw_data(self, stock):
#         with open(f'{os.getcwd()}/data/raw_{stock}_file.json') as f:
#             data = json.load(f)
#             return data

#     def fetch_data(self, stock: str, start_date=None, end_date=None):
#         """
#         Fetch data from the database for analysis.

#         :param stock: The stock symbol for which to fetch data.
#         :param start_date: Optional start date for filtering data (format: 'YYYY-MM-DD').
#         :param end_date: Optional end date for filtering data (format: 'YYYY-MM-DD').
#         :return: A Pandas DataFrame containing the requested data.
#         """
#         query = f"SELECT * FROM {self.table_name} WHERE symbol = ?"
#         params = [stock]

#         if start_date:
#             query += " AND date >= ?"
#             params.append(start_date)
#         if end_date:
#             query += " AND date <= ?"
#             params.append(end_date)

#         try:
#             # Explicitly open a new connection
#             # self.datastore.open_db() # Ensure connection is refreshed
#             self.logger.info(f"Fetching data for {stock} from the database")
#             df = self.datastore.get_data_by_symbol(self.table_name, stock)
#             self.logger.info(f"Data for {stock} fetched successfully")
#             return df
#         except Exception as e:
#             self.logger.error(f"Error fetching data for {stock}: {str(e)}")
#             return pd.DataFrame()  # Return an empty DataFrame on error
#         #finally:
#         #    self.datastore.close_db() #  dont know why this doesnt work



