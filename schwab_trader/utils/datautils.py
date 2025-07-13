import time
import os
from datetime import datetime
import pandas as pd
from utils.aggregate import Aggregator 
from dotenv import load_dotenv
from pathlib import Path
from utils.authenticator import Authenticator

parent_directory = Path.cwd().parent
# Build the path to the .env file
env_file_path = parent_directory / '.venv' / 'env' / '.env'
# Load the .env file
load_dotenv(env_file_path)
auth = Authenticator()
apikey = auth.apikey
secret = auth.secret

class DictHandler:
    def __init__(self, data=None):
        """
        Initialize with a dictionary (or another nested structure like list).
        """
        self.data = data if data is not None else {}

    def find_key(self, target_key, partial=False, case_insensitive=False, max_depth=float('inf')):
        """
        Search for a key in nested dictionaries and lists and return the first matching value.
        
        Args:
            target_key (str): The key to search for.
            partial (bool): Whether to search for partial matches of the key. Default is False.
            case_insensitive (bool): Whether the search should ignore case. Default is False.
            max_depth (int/float): Maximum depth to search. Default is infinite.
        
        Returns:
            value: The value of the matching key, or None if not found.
        """
        stack = [self.data]  # Stack holds the current data
        while stack:
            current_data = stack.pop()

            if isinstance(current_data, dict):
                for k, v in current_data.items():
                    # Prepare key for comparison based on case_insensitive and partial settings
                    k_to_compare = k.lower() if case_insensitive else k
                    key_to_compare = target_key.lower() if case_insensitive else target_key

                    if (partial and key_to_compare in k_to_compare) or (k_to_compare == key_to_compare):
                        return v  # Return the first matching value

                    # Add the current value (v) to the stack for further exploration
                    stack.append(v)
            
            elif isinstance(current_data, list):
                for item in current_data:
                    stack.append(item)

        return None  # Return None if the key is not found

def get_epoch_dates(start_date: str, end_date: str):
    """
    Given a start date and end date in 'YYYY-MM-DD' format, 
    returns the epoch timestamps for both the start and end dates.

    :param start_date: The starting date in 'YYYY-MM-DD' format.
    :param end_date: The ending date in 'YYYY-MM-DD' format.
    :return: Tuple of epoch timestamps (start_date_epoch, end_date_epoch).
    """

    # Convert the start and end dates from string to datetime objects
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    # Convert the start and end dates to epoch timestamps
    start_date_epoch = int(time.mktime(start_date_obj.timetuple()))
    end_date_epoch = int(time.mktime(end_date_obj.timetuple()))

    # Return both the start and end epoch timestamps
    return start_date_epoch, end_date_epoch

def filter_data_by_date(df: pd.DataFrame, start_date_epoch: int, end_date_epoch: int):
    """
    Filters the dataframe to include only rows between the start and end dates (inclusive) 
    using the provided epoch timestamps.
    
    :param df: The DataFrame containing the data.
    :param start_date_epoch: The start date epoch timestamp (in milliseconds or seconds).
    :param end_date_epoch: The end date epoch timestamp (in milliseconds or seconds).
    :return: Filtered DataFrame between the given dates.
    """
    
    # Check if the Date column is in seconds or milliseconds
    if df['Date'].max() > 1e9:  # Assuming milliseconds (greater than 1e9)
        # Convert start and end date to milliseconds if they are in seconds
        start_date_epoch *= 1000
        end_date_epoch *= 1000
    
    # Ensure that Date is in epoch (in milliseconds or seconds)
    if df['Date'].dtype != 'int64':
        raise ValueError("The 'Date' column must be in epoch format (int64).")
    
    # Filter the dataframe based on the epoch dates
    filtered_df = df[(df['Date'] >= start_date_epoch) & (df['Date'] <= end_date_epoch)]
    
    return filtered_df

def load_stock_Data(stock_list: list[str]):
    aggregator = Aggregator(apikey=auth.apikey, secret=auth.secret)
    """ Load data into memory for whatever reason"""
    for idx, stock in enumerate(stock_list):
        frame = aggregator.get_processed_data_files(stock=stock)
    return frame


def epoch_to_date(epoch_timestamp):
    return datetime.fromtimestamp(epoch_timestamp).strftime('%Y-%m-%d')