#%%
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import json
from unittest.mock import patch, MagicMock, mock_open
from utils.aggregate import Aggregator
from datetime import datetime, timedelta


class TestAggregator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.apikey = 'test_api_key'
        cls.secret = 'test_secret_key'
        cls.aggregator = Aggregator(cls.apikey, cls.secret)
        cls.stock = 'AAPL'
        cls.start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cls.end_date = datetime.now().strftime('%Y-%m-%d')
        cls.mock_data = {'symbol': cls.stock, 'candles': [{'datetime': '2023-06-01', 'open': 150.0, 'high': 155.0, 'low': 148.0, 'close': 152.0, 'volume': 1000000}]}

    @patch('aggregator.SchwabClient.daily_price_history')
    @patch('aggregator.write_json')
    def test_raw_data_store(self, mock_write_json, mock_daily_price_history):
        mock_daily_price_history.return_value = self.mock_data
        result = self.aggregator.raw_data_store(self.stock, self.start_date, self.end_date)
        mock_write_json.assert_called_once()
        self.assertIn('File Generated!', result)

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps(mock_data))
    @patch('aggregator.write_json')
    def test_update_raw_data(self, mock_write_json, mock_open):
        new_data = [{'datetime': '2023-06-02', 'open': 152.0, 'high': 157.0, 'low': 150.0, 'close': 155.0, 'volume': 1200000}]
        result = self.aggregator.update_raw_data(self.stock, new_data)
        mock_write_json.assert_called_once()
        self.assertIn('Raw data updated!', result)

    @patch('aggregator.DataStore.create_database')
    @patch('aggregator.DataStore.fill_database')
    @patch('aggregator.DataStore.commit')
    @patch('aggregator.DataStore.close_db')
    def test_store_processed_data(self, mock_close_db, mock_commit, mock_fill_database, mock_create_database):
        processed_data = self.mock_data['candles']
        result = self.aggregator.store_processed_data(self.stock, processed_data)
        mock_create_database.assert_called_once_with(f"{self.stock}_data")
        mock_fill_database.assert_called_once()
        mock_commit.assert_called_once()
        mock_close_db.assert_called_once()
        self.assertIn('Processed data stored!', result)

    @patch('aggregator.DataStore.fill_database')
    @patch('aggregator.DataStore.commit')
    @patch('aggregator.DataStore.close_db')
    def test_update_processed_data(self, mock_close_db, mock_commit, mock_fill_database):
        new_processed_data = self.mock_data['candles']
        result = self.aggregator.update_processed_data(self.stock, new_processed_data)
        mock_fill_database.assert_called_once()
        mock_commit.assert_called_once()
        mock_close_db.assert_called_once()
        self.assertIn('Processed data updated!', result)

    @patch.object(Aggregator, 'raw_data_store')
    @patch.object(Aggregator, 'update_raw_data')
    @patch.object(Aggregator, 'store_processed_data')
    @patch.object(Aggregator, 'update_processed_data')
    def test_aggregate_data(self, mock_update_processed_data, mock_store_processed_data, mock_update_raw_data, mock_raw_data_store):
        self.aggregator.aggregate_data([self.stock], self.start_date, self.end_date)
        mock_raw_data_store.assert_called_once_with(self.stock, self.start_date, self.end_date)
        mock_update_raw_data.assert_called_once()
        mock_store_processed_data.assert_called_once()
        mock_update_processed_data.assert_called_once()

#%%
if __name__ == '__main__':
    unittest.main()

# %%
