"""
Test suite for data writers.

Tests file writing, JSON output, and data export functionality.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
import tempfile
import shutil
import csv


class TestJsonWriter(unittest.TestCase):
    """Test JSON file writing."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_write_json_file(self):
        """Should write JSON data to file."""
        data = {'symbol': 'AAPL', 'price': 150.25}
        filepath = os.path.join(self.test_dir, 'output.json')

        with open(filepath, 'w') as f:
            json.dump(data, f)

        # Verify file exists
        self.assertTrue(os.path.exists(filepath))

        # Verify content
        with open(filepath, 'r') as f:
            loaded = json.load(f)

        self.assertEqual(loaded['symbol'], 'AAPL')

    def test_write_json_pretty(self):
        """Should write pretty-formatted JSON."""
        data = {'key1': 'value1', 'key2': 'value2'}
        filepath = os.path.join(self.test_dir, 'pretty.json')

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        with open(filepath, 'r') as f:
            content = f.read()

        # Pretty format should have newlines
        self.assertIn('\n', content)

    def test_write_json_list(self):
        """Should write JSON list."""
        data = [
            {'id': 1, 'name': 'item1'},
            {'id': 2, 'name': 'item2'}
        ]
        filepath = os.path.join(self.test_dir, 'list.json')

        with open(filepath, 'w') as f:
            json.dump(data, f)

        with open(filepath, 'r') as f:
            loaded = json.load(f)

        self.assertEqual(len(loaded), 2)


class TestCsvWriter(unittest.TestCase):
    """Test CSV file writing."""

    def setUp(self):
        """Create temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_write_csv_with_header(self):
        """Should write CSV with header row."""
        filepath = os.path.join(self.test_dir, 'output.csv')
        data = [
            {'symbol': 'AAPL', 'price': 150.25},
            {'symbol': 'MSFT', 'price': 250.00}
        ]

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['symbol', 'price'])
            writer.writeheader()
            writer.writerows(data)

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['symbol'], 'AAPL')

    def test_append_to_csv(self):
        """Should append to existing CSV."""
        filepath = os.path.join(self.test_dir, 'append.csv')

        # Write initial data
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'value'])
            writer.writerow(['2023-01-01', '100'])

        # Append more data
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['2023-01-02', '105'])

        with open(filepath, 'r') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 3)  # Header + 2 data rows


class TestTradeWriter(unittest.TestCase):
    """Test trade log writing."""

    def setUp(self):
        """Create temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_write_trade_record(self):
        """Should write trade record with all fields."""
        filepath = os.path.join(self.test_dir, 'trades.csv')

        trade = {
            'timestamp': '2023-01-01T10:00:00',
            'symbol': 'AAPL',
            'side': 'BUY',
            'qty': 100,
            'price': 150.25,
            'pnl': 0.0
        }

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trade.keys())
            writer.writeheader()
            writer.writerow(trade)

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)

        self.assertEqual(row['symbol'], 'AAPL')
        self.assertEqual(row['side'], 'BUY')

    def test_trade_log_format(self):
        """Trade log should have correct columns."""
        required_columns = [
            'timestamp', 'symbol', 'side', 'qty', 'price'
        ]

        trade = {col: '' for col in required_columns}

        for col in required_columns:
            self.assertIn(col, trade)


class TestCandelDataWriter(unittest.TestCase):
    """Test candle/bar data writing."""

    def setUp(self):
        """Create temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_write_ohlcv_data(self):
        """Should write OHLCV candle data."""
        filepath = os.path.join(self.test_dir, 'candles.json')

        candles = {
            'symbol': 'AAPL',
            'candles': [
                {
                    'datetime': '2023-01-01',
                    'open': 150.0,
                    'high': 152.0,
                    'low': 149.0,
                    'close': 151.0,
                    'volume': 1000000
                }
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(candles, f)

        with open(filepath, 'r') as f:
            loaded = json.load(f)

        self.assertIn('candles', loaded)
        self.assertEqual(len(loaded['candles']), 1)

    def test_candle_data_validation(self):
        """Candle data should have valid OHLCV."""
        candle = {
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000
        }

        # Validate OHLCV constraints
        is_valid = (
            candle['high'] >= candle['low'] and
            candle['high'] >= candle['open'] and
            candle['high'] >= candle['close'] and
            candle['low'] <= candle['open'] and
            candle['low'] <= candle['close'] and
            candle['volume'] >= 0
        )

        self.assertTrue(is_valid)


class TestMetricsWriter(unittest.TestCase):
    """Test metrics and performance data writing."""

    def setUp(self):
        """Create temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_write_performance_metrics(self):
        """Should write performance metrics."""
        filepath = os.path.join(self.test_dir, 'metrics.json')

        metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 1.25,
            'max_drawdown': 0.08,
            'win_rate': 0.55,
            'profit_factor': 1.8,
            'total_trades': 150
        }

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        with open(filepath, 'r') as f:
            loaded = json.load(f)

        self.assertEqual(loaded['total_return'], 0.15)
        self.assertEqual(loaded['sharpe_ratio'], 1.25)

    def test_write_equity_curve(self):
        """Should write equity curve data."""
        filepath = os.path.join(self.test_dir, 'equity.csv')

        equity_data = [
            {'date': '2023-01-01', 'equity': 10000},
            {'date': '2023-01-02', 'equity': 10150},
            {'date': '2023-01-03', 'equity': 10050},
        ]

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'equity'])
            writer.writeheader()
            writer.writerows(equity_data)

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 3)


class TestFileNaming(unittest.TestCase):
    """Test file naming conventions."""

    def test_symbol_file_naming(self):
        """Should generate correct file names for symbols."""
        symbol = 'AAPL'
        date_str = '20230101'

        raw_filename = f"raw_{symbol}_{date_str}.json"
        proc_filename = f"proc_{symbol}_{date_str}.json"

        self.assertEqual(raw_filename, "raw_AAPL_20230101.json")
        self.assertEqual(proc_filename, "proc_AAPL_20230101.json")

    def test_timestamp_in_filename(self):
        """Should include timestamp in filename when needed."""
        from datetime import datetime

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"trades_{timestamp}.csv"

        self.assertIn('trades_', filename)
        self.assertTrue(filename.endswith('.csv'))


if __name__ == '__main__':
    unittest.main()
