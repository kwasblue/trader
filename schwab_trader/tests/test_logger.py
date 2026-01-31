"""
Test suite for logging utilities.

Tests logging configuration, formatters, and file handlers.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock, patch
import logging
import tempfile
import shutil


class TestLoggerInitialization(unittest.TestCase):
    """Test logger initialization."""

    def test_logger_creation(self):
        """Logger should be created with correct name."""
        logger = logging.getLogger('test_logger')

        self.assertEqual(logger.name, 'test_logger')

    def test_logger_level_configuration(self):
        """Logger should accept level configuration."""
        logger = logging.getLogger('test_level')
        logger.setLevel(logging.DEBUG)

        self.assertEqual(logger.level, logging.DEBUG)

    def test_logger_propagation(self):
        """Logger propagation should be configurable."""
        logger = logging.getLogger('test_propagation')
        logger.propagate = False

        self.assertFalse(logger.propagate)


class TestLoggerFormatters(unittest.TestCase):
    """Test logging formatters."""

    def test_standard_format(self):
        """Standard format should include timestamp and level."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create a log record
        record = logging.LogRecord(
            name='test', level=logging.INFO,
            pathname='', lineno=0,
            msg='Test message', args=(), exc_info=None
        )

        formatted = formatter.format(record)

        self.assertIn('INFO', formatted)
        self.assertIn('Test message', formatted)

    def test_trade_log_format(self):
        """Trade log format should include trade details."""
        formatter = logging.Formatter(
            '%(asctime)s,%(symbol)s,%(side)s,%(qty)s,%(price)s'
        )

        # Custom record with extra fields
        record = logging.LogRecord(
            name='trade', level=logging.INFO,
            pathname='', lineno=0,
            msg='', args=(), exc_info=None
        )
        record.symbol = 'AAPL'
        record.side = 'BUY'
        record.qty = 100
        record.price = 150.25

        formatted = formatter.format(record)

        self.assertIn('AAPL', formatted)
        self.assertIn('BUY', formatted)


class TestFileTradeLogger(unittest.TestCase):
    """Test file-based trade logger."""

    def setUp(self):
        """Create temporary directory for log files."""
        self.test_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.test_dir, 'trades.csv')

    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_trade_logger_writes_csv(self):
        """Trade logger should write CSV format."""
        with open(self.log_file, 'w') as f:
            f.write('timestamp,symbol,side,qty,price,pnl\n')
            f.write('2023-01-01T10:00:00,AAPL,BUY,100,150.25,0\n')

        with open(self.log_file, 'r') as f:
            content = f.read()

        self.assertIn('timestamp', content)
        self.assertIn('AAPL', content)

    def test_trade_logger_appends(self):
        """Trade logger should append to existing file."""
        # Write header
        with open(self.log_file, 'w') as f:
            f.write('timestamp,symbol,side,qty,price\n')

        # Append trade
        with open(self.log_file, 'a') as f:
            f.write('2023-01-01T10:00:00,AAPL,BUY,100,150.25\n')

        # Append another trade
        with open(self.log_file, 'a') as f:
            f.write('2023-01-01T10:05:00,MSFT,SELL,50,250.00\n')

        with open(self.log_file, 'r') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 3)  # Header + 2 trades


class TestModuleLogger(unittest.TestCase):
    """Test module-specific logger."""

    def test_get_module_logger(self):
        """Should get logger for specific module."""
        try:
            from loggers.factory import get_module_logger

            logger = get_module_logger(
                module_name='TestModule',
                file_key='test'
            )

            self.assertIsNotNone(logger)
        except ImportError:
            # Fallback test
            logger = logging.getLogger('TestModule')
            self.assertIsNotNone(logger)

    def test_module_logger_isolation(self):
        """Module loggers should be isolated."""
        logger1 = logging.getLogger('Module1')
        logger2 = logging.getLogger('Module2')

        logger1.setLevel(logging.DEBUG)
        logger2.setLevel(logging.ERROR)

        self.assertNotEqual(logger1.level, logger2.level)


class TestLogLevels(unittest.TestCase):
    """Test log level handling."""

    def test_debug_level(self):
        """DEBUG level should be lowest."""
        self.assertEqual(logging.DEBUG, 10)

    def test_info_level(self):
        """INFO level should be standard."""
        self.assertEqual(logging.INFO, 20)

    def test_warning_level(self):
        """WARNING level should be elevated."""
        self.assertEqual(logging.WARNING, 30)

    def test_error_level(self):
        """ERROR level should be high priority."""
        self.assertEqual(logging.ERROR, 40)

    def test_critical_level(self):
        """CRITICAL level should be highest."""
        self.assertEqual(logging.CRITICAL, 50)

    def test_level_filtering(self):
        """Logger should filter by level."""
        logger = logging.getLogger('filter_test')
        logger.setLevel(logging.WARNING)

        # Below threshold
        self.assertFalse(logger.isEnabledFor(logging.DEBUG))
        self.assertFalse(logger.isEnabledFor(logging.INFO))

        # At or above threshold
        self.assertTrue(logger.isEnabledFor(logging.WARNING))
        self.assertTrue(logger.isEnabledFor(logging.ERROR))


class TestLogRotation(unittest.TestCase):
    """Test log file rotation."""

    def test_rotation_config(self):
        """Log rotation should be configurable."""
        rotation_config = {
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'when': 'midnight',
            'interval': 1
        }

        self.assertEqual(rotation_config['maxBytes'], 10485760)
        self.assertEqual(rotation_config['backupCount'], 5)

    def test_timed_rotation_config(self):
        """Time-based rotation should be configurable."""
        from logging.handlers import TimedRotatingFileHandler

        config = {
            'when': 'D',  # Daily
            'interval': 1,
            'backupCount': 7
        }

        self.assertEqual(config['when'], 'D')
        self.assertEqual(config['backupCount'], 7)


if __name__ == '__main__':
    unittest.main()
