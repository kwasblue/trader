import logging
import os
from logging.handlers import RotatingFileHandler


class UniqueMessageFilter(logging.Filter):
    """Suppress consecutive duplicate log messages."""

    def __init__(self):
        super().__init__()
        self.last_message = None

    def filter(self, record):
        msg = record.getMessage()
        if msg != self.last_message:
            self.last_message = msg
            return True
        return False


class Logger:
    """Thread-safe rotating logger with duplicate suppression."""

    def __init__(
        self,
        log_file: str,
        logger_name: str,
        log_dir: str = "logs",
        level: int = logging.DEBUG,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S",
        propagate: bool = False,
        max_bytes: int = 5_000_000,
        backup_count: int = 5,
        console: bool = False,
    ):
        os.makedirs(log_dir, exist_ok=True)
        full_log_path = os.path.join(log_dir, log_file)

        # Always use a *real* logging.Logger, never self.logger recursion
        _logger = logging.getLogger(logger_name)
        _logger.setLevel(level)
        _logger.propagate = propagate

        # Avoid duplicate handlers if logger already exists
        if not _logger.handlers:
            fmt = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt=timestamp_format,
            )

            fh = RotatingFileHandler(full_log_path, maxBytes=max_bytes, backupCount=backup_count)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            fh.addFilter(UniqueMessageFilter())
            _logger.addHandler(fh)

            if console:
                ch = logging.StreamHandler()
                ch.setLevel(level)
                ch.setFormatter(fmt)
                _logger.addHandler(ch)

        self._logger = _logger
        self._logger.debug(f"Logger '{logger_name}' initialized → {full_log_path}")

    # ---------------------------------------------------------------------
    # Convenience passthroughs
    # ---------------------------------------------------------------------
    def get_logger(self):
        return self._logger

    def debug(self, msg: str):
        self._logger.debug(msg)

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def critical(self, msg: str):
        self._logger.critical(msg)
