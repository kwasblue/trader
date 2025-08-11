# keep your UniqueMessageFilter as-is, add rotation
import logging, os
from logging.handlers import RotatingFileHandler

class UniqueMessageFilter(logging.Filter):
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
    def __init__(
        self,
        log_file: str,
        logger_name: str,
        log_dir: str = "logs",
        level: int = logging.INFO,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S",
        propagate: bool = True,
        max_bytes: int = 5_000_000,
        backup_count: int = 5,
        console: bool = False,
    ):
        os.makedirs(log_dir, exist_ok=True)
        full_log_file = os.path.join(log_dir, log_file)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        self.logger.propagate = propagate

        if not self.logger.hasHandlers():
            fmt = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt=timestamp_format,
            )

            fh = RotatingFileHandler(full_log_file, maxBytes=max_bytes, backupCount=backup_count)
            fh.setLevel(level)
            fh.addFilter(UniqueMessageFilter())
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

            if console:
                ch = logging.StreamHandler()
                ch.setLevel(level)
                ch.setFormatter(fmt)
                self.logger.addHandler(ch)

        self.logger.debug("Logger initialized.")

    def get_logger(self):
        return self.logger

    # convenience
    def debug(self, m): self.logger.debug(m)
    def info(self, m): self.logger.info(m)
    def warning(self, m): self.logger.warning(m)
    def error(self, m): self.logger.error(m)
    def critical(self, m): self.logger.critical(m)
