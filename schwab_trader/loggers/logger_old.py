import logging
import os

class UniqueMessageFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.last_message = None

    def filter(self, record):
        current_message = record.getMessage()
        if current_message != self.last_message:
            self.last_message = current_message
            return True
        return False

class Logger:
    def __init__(self, log_file: str, logger_name: str, log_dir: str = 'logs', level: int = logging.DEBUG, timestamp_format: str = '%Y-%m-%d %H:%M:%S'):
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Full path to the log file
        full_log_file = os.path.join(log_dir, log_file)
        
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        
        # Create handlers only if not already added
        if not self.logger.hasHandlers():
            # Create file handler
            file_handler = logging.FileHandler(full_log_file)
            file_handler.setLevel(level)
            
            # Add unique message filter to avoid duplicate log messages
            unique_message_filter = UniqueMessageFilter()
            file_handler.addFilter(unique_message_filter)
            
            # Create formatter with customizable timestamp format
            formatter = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt=timestamp_format)
            file_handler.setFormatter(formatter)
            
            # Add file handler to logger
            self.logger.addHandler(file_handler)
            
            # Optionally add console logging
            # console_handler = logging.StreamHandler()
            # console_handler.setLevel(level)
            # console_handler.setFormatter(formatter)
            # self.logger.addHandler(console_handler)
        
        # Debugging information
        self.logger.debug("Logger initialized.")
        self.logger.debug(f"Log file: {full_log_file}")
        self.logger.debug(f"Logger name: {logger_name}")

    def get_logger(self):
        return self.logger

    # Convenience methods for different log levels
    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)


# Usage Example
#if __name__ == "__main__":
#    logger_instance = Logger('application.log', 'MyAppLogger')
#    logger = logger_instance.get_logger()
#    logger.info("This is an info message.")
#    logger.error("This is an error message.")
