import logging
from termcolor import colored
from datetime import datetime
import os


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': 'yellow',
        'INFO': 'green',
        'DEBUG': 'blue',
        'CRITICAL': 'red',
        'ERROR': 'red'
    }

    def format(self, record):
        log_message = super().format(record)
        return colored(log_message, self.COLORS.get(record.levelname))

def create_logs_directory():
    if not os.path.exists(os.path.join(".", "logs")):
        os.makedirs(os.path.join(".", "logs"))

def setup_logger(log_file, log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_file (str): Path to the log file.
        log_level (int): Logging level (e.g. logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: Configured logger.
    """
    create_logs_directory()
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    # Create file handler which logs messages to a specified file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    # Regular formatter for file handler
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Create console handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    # Colored formatter for console handler
    console_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_filename():
    # Get current date and time
    current_datetime = datetime.now()

    # Convert to string format
    current_datetime_str = current_datetime.strftime('%Y-%m-%d_%H:%M:%S')

    return current_datetime_str + ".log"

# Usage
# log_file = 'app.log'
# logger = setup_logger(log_file)
# logger.info("This is an info message")
# logger.error("This is an error message")
# logger.debug("this  is a debug message")
