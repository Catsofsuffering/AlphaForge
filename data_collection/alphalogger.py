import logging
import colorlog
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "runlog"
LOG_DIR.mkdir(exist_ok=True)
# Configuration for log color output in the terminal
log_colors_config = {
    "DEBUG": "white",
    "INFO": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

# Default formats for log output

# default_formats = {
#     "color_format": "%(log_color)s%(asctime)s-%(name)s-%(filename)s-[line:%(lineno)d]-%(levelname)s: %(message)s",
#     "log_format": "%(asctime)s-%(name)s-%(filename)s-[line:%(lineno)d]-%(levelname)s: %(message)s",
# }

default_formats = {
    "color_format": "%(log_color)s%(asctime)s-%(levelname)s: %(message)s",
    "log_format": "%(asctime)s-%(levelname)s: %(message)s",
}


class AlphaLogger:
    """
    The AlphaLogger class is a Python logging utility designed to initialize and manage loggers with handlers
    for both file and console outputs. It automatically creates log files based on the current date,
    sets up log handlers with specified levels, and provides methods to log messages at different severity levels.

    Attributes:
        __now_time (str): The current date formatted as YYYY-MM-DD.
        __all_log_path (str): The path to the log file for all log messages.
        __error_log_path (str): The path to the log file for error messages only.
        __logger (logging.Logger): The logger instance used for logging messages.
        __console_handle (colorlog.StreamHandler): The console handler instance for logging to the console.
        __all_logger_handler (logging.handlers.RotatingFileHandler): The file handler instance for all log messages.
        __error_logger_handler (logging.handlers.RotatingFileHandler): The file handler instance for error log messages.

    Methods:
        __init_logger_handler(log_path): Initializes a file handler for a specified log path.
        __init_console_handle(): Initializes the console handler with colorlog support.
        __set_log_handler(logger_handler, level=logging.DEBUG): Configures the log level for a handler and adds it to the logger.
        __set_color_handle(console_handle): Adds the color-formatted console handler to the logger.
        __set_color_formatter(console_handle, color_config): Sets the formatter for the console handler with specified color configurations.
        __set_log_formatter(file_handler): Sets the formatter for file handlers with a default log format.
        __close_handler(file_handler): Closes a file handler.
        __console(level, message): Logs a message at the specified level and manages handlers accordingly.
        debug(message): Logs a debug message.
        info(message): Logs an informational message.
        warning(message): Logs a warning message.
        error(message): Logs an error message.
        critical(message): Logs a critical message.
    """

    def __init__(self):
        # Current date formatted
        self.__now_time = datetime.now().strftime("%Y-%m-%d")
        # Path for all log information file
        self.__all_log_path = str(LOG_DIR / (self.__now_time + "-all" + ".log"))
        self.__info_log_path = str(LOG_DIR / (self.__now_time + "-info" + ".log"))
        # Path for error log information file
        self.__error_log_path = str(LOG_DIR / (self.__now_time + "-error" + ".log"))
        # Create a logger instance
        self.__logger = logging.getLogger()
        # Set the default log level
        self.__logger.setLevel(logging.DEBUG)

    @staticmethod
    def __init_logger_handler(log_path):
        """
        Initializes a rotating file handler for logging to a file.

        Args:
            log_path (str): The path to the log file where messages will be written.

        Returns:
            logging.handlers.RotatingFileHandler: A configured file handler that will rotate logs.
        """
        logger_handler = RotatingFileHandler(filename=log_path, maxBytes=1 * 1024 * 1024, backupCount=3, encoding="utf-8")
        return logger_handler

    @staticmethod
    def __init_console_handle():
        """
        Initializes a console handler that uses colorlog for colorful output.

        Returns:
            colorlog.StreamHandler: A console handler that formats log messages with colors.
        """
        console_handle = colorlog.StreamHandler()
        return console_handle

    def __set_log_handler(self, logger_handler, level=logging.DEBUG):
        """
        Configures the log level for a handler and adds it to the logger.

        Args:
            logger_handler (logging.Handler): The handler to configure and add.
            level (logging.level): The log level to set for the handler.
        """
        logger_handler.setLevel(level=level)
        self.__logger.addHandler(logger_handler)

    def __set_color_handle(self, console_handle):
        """
        Adds the color-formatted console handler to the logger.

        Args:
            console_handle (colorlog.StreamHandler): The console handler to add.
        """
        # SET THE CONSOLE LOG
        console_handle.setLevel(logging.INFO)
        self.__logger.addHandler(console_handle)

    @staticmethod
    def __set_color_formatter(console_handle, color_config):
        """
        Sets the formatter for the console handler with specified color configurations.

        Args:
            console_handle (colorlog.StreamHandler): The console handler to format.
            color_config (dict): A dictionary containing color configurations for log levels.
        """
        formatter = colorlog.ColoredFormatter(default_formats["color_format"], log_colors=color_config)
        console_handle.setFormatter(formatter)

    @staticmethod
    def __set_log_formatter(file_handler):
        """
        Sets the formatter for file handlers with a default log format.

        Args:
            file_handler (logging.Handler): The file handler to format.
        """
        formatter = logging.Formatter(default_formats["log_format"], datefmt="%a, %d %b %Y %H:%M:%S")
        file_handler.setFormatter(formatter)

    @staticmethod
    def __close_handler(file_handler):
        """
        Closes a file handler to ensure resources are properly released.

        Args:
            file_handler (logging.Handler): The file handler to close.
        """
        file_handler.close()

    def __console(self, level, message):
        """
        Logs a message at the specified level and manages handlers accordingly.

        Args:
            level (str): The log level ('info', 'debug', 'warning', 'error', 'critical').
            message (str): The message to log.
        """
        all_logger_handler = self.__init_logger_handler(self.__all_log_path)
        error_logger_handler = self.__init_logger_handler(self.__error_log_path)
        console_handle = self.__init_console_handle()

        self.__set_log_formatter(all_logger_handler)
        self.__set_log_formatter(error_logger_handler)
        self.__set_color_formatter(console_handle, log_colors_config)

        self.__set_log_handler(all_logger_handler)
        self.__set_log_handler(error_logger_handler, level=logging.ERROR)
        self.__set_color_handle(console_handle)

        if level == "info":
            self.__logger.info(message)
        elif level == "debug":
            self.__logger.debug(message)
        elif level == "warning":
            self.__logger.warning(message)
        elif level == "error":
            self.__logger.error(message)
        elif level == "critical":
            self.__logger.critical(message)

        self.__logger.removeHandler(all_logger_handler)
        self.__logger.removeHandler(error_logger_handler)
        self.__logger.removeHandler(console_handle)

        # self.__close_handler(all_logger_handler)
        self.__close_handler(error_logger_handler)

    def debug(self, message):
        self.__console("debug", message)

    def info(self, message):
        self.__console("info", message)

    def warning(self, message):
        self.__console("warning", message)

    def error(self, message):
        self.__console("error", message)

    def critical(self, message):
        self.__console("critical", message)


if __name__ == "__main__":
    # Usage example
    log = AlphaLogger()
    log.info("This is an info message")
    log.debug("This is a debug message")
    log.warning("This is a warning message")
    log.error("This is an error message")
    log.critical("This is a critical message")
