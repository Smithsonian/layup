import logging
import sys
import threading

from datetime import datetime
from multiprocessing import Manager
from pathlib import Path


class LayupLogger:

    def __init__(self, log_directory="."):
        self.q = Manager().Queue(-1)
        self._prepare_logger(log_directory)
        self._start_logger()

    def get_logger(self, name):
        """Convenience function to return a logger under the top level logger.
        This is identical to calling:
        `logger = logging.getLogger(__name__)`

        Parameters
        ----------
        name : str
            The name to use when emitting messages using this logger.

        Returns
        -------
        Logger
            The logger to use to emit message.
        """
        return logging.getLogger(name)

    def __enter__(self):
        """Entry point for using LayupLogger as a context manager

        Returns
        -------
        self
            An instance of the LayupLogger object
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when the context manager exits. Used only to call _stop_logger
        to terminate the loop in the queue and kill the queue thread.
        """
        self._stop_logger()

    def _prepare_logger(self, log_directory="."):
        """Setup for the primary logger.

        Parameters
        ----------
        log_directory : str, optional
            The directory to place the log files, by default "."

        Returns
        -------
        Logger
            The top level logger.
        """

        logger = logging.getLogger("layup")

        # This logger handles all messages >= DEBUG
        logger.setLevel(logging.DEBUG)

        # The format of the log messages
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(process)d - %(levelname)s - %(message)s")

        # Console handler - all messages >= INFO will be recorded to STDERR
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        # Configure log files
        log_location = Path(log_directory)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_file_base_name = f"layup-{timestamp}"
        log_file_info = log_location / f"{log_file_base_name}.log"
        log_file_error = log_location / f"{log_file_base_name}.err"

        # File handler that will record all messages >= DEBUG
        file_handler_info = logging.FileHandler(log_file_info)
        file_handler_info.setFormatter(formatter)
        file_handler_info.setLevel(logging.DEBUG)

        # File handler that will record all messaged >- ERROR
        file_handler_error = logging.FileHandler(log_file_error)
        file_handler_error.setFormatter(formatter)
        file_handler_error.setLevel(logging.ERROR)

        # Add the handlers to the logger
        logger.addHandler(file_handler_info)
        logger.addHandler(file_handler_error)
        logger.addHandler(console_handler)

        # Return the top level logger
        return logger

    def _start_logger(self):
        """Start the thread that runs the log listener queue"""
        self.logger_thread = threading.Thread(target=self._logger_listener)
        self.logger_thread.start()

    def _logger_listener(self):
        """An infinite loop that reads off the queue and emits log records to
        the top level logger.
        """
        while True:
            record = self.q.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)

    def _stop_logger(self):
        """End the infinite loop and kill the thread where the queue was running."""
        self.q.put(None)
        self.logger_thread.join()
