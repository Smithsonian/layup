import logging
import sys

from pathlib import Path


def prepare_logger(log_directory="."):

    logger = logging.getLogger("layup")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    log_location = Path(log_directory)
    log_file_info = log_location / f"layup.log"
    log_file_error = log_location / f"layup.err"

    file_handler_info = logging.FileHandler(log_file_info)
    file_handler_info.setFormatter(formatter)
    file_handler_info.setLevel(logging.DEBUG)

    file_handler_error = logging.FileHandler(log_file_error)
    file_handler_error.setFormatter(formatter)
    file_handler_error.setLevel(logging.ERROR)

    logger.addHandler(file_handler_info)
    logger.addHandler(file_handler_error)
    logger.addHandler(console_handler)
    return logger
