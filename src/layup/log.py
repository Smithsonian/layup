import logging
import os

import numpy as np

from layup.utilities.data_processing_utilities import process_data, process_data_by_id

logger = logging.getLogger(__name__)


def _apply_log(data, primary_id_column_name=None):
    logger.debug(f"In `_apply_log` with data = {data}")
    return data


def log(samples: int, num_workers: int):
    logger.info("In `log` function.")

    # Create random data
    data = np.random.rand(samples)

    # Parallelize the call to `_apply_log`
    return process_data(data, num_workers, _apply_log)


def log_by_id(samples: int, num_workers: int):
    logger.info("In `log_by_id` function.")

    # Create a random recarray
    primary_id_column_name = "id"
    dtype = [(primary_id_column_name, "i4"), ("value", "f8")]
    id = np.random.randint(0, 4, samples)
    value = np.random.rand(samples)
    data = np.rec.fromarrays([id, value], dtype=dtype)

    # Parallelize the call to `_apply_log`
    return process_data_by_id(data, num_workers, _apply_log, primary_id_column_name)


def log_cli():
    logger.info(f"In `log_cli` function.")

    samples = 25
    num_workers = os.cpu_count()

    result = log(samples, num_workers)
    logger.debug(f"Processed data with this length: {len(result)}")

    result = log_by_id(samples, num_workers)
    logger.debug(f"Processed data with this length: {len(result)}")

    logger.error("I think you're great.")
