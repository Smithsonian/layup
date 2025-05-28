import logging
import os

import numpy as np

from layup.utilities.data_processing_utilities import process_data

logger = logging.getLogger(__name__)


def _apply_log(data):
    logger.debug(f"In `_apply_log` with data = {data}")
    return data


def log():
    logger.info("In `log` function.")
    N = 25
    data = np.random.rand(N)
    num_workers = os.cpu_count()

    # Parallelize the call to `_apply_log`
    return process_data(data, num_workers, _apply_log)


def log_cli():
    logger.info(f"In `log_cli` function.")
    result = log()
    logger.debug(f"Processed data with this length: {len(result)}")
    logger.error("I think you're great.")
