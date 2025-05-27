import logging
import logging.handlers
import os

import numpy as np

from layup.utilities.layup_logging import LayupLogger
from layup.utilities.data_processing_utilities import process_data

logger = logging.getLogger(__name__)


def _apply_log(data, log_queue):
    # Create a new logger and handler just for the `_apply_log` function that
    # will add messages to the queue. The queue will then handle emitting the
    # records to the appropriate handlers.
    qh = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger(__name__)
    root.setLevel(logging.DEBUG)
    root.addHandler(qh)
    root.debug(f"In `_apply_log` with data = {data}")
    return data


def log(layup_logger):

    logger.info("In `log` function.")
    N = 25
    data = np.random.rand(N)
    num_workers = os.cpu_count()

    # Parallelize the call to `_apply_log`
    return process_data(data, num_workers, _apply_log, log_queue=layup_logger.q)


def log_cli(
    layup_logger: LayupLogger = None,
):

    logger.info(f"In `log_cli` function.")

    result = log(layup_logger)

    logger.debug(f"Processed data with this length: {len(result)}")
    logger.error("I think you're great.")
