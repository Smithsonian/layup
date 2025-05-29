"""This module is meant to be an example for how to logging works in Layup.
It has a corresponding layup_cmdline verb, `log` as well."""

# These imports are needed only for the demo
import os
import numpy as np
from layup.utilities.data_processing_utilities import process_data, process_data_by_id

# NOTE - The following is the "configuration" required to log from this module.
import logging

logger = logging.getLogger(__name__)


def _apply_log(data, primary_id_column_name=None):
    """Simple function that is executed in parallel across all processes."""
    logger.debug(f"In `_apply_log` with data = {data}")
    return data


def log_by_chunk(samples: int, num_workers: int):
    """This example function will generate a random dataset with length `samples`.
    The example dataset is then passed to the multiprocessing function that will
    distribute the data evenly across all available CPU workers."""

    logger.info("In `log` function.")

    # Create random data
    data = np.random.rand(samples)

    # Parallelize the call to `_apply_log`
    return process_data(data, num_workers, _apply_log)


def log_by_id(samples: int, num_workers: int):
    """This example function will generate a random dataset with length `samples`.
    The data will have an `id` column that will take a value between 0 and 4 and
    a `value` column with a random value between 0 and 1. The example dataset is
    then passed to the multiprocessing function that will group data by id and then
    distribute the subsets of data evenly across all available CPU workers."""
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
    """This function is called by `src/layup_cmdline/log.py`. Here it demonstrates
    the use of a few different logging levels and calls the two example functions
    that will trigger the two modes of multiprocessing currently implemented in
    Layup."""

    logger.info(f"In `log_cli` function.")

    samples = 25
    num_workers = os.cpu_count()

    result = log_by_chunk(samples, num_workers)
    logger.debug(f"Processed data with this length: {len(result)}")

    result = log_by_id(samples, num_workers)
    logger.debug(f"Processed data with this length: {len(result)}")

    # This message will be sent to the terminal, .log, and .err files
    logger.error("I think you're great.")
