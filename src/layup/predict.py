import os
from argparse import Namespace
from pathlib import Path
from layup.utilities.data_processing_utilities import process_data


def _predict(data, other_arg_1=None):
    """This function is called by the parallelization function to call the C++ code."""
    for row in data:
        # Here you would call the C++ prediction code with the row data.
        # cpp_predict_function(row, other_arg_1)
        pass
    pass


def predict(data, num_workers=-1):
    """This is the stub that will be used when calling predict from a notebook"""
    if num_workers < 0:
        num_workers = os.cpu_count()

    return process_data(data, n_workers=num_workers, func=_predict, other_arg_1=None)


def predict_cli(
    cli_args: Namespace,
    start_date: float,
    end_date: float,
    timestep_day: float,
    output_file: str,
    cache_dir: Path,
):
    """This is the stub that will used when calling predict from the command line"""

    num_workers = cli_args.n

    if num_workers < 0:
        num_workers = os.cpu_count()

    data = None

    results = predict(data, num_workers=num_workers)

    # Write the results to the output file
    pass
