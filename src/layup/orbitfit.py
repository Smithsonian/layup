import os
import logging

from pathlib import Path
from typing import Literal

import numpy as np

from layup.utilities.data_processing_utilities import process_data
from layup.utilities.file_io import CSVDataReader, HDF5DataReader
from layup.utilities.file_io.file_output import write_csv, write_hdf5

logger = logging.getLogger(__name__)

INPUT_FORMAT_READERS = {
    "MPC80col": None,
    "ADES_csv": CSVDataReader,
    "ADES_psv": None,
    "ADES_xml": None,
    "ADES_hdf5": HDF5DataReader,
}


def _orbitfit(data, cache_dir:str):
    """This function will contain all of the calls to the c++ code that will
    calculate an orbit given a set of observations.

    This is function that is passed to the parallelizer.

    Parameters
    ----------
    data : numpy structured array
        The object data to derive an orbit for
    cache_dir : str
        The directory where the required orbital files are stored
    """
    return data


def orbitfit(data, cache_dir:str, num_workers=1):
    """This is the function that you would call interactively. i.e. from a notebook

    Parameters
    ----------
    data : numpy structured array
        The object data to derive an orbit for
    cache_dir : str
        The directory where the required orbital files are stored
    num_workers : int
        The number of workers to use for parallel processing. Default is 1
    """
    if num_workers == 1:
        return _orbitfit(data, cache_dir)
    return process_data(data, num_workers, _orbitfit, cache_dir=cache_dir)


def orbitfit_cli(
        input: str,
        input_file_format: Literal["MPC80col", "ADES_csv", "ADES_psv", "ADES_xml", "ADES_hdf5"],
        output_file_stem: str,
        output_file_format: Literal["csv", "hdf5"] = "csv",
        chunk_size: int = 10_000,
        num_workers: int = -1,
        cli_args: dict = None,
):
    """This is the function that is called from the command line

    Parameters
    ----------
    input : str
        Path to the input data file.
    input_file_format : Literal[MPC80col, ADES_csv, ADES_psv, ADES_xml, ADES_hdf5]
        The format of the input data file.
    output_file_stem : str
        The stem of the output file.
    output_file_format : Literal[csv, hdf5] optional (default="csv")
        The format of the output file. Must be one of: "csv", "hdf5"
    num_workers : int, optional (default=-1)
        The number of workers to use for parallel processing of the individual
        chunk. If -1, the number of workers will be set to the number of CPUs on
        the system. The default is 1 worker.
    cli_args : argparse, optional (default=None)
        The argparse object that was created when running from the CLI.
    """

    input_file = Path(input)
    if output_file_format == "csv":
        output_file = Path(f"{output_file_stem}.{output_file_format.lower()}")
    else:
        output_file = (
            Path(f"{output_file_stem}")
            if output_file_stem.endswith(".h5")
            else Path(f"{output_file_stem}.h5")
        )
    output_directory = output_file.parent.resolve()

    if num_workers < 0:
        num_workers = os.cpu_count()

    # Check that input file exists
    if not input_file.exists():
        logger.error(f"Input file {input_file} does not exist")

    # Check that output directory exists
    if not output_directory.exists():
        logger.error(f"Output directory {output_directory} does not exist")

    # Check that chunk size is a positive integer
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        logger.error("Chunk size must be a positive integer")

    # Check that the file format is valid
    if output_file_format.lower() not in ["csv", "hdf5"]:
        logger.error("File format must be 'csv' or 'hdf5'")

    reader_class = INPUT_FORMAT_READERS[input_file_format]
    if reader_class is None:
        logger.error(f"File format {input_file_format} is not supported")

    #! Don't know if this is the correct way to instantiate the reader
    reader = reader_class(input_file, format_column_name="FORMAT")

    # Force the reader to build the id table, and then use the internal `obj_id_table`
    # to get the unique object ids and chunk along those
    reader._build_id_map()
    total_unique_ids = np.unique(reader.obj_id_table).size
    chunks = [(i, min(i + chunk_size, total_unique_ids)) for i in range(0, total_unique_ids, chunk_size)]

    cache_dir = cli_args.ar_data_file_path if cli_args else None

    for chunk_start, chunk_end in chunks:
        data = reader.read_objects(reader.obj_id_table[chunk_start:chunk_end])
        fit_orbits = orbitfit(data, cache_dir=cache_dir, num_workers=num_workers)

        if output_file_format == "hdf5":
            write_hdf5(fit_orbits, output_file, key="data")
        else:
            write_csv(fit_orbits, output_file)

    print(f"Data has been written to {output_file}")
