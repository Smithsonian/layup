import logging
import os
from pathlib import Path
from typing import Literal
import numpy as np
from sorcha.ephemeris.simulation_setup import create_assist_ephemeris

from layup.convert import get_output_column_names_and_types, convert
from layup.utilities.file_io import CSVDataReader, HDF5DataReader
from layup.utilities.file_io.file_output import write_csv, write_hdf5
from layup.utilities.data_processing_utilities import (
    get_format,
    process_data,
)

logger = logging.getLogger(__name__)

INPUT_READERS = {
    "csv": CSVDataReader,
    "hdf5": HDF5DataReader,
}


def _apply_comet(data, cache_dir=None, primary_id_column_name=None):
    """
    Determines original orbit for comets

    Parameters
    ----------
    data : numpy structured array
        The data to use comet function on.
    cache_dir : str, optional
        The base directory for downloaded files.
    primary_id_column_name : str, optional
        The name of the column in the data that contains the primary ID of the object.

    Returns
    -------
    data : numpy structured array
        The comet data output
    """

    data_kep = convert(
                data,
                convert_to="KEP",
                cache_dir=cache_dir,
                primary_id_column_name=primary_id_column_name,
            )
    to_delete = []

    for i in range(len(data)):

        a = data_kep['a'][i]
        if np.isinf(a) or a<200:
            to_delete.append(i)
    data = np.delete(data, to_delete)


    return data


def comet(data, num_workers=1, cache_dir=None, primary_id_column_name="ObjID"):
    """
    _apply_comet wrapper with support for parallel processing

    Parameters
    ----------
    data : numpy structured array
        The data to use comet function on.
    num_workers : int, optional (default=1)
        The number of workers to use for parallel processing.
    primary_id_column_name : str, optional (default="ObjID")
        The name of the column in the data that contains the primary ID of the object.

    Returns
    -------
    data : numpy structured array
        The comet data output
    """

    if num_workers == 1:
        return _apply_comet(data, cache_dir=cache_dir, primary_id_column_name=primary_id_column_name)
    # Parallelize the conversion of the data across the requested number of workers
    return process_data(
        data,
        num_workers,
        _apply_comet,
        cache_dir=cache_dir,
        primary_id_column_name=primary_id_column_name,
    )


def comet_cli(
    input: str,
    output_file_stem: str,
    file_format: Literal["csv", "hdf5"] = "csv",
    chunk_size: int = 10_000,
    num_workers: int = -1,
    cli_args: dict = None,
):
    """
    Determines original orbit for comets with support for parallel processing.

     Note that the output file will be written in the caller's current working directory.

     Parameters
     ----------
     input : str
         The path to the input file.
     output_file_stem : str
         The stem of the output file.
     file_format : str, optional (default="csv")
         The format of the output file. Must be one of: "csv", "hdf5"
     chunk_size : int, optional (default=10_000)
         The number of rows to read in at a time.
     num_workers : int, optional (default=-1)
         The number of workers to use for parallel processing of the individual
         chunk. If -1, the number of workers will be set to the number of CPUs on
         the system.
     cli_args : argparse, optional (default=None)
         The argparse object that was created when running from the CLI.
    """

    input_file = Path(input)
    if file_format == "csv":
        output_file = Path(f"{output_file_stem}.{file_format.lower()}")
    else:
        output_file = (
            Path(f"{output_file_stem}")
            if output_file_stem.endswith(".h5")
            else Path(f"{output_file_stem}.h5")
        )

    primary_id_column_name = cli_args.primary_id_column_name if cli_args else "ObjID"

    if num_workers < 0:
        num_workers = os.cpu_count()

    # Open the input file and read the first line
    reader_class = INPUT_READERS.get(file_format)
    if reader_class is None:
        logger.error(f"Invalid file format: {file_format}. Must be one of: 'csv', 'hdf5'.")
        raise ValueError(f"Invalid file format: {file_format}. Must be one of: 'csv', 'hdf5'.")

    sample_reader = reader_class(
        input_file,
        format_column_name="FORMAT",
        primary_id_column_name=primary_id_column_name,
    )

    sample_data = sample_reader.read_rows(block_start=0, block_size=1)

    # Check orbit format in the file
    input_format = get_format(sample_data)

    # Reopen the file now that we know the input format and can validate the column names
    required_columns_names, _ = get_output_column_names_and_types(
        primary_id_column_name,
        has_covariance=False,  # Change for function
        extra_cols_to_keep=[],
    )

    required_columns = required_columns_names[input_format]
    full_reader = reader_class(
        input_file,
        format_column_name="FORMAT",
        primary_id_column_name=primary_id_column_name,
        required_column_names=required_columns,
    )

    # Calculate the start and end indices for each chunk, as a list of tuples
    # of the form (start, end) where start is the starting index of the chunk
    # and the last index of the chunk + 1.
    total_rows = full_reader.get_row_count()
    chunks = [(i, min(i + chunk_size, total_rows)) for i in range(0, total_rows, chunk_size)]

    cache_dir = cli_args.ar_data_file_path if cli_args else None

    for chunk_start, chunk_end in chunks:
        # Read the chunk of data
        chunk_data = full_reader.read_rows(block_start=chunk_start, block_size=chunk_end - chunk_start)
        if get_format(chunk_data) != "COM":
            chunk_data = convert(
                chunk_data,
                convert_to="COM",
                num_workers=num_workers,
                cache_dir=cache_dir,
                primary_id_column_name=primary_id_column_name,
            )
        # Parallelize conversion of this chunk of data.
        comet_data = comet(
            chunk_data,
            num_workers=num_workers,
            cache_dir=cache_dir,
            primary_id_column_name=primary_id_column_name,
        )
        # Write out the converted data in in the requested file format.
        if file_format == "hdf5":
            write_hdf5(comet_data, output_file, key="data")
        else:
            write_csv(comet_data, output_file)

    logger.info(f"Data has been written to {output_file}")
