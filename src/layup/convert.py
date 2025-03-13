import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np

from concurrent.futures import ProcessPoolExecutor

from sorcha.ephemeris.simulation_geometry import equatorial_to_ecliptic
from sorcha.ephemeris.simulation_setup import create_assist_ephemeris
from sorcha.ephemeris.simulation_parsing import parse_orbit_row
from sorcha.ephemeris.orbit_conversion_utilities import universal_cometary, universal_keplerian

from layup.utilities.file_io.CSVReader import CSVDataReader
from layup.utilities.file_io.HDF5Reader import HDF5DataReader
from layup.utilities.file_io.file_output import write_csv, write_hdf5
from layup.utilities.layup_configs import LayupConfigs

logger = logging.getLogger(__name__)


REQUIRED_COLUMN_NAMES = {
    "BCART": ["ObjID", "FORMAT", "x", "y", "z", "xdot", "ydot", "zdot", "epochMJD_TDB"],
    "BCOM": ["ObjID", "FORMAT", "q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB", "epochMJD_TDB"],
    "BKEP": ["ObjID", "FORMAT", "a", "e", "inc", "node", "argPeri", "ma", "epochMJD_TDB"],
    "CART": ["ObjID", "FORMAT", "x", "y", "z", "xdot", "ydot", "zdot", "epochMJD_TDB"],
    "COM": ["ObjID", "FORMAT", "q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB", "epochMJD_TDB"],
    "KEP": ["ObjID", "FORMAT", "a", "e", "inc", "node", "argPeri", "ma", "epochMJD_TDB"],
}

DEFAULT_COLUMN_DTYPES = ["<U12", "<U5", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8"]

# Add this to MJD to convert to JD
MJD_TO_JD_CONVERSTION = 2400000.5


def process_data(data, n_workers, func, **kwargs):
    """
    Process a structured numpy array in parallel for a given function and keyword arguments

    Parameters
    ----------
    data : numpy structured array
        The data to process.
    n_workers : int
        The number of workers to use for parallel processing.
    func : function
        The function to apply to each block of data within parallel.
    **kwargs : dictionary
        Extra arguments to pass to the function.

    Returns
    -------
    res : numpy structured array
        The processed data concatenated from each function result
    """
    # Divide our data into blocks to be processed by each worker
    block_size = max(1, int(len(data) / n_workers))
    # Create a list of tuples of the form (start, end) where start is the starting index of the block
    # and end is the last index of the block + 1.
    blocks = [(i, min(i + block_size, len(data))) for i in range(0, len(data), block_size)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a future applying the function to each block of data
        futures = [executor.submit(func, data[start:end], **kwargs) for start, end in blocks]
        # Concatenate all processed blocks together as our final result
        return np.concatenate([future.result() for future in futures])


def _apply_convert(data, convert_to):
    """
    Apply the appropriate conversion function to the data

    Parameters
    ----------
    data : numpy structured array
        The data to convert.
    convert_to : str
        The orbital format to convert the data to. Must be one of: "BCART", "BCOM", "BKEP", "CART", "COM", "KEP"

    Returns
    -------
    data : numpy structured array
        The converted data
    """

    config = LayupConfigs()
    ephem, gm_sun, gm_total = create_assist_ephemeris(None, config.auxiliary)

    convert_from = data["FORMAT"][0]
    results = []

    columns_to_convert = {
        "BCOM": ["inc", "node", "argPeri"],
        "COM": ["inc", "node", "argPeri"],
        "BKEP": ["inc", "node", "argPeri", "ma"],
        "KEP": ["inc", "node", "argPeri", "ma"],
    }

    for d in data:
        # `out` is a tuple
        bcart_row = parse_orbit_row(d, d["epochMJD_TDB"] + MJD_TO_JD_CONVERSTION, ephem, {}, gm_sun, gm_total)
        output_dtype = [
            (col, dtype) for col, dtype in zip(REQUIRED_COLUMN_NAMES[convert_to], DEFAULT_COLUMN_DTYPES)
        ]

        # Unpack the tuple to make the code more readable
        (
            x,
            y,
            z,
            xdot,
            ydot,
            zdot,
        ) = bcart_row

        if convert_to == "BCART":
            # return without additional modifications
            ecliptic_coords = np.array(equatorial_to_ecliptic([x, y, z]))
            ecliptic_velocities = np.array(equatorial_to_ecliptic([xdot, ydot, zdot]))

            row = tuple(np.concatenate([ecliptic_coords, ecliptic_velocities]))

        elif convert_to == "CART":
            sun = ephem.get_particle("Sun", d["epochMJD_TDB"] + MJD_TO_JD_CONVERSTION - ephem.jd_ref)
            equitorial_coords = np.array((x, y, z)) - np.array((sun.x, sun.y, sun.z))
            equitorial_velocities = np.array((xdot, ydot, zdot)) - np.array((sun.vx, sun.vy, sun.vz))
            ecliptic_coords = np.array(equatorial_to_ecliptic(equitorial_coords))
            ecliptic_velocities = np.array(equatorial_to_ecliptic(equitorial_velocities))

            row = tuple(np.concatenate([ecliptic_coords, ecliptic_velocities]))

        elif convert_to == "BCOM":
            # Use universal_cometary to convert to BCOM with mu = gm_total
            ecliptic_coords = np.array(equatorial_to_ecliptic([x, y, z]))
            ecliptic_velocities = np.array(equatorial_to_ecliptic([xdot, ydot, zdot]))

            x, y, z, xdot, ydot, zdot = tuple(np.concatenate([ecliptic_coords, ecliptic_velocities]))

            row = universal_cometary(gm_total, x, y, z, xdot, ydot, zdot, d["epochMJD_TDB"])
        elif convert_to == "COM":
            # Use universal_cometary to convert to COM with mu = gm_sun
            sun = ephem.get_particle("Sun", d["epochMJD_TDB"] + MJD_TO_JD_CONVERSTION - ephem.jd_ref)

            equitorial_coords = np.array((x, y, z)) - np.array((sun.x, sun.y, sun.z))
            equitorial_velocities = np.array((xdot, ydot, zdot)) - np.array((sun.vx, sun.vy, sun.vz))
            ecliptic_coords = np.array(equatorial_to_ecliptic(equitorial_coords))
            ecliptic_velocities = np.array(equatorial_to_ecliptic(equitorial_velocities))

            x, y, z, xdot, ydot, zdot = tuple(np.concatenate([ecliptic_coords, ecliptic_velocities]))
            row = universal_cometary(gm_sun, x, y, z, xdot, ydot, zdot, d["epochMJD_TDB"])
        elif convert_to == "BKEP":
            # Use universal_keplerian to convert to BKEP with mu = gm_total
            ecliptic_coords = np.array(equatorial_to_ecliptic([x, y, z]))
            ecliptic_velocities = np.array(equatorial_to_ecliptic([xdot, ydot, zdot]))

            x, y, z, xdot, ydot, zdot = tuple(np.concatenate([ecliptic_coords, ecliptic_velocities]))
            row = universal_keplerian(gm_total, x, y, z, xdot, ydot, zdot, d["epochMJD_TDB"])
        elif convert_to == "KEP":
            # Use universal_keplerian to convert to KEP with mu = gm_sun
            sun = ephem.get_particle("Sun", d["epochMJD_TDB"] + MJD_TO_JD_CONVERSTION - ephem.jd_ref)

            equitorial_coords = np.array((x, y, z)) - np.array((sun.x, sun.y, sun.z))
            equitorial_velocities = np.array((xdot, ydot, zdot)) - np.array((sun.vx, sun.vy, sun.vz))
            ecliptic_coords = np.array(equatorial_to_ecliptic(equitorial_coords))
            ecliptic_velocities = np.array(equatorial_to_ecliptic(equitorial_velocities))

            x, y, z, xdot, ydot, zdot = tuple(np.concatenate([ecliptic_coords, ecliptic_velocities]))
            row = universal_keplerian(gm_sun, x, y, z, xdot, ydot, zdot, d["epochMJD_TDB"])

        result_struct_array = np.array(
            [(d["ObjID"], convert_to) + row + (d["epochMJD_TDB"],)],
            dtype=output_dtype,
        )

        results.append(result_struct_array)

    output = np.squeeze(np.array(results))
    cols = columns_to_convert.get(convert_to, [])

    for col in cols:
        output[col] = output[col] * 180 / np.pi
        for i in range(len(output[col])):
            if output[col][i] < 0:
                output[col][i] += 360

    return output


def convert(data, convert_to, num_workers=1):
    """
    Convert a structured numpy array to a different orbital format with support for parallel processing

    Parameters
    ----------
    data : numpy structured array
        The data to convert.
    convert_to : str
        The format to convert the data to. Must be one of: "BCART", "BCOM", "BKEP", "CART", "COM", "KEP"
    num_workers : int, optional (default=1)
        The number of workers to use for parallel processing.

    Returns
    -------
    data : numpy structured array
        The converted data
    """

    if num_workers == 1:
        return _apply_convert(data, convert_to)
    # Parallelize the conversion of the data across the requested number of workers
    return process_data(data, num_workers, _apply_convert, convert_to=convert_to)


def convert_cli(
    input: str,
    output_file_stem: str,
    convert_to: Literal["BCART", "BCOM", "BKEP", "CART", "COM", "KEP"],
    file_format: Literal["csv", "hdf5"] = "csv",
    chunk_size: int = 10_000,
    num_workers: int = 1,
):
    """
    Convert an orbit file from one format to another with support for parallel processing.

    Note that the output file will be written in the caller's current working directory.

    Parameters
    ----------
    input : str
        The path to the input file.
    output_file_stem : str
        The stem of the output file.
    convert_to : str
        The format to convert the input file to. Must be one of: "BCART", "BCOM", "BKEP", "CART", "COM", "KEP"
    file_format : str, optional (default="csv")
        The format of the output file. Must be one of: "csv", "hdf5"
    chunk_size : int, optional (default=10_000)
        The number of rows to read in at a time.
    num_workers : int, optional (default=1)
        The number of workers to use for parallel processing of the individual
        chunk. If -1, the number of workers will be set to the number of CPUs on
        the system. The default is 1 worker.
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
    if file_format.lower() not in ["csv", "hdf5"]:
        logger.error("File format must be 'csv' or 'hdf5'")

    # Check that the conversion type is valid
    if convert_to not in ["BCART", "BCOM", "BKEP", "CART", "COM", "KEP"]:
        logger.error("Conversion type must be 'BCART', 'BCOM', 'BKEP', 'CART', 'COM', or 'KEP'")

    # Open the input file and read the first line
    if file_format == "hdf5":
        sample_reader = HDF5DataReader(
            input_file,
            format_column_name="FORMAT",
        )
    else:
        sample_reader = CSVDataReader(input_file, format_column_name="FORMAT")

    sample_data = sample_reader.read_rows(block_start=0, block_size=1)

    # Check orbit format in the file
    input_format = None
    if "FORMAT" in sample_data.dtype.names:
        input_format = sample_data["FORMAT"][0]
        if input_format not in ["BCART", "BCOM", "BKEP", "CART", "COM", "KEP"]:
            logger.error(f"Input file contains invalid 'FORMAT' column: {input_format}")
    else:
        logger.error("Input file does not contain 'FORMAT' column")

    # Check that the input format is not already the desired format
    if convert_to == input_format:
        logger.error("Input file is already in the desired format")

    # Reopen the file now that we know the input format and can validate the column names
    required_columns = REQUIRED_COLUMN_NAMES[input_format]
    if file_format == "hdf5":
        reader = HDF5DataReader(
            input_file, format_column_name="FORMAT", required_column_names=required_columns
        )
    else:
        reader = CSVDataReader(
            input_file, format_column_name="FORMAT", required_column_names=required_columns
        )

    # Calculate the start and end indices for each chunk, as a list of tuples
    # of the form (start, end) where start is the starting index of the chunk
    # and the last index of the chunk + 1.
    total_rows = reader.get_row_count()
    chunks = [(i, min(i + chunk_size, total_rows)) for i in range(0, total_rows, chunk_size)]

    for chunk_start, chunk_end in chunks:
        # Read the chunk of data
        chunk_data = reader.read_rows(block_start=chunk_start, block_size=chunk_end - chunk_start)
        # Parallelize conversion of this chunk of data.
        converted_data = convert(chunk_data, convert_to, num_workers=num_workers)
        # Write out the converted data in in the requested file format.
        if file_format == "hdf5":
            write_hdf5(converted_data, output_file, key="data")
        else:
            write_csv(converted_data, output_file)

    print(f"Data has been written to {output_file}")
