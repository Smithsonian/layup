import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pooch
import spiceypy as spice
from numpy.lib import recfunctions as rfn

from layup.routines import Observation, get_ephem, run_from_vector
from layup.utilities.data_processing_utilities import LayupObservatory, process_data_by_id
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

# Define a structured dtype to match the OrbfitResult fields
_RESULT_DTYPES = np.dtype(
    [
        ("provID", "O"),  # Object ID
        ("csq", "f8"),  # Chi-square value
        ("ndof", "i4"),  # Number of degrees of freedom
        ("x", "f8"),  # The first of 6 state vector elements
        ("y", "f8"),
        ("z", "f8"),
        ("vx", "f8"),
        ("vy", "f8"),
        ("vz", "f8"),  # The last of 6 state vector elements
        ("epoch", "f8"),  # Epoch
        ("niter", "i4"),  # Number of iterations
        ("method", "O"),  # Method used for orbit fitting
        ("flag", "i4"),  # Single-character flag indicating success of the fit
        ("FORMAT", "O"),  # Orbit format
    ]
    + [(f"cov_0{i}", "f8") for i in range(10)]  # Flat covariance matrix (first 10 elements)
    + [(f"cov_{i}", "f8") for i in range(10, 36)]  # Flat covariance matrix (remaining 26 elements)
)


def _orbitfit(data, cache_dir: str):
    """This function will contain all of the calls to the c++ code that will
    calculate an orbit given a set of observations. Note that all observations
    should correspond to the same object.

    This is function that is passed to the parallelizer.

    Parameters
    ----------
    data : numpy structured array
        The object data to derive an orbit for
    cache_dir : str
        The directory where the required orbital files are stored
    """
    if len(data) == 0:
        return np.array([], dtype=_RESULT_DTYPES)

    # Convert the astrometry data to a list of Observations
    # Reminder to label the units.  Within an Observation struct,
    # and internal to the C++ code in general, we are using
    # radians.
    observations = [
        Observation.from_astrometry(
            d["ra"] * np.pi / 180.0,
            d["dec"] * np.pi / 180.0,
            spice.j2000() + d["et"] / (24 * 60 * 60),  # Convert ET to JD TDB
            [d["x"], d["y"], d["z"]],  # Barycentric position
            [d["vx"], d["vy"], d["vz"]],  # Barycentric velocity
        )
        for d in data
    ]

    # if cache_dir is not provided, use the default os_cache
    if cache_dir is None:
        kernels_loc = str(pooch.os_cache("layup"))
    else:
        kernels_loc = str(cache_dir)

    # Perform the orbit fitting
    res = run_from_vector(get_ephem(kernels_loc), observations)

    # Populate our output structured array with the orbit fit results
    success = res.flag == 0
    cov_matrix = tuple(res.cov[i] for i in range(36)) if success else (np.nan,) * 36
    output = np.array(
        [
            (
                data["provID"][0],
                res.csq,
                res.ndof,
            )
            + tuple(res.state[i] for i in range(6))  # Flat state vector
            + (
                res.epoch,
                res.niter,
                res.method,
                res.flag,
                "BCART",  # The base format returned by the C++ code
            )
            + cov_matrix  # Flat covariance matrix
        ],
        dtype=_RESULT_DTYPES,
    )

    return output


def orbitfit(data, cache_dir: str, num_workers=1, primary_id_column_name="provID"):
    """This is the function that you would call interactively. i.e. from a notebook

    Parameters
    ----------
    data : numpy structured array
        The object data to derive an orbit for
    cache_dir : str
        The directory where the required orbital files are stored
    num_workers : int
        The number of workers to use for parallel processing. Default is 1
    primary_id_column_name : str
        The name of the primary identifier column for the objects. Default is "provID".
    """

    layup_observatory = LayupObservatory()

    # The units of et are seconds (from J2000).
    et_col = np.array([spice.str2et(row["obstime"]) for row in data], dtype="<f8")
    data = rfn.append_fields(data, "et", et_col, usemask=False, asrecarray=True)

    pos_vel = layup_observatory.obscodes_to_barycentric(data)
    data = rfn.merge_arrays([data, pos_vel], flatten=True, asrecarray=True, usemask=False)

    return process_data_by_id(
        data, num_workers, _orbitfit, primary_id_column_name=primary_id_column_name, cache_dir=cache_dir
    )


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

    _primary_id_column_name = "provID"

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

    reader = reader_class(input_file, primary_id_column_name=_primary_id_column_name)

    chunks = _create_chunks(reader, chunk_size)

    if cli_args is not None:
        cache_dir = cli_args.ar_data_file_path
    else:
        cache_dir = None

    for chunk in chunks:
        data = reader.read_objects(chunk)

        logger.info(f"Processing {len(data)} rows for {chunk}")

        fit_orbits = orbitfit(
            data,
            cache_dir=cache_dir,
            num_workers=num_workers,
            primary_id_column_name=_primary_id_column_name,
        )

        if output_file_format == "hdf5":
            write_hdf5(fit_orbits, output_file, key="data")
        else:
            write_csv(fit_orbits, output_file)

    print(f"Data has been written to {output_file}")


def _create_chunks(reader, chunk_size):
    """For a given reader create a list of lists of object ids such that the total
    number of entries in the file for all object ids in a given list, will be
    less than the chunk size.

    Parameters
    ----------
    reader : ObjectDataReader
        The file reader object for the input file
    chunk_size : int
        The maximum number of rows to be included in a single list of ids

    Returns
    -------
    chunks : list[list[ObjIds]]
        A list of lists of object ids that can be passed to the reader's read_objects
        method.
    """
    # Force the reader to build the id table and id count dictionary
    reader._build_id_map()

    # Find all object ids with more rows than the max allowed number of rows.
    exceeds_id_list = []
    for k, v in reader.obj_id_counts.items():
        if v > chunk_size:
            exceeds_id_list.append(k)

    # Log an error if the any of the objects have more rows than the chunk size
    if exceeds_id_list:
        logger.error("The following objects have more rows than the max allowed number of rows.")
        for k in exceeds_id_list:
            logger.error(f"Object id {k} has {reader.obj_id_counts[k]} rows")
        raise ValueError("At least one object has more rows than the max allowed number of rows.")

    chunks = []
    obj_ids_in_chunk = []
    accumulator = 0

    # Loop over the object id counts dictionary
    for k, v in reader.obj_id_counts.items():
        # Check if the chunk size is exceeded, if so, save the current chunk and start a new chunk
        if accumulator + v > chunk_size:
            chunks.append(obj_ids_in_chunk)
            obj_ids_in_chunk = []
            accumulator = 0

        # Increase the accumulator and add the object id to the current chunk
        accumulator += v
        obj_ids_in_chunk.append(k)

    # Add the last chunk if it is not empty
    if obj_ids_in_chunk:
        chunks.append(obj_ids_in_chunk)

    return chunks
