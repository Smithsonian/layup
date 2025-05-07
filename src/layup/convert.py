import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
from sorcha.ephemeris.simulation_geometry import equatorial_to_ecliptic
from sorcha.ephemeris.simulation_parsing import parse_orbit_row
from sorcha.ephemeris.simulation_setup import _create_assist_ephemeris

from layup.utilities.data_processing_utilities import process_data
from layup.utilities.file_io import CSVDataReader, HDF5DataReader
from layup.utilities.file_io.file_output import write_csv, write_hdf5
from layup.utilities.layup_configs import LayupConfigs
from layup.utilities.orbit_conversion import (
    covariance_cometary_xyz,
    covariance_keplerian_xyz,
    covariance_xyz_cometary,
    covariance_xyz_keplerian,
    universal_cometary,
    universal_keplerian,
)

logger = logging.getLogger(__name__)


# Columns which use degrees as units in each orbit format
degree_columns = {
    "BCOM": ["inc", "node", "argPeri"],
    "COM": ["inc", "node", "argPeri"],
    "BKEP": ["inc", "node", "argPeri", "ma"],
    "KEP": ["inc", "node", "argPeri", "ma"],
}

# Add this to MJD to convert to JD
MJD_TO_JD_CONVERSTION = 2400000.5

INPUT_READERS = {
    "csv": CSVDataReader,
    "hdf5": HDF5DataReader,
}


def get_output_column_names_and_types(primary_id_column_name, has_covariance):
    """
    Get the output column names and types for the converted data.

    Parameters
    ----------
    primary_id_column_name : str
        The name of the column in the data that contains the primary ID of the object.
    has_covariance : bool
        Whether the data has covariance information.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - A dictionary mapping orbit formats to the required column names for that format.
        - A list of default column dtypes for the output data.
    """

    # Required column names for each orbit format
    required_column_names = {
        "BCART": [primary_id_column_name, "FORMAT", "x", "y", "z", "xdot", "ydot", "zdot", "epochMJD_TDB"],
        "BCOM": [
            primary_id_column_name,
            "FORMAT",
            "q",
            "e",
            "inc",
            "node",
            "argPeri",
            "t_p_MJD_TDB",
            "epochMJD_TDB",
        ],
        "BKEP": [primary_id_column_name, "FORMAT", "a", "e", "inc", "node", "argPeri", "ma", "epochMJD_TDB"],
        "CART": [primary_id_column_name, "FORMAT", "x", "y", "z", "xdot", "ydot", "zdot", "epochMJD_TDB"],
        "COM": [
            primary_id_column_name,
            "FORMAT",
            "q",
            "e",
            "inc",
            "node",
            "argPeri",
            "t_p_MJD_TDB",
            "epochMJD_TDB",
        ],
        "KEP": [primary_id_column_name, "FORMAT", "a", "e", "inc", "node", "argPeri", "ma", "epochMJD_TDB"],
    }
    # Default column dtypes across all orbit formats. Note that the ordering of the dtypes matches
    # the ordering of the column names in REQUIRED_COLUMN_NAMES.
    default_column_dtypes = ["<U12", "<U5", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8"]
    if has_covariance:
        # Flattened 6x6 covariance matrix
        default_column_dtypes += ["<f8"] * 36
    for format in required_column_names:
        # Add the covariance columns to the required column names
        if has_covariance:
            required_column_names[format] += [f"cov_0{i}" for i in range(10)]
            required_column_names[format] += [f"cov_{i}" for i in range(10, 36)]

    return required_column_names, default_column_dtypes


def parse_covariance_row_to_BCART(row, init_format, cov, ephem, gm_total, gm_sun):
    # Converts the covariance row and a covariance matrix into BCART
    if init_format == "BCART" or init_format == "CART":
        # It is simply a translation of the covariance matrix
        return cov
    elif init_format == "BCOM":
        # Convert the covariance matrix from BCOM to BCART
        # TODO how to actually handle this.
        cov = covariance_xyz_cometary(
            gm_total,
            row["q"],
            row["e"],
            row["inc"],
            row["node"],
            row["argPeri"],
            row["tp"],
            row["epochMJD_TDB"],
            cov,
        )
    elif init_format == "COM":
        # Convert the covariance matrix from COM to BCART
        # TODO actually handle this
        cov = covariance_xyz_cometary(
            gm_sun,
            row["q"],
            row["e"],
            row["inc"],
            row["node"],
            row["argPeri"],
            row["tp"],
            row["epochMJD_TDB"],
            cov,
        )
    elif init_format in ["KEP", "BKEP"]:
        if init_format == "KEP":
            # Handle the conversion to barycentric coordinates
            # TODO how to appropriately use the Sun?
            sun = ephem.get_particle("Sun", row["epochMJD_TDB"] + MJD_TO_JD_CONVERSTION - ephem.jd_ref)
        # Convert the covariance matrix from BKEP to BCART
        a = row["a"]
        e = row["e"]
        incl = row["inc"] * np.pi / 180.0
        longnode = row["node"] * np.pi / 180.0
        argperi = row["argPeri"] * np.pi / 180.0
        M = row["ma"] * np.pi / 180
        if np.pi < M:
            M -= 2 * np.pi

        mu = gm_total if init_format == "BKEP" else gm_sun
        cov = covariance_xyz_keplerian(mu, a, e, incl, longnode, argperi, M, row["epochMJD_TDB"], cov)

    else:
        raise ValueError(f"Invalid input format: {init_format}")
    return cov


def _apply_convert(data, convert_to, cache_dir=None, primary_id_column_name=None):
    """
    Apply the appropriate conversion function to the data

    Parameters
    ----------
    data : numpy structured array
        The data to convert.
    convert_to : str
        The orbital format to convert the data to. Must be one of: "BCART", "BCOM", "BKEP", "CART", "COM", "KEP"
    cache_dir : str, optional
        The base directory for downloaded files.
    primary_id_column_name : str, optional
        The name of the column in the data that contains the primary ID of the object.

    Returns
    -------
    data : numpy structured array
        The converted data
    """
    if len(data) == 0:
        return data

    if convert_to not in ["BCART", "BCOM", "BKEP", "CART", "COM", "KEP"]:
        raise ValueError("Invalid conversion type")
    # TODO validate all cov column names are present. Maybe make helper function
    has_covariance = "cov_00" in data.dtype.names
    logger.debug(f"Data does not have covariance: {not has_covariance}")

    required_colum_names, default_column_dtypes = get_output_column_names_and_types(
        primary_id_column_name, has_covariance
    )

    # Fetch layup configs to get the necessary auxiliary data
    config = LayupConfigs()
    ephem, gm_sun, gm_total = _create_assist_ephemeris(config.auxiliary, cache_dir)

    # Construct the output dtype for the converted data
    output_dtype = [
        (col, dtype)
        for col, dtype in zip(required_colum_names[convert_to], default_column_dtypes, strict=False)
    ]

    # For each row in the data, convert the orbit to the desired format
    results = []
    for d in data:
        # Regardless of the input format, parse_orbit row will always return the Barycentric Cartesian coordinates
        if d["FORMAT"] in ["BCART", "CART"]:
            # We don't use parse_orbit_row here because we already have the equatorial coordinates
            x, y, z, xdot, ydot, zdot = d["x"], d["y"], d["z"], d["xdot"], d["ydot"], d["zdot"]
            if d["FORMAT"] == "CART":
                # Convert to BCART by adding the Sun's position and velocity to the Cartesian coordinates
                sun = ephem.get_particle("Sun", d["epochMJD_TDB"] + MJD_TO_JD_CONVERSTION - ephem.jd_ref)
                x += sun.x
                y += sun.y
                z += sun.z
                xdot += sun.vx
                ydot += sun.vy
                zdot += sun.vz
        else:
            x, y, z, xdot, ydot, zdot = parse_orbit_row(
                d, d["epochMJD_TDB"] + MJD_TO_JD_CONVERSTION, ephem, {}, gm_sun, gm_total
            )
        cov = np.full((6, 6), np.nan)
        if has_covariance:
            flat_cov = [d[f"cov_0{i}"] for i in range(10)] + [d[f"cov_{i}"] for i in range(10, 36)]
            # Populate our covariance matrix
            for i in range(6):
                for j in range(6):
                    # flatten the covariance matrix
                    cov[i][j] = flat_cov[(i * 6) + j]
            cov = parse_covariance_row_to_BCART(d, d["FORMAT"], cov, ephem, gm_total, gm_sun)
        if convert_to == "BCART":
            # Already in equatorial BCART so simply use the parsed coordinates
            row = x, y, z, xdot, ydot, zdot

        elif convert_to == "CART":
            # Convert to CART by subtracting the Sun's position and velocity from the Barycentric Cartesian coordinates
            sun = ephem.get_particle("Sun", d["epochMJD_TDB"] + MJD_TO_JD_CONVERSTION - ephem.jd_ref)
            equatorial_coords = np.array((x, y, z)) - np.array((sun.x, sun.y, sun.z))
            equatorial_velocities = np.array((xdot, ydot, zdot)) - np.array((sun.vx, sun.vy, sun.vz))

            row = tuple(np.concatenate([equatorial_coords, equatorial_velocities]))

        elif convert_to == "BCOM":
            if has_covariance:
                cov = covariance_cometary_xyz(gm_total, x, y, z, xdot, ydot, zdot, d["epochMJD_TDB"], cov)
            # Convert back to ecliptic from parse_orbit_row's equatorial output.
            ecliptic_coords = np.array(equatorial_to_ecliptic([x, y, z]))
            ecliptic_velocities = np.array(equatorial_to_ecliptic([xdot, ydot, zdot]))
            x, y, z, xdot, ydot, zdot = tuple(np.concatenate([ecliptic_coords, ecliptic_velocities]))

            # Use universal_cometary to convert to BCOM with mu = gm_total (used for barycentric)
            row = universal_cometary(gm_total, x, y, z, xdot, ydot, zdot, d["epochMJD_TDB"])
        elif convert_to == "COM":
            # Convert out of barycentric by subtracting the Sun's position and velocity from the Barycentric Cartesian coordinates
            sun = ephem.get_particle("Sun", d["epochMJD_TDB"] + MJD_TO_JD_CONVERSTION - ephem.jd_ref)
            equatorial_coords = np.array((x, y, z)) - np.array((sun.x, sun.y, sun.z))
            equatorial_velocities = np.array((xdot, ydot, zdot)) - np.array((sun.vx, sun.vy, sun.vz))
            if has_covariance:
                cov = covariance_cometary_xyz(gm_sun, x, y, z, xdot, ydot, zdot, d["epochMJD_TDB"], cov)

            # Convert back to ecliptic from parse_orbit_row's equatorial output.
            ecliptic_coords = np.array(equatorial_to_ecliptic(equatorial_coords))
            ecliptic_velocities = np.array(equatorial_to_ecliptic(equatorial_velocities))
            x, y, z, xdot, ydot, zdot = tuple(np.concatenate([ecliptic_coords, ecliptic_velocities]))

            # Use universal_cometary to convert to COM with mu = gm_sun
            row = universal_cometary(gm_sun, x, y, z, xdot, ydot, zdot, d["epochMJD_TDB"])

        elif convert_to == "BKEP":
            if has_covariance:
                cov = covariance_keplerian_xyz(gm_total, x, y, z, xdot, ydot, zdot, d["epochMJD_TDB"], cov)

            # Convert back to ecliptic from parse_orbit_row's equatorial output.
            ecliptic_coords = np.array(equatorial_to_ecliptic([x, y, z]))
            ecliptic_velocities = np.array(equatorial_to_ecliptic([xdot, ydot, zdot]))
            x, y, z, xdot, ydot, zdot = tuple(np.concatenate([ecliptic_coords, ecliptic_velocities]))

            # Use universal_keplerian to convert to BKEP with mu = gm_total
            row = universal_keplerian(gm_total, x, y, z, xdot, ydot, zdot, d["epochMJD_TDB"])
        elif convert_to == "KEP":
            # Convert out of barycentric by subtracting the Sun's position and velocity from the Barycentric Cartesian coordinates
            sun = ephem.get_particle("Sun", d["epochMJD_TDB"] + MJD_TO_JD_CONVERSTION - ephem.jd_ref)
            equatorial_coords = np.array((x, y, z)) - np.array((sun.x, sun.y, sun.z))
            equatorial_velocities = np.array((xdot, ydot, zdot)) - np.array((sun.vx, sun.vy, sun.vz))

            if has_covariance:
                cov = covariance_keplerian_xyz(gm_sun, x, y, z, xdot, ydot, zdot, d["epochMJD_TDB"], cov)

            # Convert back to ecliptic from parse_orbit_row's equatorial output.
            ecliptic_coords = np.array(equatorial_to_ecliptic(equatorial_coords))
            ecliptic_velocities = np.array(equatorial_to_ecliptic(equatorial_velocities))

            # Use universal_keplerian to convert to KEP with mu = gm_sun
            x, y, z, xdot, ydot, zdot = tuple(np.concatenate([ecliptic_coords, ecliptic_velocities]))
            row = universal_keplerian(gm_sun, x, y, z, xdot, ydot, zdot, d["epochMJD_TDB"])

        # If the covariance matrix is present, convert it to a flattened tuple.
        cov_res = tuple(val for val in cov.flatten()) if has_covariance else tuple()
        # output = (d[primary_id_column_name], convert_to) + row + (d["epochMJD_TDB"],) + cov_res
        # raise ValueError(f"{len(output)} {len(output_dtype)} " + str(output))
        # Turn our converted row into a structured array
        result_struct_array = np.array(
            [(d[primary_id_column_name], convert_to) + row + (d["epochMJD_TDB"],) + cov_res],
            dtype=output_dtype,
        )
        results.append(result_struct_array)

    # Convert the list of results to a numpy structured array
    output = np.squeeze(np.array(results)) if len(results) > 1 else results[0]

    # The outputs of the sorcha orbit conversion utilities are always in radians, so convert to degrees for any such columns.
    for col in degree_columns.get(convert_to, []):
        # Convert from radians to degrees and wrap to [0, 360)
        output[col] = (output[col] * 180 / np.pi) % 360

    return output


def convert(data, convert_to, num_workers=1, cache_dir=None, primary_id_column_name="ObjID"):
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
    primary_id_column_name : str, optional (default="ObjID")
        The name of the column in the data that contains the primary ID of the object.

    Returns
    -------
    data : numpy structured array
        The converted data
    """

    if num_workers == 1:
        return _apply_convert(
            data, convert_to, cache_dir=cache_dir, primary_id_column_name=primary_id_column_name
        )
    # Parallelize the conversion of the data across the requested number of workers
    return process_data(
        data,
        num_workers,
        _apply_convert,
        convert_to=convert_to,
        cache_dir=cache_dir,
        primary_id_column_name=primary_id_column_name,
    )


def convert_cli(
    input: str,
    output_file_stem: str,
    convert_to: Literal["BCART", "BCOM", "BKEP", "CART", "COM", "KEP"],
    file_format: Literal["csv", "hdf5"] = "csv",
    chunk_size: int = 10_000,
    num_workers: int = -1,
    cli_args: dict = None,
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
    required_columns_names, _ = get_output_column_names_and_types(
        primary_id_column_name,
        has_covariance=False,  # Change for function
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
        # Parallelize conversion of this chunk of data.
        converted_data = convert(
            chunk_data,
            convert_to,
            num_workers=num_workers,
            cache_dir=cache_dir,
            primary_id_column_name=primary_id_column_name,
        )
        # Write out the converted data in in the requested file format.
        if file_format == "hdf5":
            write_hdf5(converted_data, output_file, key="data")
        else:
            write_csv(converted_data, output_file)

    print(f"Data has been written to {output_file}")
