import logging
import os
from pathlib import Path
from typing import Literal
import numpy as np

from layup.utilities.layup_configs import LayupConfigs
from layup.utilities.bootstrap_utilties.download_utilities import make_retriever
import assist
import rebound

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

def create_assist_ephemeris(auxconfigs, cache_dir=None):
    """
    Create an ASSIST ephemeris object and compute GM values. Modified from sorcha.
    """
    retriever = make_retriever(auxconfigs, cache_dir)

    planet_path = retriever.fetch(auxconfigs.jpl_planets)
    small_bodies_path = retriever.fetch(auxconfigs.jpl_small_bodies)

    ephem = assist.Ephem(planets_path=planet_path, asteroids_path=small_bodies_path)
    gm_sun = ephem.get_particle("Sun", 0).m
    gm_total = sum(ephem.get_particle(i, 0).m for i in range(27))

    return ephem, gm_sun, gm_total

def generate_assist_simulation_from_cartesian(row, epoch, ephem, gm_total):
    """
    Given a row with x, y, z, vx, vy, vz, create a REBOUND+ASSIST simulation. Modified from sorcha.
    """
    x, y, z = row["x"], row["y"], row["z"]
    vx, vy, vz = row["xdot"], row["ydot"], row["zdot"]

    if np.any(np.isnan([x, y, z, vx, vy, vz])):
        return None

    sim = rebound.Simulation()
    sim.t = epoch - ephem.jd_ref
    sim.dt = 10.0
    sim.ri_ias15.adaptive_mode = 1
    sim.add(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)

    ex = assist.Extras(sim, ephem)
    ex.forces = [f for f in ex.forces if f != "GR_EIH"] + ["GR_SIMPLE"]

    return sim


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
    layup_config = LayupConfigs()
    auxconfigs = layup_config.auxiliary

    ephem, gm_sun, gm_total = create_assist_ephemeris(auxconfigs, cache_dir=cache_dir)

    output_dtype = data.dtype.descr + [('inv_a0', 'f8')]
    out = np.empty(data.shape, dtype=output_dtype)
    for name in data.dtype.names:
        out[name] = data[name]

    for i, row in enumerate(data):
        try:
            epoch = row["epochMJD_TDB"] + 2400000.5

            sim = generate_assist_simulation_from_cartesian(row, epoch, ephem, gm_total)
            if sim is None:
                obj_id = row[primary_id_column_name] if primary_id_column_name in row.dtype.names else i
                logger.warning(f"Row {i} ({obj_id}): invalid Cartesian state. Skipping.")
                out[i]["inv_a0"] = np.nan
                continue

            primary = rebound.Particle(m=gm_total)

            max_steps = 10_000_000
            step = 0
            deltaT = 10.0  # days
            threshold_distance = 250.0  # AU

            # Get initial orbit at t0
            oi = sim.particles[0].orbit(primary=primary)

            # Take one backward step to get a initial distance
            sim.integrate(sim.t - deltaT)
            of = sim.particles[0].orbit(primary=primary)
            initial_outward = of.d > oi.d  # True if the particle is moving outward initially when integrating backward


            while step < max_steps:
                # Integrate backward
                sim.integrate(sim.t - deltaT)
                step += 1

                of = sim.particles[0].orbit(primary=primary)

                # If we've reached the large heliocentric distance, use this orbit
                if of.d > threshold_distance:
                    out[i]["inv_a0"] = 1.0 / of.a
                    break

                # Flip direction tracker if needed (to allow crossing perihelion)
                if initial_outward and of.d < oi.d:
                    initial_outward = False

                # If particle starts returning inward after outbound phase, we missed aphelion so calculate a and break 
                if not initial_outward and of.d > oi.d:
                    logger.warning(f"Row {i} appears to be turning inward again. Breaking.")
                    out[i]["inv_a0"] = 1.0 / of.a  
                    break

                oi = of  # update previous orbit for next step

            else:
                logger.warning(f"{obj_id} exceeded max steps before reaching 250 AU.")
                out[i]["inv_a0"] = np.nan


        except Exception as e:
            obj_id = row[primary_id_column_name] if primary_id_column_name in row.dtype.names else i
            logger.warning(f"{obj_id}: integration error — {e}")
            out[i]["inv_a0"] = np.nan

    return out


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
        if get_format(chunk_data) != "BCART_EQ":
            chunk_data = convert(
                chunk_data,
                convert_to="BCART_EQ",
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

    print(f"Data has been written to {output_file}")
