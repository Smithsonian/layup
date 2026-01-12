import logging
import os
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import rebound, assist
from sorcha.ephemeris.simulation_setup import create_assist_ephemeris, generate_simulations

from layup.convert import get_output_column_names_and_types, convert
from layup.utilities.file_io import CSVDataReader, HDF5DataReader
from layup.utilities.file_io.file_output import write_csv, write_hdf5
from layup.utilities.data_processing_utilities import (
    get_format,
    process_data,
)

# These are the maximum and minimum dates that the ASSIST ephemeris file allows for
ASSIST_TIMEFRAME_MAX_MJD = 236455
ASSIST_TIMEFRAME_MIN_MJD = -163545
logger = logging.getLogger(__name__)

INPUT_READERS = {
    "csv": CSVDataReader,
    "hdf5": HDF5DataReader,
}


def _remove_spc(data):
    """
    Removes Short Period Comets (SPCs) from the input data

    Parameters
    ----------
    data : numpy structured array
        The data to remove SPCs from.

    Returns
    -------
    data : numpy structured array
        The data with SPCs removed.
    """

    # Convert to BCOM if not already
    if get_format(data) != "BCOM":
        data_BCOM = convert(
            data,
            convert_to="BCOM",
        )
    else:
        data_BCOM = data
    to_delete = []
    for i in range(len(data_BCOM)):
        if data_BCOM["e"][i] < 1:
            a = data_BCOM["q"][i] / (1 - data_BCOM["e"][i])
            if np.isinf(a) or a < 250 or data_BCOM["q"][i] > 250:
                to_delete.append(i)
    data = np.delete(data, to_delete)

    return data


def _assist_integrate(sim, ex, dt, ephem, include_assist=True):
    """
    Integrates the simulation across a specified time, and returns the orbits before and after integrating.

    Parameters
    ----------
    sim : REBOUND simulation
        The data to use comet function on.
    ephem : ASSIST Ephem Object
        The ASSIST Ephemeris.
    ex : ASSIST Extras object
        The ASSIST extras.
    dt : int
        The timestep to integrate across.
    include_assist : bool, optional
        If True, the simulation will run the integration through ASSIST, otherwise it will be a pure REBOUND integration. Default is True.

    Returns
    -------
    oi : REBOUND Orbit instance
        The orbit of the simulated particle before integration.
    of : REBOUND Orbit instance
        The orbit of the simulated particle after integration.
    sim : REBOUND simulation
        The simulation after integration.
    """
    if include_assist == True:
        primary = ephem.get_particle("sun", sim.t)
    else:
        primary = sim.particles[0]
    oi = sim.particles[-1].orbit(primary=primary)

    if include_assist == True:
        ex.integrate_or_interpolate(sim.t + dt)

    else:
        sim.integrate(sim.t + dt)

    if include_assist == True:
        primary = ephem.get_particle("sun", sim.t)
    else:
        primary = sim.particles[0]
    of = sim.particles[-1].orbit(primary=primary)

    return oi, of, sim


def _direction_of_integration(sim, ex, step, ephem, include_assist=True):
    """
    Determines if the simulation is approaching or receding from d=250au, and from this sets the timestep to be positive or negative in order to approach this distance.

    Parameters
    ----------
    sim : REBOUND simulation
        The data to use comet function on.
    step : int
        The timestep to integrate across.
    ephem : ASSIST Ephem Object
        The ASSIST Ephemeris.
    ex : ASSIST Extras object
        The ASSIST extras.
    dt : int
        The timestep to integrate across.
    include_assist : bool, optional
        If True, the simulation will run the integration through ASSIST, otherwise it will be a pure REBOUND integration. Default is True.

    Returns
    -------
    dt : int
        The timestep modified to be positive or negative, depending on the direction the comet must be integrated to get to d=250au
    oi : REBOUND Orbit instance
        The orbit of the simulated particle before integration.
    of : REBOUND Orbit instance
        The orbit of the simulated particle after integration.
    """

    oi, of, sim = _assist_integrate(
        sim, ex, step, ephem, include_assist=include_assist
    )  # Get initial values of oi, of
    if oi.d < of.d:
        # Moving outwards initially
        dt = -abs(step)
        oi, of, sim = _assist_integrate(sim, ex, dt, ephem, include_assist=include_assist)

        while of.d < oi.d:  # Returns to its perihelion
            oi, of, sim = _assist_integrate(sim, ex, dt, ephem, include_assist=include_assist)

    else:
        # Moving inwards; if already passed 250au go back, otherwise go forward
        if of.d > 250:
            dt = abs(step)

        else:
            dt = -abs(step)
            oi, of, sim = _assist_integrate(sim, ex, dt, ephem, include_assist=include_assist)

    return dt, oi, of


def _apply_comet(data, args, aux=None, cache_dir=None, primary_id_column_name=None):
    """
    Determines original orbit for comets

    Parameters
    ----------
    data : numpy structured array
        The data to use comet function on.
    args : argparse
        The argparse object that was created when running from the CLI.
    aux : AuxiliaryConfigs object
        The LayUp auxiliary arguments
    cache_dir : str, optional
        The base directory for downloaded files.
    primary_id_column_name : str, optional
        The name of the column in the data that contains the primary ID of the object.

    Returns
    -------
    output : numpy structured array
        The comet data output
    """

    # Check for short period comets, remove them
    data = _remove_spc(data)
    ephem, Msun, Mtot = create_assist_ephemeris(args, aux)

    output = {objid: (np.nan, np.nan, np.nan, np.nan) for objid in data["ObjID"]}

    # Convert to pandas to use generate_simulations
    cols = data.dtype.names
    orbit_df = pd.DataFrame(data, columns=cols, index=data["ObjID"])
    sim_dict = generate_simulations(ephem, Msun, Mtot, orbit_df, args)
    step = 10  # Guess to begin with
    rebound_only = []

    for comet in sim_dict:

        sim = sim_dict[comet]["sim"]
        ex = sim_dict[comet]["ex"]

        dt, oi, of = _direction_of_integration(
            sim, ex, step, ephem
        )  # Decide whether to go backwards in time or forwards

        if dt > 0:
            while of.d > 250 and oi.d > of.d and sim.t < ASSIST_TIMEFRAME_MAX_MJD:
                oi, of, sim = _assist_integrate(sim, ex, dt, ephem, include_assist=True)

        else:
            while of.d < 250 and oi.d < of.d and sim.t > ASSIST_TIMEFRAME_MIN_MJD:
                oi, of, sim = _assist_integrate(sim, ex, dt, ephem, include_assist=True)

        if sim.t >= ASSIST_TIMEFRAME_MAX_MJD or sim.t <= ASSIST_TIMEFRAME_MIN_MJD:
            # If comet goes outside assist timeframe, continue the simulation in pure rebound
            rebound_only.append(comet)
            print(f"{comet} has exceeded the timeframe of the ASSIST Ephemeris")
        else:
            output[comet] = (1 / of.a, of.a, of.d, of.e)

    # Repeat steps as before, only as a rebound simulation
    sim_dict = generate_simulations(ephem, Msun, Mtot, orbit_df[orbit_df["ObjID"].isin(rebound_only)], args)
    step = 10
    for comet in rebound_only:
        print(f"Running rebound only for {comet}")
        sim = sim_dict[comet]["sim"]
        ex = sim_dict[comet]["ex"]

        primary = ephem.get_particle("sun", sim.t)
        oi = sim.particles[-1].orbit(primary=primary).d
        sim = assist.simulation_convert_to_rebound(sim, ephem)
        primary = ephem.get_particle("sun", sim.t)
        of = sim.particles[-1].orbit(primary=primary).d
        dt, oi, of = _direction_of_integration(
            sim, ex, step, ephem, include_assist=False
        )  # Decide whether to go backwards in time or forwards

        if dt > 0:
            while of.d > 250 and oi.d > of.d:
                oi, of, sim = _assist_integrate(sim, ex, dt, ephem, include_assist=False)

        else:
            while of.d < 250 and oi.d < of.d:
                oi, of, sim = _assist_integrate(sim, ex, dt, ephem, include_assist=False)
        output[comet] = (1 / of.a, of.a, of.d, of.e)

    # turn output into an array
    output = np.array(
        [tuple([comet, *output[comet]]) for comet in output],
        dtype=[
            ("ObjID", "<U16"),
            ("inv_ao", float),
            ("ao_barycentric", float),
            ("d_ao", float),
            ("e_ao", float),
        ],
    )

    if args.code_format:  # Put into same format as CODE Catalogue, if requested
        output["inv_ao"] *= 1e6
        output.dtype.names = ("ObjID", "inv_ao_CODE", "ao_barycentric", "d_ao", "e_ao")

    return output


def comet(data, num_workers=1, cache_dir=None, primary_id_column_name="ObjID", args=None, aux=None):
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
        return _apply_comet(
            data, args, cache_dir=cache_dir, primary_id_column_name=primary_id_column_name, aux=aux
        )
    # Parallelize the conversion of the data across the requested number of workers
    return process_data(
        data,
        num_workers,
        _apply_comet,
        args=args,
        cache_dir=cache_dir,
        primary_id_column_name=primary_id_column_name,
        aux=aux,
    )


def comet_cli(
    input: str,
    output_file_stem: str,
    file_format: Literal["csv", "hdf5"] = "csv",
    chunk_size: int = 10_000,
    num_workers: int = -1,
    cli_args: dict = None,
    aux=None,
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
            args=cli_args,
            aux=aux,
        )
        # Write out the converted data in in the requested file format.
        if file_format == "hdf5":
            write_hdf5(comet_data, output_file, key="data")
        else:
            write_csv(comet_data, output_file)

    logger.info(f"Data has been written to {output_file}")
