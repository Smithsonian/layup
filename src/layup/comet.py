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

logger = logging.getLogger(__name__)

INPUT_READERS = {
    "csv": CSVDataReader,
    "hdf5": HDF5DataReader,
}

def _remove_spc(data):
    # Check for short period comets, remove them
    to_delete = []
    for i in range(len(data)):
        if data['e'][i] < 1:
            a = data['q'][i]/(1 - data['e'][i])
            if np.isinf(a) or a<250 or data['q'][i] > 250:
                to_delete.append(i)
    data = np.delete(data, to_delete)

    return data

def _assist_integrate(sim, primary, ex, dt, include_assist=True):
    # Attempts to integrate the simulation, failing that, switches to rebound and tries again
    oi = sim.particles[0].orbit(primary=primary)

    if include_assist==True:
        ex.integrate_or_interpolate(sim.t+dt)
        
    else:
        sim.integrate(sim.t+dt)

    of = sim.particles[0].orbit(primary=primary)

    return oi, of, sim

def _direction_of_integration(sim, step, primary, ex, include_assist=True):
    oi, of, sim = _assist_integrate(sim, primary, ex, step, include_assist=include_assist) # Get initial values of oi, of
    if oi.d < of.d:
        # Moving outwards initially
        dt = -abs(step)
        oi, of, sim = _assist_integrate(sim, primary, ex, dt, include_assist=include_assist)

        while of.d < oi.d: # Returns to its perihelion
            oi, of, sim = _assist_integrate(sim, primary, ex, dt, include_assist=include_assist)
            #print(of.d)

    else:
        # Moving inwards; if already passed 250au go back, otherwise go forward
        if of.d > 250:
            dt = abs(step)

        else:
            dt = -abs(step)
            oi, of, sim = _assist_integrate(sim, primary, ex, dt, include_assist=include_assist)

    return dt, oi, of

def _apply_comet(data, args, aux=None, cache_dir=None, primary_id_column_name=None):
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

    # Assumes provided data is in COM format
    # Check for short period comets, remove them
    data = _remove_spc(data)
    ephem, Msun, Mtot = create_assist_ephemeris(args, aux)
    primary = rebound.Particle(m=Msun)
    output = np.array([tuple([objid, 0, 0, 0, 0]) for objid in data["ObjID"]], dtype=[("ObjID", "<U16"), ("inv_ao", float), ("ao", float), ("d_ao", float), ("e_ao", float)])

    cols = data.dtype.names
    orbit_df = pd.DataFrame(data, columns=cols, index=data['ObjID'])
    sim_dict = generate_simulations(ephem, Msun, Mtot, orbit_df, args)
    step = 10
    repeat = []
    repeat_index = []
    for i in range(len(sim_dict)):
        comet = list(sim_dict)[i]
        print(comet)
        sim = sim_dict[comet]['sim']
        ex = sim_dict[comet]['ex']
        dt, oi, of = _direction_of_integration(sim, step, primary, ex) # Decide whether to go backwards in time or forwards
        print(dt)
        if dt > 0:
            while of.d>250 and oi.d > of.d:
                oi, of, sim = _assist_integrate(sim, primary, ex, dt, include_assist=True)

        else:
            while of.d < 250 and oi.d < of.d:
                oi, of, sim = _assist_integrate(sim, primary, ex, dt, include_assist=True)
                #print(of.d)


        if np.isnan(of.d):
            repeat.append(comet)
            repeat_index.append(i)
            print(f"Comet {comet} has exceeded the timeframe of the ASSIST Ephemeris")
        else:
            output['inv_ao'][i] = 1/of.a
            output['ao'][i] = of.a
            output['d_ao'][i] = of.d
            output['e_ao'][i] = of.e 

    sim_dict = generate_simulations(ephem, Msun, Mtot, orbit_df, args)#orbit_df[orbit_df["ObjID"].isin(repeat)], args)
    step=10
    for comet in sim_dict:#repeat:
        i = np.where(orbit_df["ObjID"] == comet)
        print(f"Running rebound only for {comet}")
        sim = sim_dict[comet]['sim']
        ex = sim_dict[comet]['ex']
        oi = sim.particles[0].orbit(primary=primary).d
        sim = assist.simulation_convert_to_rebound(sim, ephem)
        of = sim.particles[0].orbit(primary=primary).d
        print(oi/of)
        dt, oi, of = _direction_of_integration(sim, step, primary, ex, include_assist=False) # Decide whether to go backwards in time or forwards

        if dt > 0:
            while of.d>250 and oi.d > of.d:
                oi, of, sim = _assist_integrate(sim, primary, ex, dt, include_assist=False)

        else:
            while of.d < 250 and oi.d < of.d:
                oi, of, sim = _assist_integrate(sim, primary, ex, dt, include_assist=False)
                #print(of.d)
        output['inv_ao'][i] = 1/of.a
        output['ao'][i] = of.a
        output['d_ao'][i] = of.d
        output['e_ao'][i] = of.e 

    if args.code_format: # Put into same format as CODE, if requested
         output['inv_ao'] *= 1e6
         output.dtype.names = ('ObjID','inv_ao_CODE','ao' ,'d_ao','e_ao')


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
        return _apply_comet(data, args, cache_dir=cache_dir, primary_id_column_name=primary_id_column_name, aux=aux)
    # Parallelize the conversion of the data across the requested number of workers
    return process_data(
        data,
        num_workers,
        _apply_comet,
        args=args,
        cache_dir=cache_dir,
        primary_id_column_name=primary_id_column_name,
        aux=aux
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
        if get_format(chunk_data) != "BCOM":
            chunk_data = convert(
                chunk_data,
                convert_to="BCOM",
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
            aux=aux
        )
        # Write out the converted data in in the requested file format.
        if file_format == "hdf5":
            write_hdf5(comet_data, output_file, key="data")
        else:
            write_csv(comet_data, output_file)

    logger.info(f"Data has been written to {output_file}")
