import os
from argparse import Namespace
from pathlib import Path
from layup.utilities.data_processing_utilities import process_data
from layup.routines import predict_sequence, Observation, FitResult, get_ephem, numpy_to_eigen
from layup.utilities.file_io import CSVDataReader
from layup.utilities.data_processing_utilities import LayupObservatory, parse_fit_result
from layup.utilities.datetime_conversions import convert_tdb_date_to_julian_date
import spiceypy as spice
import numpy as np
import pooch

def _predict(data, obs_pos_vel, times, cache_dir):
    """This function is called by the parallelization function to call the C++ code."""
    if cache_dir is None:
        kernels_loc = str(pooch.os_cache("layup"))
    else:
        kernels_loc = str(cache_dir)

    observations = []
    for i, pos in enumerate(obs_pos_vel):
        obs = Observation()
        obs.observer_position = [pos["x"], pos["y"], pos["z"]]
        obs.observer_velocity = [pos["vx"], pos["vy"], pos["vz"]]
        obs.epoch = times[i]
        observations.append(obs)

    predict_results = []
    for row in data:
        fit = parse_fit_result(row)

        cov = []
        for i in range(10):
            cov.append(row[f"cov_0{i}"])
        for i in range(10, 36):
            cov.append(row[f"cov_{i}"])
        cov = np.array(cov)

        print(cov)

        predict_results.append(
            predict_sequence(
                get_ephem(kernels_loc),
                fit,
                observations,
                numpy_to_eigen(cov, 6, 6)
            )
        )

    for p in predict_results:
        for r in p:
            print(r)
            print(r.rho)
            print(r.obs_cov)
            print(r.epoch)

    return np.array([1])


def predict(data, obscode, times, primary_id_column_name="provID", num_workers=-1, cache_dir=None):
    """This is the stub that will be used when calling predict from a notebook"""
    if num_workers < 0:
        num_workers = os.cpu_count()

    Layup_observatory = LayupObservatory()

    times_et = np.array([
        spice.str2et(f"jd {t} tdb") for t in times
    ], dtype="<f8")

    obs_data  = np.array([
        (obscode, t) for t in times_et
    ], dtype=[("stn", "<U10"), ("et", "<f8")])

    obs_pos_vel = Layup_observatory.obscodes_to_barycentric(obs_data)

    return process_data(data,
        n_workers=num_workers,
        func=_predict,
        obs_pos_vel=obs_pos_vel,
        times=times,
        cache_dir=cache_dir,
    )


def predict_cli(
    cli_args: Namespace,
    input_file: str,
    start_date: float,
    end_date: float,
    timestep_day: float,
    output_file: str,
    cache_dir: Path,
):
    """This is the stub that will used when calling predict from the command line"""

    print(cli_args)
    num_workers = cli_args.n
    _primary_id_column_name = "provID"

    if num_workers < 0:
        num_workers = os.cpu_count()

    times = np.arange(
        start_date,
        end_date+1
    )

    reader = CSVDataReader(
        input_file,
        primary_id_column_name=_primary_id_column_name,
        sep="csv"
    )

    chunks = _create_chunks(reader, chunk_size=cli_args.chunk)

    for chunk in chunks:
        # Read the objects from the file
        data = reader.read_objects(chunk)

        predictions = predict(
            data,
            obscode=cli_args.station,
            times=times,
            num_workers=cli_args.n,
            cache_dir=cache_dir
        )

    # data = None

    # results = predict(data, num_workers=num_workers)

    # Write the results to the output file
    pass

# TODO: abstract this away
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
