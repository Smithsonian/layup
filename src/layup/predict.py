import os
from argparse import Namespace
from pathlib import Path
from layup.utilities.data_processing_utilities import process_data
from layup.routines import predict_sequence, Observation, FitResult, get_ephem, numpy_to_eigen
from layup.utilities.file_io import CSVDataReader
from layup.utilities.data_processing_utilities import LayupObservatory, parse_fit_result, create_chunks
from layup.utilities.datetime_conversions import convert_tdb_date_to_julian_date
from layup.utilities.file_io.file_output import write_csv
import spiceypy as spice
import numpy as np
import pooch


def _get_result_dtypes(primary_id_column_name: str):
    """Helper function to create the result dtype with the correct primary ID column name."""
    # Define a structured dtype to match the OrbfitResult fields
    return np.dtype(
        [
            (primary_id_column_name, "O"),  # Object ID
            ("rho_x", "f8"),  # The first of the 3 rho unit vector
            ("rho_y", "f8"),
            ("rho_z", "f8"),
            ("obs_cov0", "f8"),  # The first of 4 of the observer covariance
            ("obs_cov1", "f8"),
            ("obs_cov2", "f8"),
            ("obs_cov3", "f8"),
            ("epoch", "f8"),  # Time for prediction
        ]
    )


def _predict(data, obs_pos_vel, times, cache_dir, primary_id_column_name):
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

        pred_res = predict_sequence(get_ephem(kernels_loc), fit, observations, numpy_to_eigen(cov, 6, 6))

        for pred in pred_res:
            predict_results.append(
                (
                    row[primary_id_column_name],
                    pred.rho[0],
                    pred.rho[1],
                    pred.rho[2],
                    pred.obs_cov[0],
                    pred.obs_cov[1],
                    pred.obs_cov[2],
                    pred.obs_cov[3],
                    pred.epoch,
                )
            )

    return np.array(predict_results, dtype=_get_result_dtypes(primary_id_column_name))


def predict(data, obscode, times, primary_id_column_name="provID", num_workers=-1, cache_dir=None):
    """This is the stub that will be used when calling predict from a notebook"""
    if num_workers < 0:
        num_workers = os.cpu_count()

    Layup_observatory = LayupObservatory()

    times_et = np.array([spice.str2et(f"jd {t} tdb") for t in times], dtype="<f8")

    obs_data = np.array([(obscode, t) for t in times_et], dtype=[("stn", "<U10"), ("et", "<f8")])

    obs_pos_vel = Layup_observatory.obscodes_to_barycentric(obs_data)

    return process_data(
        data,
        n_workers=num_workers,
        func=_predict,
        obs_pos_vel=obs_pos_vel,
        times=times,
        cache_dir=cache_dir,
        primary_id_column_name=primary_id_column_name,
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

    times = np.arange(start_date, end_date + 1, step=timestep_day)

    reader = CSVDataReader(input_file, primary_id_column_name=_primary_id_column_name, sep="csv")

    chunks = create_chunks(reader, chunk_size=cli_args.chunk)

    for chunk in chunks:
        # Read the objects from the file
        data = reader.read_objects(chunk)

        predictions = predict(
            data,
            obscode=cli_args.station,
            times=times,
            num_workers=cli_args.n,
            cache_dir=cache_dir,
            primary_id_column_name=cli_args.primary_id_column_name,
        )

        if len(predictions) > 0:
            write_csv(predictions, output_file)
    pass
