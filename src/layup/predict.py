import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pooch
import spiceypy as spice
from sorcha.ephemeris.simulation_geometry import vec2ra_dec

from layup.convert import convert
from layup.routines import Observation, get_ephem, numpy_to_eigen, predict_sequence
from layup.utilities.data_processing_utilities import (
    LayupObservatory,
    create_chunks,
    get_format,
    layup_furnish_spiceypy,
    parse_fit_result,
    process_data,
    skyplane_cov_to_radec_cov,
)
from layup.utilities.file_io import CSVDataReader
from layup.utilities.file_io.file_output import write_csv


def _get_result_dtypes(primary_id_column_name: str):
    """Helper function to create the result dtype with the correct primary ID column name."""
    # Define a structured dtype to match the OrbfitResult fields
    return np.dtype(
        [
            (primary_id_column_name, "O"),  # Object ID
            ("epoch_UTC", "O"),  # Time for prediction in UTC
            ("epoch_JD_TDB", "f8"),  # Time for prediction in JD TDB
            ("ra_deg", "f8"),
            ("dec_deg", "f8"),
            ("rho_x", "f8"),  # The first of the 3 rho unit vector
            ("rho_y", "f8"),
            ("rho_z", "f8"),
            ("obs_cov0", "f8"),  # The first of 4 of the observer covariance
            ("obs_cov1", "f8"),
            ("obs_cov2", "f8"),
            ("obs_cov3", "f8"),
        ]
    )


def _predict(data, obs_pos_vel, times, cache_dir, primary_id_column_name):
    """This function is called by the parallelization function to call the C++ code.

    Parameters
    ----------
    data : nump structured array
        The data to be processed.
    obs_pos_vel : numpy structured array
        The observer position and velocity.
    times : list
        The times for the predictions, in jd_tdb.
    cache_dir : str
        The directory to the cached kernels.
    primary_id_column_name : str
        The name of the primary ID column.

    Returns
    -------
    numpy structured array with the flattened results
    """
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

    # Load kernels for time conversion.
    layup_furnish_spiceypy(kernels_loc)

    predict_results = []
    for row in data:

        # get the fit result (we don't need the csq and ndof values)
        fit = parse_fit_result(row, orbit_colm_flag=False)
        pred_res = predict_sequence(get_ephem(kernels_loc), fit, observations, numpy_to_eigen(fit.cov, 6, 6))

        for pred in pred_res:
            et = spice.str2et(f"jd {pred.epoch} tdb")
            utc_time = spice.et2utc(et, "C", 0)
            predict_results.append(
                (
                    row[primary_id_column_name],
                    utc_time,
                    pred.epoch,  # JD TDB
                    pred.rho[0],  # place holder
                    pred.rho[0],  # place holder
                    pred.rho[0],
                    pred.rho[1],
                    pred.rho[2],
                    pred.obs_cov[0],
                    pred.obs_cov[1],
                    pred.obs_cov[2],
                    pred.obs_cov[3],
                )
            )

    results = np.array(predict_results, dtype=_get_result_dtypes(primary_id_column_name))
    results["ra_deg"], results["dec_deg"] = vec2ra_dec([results["rho_x"], results["rho_y"], results["rho_z"]])
    results["a_arcsec"], results["b_arcsec"], results["PA_deg"] = skyplane_cov_to_radec_cov(
        results["ra_deg"], results["dec_deg"], results["obs_cov0"], results["obs_cov3"], results["obs_cov1"]
    )

    return results


def predict(data, obscode, times, primary_id_column_name="provID", num_workers=-1, cache_dir=None):
    """The function to all that predict functionality interactively, i.e from a notebook or a script.

    Parameters
    ----------
    data : numpy structured array
        The data to be processed.
    obscode : str
        The observer code.
    times : list
        The times for the predictions, in jd_tdb.
    primary_id_column_name : str
        The name of the primary ID column.
    num_workers : int
        The number of workers to use for parallelization. If -1, use all available cores.
    cache_dir : str or None
        The directory to the cached kernels. If None, use the default cache directory.

    Returns
    -------
    numpy structured array with the flattened results
    """
    if num_workers < 0:
        num_workers = os.cpu_count()

    Layup_observatory = LayupObservatory(cache_dir=cache_dir)

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
    """The function for calling predict through the command line interface.

    Parameters
    ----------
    cli_args : Namespace
        The command line arguments.
    input_file : str
        The input file to read the data from.
    start_date : float
        The start date for the predictions, in jd_tdb.
    end_date : float
        The end date for the predictions, in jd_tdb.
    timestep_day : float
        The time step for the predictions, in days.
    output_file : str
        The output file to write the predictions to.
    cache_dir : Path
        The directory to the cached kernels.
    """

    num_workers = cli_args.n

    if num_workers < 0:
        num_workers = os.cpu_count()

    times = np.arange(start_date, end_date + timestep_day, step=timestep_day)

    reader = CSVDataReader(input_file, primary_id_column_name=cli_args.primary_id_column_name, sep="csv")

    chunks = create_chunks(reader, chunk_size=cli_args.chunk)

    for chunk in chunks:
        # Read the objects from the file
        data = reader.read_objects(chunk)

        if get_format(data) != "BCART_EQ":
            data = convert(
                data,
                "BCART_EQ",
                cache_dir=cache_dir,
                primary_id_column_name=cli_args.primary_id_column_name,
            )

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
