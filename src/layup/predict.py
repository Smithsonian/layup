import logging
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pooch
import spiceypy as spice
from sorcha.ephemeris.simulation_geometry import vec2ra_dec, integrate_light_time
from sorcha.ephemeris.simulation_setup import create_assist_ephemeris, furnish_spiceypy, generate_simulations
from sorcha.ephemeris.simulation_driver import (
    EphemerisGeometryParameters,
    calculate_rates_and_geometry,
)
from pandas import DataFrame, Series
from numpy.lib.recfunctions import join_by

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

logger = logging.getLogger(__name__)

# The list of required input column names. Note: This should not include the
# primary id column name.
REQUIRED_INPUT_COLUMN_NAMES = [
    "epochMJD_TDB",
    "FORMAT",
]


def _get_on_sky_data(orbits_df, observations, predictions, args, configs):
    # Create simulations
    ephem, gm_sun, gm_total = create_assist_ephemeris(args, configs.auxiliary)
    furnish_spiceypy(args, configs.auxiliary)
    sim_dict = generate_simulations(ephem, gm_sun, gm_total, orbits_df, args)

    # For each predicted position, want to generate on-sky info
    rates = [(objid, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) for objid in predictions["provID"]]
    rates = np.array(
        rates,
        dtype=[
            ("provID", "<U16"),
            ("epoch_JD_TDB", "f8"),
            ("RARateCosDec_deg_day", "f8"),
            ("DecRate_deg_day", "f8"),
            ("Obj_Sun_x_LTC_km", "f8"),
            ("Obj_Sun_y_LTC_km", "f8"),
            ("Obj_Sun_z_LTC_km", "f8"),
            ("Obj_Sun_vx_LTC_km_s", "f8"),
            ("Obj_Sun_vy_LTC_km_s", "f8"),
            ("Obj_Sun_vz_LTC_km_s", "f8"),
            ("phase_deg", "f8"),
        ],
    )

    rows = [(objid, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) for objid in predictions["provID"]]
    cols = [
            ("FieldID", '<U16'),
            ("fieldJD_TDB", 'f8'),
            ("r_obs_x", 'f8'),
            ("r_obs_y", 'f8'),
            ("r_obs_z", 'f8'),
            ("v_obs_x", 'f8'),
            ("v_obs_y", 'f8'),
            ("v_obs_z", 'f8'),
            ("r_sun_x", 'f8'),
            ("r_sun_y", 'f8'),
            ("r_sun_z", 'f8'),
            ("v_sun_x", 'f8'),
            ("v_sun_y", 'f8'),
            ("v_sun_z", 'f8'),
    ]
    rows = np.array(rows, dtype=cols)
    for i, pred in enumerate(predictions):
        # Setup - define values used later
        provid = pred["provID"]
        ephem_geom_params = EphemerisGeometryParameters()
        ephem_geom_params.obj_id = provid

        v = sim_dict[provid]
        sim, ex = v["sim"], v["ex"]
        obs_pos = observations[i].observer_position
        obs_vel = observations[i].observer_velocity
        sun_pos = ephem.get_particle("sun", pred["epoch_JD_TDB"] - ephem.jd_ref).xyz
        sun_vel = ephem.get_particle("sun", pred["epoch_JD_TDB"] - ephem.jd_ref).vxyz
        # Get rest of geometry params
        (
            ephem_geom_params.rho,
            ephem_geom_params.rho_mag,
            _,
            ephem_geom_params.r_ast,
            ephem_geom_params.v_ast,
        ) = integrate_light_time(sim, ex, pred["epoch_JD_TDB"] - ephem.jd_ref, obs_pos, lt0=0.01)
        ephem_geom_params.rho_hat = ephem_geom_params.rho / ephem_geom_params.rho_mag

        # Formatting the pointing data
        cols = (
            "FieldID",
            "fieldJD_TDB",
            "r_obs_x",
            "r_obs_y",
            "r_obs_z",
            "v_obs_x",
            "v_obs_y",
            "v_obs_z",
            "r_sun_x",
            "r_sun_y",
            "r_sun_z",
            "v_sun_x",
            "v_sun_y",
            "v_sun_z",
        )
        # Make pointing a Series because input needs to be 1darray, every value is a float (even the fieldID) for performance (including a string in a series of floats/ints reduces performance dramatically, as per https://stackoverflow.com/questions/52129791/how-can-i-have-different-types-in-pandas-series-if-pandas-series-uses-numpy)
        pointing = Series(
            (
                i,
                pred["epoch_JD_TDB"],
                *obs_pos,
                *obs_vel,
                *sun_pos,
                *sun_vel,
            ),
            index=cols,
        )

        onsky = calculate_rates_and_geometry(pointing, ephem_geom_params)
        # Only want certain returned values: on-sky rates and phase angle
        onsky_desired = (onsky[n] for n in [7, 9, 10, 11, 12, 13, 14, 15, 22])

        rates[i] = (provid, pred["epoch_JD_TDB"], *onsky_desired)
    return rates


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
            ("obs_cov_xx", "f8"),  # The first of 4 of the observer covariance
            ("obs_cov_yy", "f8"),
            ("obs_cov_xy", "f8"),
            ("a_arcsec", "f8"),
            ("b_arcsec", "f8"),
            ("PA_deg", "f8"),
        ]
    )


def _convert_to_sg(data):
    """This function appends two columns of the RA and Dec in sexagesimal to the input array.

    Parameters
    ----------
    data : numpy structured array
        The data to be processed.

    Returns
    -------
    input array with ra and dec in sexagesimal appended, called ra_str_hms and dec_str_dms respectively.
    """
    ra_deg = (data["ra_deg"] / 15) % 24  # Ensuring ra is within 24 hours/360 degrees
    ra_h = ra_deg.astype(int)
    dec_deg = data["dec_deg"]
    dec_d = dec_deg.astype(int)
    ra_decimal = (ra_deg % 1) * 60
    ra_m = ra_decimal.astype(int)
    dec_decimal = (np.abs(dec_deg) % 1) * 60
    dec_m = dec_decimal.astype(int)
    ra_s = (ra_decimal % 1) * 60  # Take decimal portion again for arcseconds
    dec_s = (dec_decimal % 1) * 60

    ra = np.empty(len(ra_h), dtype="<U16")
    dec = np.empty(len(ra_h), dtype="<U16")
    
    for i in range(len(ra_h)):

        ra[i] = f"{ra_h[i]:02} {ra_m[i]:02} {ra_s[i]:05.2f}"  # Same format as
        dec[i] = f"{'-' if dec_deg[i] < 0 else '+'}{dec_d[i]:02} {dec_m[i]:02} {dec_s[i]:04.1f}"  # JPL Horizons

    return np.lib.recfunctions.append_fields(data, ["ra_str_hms", "dec_str_dms"], [ra, dec], usemask=False)


def _predict(data, obs_pos_vel, times, cache_dir, primary_id_column_name, args, configs):
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
                    pred.obs_cov[2],
                    pred.obs_cov[1],
                    0.0,
                    0.0,
                    0.0,
                )
            )

    results = np.array(predict_results, dtype=_get_result_dtypes(primary_id_column_name))
    results["ra_deg"], results["dec_deg"] = vec2ra_dec([results["rho_x"], results["rho_y"], results["rho_z"]])
    results["a_arcsec"], results["b_arcsec"], results["PA_deg"] = skyplane_cov_to_radec_cov(
        results["ra_deg"],
        results["dec_deg"],
        results["obs_cov_xx"],
        results["obs_cov_yy"],
        results["obs_cov_xy"],
    )
    if args.onsky_data:  # Get onsky data if flagged
        # generate_simulations (used in _get_on_sky_data) doesn't accept BCART_EQ, accepts COM, KEP, CART and their barycentric equivalents
        data = convert(
            data,
            "BCART",
            cache_dir=cache_dir,
            primary_id_column_name=args.primary_id_column_name,
        )
        cols = data.dtype.names
        orbits_df = DataFrame(data, columns=cols, index=data["provID"])
        orbits_df = orbits_df.rename(columns={"provID": "ObjID"})
        onsky_results = _get_on_sky_data(orbits_df, observations, results, args, configs)
        results = join_by(["provID", "epoch_JD_TDB"], results, onsky_results, usemask=False, asrecarray=True)

    return results


def predict(
    data,
    obscode,
    times,
    primary_id_column_name="provID",
    num_workers=-1,
    cache_dir=None,
    args=None,
    configs=None,
):
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
        args=args,
        configs=configs,
    )


def predict_cli(
    cli_args: Namespace,
    input_file: str,
    start_date: float,
    end_date: float,
    timestep_day: float,
    output_file: str,
    cache_dir: Path,
    configs: Namespace,
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

    reader = CSVDataReader(
        input_file,
        primary_id_column_name=cli_args.primary_id_column_name,
        sep="csv",
        required_columns=REQUIRED_INPUT_COLUMN_NAMES,
    )

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
            args=cli_args,
            configs=configs,
        )

        if len(predictions) > 0:
            if cli_args.sexagesimal:
                predictions = _convert_to_sg(predictions)
                write_csv(predictions, output_file, move_columns={"ra_str_hms": 3, "dec_str_dms": 4})
            else:
                write_csv(predictions, output_file)
