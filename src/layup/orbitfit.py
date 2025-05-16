import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pooch
import spiceypy as spice

from numpy.lib import recfunctions as rfn

from layup.routines import (
    FitResult,
    Observation,
    gauss,
    get_ephem,
    run_from_vector_with_initial_guess,
)
from layup.convert import convert

from layup.utilities.astrometric_uncertainty import data_weight_Veres2017
from layup.utilities.data_processing_utilities import (
    LayupObservatory,
    create_chunks,
    get_cov_columns,
    get_format,
    parse_fit_result,
    process_data_by_id,
)
from layup.utilities.datetime_conversions import convert_tdb_date_to_julian_date
from layup.utilities.debiasing import debias, generate_bias_dict
from layup.utilities.file_io import CSVDataReader, HDF5DataReader, Obs80DataReader
from layup.utilities.file_io.file_output import write_csv, write_hdf5

logger = logging.getLogger(__name__)

INPUT_FORMAT_READERS = {
    "MPC80col": (Obs80DataReader, None),
    "ADES_csv": (CSVDataReader, "csv"),
    "ADES_psv": (CSVDataReader, "psv"),
    "ADES_xml": (None, None),
    "ADES_hdf5": (HDF5DataReader, None),
}

GMtotal = 0.0002963092748799319
AU_M = 149597870700
SPEED_OF_LIGHT = 2.99792458e8 * 86400.0 / AU_M


def _get_result_dtypes(primary_id_column_name: str):
    """Helper function to create the result dtype with the correct primary ID column name."""
    # Define a structured dtype to match the OrbfitResult fields
    return np.dtype(
        [
            (primary_id_column_name, "O"),  # Object ID
            ("csq", "f8"),  # Chi-square value
            ("ndof", "i4"),  # Number of degrees of freedom
            ("x", "f8"),  # The first of 6 state vector elements
            ("y", "f8"),
            ("z", "f8"),
            ("xdot", "f8"),
            ("ydot", "f8"),
            ("zdot", "f8"),  # The last of 6 state vector elements
            ("epochMJD_TDB", "f8"),  # Epoch
            ("niter", "i4"),  # Number of iterations
            ("method", "O"),  # Method used for orbit fitting
            ("flag", "i4"),  # Single-character flag indicating success of the fit
            ("FORMAT", "O"),  # Orbit format
        ]
        + [(col_name, "f8") for col_name in get_cov_columns()]  # Flat covariance matrix (36 elements)
    )


def _is_occultation(data):
    """Check if a given data point is an occultation measurement.
    An occultation measurement is indicated by the lack of ra and dec values.

    Parameters
    ----------
    data : numpy structured array
        The object data to check for occultation measurements.

    Returns
    -------
    bool
        True if the data point is an occultation measurement, False otherwise."""
    return data["ra"] is None or data["dec"] is None


def _use_star_astrometry(data):
    """Use occulting star's astrometry to replace the ra and dec values.

    Notes
    -----
    The units are a bit odd here. starra and stardec are in degrees. However, deltara
    and deltadec are in arcseconds. Thus will either convert to degrees or radians
    depending on the context.

    For more details see the ADES description here:
    https://github.com/IAU-ADES/ADES-Master/blob/master/ADES_Description.pdf

    Parameters
    ----------
    data : numpy structured array
        The object data to replace the ra and dec values.

    Returns
    -------
    data : numpy structured array
        The object data with the ra and dec values replaced by the star's astrometry.
    """
    data["ra"] = data["starra"] + (data["deltra"] / 3600) / np.cos(data["stardec"] * np.pi / 180.0)
    data["dec"] = data["stardec"] + (data["deltadec"] / 3600)
    return data


def _split_by_index(input_list, indices):
    """Split an input list into a list of sublists, with break
    points at the provided indices.

    Parameters
    ----------
    input_list : a list of, in this case, integers.
    indices : a list of indices at which to split the input list.

    Returns
    -------
    list of lists."""
    result = []
    sublist = []
    for i, _ in enumerate(input_list):
        if i in indices:
            result.append(sublist)
            sublist = [i]
        else:
            sublist.append(i)
    result.append(sublist)  # Append the last sublist
    return result


def _time_distance(ic0, ic1, jds):
    """Find the separation in time between two set of indices
    (index chunks), given the corresponding set of julian dates.

    Parameters
    ----------
    ic0 : a list of indices (index chunk 0)
    ic1 : a list of indices (index chunk 1)
    jds : a collection of julian dates.

    Returns
    -------
    float : the size of the time gap separating the two chunks.
    """
    if ic0 == ic1:
        return 0.0
    jds0 = jds[ic0]
    jds1 = jds[ic1]
    td0 = np.abs(jds0.min() - jds1.max())
    td1 = np.abs(jds1.min() - jds0.max())
    return min(td0, td1)


def _nearest_chunk(target, index_chunks, jds, self_match=False):
    """Given an index chunk (target) and the set of all the index
    chunks, find the index chunk that is nearest in time, that has
    the smallest separation in time.

    Parameters
    ----------
    target : a list of indices (index chunk)
    index_chunks : a list of lists of indices
    jds : a collection of julian dates.
    self_match : allow the target to match to itself

    Returns
    -------
    list : the nearest index chunk.
    float : the separation in time between the target and the nearest chunk.

    """
    min_dist = np.inf
    nc = None
    for i, ic in enumerate(index_chunks):
        if ic == target and self_match:
            min_dist = 0.0
            nc = ic
        elif ic == target:
            continue
        else:
            td = _time_distance(target, ic, jds)
            if td < min_dist:
                min_dist = td
                nc = ic
    return nc, min_dist


def _next_nearest(chunk_sequence, index_chunks, jds):
    """Given a sequence of index chunks and a collection of other index
    chunks, find the index chunk that is nearest in time to the sequence.
    This will be the next one to include in the fit.

    Parameters
    ----------
    chunk_sequence : a list of lists of indices
    index_chunks : a list of lists of indices
    jds : a collection of julian dates.
    self_match : allow the target to match to itself

    Returns
    -------
    list : the nearest index chunk.
    """
    nc = min([_nearest_chunk(target, index_chunks, jds) for target in chunk_sequence], key=lambda x: x[1])
    return nc[0]


def _iterate_sequence(sequence, other_chunks, jds):
    """Given a sequence of index chunks and a collection of other index
    chunks, iteratively build the sequence.

    Parameters
    ----------
    sequence : a list of lists of indices (the start of the sequence)
    other_chunks : a list of lists of indices
    jds : a collection of julian dates.

    Returns
    -------
    list of lists : the resulting list of lists of indices
    """
    seq = sequence.copy()
    ocs = other_chunks.copy()
    while ocs != []:
        nc = _next_nearest(seq, ocs, jds)
        seq.append(nc)
        ocs.remove(nc)
    return seq


def _build_sequence(jds, sep_dt=90.0):
    """Given a set of julian dates, in the order of the observations,
    split the observations into sets of indices that have gaps of at
    least sep_dt (days) between them.  These are index chunks.  Then
    develop a sequence of the index chunks that will be fit in order.
    The first chunk will have the longest time span.  The next will
    be the closest in time to the first.  The next will be the closest
    in time to the first two, etc.

    Parameters
    ----------
    jds : a collection of julian dates in order of the observations (sorted).
    sep_dt: float the mininum separation between chunks


    Returns
    -------
    list of lists : the resulting sequence of lists of indices that
    will be fit.
    """
    intervals = jds[1:] - jds[:-1]
    intervals = np.insert(intervals, 0, 0.0, axis=0)
    index_chunks = _split_by_index(jds, np.argwhere(intervals > sep_dt))
    start_index = np.argmax([jds[ic].max() - jds[ic].min() for ic in index_chunks])
    start_chunk = index_chunks[start_index]
    remainder = index_chunks.copy()
    remainder.remove(start_chunk)
    seq = _iterate_sequence([start_chunk], remainder, jds)
    return seq


def create_empty_result(id, dtypes):
    """Create an empty return object

    Parameters
    ----------
    id : str
        The id of the object to provide an empty result for
    dtypes : np.array
        The list of datatypes for the structured array.

    Returns
    -------
    np.array
        Empty numpy structured array
    """
    return np.array(
        [
            (
                id,
                np.nan,  # csq
                0,  # ndof
            )
            + (np.nan,) * 6  # Flat state vector
            + (
                np.nan,  # epoch
                0,  # niter
                np.nan,  # method
                -1,  # flag
                "NONE",  # format
            )
            + (np.nan,) * 36  # Flat covariance matrix
        ],
        dtype=dtypes,
    )


def do_fit(observations, seq, cache_dir):
    """Carry out an orbit fit to the observations in a
    series of steps.  A list of lists of observation indices
    specifies the order in which the fit proceeds.

    A Gauss preliminary order is fit for the 0-th segment,
    using the first, middle, and last observations in that
    segement.

    Then an orbit fit is done on the 0-th segment, using the
    initial orbit from Gauss.

    Parameters
    ----------
    observations : list
        A time-ordered list of observations
    seq : list of lists
        A list of lists of observation indices.

    Returns
    -------
    FitResult
        The result of the orbit fit.a
    """
    # Get gauss solution, using the first, middle, and last observation
    # of the primary sequence
    idx0, idx1, idx2 = seq[0][0], seq[0][int(len(seq[0]) / 2)], seq[0][-1]
    logger.debug(f"Sequence indexs passed to gauss: {idx0}, {idx1}, {idx2}")
    solns = gauss(GMtotal, observations[idx0], observations[idx1], observations[idx2], 0.0001, SPEED_OF_LIGHT)

    # If gauss fails, try something else.
    if not solns:
        logger.debug("gauss failed")
        x = FitResult()
        x.flag = 5
        return x

    assist_ephem = get_ephem(cache_dir)

    #! I think this can be a `for/else loop...`
    # Fit primary interval, starting with gauss solution
    x = solns[0]
    obs = [observations[i] for i in seq[0]]
    x = run_from_vector_with_initial_guess(assist_ephem, x, obs)

    if (x.flag != 0) and len(solns) > 1:
        x = solns[1]
        obs = [observations[i] for i in seq[0]]
        x = run_from_vector_with_initial_guess(assist_ephem, x, obs)
    elif (x.flag != 0) and len(solns) > 2:
        x = solns[2]
        obs = [observations[i] for i in seq[0]]
        x = run_from_vector_with_initial_guess(assist_ephem, x, obs)
    if x.flag != 0:
        logger.debug(f"Primary interval failed. Total observations: {len(obs)}")
        x.flag = 3  # caution
        return x

    # Attempt to fit all the data, given the fit of the primary interval
    obs = observations
    x = run_from_vector_with_initial_guess(assist_ephem, x, obs)

    # If that failed, build up the solution slowly
    if x.flag != 0:
        obs = []
        x = solns[0]
        for i, sq in enumerate(seq):
            obs += [observations[i] for i in sq]
            print(i, "of", len(seq), obs[0], sq)
            x = run_from_vector_with_initial_guess(assist_ephem, x, obs)
            print("flag:", x.flag)
            if x.flag != 0:
                x.flag = 4
                break
            logger.debug(f"Result `state`: {x.state}")
            logger.debug(f"Epoch: {x.epoch}, CSQ: {x.csq}, ndof: {x.ndof}, num obs: {len(obs)}")
    else:
        logger.debug(f"Result `state`: {x.state}")
        logger.debug(f"Epoch: {x.epoch}, CSQ: {x.csq}, ndof: {x.ndof}, num obs: {len(obs)}")

    return x


def _orbitfit(
    data,
    cache_dir: str,
    primary_id_column_name: str,
    initial_guess=None,
    bias_dict: dict = None,
    sort_array: bool = True,
    weight_data: bool = False,
):
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
    primary_id_column_name : str
        The name of the primary identifier column for the objects.
    initial_guess : numpy structured array
        Optional guess data to use for the orbit fit. Default is None.
    bias_dict : dict
        A dictionary containing bias corrections for different catalogs.
    sort_array : bool
        Whether to sort the observations by obstime before processing. Default is True.
    weight_data : bool
        Whether to apply data weighting based on the observation code, date, catalog
        and program. Default is False.
    """
    _RESULT_DTYPES = _get_result_dtypes(primary_id_column_name)
    if len(data) == 0:
        return np.array([], dtype=_RESULT_DTYPES)

    if primary_id_column_name not in data.dtype.names:
        raise ValueError(f"Column {primary_id_column_name} not found in requested data to orbit fit.")
    if initial_guess is not None:
        if primary_id_column_name not in initial_guess.dtype.names:
            raise ValueError(f"Column {primary_id_column_name} not found in intial guess data to orbit fit.")
        # Filter the initial guess data to only include the row for this current object.
        initial_guess = initial_guess[
            initial_guess[primary_id_column_name] == data[primary_id_column_name][0]
        ]
        if len(initial_guess) == 0:
            raise ValueError(
                f"Initial guess data does not contain any rows for {primary_id_column_name} = {data[primary_id_column_name][0]}"
            )
        if initial_guess["flag"] != 0:
            logger.debug("Initial guess data is from a failed run. Using default initial guess.")
            initial_guess = None

    if not _is_valid_data(data):  # checks data being supplied to c++ code is valid
        output = create_empty_result(id=data[primary_id_column_name][0], dtypes=_RESULT_DTYPES)
    else:
        # sort the observations by the obstime if specified by the user
        if sort_array:
            data = np.sort(data, order="obstime", kind="mergesort")

        # Check if certain columns are present in the data
        column_names = data.dtype.names
        astcat_column_present = "astcat" in column_names
        program_column_present = "program" in column_names
        position_rates_columns_present = all(col in column_names for col in ["rarate", "decrate"])

        # Accommodate occultation measurements. These measurements are implied when
        # the "ra" and "dec" columns are None. In this case, we will use the "starra"
        # and "stardec" columns.
        for d in data:
            if _is_occultation(d):
                d = _use_star_astrometry(d)

        # bias_dict will be a dictionary when the debias flag is set to True.
        if bias_dict is not None:
            for d in data:
                d["ra"], d["dec"] = debias(
                    ra=d["ra"],
                    dec=d["dec"],
                    epoch_jd_tdb=convert_tdb_date_to_julian_date(d["obstime"], cache_dir),
                    catalog=d["astcat"] if astcat_column_present else None,
                    bias_dict=bias_dict,
                )

        # Convert the astrometry data to a list of Observations
        # Reminder to label the units.  Within an Observation struct,
        # and internal to the C++ code in general, we are using
        # radians.
        observations = []
        for d in data:
            if position_rates_columns_present and (not np.isnan(d["rarate"]) and not np.isnan(d["decrate"])):
                o = Observation.from_streak_with_id(
                    str(d[primary_id_column_name]),
                    d["ra"] * np.pi / 180.0,
                    d["dec"] * np.pi / 180.0,
                    d["rarate"],
                    d["decrate"],
                    convert_tdb_date_to_julian_date(d["obstime"], cache_dir),  # Convert obstime to JD TDB
                    [d["x"], d["y"], d["z"]],  # Barycentric position
                    [d["vx"], d["vy"], d["vz"]],  # Barycentric velocity
                )
            else:
                o = Observation.from_astrometry_with_id(
                    str(d[primary_id_column_name]),
                    d["ra"] * np.pi / 180.0,
                    d["dec"] * np.pi / 180.0,
                    convert_tdb_date_to_julian_date(d["obstime"], cache_dir),  # Convert obstime to JD TDB
                    [d["x"], d["y"], d["z"]],  # Barycentric position
                    [d["vx"], d["vy"], d["vz"]],  # Barycentric velocity
                )

            if weight_data:
                data_weight = data_weight_Veres2017(
                    obsCode=d["stn"],
                    jd_tdb=convert_tdb_date_to_julian_date(d["obstime"], cache_dir),
                    catalog=d["astcat"] if astcat_column_present else None,
                    program=d["program"] if program_column_present else None,
                )

                o.ra_unc = data_weight
                o.dec_unc = data_weight

            observations.append(o)

        # if cache_dir is not provided, use the default os_cache
        if cache_dir is None:
            kernels_loc = str(pooch.os_cache("layup"))
        else:
            kernels_loc = str(cache_dir)

        jds = convert_tdb_date_to_julian_date(data["obstime"])
        sequence = _build_sequence(jds, sep_dt=90.0)

        # Perform the orbit fitting
        if initial_guess is None or initial_guess["flag"] != 0:
            # res = run_from_vector(get_ephem(kernels_loc), observations)
            res = do_fit(observations=observations, seq=sequence, cache_dir=kernels_loc)
        else:
            guess_to_use = parse_fit_result(initial_guess)
            res = run_from_vector_with_initial_guess(get_ephem(kernels_loc), guess_to_use, observations)
        # Populate our output structured array with the orbit fit results
        success = res.flag == 0
        cov_matrix = tuple(res.cov[i] for i in range(36)) if success else (np.nan,) * 36
        output = np.array(
            [
                (
                    data[primary_id_column_name][0],
                    (res.csq if success else np.nan),
                    res.ndof,
                )
                + (tuple(res.state[i] for i in range(6)) if success else (np.nan,) * 6)  # Flat state vector
                + (
                    ((res.epoch - 2400000.5) if success else np.nan),
                    res.niter,
                    res.method,
                    res.flag,
                    ("BCART_EQ" if success else "NONE"),  # The base format returned by the C++ code
                )
                + cov_matrix  # Flat covariance matrix
            ],
            dtype=_RESULT_DTYPES,
        )

    return output


def orbitfit(
    data,
    cache_dir: str,
    initial_guess=None,
    num_workers=1,
    primary_id_column_name="provID",
    debias=False,
    weight_data=False,
):
    """This is the function that you would call interactively. i.e. from a notebook

    Parameters
    ----------
    data : numpy structured array
        The object data to derive an orbit for
    cache_dir : str
        The directory where the required orbital files are stored
    initial_guess : numpy structured array
        Optional initial guess data to use for the orbit fit. Default is None.
    num_workers : int
        The number of workers to use for parallel processing. Default is 1
    primary_id_column_name : str
        The name of the primary identifier column for the objects. Default is "provID".
    debias : bool
        Whether to apply debiasing corrections to the observations. Default is False.
    weight_data : bool
        Whether to apply data weighting based on the observation code, date, catalog
        and program. Default is False.
    """

    layup_observatory = LayupObservatory(cache_dir=cache_dir)

    # The units of et are seconds (from J2000). This new column is used by
    # data_processing_utilities.obscodes_to_barycentric.
    et_col = np.array([spice.str2et(row["obstime"]) for row in data], dtype="<f8")
    data = rfn.append_fields(data, "et", et_col, usemask=False, asrecarray=True)

    pos_vel = layup_observatory.obscodes_to_barycentric(data)
    data = rfn.merge_arrays([data, pos_vel], flatten=True, asrecarray=True, usemask=False)

    bias_dict = None
    if debias:
        bias_dict = generate_bias_dict(cache_dir)

    return process_data_by_id(
        data,
        num_workers,
        _orbitfit,
        primary_id_column_name=primary_id_column_name,
        cache_dir=cache_dir,
        initial_guess=initial_guess,
        bias_dict=bias_dict,
        weight_data=weight_data,
    )


def orbitfit_cli(
    input: str,
    input_file_format: Literal["MPC80col", "ADES_csv", "ADES_psv", "ADES_xml", "ADES_hdf5"],
    output_file_stem: str,
    output_file_format: Literal["csv", "hdf5"] = "csv",
    chunk_size: int = 10_000,
    num_workers: int = -1,
    cli_args: Optional[Namespace] = None,
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
    cli_args : argparse.Namespace, optional (default=None)
        The argparse object that was created when running from the CLI.
    """

    if cli_args is not None:
        cache_dir = cli_args.ar_data_file_path
        debias = cli_args.debias
        guess_file = Path(cli_args.g) if cli_args.g is not None else None
        weight_data = cli_args.weight_data
        output_orbit_format = cli_args.output_orbit_format
    else:
        cache_dir = None
        debias = False
        guess_file = None
        weight_data = False
        output_orbit_format = "COM"  # Default output orbit format.

    _primary_id_column_name = cli_args.primary_id_column_name

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

    # If splitting the output has been requested, then we'll create a second output
    # file with "_flagged" appended to the stem. i.e. if the user provided "output.h5
    # then the flagged output will be "output_flagged.h5".
    if cli_args.separate_flagged:
        output_file_stem_flagged = output_file_stem
        if output_file_format == "csv":
            output_file_flagged = Path(f"{output_file_stem_flagged}_flagged.{output_file_format.lower()}")
        else:
            output_file_stem_parts = output_file_stem_flagged.split(".")
            if len(output_file_stem_parts) > 1:
                output_file_stem_flagged = output_file_stem_parts[0] + "_flagged." + output_file_stem_parts[1]
            else:
                output_file_stem_flagged = output_file_stem_flagged + "_flagged"

            output_file_flagged = (
                Path(f"{output_file_stem_flagged}")
                if output_file_stem_flagged.endswith(".h5")
                else Path(f"{output_file_stem_flagged}.h5")
            )

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

    reader_class, separator = INPUT_FORMAT_READERS[input_file_format]
    if reader_class is None:
        logger.error(f"File format {input_file_format} is not supported")

    reader = reader_class(input_file, primary_id_column_name=_primary_id_column_name, sep=separator)
    if guess_file is not None:
        # Check that the guess file exists
        if not guess_file.exists():
            logger.error(f"Guess file {guess_file} does not exist")
            raise FileNotFoundError(f"Guess file {guess_file} does not exist")
        # Check that the guess file is not the same path as the input file
        if os.path.abspath(guess_file) == os.path.abspath(input_file):
            logger.error("Guess file cannot be the same as the input file")
            raise ValueError("Guess file cannot be the same as the input file")
        # Set up our initial guess file reader. Assumes a matching file format and primary id column name
        # as the input file.
        guess_reader = reader_class(guess_file, primary_id_column_name=_primary_id_column_name, sep=separator)

    chunks = create_chunks(reader, chunk_size)

    for chunk in chunks:
        data = reader.read_objects(chunk)
        initial_guess = None
        if guess_file is not None:
            # Get the guesses for all the objects in the current chunk.
            initial_guess = guess_reader.read_objects(chunk)
            if len(initial_guess) != 0 and get_format(initial_guess) != "BCART_EQ":
                # If the initial guess is not in the BCART_EQ format, convert it to BCART_EQ
                initial_guess = convert(
                    initial_guess,
                    convert_to="BCART_EQ",
                    num_workers=num_workers,
                    cache_dir=cache_dir,
                    primary_id_column_name=_primary_id_column_name,
                )

        logger.info(f"Processing {len(data)} rows for {chunk}")

        fit_orbits = orbitfit(
            data,
            cache_dir=cache_dir,
            initial_guess=initial_guess,
            num_workers=num_workers,
            primary_id_column_name=_primary_id_column_name,
            debias=debias,
            weight_data=weight_data,
        )

        # Convert the fit_orbits to the preferred output format
        if output_orbit_format != "BCART_EQ":
            fit_orbits = convert(
                fit_orbits,
                convert_to=output_orbit_format,
                num_workers=num_workers,
                cache_dir=cache_dir,
                primary_id_column_name=_primary_id_column_name,
            )

        if cli_args.separate_flagged:
            # Split the results into two files: one for successful fits and one for failed fits
            success_mask = fit_orbits["flag"] == 0
            fit_orbits_success = fit_orbits[success_mask]
            fit_orbits_failed = fit_orbits[~success_mask]

            if output_file_format == "hdf5":
                if len(fit_orbits_success) > 0:
                    write_hdf5(fit_orbits_success, output_file, key="data")

                if len(fit_orbits_failed) > 0:
                    write_hdf5(
                        fit_orbits_failed[[_primary_id_column_name, "method", "flag"]],
                        output_file_flagged,
                        key="data",
                    )
            else:  # csv output format
                if len(fit_orbits_success) > 0:
                    write_csv(fit_orbits_success, output_file)

                if len(fit_orbits_failed) > 0:
                    write_csv(
                        fit_orbits_failed[[_primary_id_column_name, "method", "flag"]], output_file_flagged
                    )

        else:  # All results go to a single output file
            if output_file_format == "hdf5":
                write_hdf5(fit_orbits, output_file, key="data")
            else:
                write_csv(fit_orbits, output_file)

    print(f"Data has been written to {output_file}")


def _is_valid_data(data):
    """
    Check if the input data contains all valid values.

    Parameters
    ----------
    data : numpy structured array
        The object data to validate.

    Returns
    -------
    bool
        True if the data is valid, False otherwise.
    """
    valid_conditions = [
        len(data) >= 3,
        np.all(
            data["et"] >= -6279962400.00
        ),  # excludes all datasets before 1801, data["et"] = 0 is j2000, 6279962400.00 is seconds between 1801 and j2000
        np.all(is_numeric(data["ra"])),
        np.all(is_numeric(data["dec"])),
        np.all(is_numeric(data["x"])),
        np.all(is_numeric(data["y"])),
        np.all(is_numeric(data["z"])),
        np.all(is_numeric(data["vx"])),
        np.all(is_numeric(data["vy"])),
        np.all(is_numeric(data["vz"])),
    ]
    return all(valid_conditions)


def is_numeric(obj):  # checks object is numeric by checking object has all required attributes
    attrs = ["__add__", "__sub__", "__mul__", "__truediv__", "__pow__"]
    return all(hasattr(obj, attr) for attr in attrs)
