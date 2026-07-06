import logging
import os
import re
from argparse import Namespace
from dataclasses import dataclass
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
    run_bk_iod,
    run_bk_native_fit,
    run_from_vector_with_initial_guess,
    run_sequential_update,
)

try:
    from layup.routines import (
        get_ias15_adaptive_mode,
        set_ias15_adaptive_mode,
    )
except ImportError:  # extension not rebuilt yet
    get_ias15_adaptive_mode = lambda: -1
    set_ias15_adaptive_mode = lambda m: None
# _MU_SUN (= heliocentric GM = k^2) is used by the BK-native fit for the
# bound-orbit energy prior on gdot; SPEED_OF_LIGHT (au/day) by the radar ingest.
from layup.constants import MU_SUN as _MU_SUN, SPEED_OF_LIGHT
from layup.convert import convert
from layup.iod import filter_candidates_by_residual, get_iod, iod_methods

from layup.utilities.astrometric_uncertainty import astrometric_uncertainty_Veres2017
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
from layup.utilities.file_io import (
    ADESXMLDataReader,
    CSVDataReader,
    HDF5DataReader,
    Obs80DataReader,
)
from layup.utilities.file_io.file_output import append_hdf5, write_csv, write_hdf5

logger = logging.getLogger(__name__)

# Observed sky-motion rates (ADES `raRate`/`decRate`) arrive in arcsec/hour and
# follow the great-circle convention: `raRate` is cos(Dec)*dRA/dt, NOT the bare
# coordinate rate dRA/dt. This matches Sorcha's `RARateCosDec` output (which
# projects drho_hat/dt onto A = (-sinRA, cosRA, 0)) and layup's own residual,
# which projects onto the same tangent vector `a_vec` -- so an observed rate is
# compared directly to omega.a_vec with no extra cos(Dec) factor. Internally the
# fitter works in radians and AU/day, so omega = d(rho_hat)/dt is in rad/day;
# convert the observed rates from arcsec/hour to rad/day at ingest.
ARCSEC_PER_HOUR_TO_RAD_PER_DAY = (np.pi / 180.0 / 3600.0) * 24.0

# Radar (delay/Doppler) observables arrive in JPL units: round-trip delay in
# microseconds and Doppler shift in Hz at a per-observation transmit frequency
# `freqTx` (Hz). The fitter models round-trip delay in days and round-trip
# range-rate in au/day (see RadarObservation in detection.cpp), so convert at
# ingest, mirroring the streak rate convention above:
#   delay[days]     = delay[us] * 1e-6 / 86400
#   doppler[au/day] = -c * doppler[Hz] / freqTx[Hz]
# The Doppler sign follows F = -(f_tx/c) * d(round-trip range)/dt, so a positive
# (receding) range-rate produces a negative frequency shift.
US_TO_DAYS = 1.0e-6 / 86400.0
# Fallback 1-sigma weights when the JPL uncertainty columns are absent: ~1 us of
# round-trip delay and ~1 Hz of Doppler (converted per observation via freqTx).
_DEFAULT_DELAY_UNC_DAYS = US_TO_DAYS
_DEFAULT_DOPPLER_UNC_HZ = 1.0

# The list of required input column names for the provided observations to be fit.
# Note: This should not include the primary id column name.
REQUIRED_INPUT_OBSERVATIONS_COLUMN_NAMES = [
    (
        set(["ra", "dec"]),  # Either `ra` and `dec` must be in the file
        set(["raRate", "decRate"]),  # Or `raRate` and `decRate` must be in the file
        set(["delay"]),  # Or a radar round-trip delay (us)
        set(["doppler"]),  # Or a radar Doppler shift (Hz; needs `freqTx`)
    ),
    "obsTime",
    "stn",
]

# The list of required column names for the guessed orbits (if provided).
# Note: This should now include the primary id column name.
REQUIRED_INPUT_GUESS_COLUMN_NAMES = [
    "epochMJD_TDB",
    "FORMAT",
]

INPUT_FORMAT_READERS = {
    "MPC80col": (Obs80DataReader, None),
    "ADES_csv": (CSVDataReader, "csv"),
    "ADES_psv": (CSVDataReader, "psv"),
    "ADES_xml": (ADESXMLDataReader, None),
    "ADES_hdf5": (HDF5DataReader, None),
}


def _run_fit(assist_ephem, initial_guess, observations, engine, iter_max=100):
    """Dispatch a single LM fit step to the configured engine.

    Centralizing the dispatch here keeps do_fit's IOD-then-fit pipeline
    parameterization-agnostic and lets us add new engines (e.g., a
    future distance-dispatched 'auto') with a single edit instead of
    threading the choice through every call site.

    `iter_max` is the LM iteration budget used by the multi-root picker's
    two-tier (cheap-screen then full) passes. The Cartesian engine honors
    it; the BK-native engine uses its own internal cap (it takes `mu` for
    the bound-orbit energy prior rather than an iteration budget), so
    `iter_max` is ignored on that path.
    """
    if engine == "cartesian":
        return run_from_vector_with_initial_guess(assist_ephem, initial_guess, observations, iter_max)
    if engine == "bk_native":
        return run_bk_native_fit(assist_ephem, initial_guess, observations, _MU_SUN)
    raise ValueError(f"Unknown engine {engine!r}; expected one of 'cartesian', 'bk_native'.")


# Non-gravitational Marsden parameters and their FitResult/C++ bitmask bits.
_NONGRAV_BITS = {"A1": 1, "A2": 2, "A3": 4}


def _parse_nongrav(fit_nongrav):
    """Normalize the ``fit_nongrav`` argument to ``(mask, names)``.

    Accepts ``False``/``None`` (no non-grav fit), ``True`` (== ``["A2"]``, the
    common asteroid Yarkovsky case), a string naming params (e.g. ``"A2"``,
    ``"A1A2A3"``, ``"A1,A3"``), or an iterable of names. Returns the C++ bitmask
    (bits 1/2/4 for A1/A2/A3) and the selected names ordered A1, A2, A3.
    """
    if not fit_nongrav:
        return 0, []
    if fit_nongrav is True:
        sel = {"A2"}
    elif isinstance(fit_nongrav, str):
        sel = set(re.findall(r"A[123]", fit_nongrav.upper()))
    else:
        sel = {str(s).upper() for s in fit_nongrav}
    names = [n for n in ("A1", "A2", "A3") if n in sel]
    if not names:
        raise ValueError(f"fit_nongrav={fit_nongrav!r}: expected some of 'A1', 'A2', 'A3'.")
    return sum(_NONGRAV_BITS[n] for n in names), names


# fit_nongrav="auto" model ladder (issue #357): non-grav models are tried in
# increasing complexity, and the first that converges, is well-conditioned
# (flag 0), and is statistically warranted (see NongravAutoThresholds) is adopted;
# otherwise the gravity-only fit is kept.
_AUTO_NONGRAV_LADDER = (("A2",), ("A1", "A2"), ("A1", "A2", "A3"))


@dataclass(frozen=True)
class NongravAutoThresholds:
    """Decision thresholds for adaptive non-grav selection (``fit_nongrav="auto"``,
    issue #357). Pass a customized instance to ``orbitfit`` to tune how readily a
    non-gravitational model is adopted; the defaults reproduce the standard
    "introduce a non-grav only when gravity is unacceptable and the parameter is
    statistically warranted" behavior.

    Parameters
    ----------
    accept_reduced_chi2 : float
        A gravity-only fit whose reduced chi-square is at or below this is kept
        as-is; non-grav models are tried only above it. Default 1.5.
    delta_chi2_per_param : float
        Minimum chi-square drop required per added non-grav parameter to adopt a
        model (9.0 ~ 3-sigma). Default 9.0.
    nsigma : float
        Each added non-grav parameter must exceed this many times its 1-sigma
        uncertainty to be adopted. Default 3.0.
    """

    accept_reduced_chi2: float = 1.5
    delta_chi2_per_param: float = 9.0
    nsigma: float = 3.0


_AUTO_DEFAULT_THRESHOLDS = NongravAutoThresholds()


def _gravity_fit_acceptable(csq, ndof, thresholds=_AUTO_DEFAULT_THRESHOLDS):
    """Whether a gravity-only fit is good enough that no non-grav is warranted.

    True when the reduced chi-square is at or below the acceptance threshold (or
    there are no degrees of freedom to judge it by).
    """
    return ndof <= 0 or csq / ndof <= thresholds.accept_reduced_chi2


def _nongrav_warranted(csq_gravity, res_ng, names, thresholds=_AUTO_DEFAULT_THRESHOLDS):
    """Whether adopting the (converged, well-conditioned) non-grav fit ``res_ng``
    for parameters ``names`` over the gravity-only fit is statistically warranted:
    a significant chi-square drop AND every added parameter individually significant.
    """
    if (csq_gravity - res_ng.csq) <= thresholds.delta_chi2_per_param * len(names):
        return False  # chi-square improvement not significant for the added parameter(s)
    return all(
        abs(getattr(res_ng, n.lower())) > thresholds.nsigma * getattr(res_ng, n.lower() + "_unc")
        for n in names
    )


def _select_nongrav_auto(assist_ephem, res_grav, observations, thresholds=_AUTO_DEFAULT_THRESHOLDS):
    """Adaptive non-grav selection for ``fit_nongrav="auto"`` (issue #357).

    ``res_grav`` is the converged gravity-only fit. Returns the most parsimonious
    acceptable model: the gravity-only result unless a non-grav model is both
    well-conditioned (flag 0) and statistically warranted per ``thresholds``.
    """
    if _gravity_fit_acceptable(res_grav.csq, res_grav.ndof, thresholds):
        return res_grav  # gravity-only fit is acceptable; no non-gravs needed
    for names in _AUTO_NONGRAV_LADDER:
        mask = sum(_NONGRAV_BITS[n] for n in names)
        res_ng = run_from_vector_with_initial_guess(assist_ephem, res_grav, observations, nongrav_mask=mask)
        if res_ng.flag == 0 and _nongrav_warranted(res_grav.csq, res_ng, names, thresholds):
            return res_ng  # parsimonious, well-determined non-grav model
    return res_grav  # no non-grav model is warranted


def _get_result_dtypes(primary_id_column_name: str, nongrav_names=()):
    """Helper function to create the result dtype with the correct primary ID column name.

    For each fitted non-gravitational parameter in ``nongrav_names`` (a subset of
    ``A1``/``A2``/``A3``), two columns are appended -- e.g. ``a2`` and ``a2_unc``
    (the value and its 1-sigma uncertainty, au/day^2). With no non-grav params the
    default 6-parameter output schema is unchanged.
    """
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
        + [  # non-grav params (issue #351), value + 1-sigma per fitted param
            (col, "f8") for n in nongrav_names for col in (n.lower(), n.lower() + "_unc")
        ]
    )


def _is_radar(d, column_names):
    """True if a row carries a populated radar observable (delay or Doppler).

    Radar rows have no ra/dec; they are dispatched to ``Observation.from_radar``
    rather than the astrometry/streak factories.
    """
    has_delay = "delay" in column_names and not np.isnan(d["delay"])
    has_doppler = "doppler" in column_names and not np.isnan(d["doppler"])
    return has_delay or has_doppler


def _radar_observation(objID, d, epoch_jd, column_names):
    """Build a radar ``Observation`` from a row, converting JPL units to the
    fitter's internal units.

    delay (us, round-trip) -> days; Doppler (Hz) -> round-trip range-rate
    (au/day) via the per-observation transmit frequency ``freqTx``. The
    barycentric observer position/velocity columns (x,y,z,vx,vy,vz) must already
    be present (added by ``orbitfit``); the observer acceleration columns
    (ax,ay,az) drive the two-leg light-time model and default to zero when
    absent. 1-sigma uncertainties come from ``rmsDelay``/``rmsDoppler`` when
    present, else the module defaults.
    """
    has_delay = "delay" in column_names and not np.isnan(d["delay"])
    has_doppler = "doppler" in column_names and not np.isnan(d["doppler"])

    f_tx = d["freqTx"] if "freqTx" in column_names else np.nan
    if has_doppler and (np.isnan(f_tx) or f_tx == 0.0):
        raise ValueError(f"Radar Doppler observation for {objID} requires a nonzero 'freqTx' (Hz).")

    delay_days = (d["delay"] * US_TO_DAYS) if has_delay else 0.0
    doppler_audy = (-SPEED_OF_LIGHT * d["doppler"] / f_tx) if has_doppler else 0.0

    if "rmsDelay" in column_names and not np.isnan(d["rmsDelay"]):
        delay_unc = abs(d["rmsDelay"]) * US_TO_DAYS
    else:
        delay_unc = _DEFAULT_DELAY_UNC_DAYS

    if has_doppler:
        rms_hz = (
            d["rmsDoppler"]
            if ("rmsDoppler" in column_names and not np.isnan(d["rmsDoppler"]))
            else _DEFAULT_DOPPLER_UNC_HZ
        )
        doppler_unc = abs(SPEED_OF_LIGHT * rms_hz / f_tx)
    else:
        doppler_unc = abs(SPEED_OF_LIGHT * _DEFAULT_DOPPLER_UNC_HZ / f_tx) if not np.isnan(f_tx) else 1.0

    if all(c in column_names for c in ("ax", "ay", "az")):
        observer_acc = [float(d["ax"]), float(d["ay"]), float(d["az"])]
    else:
        observer_acc = [0.0, 0.0, 0.0]

    return Observation.from_radar_with_id(
        str(objID),
        delay_days,
        doppler_audy,
        has_delay,
        has_doppler,
        epoch_jd,
        [d["x"], d["y"], d["z"]],  # Barycentric observer position
        [d["vx"], d["vy"], d["vz"]],  # Barycentric observer velocity
        delay_unc,
        doppler_unc,
        observer_acc,  # Barycentric observer acceleration (au/day^2)
    )


def _append_observer_acceleration(data, observatory, dt_sec=2.0):
    """Append barycentric observer acceleration columns (ax, ay, az; au/day^2).

    Finite-differences the barycentric station velocity from
    ``obscodes_to_barycentric`` at the observation epoch +/- ``dt_sec``. Used only
    by radar observations, whose two-leg light-time model extrapolates the station
    state back to the signal transmit time. Requires the ``et`` column (seconds
    past J2000, TDB) that ``orbitfit`` adds before computing the observer states.
    """

    def _vel(et_offset_sec):
        shifted = data.copy()
        shifted["et"] = data["et"] + et_offset_sec
        pv = np.atleast_1d(observatory.obscodes_to_barycentric(shifted))
        return np.stack([pv["vx"], pv["vy"], pv["vz"]], axis=-1)

    # d(velocity[au/day]) / d(time[day]); 2*dt_sec seconds = 2*dt_sec/86400 days.
    scale = 86400.0 / (2.0 * dt_sec)
    acc = (_vel(dt_sec) - _vel(-dt_sec)) * scale
    acc = np.atleast_2d(acc)
    return rfn.append_fields(
        data, ["ax", "ay", "az"], [acc[:, 0], acc[:, 1], acc[:, 2]], usemask=False, asrecarray=True
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
    The units are a bit odd here. raStar and decStar are in degrees. However, deltaRA
    and deltaDec are in arcseconds. Thus will either convert to degrees or radians
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
    data["ra"] = data["raStar"] + (data["deltRA"] / 3600) / np.cos(data["decStar"] * np.pi / 180.0)
    data["dec"] = data["decStar"] + (data["deltaDec"] / 3600)
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
            # non-grav columns (issue #351): NaN per a1/a2/a3 (+ _unc) that is present
            + tuple(np.nan for n in ("a1", "a2", "a3") if n in dtypes.names for _ in (0, 1))
        ],
        dtype=dtypes,
    )


def do_gauss_iod(observations, seq):
    """Backward-compat wrapper for the Gauss IOD.

    Prefer ``layup.iod.get_iod("gauss")`` for new code; this shim
    exists so callers that imported ``do_gauss_iod`` directly continue
    to work.
    """
    return get_iod("gauss")(observations, seq)


# Multi-root picker tuning. Both knobs are exposed to do_fit() callers
# in case downstream code wants to override them, but the defaults are
# what worked best on the diagnostic/scan and neo_scan datasets.
_PICKER_MIN_R_HELIO_AU = 0.3  # reject roots with r < this as unphysical
_PICKER_SCREEN_ITER_MAX = 80  # cheap LM budget for the first pass
_PICKER_FULL_ITER_MAX = 100  # full LM budget for the fallback pass

_PREFILTER_THRESHOLD_SIGMA = 1000.0  # held-out residual filter cutoff

# IAS15 adaptive-step controller used during the multi-root picker.
# With the legacy controller (mode 1), LM grinds for minutes on phantom
# Gauss roots whose trajectories pass close to Earth (the integrator
# chases ever-smaller steps to resolve the close encounter — 100-1000×
# wallclock blowup observed on diagnostic/scan). The newer (Pham, Rein
# & Spiegel 2024) controller, mode 2, steps through those encounters
# efficiently: it brings the same pathological cases from >120 s to
# sub-second with the identical recovered orbit, and unlike a step-size
# floor it is a better controller rather than a truncation, so it costs
# no accuracy on genuine close-Earth encounters. Set to -1 to leave
# ASSIST's default (legacy mode 1).
_PICKER_IAS15_ADAPTIVE_MODE = 2


# Cache of the Python-side assist.Ephem handle. The C-side get_ephem()
# from layup.routines returns the C struct; the Python residual filter
# needs the rebound/assist Python wrapper instead, so we cache one per
# cache_dir.
_assist_python_ephem_cache: dict = {}


def _get_python_ephem(cache_dir):
    """Lazy-load and cache the Python-side assist.Ephem for the filter."""
    key = str(cache_dir)
    if key in _assist_python_ephem_cache:
        return _assist_python_ephem_cache[key]
    try:
        import assist
    except ImportError:
        return None
    try:
        eph = assist.Ephem(os.path.join(key, "linux_p1550p2650.440"), os.path.join(key, "sb441-n16.bsp"))
    except Exception as e:
        logger.warning(f"assist.Ephem load failed for {cache_dir}: {e}")
        return None
    _assist_python_ephem_cache[key] = eph
    return eph


def _pick_best_root(candidates, min_r_au):
    """Pick the best converged candidate from a list of LM results.

    "Best" means smallest χ² among candidates that
      (1) report flag == 0 (LM converged), and
      (2) have heliocentric distance > min_r_au (physical orbit).
    Returns None if no candidate satisfies (1); in that case the caller
    typically retries at a larger LM budget. If (1) is met but (2)
    isn't, the smallest-χ² convergent root is still returned (better
    than nothing).
    """
    converged = [c for c in candidates if c.flag == 0]
    if not converged:
        return None
    sane = [c for c in converged if (c.state[0] ** 2 + c.state[1] ** 2 + c.state[2] ** 2) > min_r_au**2]
    pool = sane if sane else converged
    return min(pool, key=lambda c: c.csq)


def do_fit(
    observations,
    seq,
    cache_dir,
    iod="gauss",
    engine="cartesian",
    screen_iter_max: int = _PICKER_SCREEN_ITER_MAX,
    full_iter_max: int = _PICKER_FULL_ITER_MAX,
    min_r_helio_AU: float = _PICKER_MIN_R_HELIO_AU,
    prefilter_threshold_sigma: float = _PREFILTER_THRESHOLD_SIGMA,
    picker_ias15_adaptive_mode: int = _PICKER_IAS15_ADAPTIVE_MODE,
):
    """Carry out an orbit fit to a list of observations.

    Pipeline:
      1. IOD: produce one or more candidate seed orbits via the
         registered method named by `iod` (default: "gauss"). The
         registry lives in `layup.iod`; register new methods with
         `iod.register_iod(name, callable)`.
      2. Multi-root picker: run LM from every IOD candidate on the
         primary segment (`seq[0]`) at a cheap `screen_iter_max`
         budget. Pick the smallest-χ² converged candidate with
         heliocentric distance above `min_r_helio_AU`. If nothing
         converges at the cheap budget, retry at `full_iter_max`.
         Then refit on the full observation set.

    Parameters
    ----------
    observations : list
        Time-ordered list of layup Observations.
    seq : list of lists
        Per-segment index lists; seq[0] is the primary segment.
    cache_dir : str
        Directory holding the ASSIST kernels.
    iod : str
        Name of the registered IOD method (default "gauss"), or "auto" to run
        Gauss and fall back to the BK 5-parameter linear IOD (run_bk_iod) on the
        primary segment when every Gauss root fails to seed a converged fit.
    engine : str
        Which LM fitter to dispatch to.  Supported:
          - 'cartesian' (default): the existing 6D Cartesian-state fit.
          - 'bk_native': the universal Bernstein-Khushalani fit
            (run_bk_native_fit), with a fixed bound-orbit energy prior
            on gdot.  Recovers the Cartesian state at the same epoch.
    screen_iter_max, full_iter_max : int
        Two-tier LM iteration caps for the multi-root picker.
    min_r_helio_AU : float
        Lower bound on heliocentric distance for accepted IOD roots.

    Returns
    -------
    FitResult
        Best converged fit (flag == 0) when one exists, else a
        best-effort or sentinel FitResult with a non-zero flag.
    """

    # 'auto' is a strategy, not a registered IOD: seed candidates with Gauss,
    # then (after the picker, below) fall back to the BK 5-parameter linear IOD
    # if every Gauss root fails to converge. Any other value is a registry name.
    is_auto = isinstance(iod, str) and iod.lower() == "auto"
    try:
        iod_func = get_iod("gauss") if is_auto else (get_iod(iod) if isinstance(iod, str) else iod)
    except ValueError as e:
        raise ValueError(f"{e} Use iod.register_iod to add a new method.")
    # Normalize to a list: an IOD may legitimately return None (e.g. Gauss
    # finding no real roots), and the prefilter/picker below index and len() it.
    solns = list(iod_func(observations, seq) or [])

    # If the iod produced no candidates, surface a sentinel -- unless we're in
    # 'auto' mode, where the BK-IOD fallback below still has a shot.
    if not solns and not is_auto:
        logger.debug(f"IOD {iod!r} returned no candidates")
        x = FitResult()
        x.flag = 5
        return x

    # Pre-filter the IOD candidates by predicted-vs-observed residual
    # on every observation. The right Gauss root predicts the full
    # observation set within a few σ; phantom roots typically miss by
    # 10⁵+ σ. Throwing those out before any LM iteration runs cuts
    # the picker loop down to 1-2 LM fits per case in the common
    # case (vs up to 8 brute-force LMs). Loose threshold (default
    # 1000σ) so the right root is never rejected.
    py_ephem = _get_python_ephem(cache_dir)
    if py_ephem is not None and len(solns) > 1:
        before = len(solns)
        solns = filter_candidates_by_residual(
            solns, observations, py_ephem, threshold_sigma=prefilter_threshold_sigma
        )
        if len(solns) < before:
            logger.debug(
                f"IOD pre-filter: kept {len(solns)}/{before} " f"candidates at {prefilter_threshold_sigma}σ"
            )

    assist_ephem = get_ephem(cache_dir)

    # Multi-root picker. Fit every IOD candidate on the primary segment
    # at the cheap screening budget, pick the best converged root, and
    # only fall back to the full LM budget if nothing converged at the
    # cheap tier. Gauss's polynomial gives up to 8 real roots; historic
    # do_fit committed to solns[0] (largest r), which is often a
    # phantom outer-SS solution for NEO-like targets.
    #
    # During this loop we select IAS15 adaptive_mode=2 so phantom roots
    # whose trajectories pass close to Earth can't tie up the
    # integrator for minutes (100-1000× wallclock blowup observed on
    # diagnostic/scan with the legacy controller). The newer controller
    # steps through close encounters efficiently with no accuracy cost.
    # The setting is restored on every exit path.
    #
    # Each LM call dispatches through _run_fit so the picker honors the
    # selected engine. The screen/full iteration budgets apply to the
    # Cartesian engine; the BK-native engine uses its own internal cap.
    saved_mode = get_ias15_adaptive_mode()
    if picker_ias15_adaptive_mode >= 0:
        set_ias15_adaptive_mode(picker_ias15_adaptive_mode)

    obs = [observations[i] for i in seq[0]]
    try:
        candidates = [_run_fit(assist_ephem, soln, obs, engine, screen_iter_max) for soln in solns]
        x = _pick_best_root(candidates, min_r_helio_AU)
        if x is None:
            candidates = [_run_fit(assist_ephem, soln, obs, engine, full_iter_max) for soln in solns]
            x = _pick_best_root(candidates, min_r_helio_AU)
    finally:
        set_ias15_adaptive_mode(saved_mode)

    if x is None and is_auto and len(obs) >= 3:
        # Every Gauss root (if any) failed to seed a converged LM. Fall back to
        # the BK 5-parameter linear IOD on the primary segment. BK-IOD shines on
        # distant short arcs -- exactly where Gauss's three-point geometry is
        # ill-conditioned (see bk_iod.cpp's regime-of-validity note); on the
        # diagnostic scan Gauss+BK covers ~90% of cases vs ~84% for Gauss alone.
        # Epoch convention matches do_gauss_iod's middle observation.
        logger.debug(f"All {len(solns)} Gauss roots failed; trying BK-IOD fallback")
        bk_seed = run_bk_iod(obs, float(obs[len(obs) // 2].epoch), _MU_SUN)
        if bk_seed.flag == 0:
            cand = _run_fit(assist_ephem, bk_seed, obs, engine, full_iter_max)
            candidates.append(cand)
            if cand.flag == 0:
                x = cand

    if x is None:
        # Still no convergence — surface the least-bad attempt so the caller has
        # *something* to inspect, with a flag they can detect.
        if not candidates:
            # 'auto' with zero Gauss roots and no usable BK seed: nothing to
            # surface, so return an explicit no-solution sentinel.
            x = FitResult()
            x.flag = 5
            return x
        x = min(candidates, key=lambda c: c.csq)
        logger.debug(
            f"Primary interval: no root converged " f"(best csq={x.csq:.3g}, n_roots={len(candidates)})"
        )
        x.flag = 3
        return x

    # Attempt to fit all the data, given the fit of the primary interval
    primary_x = x
    obs = observations
    x = _run_fit(assist_ephem, x, obs, engine)

    # If that failed, build up the solution slowly
    if x.flag != 0:
        obs = []
        # Restart from the first IOD seed, or the converged primary fit when
        # there were no IOD candidates (the iod='auto' BK-IOD fallback path).
        x = solns[0] if solns else primary_x
        for i, sq in enumerate(seq):
            obs += [observations[i] for i in sq]
            logger.debug(f"Incremental fit segment {i} of {len(seq)} " f"(n_obs={len(obs)})")
            x = _run_fit(assist_ephem, x, obs, engine)
            if x.flag != 0:
                x.flag = 4
                break
            logger.debug(f"Result `state`: {x.state}")
            logger.debug(f"Epoch: {x.epoch}, CSQ: {x.csq}, ndof: {x.ndof}, num obs: {len(obs)}")
    else:
        logger.debug(f"Result `state`: {x.state}")
        logger.debug(f"Epoch: {x.epoch}, CSQ: {x.csq}, ndof: {x.ndof}, num obs: {len(obs)}")

    return x


def do_other_fit(iod: str):
    """This is a place holder function for future IOD implememtations that require
    more significant data manipulation."""
    raise ValueError(f"The IOD, {iod} is not supported. Please use a supported IOD.")


# Minimum observational arc (in days) generally needed to constrain an orbit.
# Below this the fit is essentially unconstrained, so a failure is most likely a
# too-short baseline rather than anything wrong with the data.
_MIN_ARC_DAYS = 1.0


def _warn_if_short_arc(jds, obj_id):
    """Emit a helpful warning when a failed fit is likely caused by too short an
    observational arc (less than ~24 hours / a single night).

    See issue #312: an orbit fit needs a baseline of more than 24 hours, so when
    a fit fails on a sub-day arc we tell the user the likely cause rather than
    leaving them with an opaque failure.
    """
    if jds is None or len(jds) == 0:
        return
    arc_days = float(np.max(jds) - np.min(jds))
    if arc_days < _MIN_ARC_DAYS:
        logger.warning(
            f"Orbit fit failed for {obj_id}: the observations span only "
            f"{arc_days * 24.0:.1f} hours.  Constraining an orbit generally requires "
            f"a baseline of more than ~24 hours (more than a single night of observations)."
        )


def _orbitfit(
    data,
    cache_dir: str,
    primary_id_column_name: str,
    initial_guess=None,
    bias_dict: dict = None,
    sort_array: bool = True,
    weight_data: bool = False,
    iod: str = "gauss",
    engine: str = "cartesian",
    fit_nongrav: bool = False,
    nongrav_auto_thresholds=None,
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
    iod : str
        The IOD used to generate an initial guess orbit. Supports 'gauss'
        (default) and 'auto' (Gauss with BK-IOD fallback).
        Default is 'gauss'.
    """
    # Fitting non-gravitational params (issue #351) uses the joint state+nongrav
    # LM, which only the Cartesian engine supports; the BK-native engine assumes a
    # 6D state. Override with a warning, mirroring the radar path.
    auto_nongrav = isinstance(fit_nongrav, str) and fit_nongrav.strip().lower() == "auto"
    if auto_nongrav:
        # 'auto' selects the non-grav model per object (issue #357); the schema
        # carries all of A1/A2/A3 and each row reports only the adopted params.
        nongrav_mask, nongrav_names = 0, ["A1", "A2", "A3"]
    else:
        nongrav_mask, nongrav_names = _parse_nongrav(fit_nongrav)
    if (nongrav_mask or auto_nongrav) and engine != "cartesian":
        logger.warning("Non-gravitational fitting requires engine='cartesian'; overriding %r.", engine)
        engine = "cartesian"

    _RESULT_DTYPES = _get_result_dtypes(primary_id_column_name, nongrav_names)
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
            data = np.sort(data, order="obsTime", kind="mergesort")

        # Check if certain columns are present in the data
        column_names = data.dtype.names
        astcat_column_present = "astCat" in column_names
        program_column_present = "program" in column_names
        position_rates_columns_present = all(col in column_names for col in ["raRate", "decRate"])
        rate_unc_columns_present = all(col in column_names for col in ["rmsRArate", "rmsDecrate"])
        radar_columns_present = any(col in column_names for col in ["delay", "doppler"])

        # Accommodate occultation measurements. These measurements are implied when
        # the "ra" and "dec" columns are None. In this case, we will use the "starra"
        # and "stardec" columns. Radar rows have no ra/dec and are skipped.
        for d in data:
            if radar_columns_present and _is_radar(d, column_names):
                continue
            if _is_occultation(d):
                d = _use_star_astrometry(d)

        # bias_dict will be a dictionary when the debias flag is set to True.
        if bias_dict is not None:
            for d in data:
                if radar_columns_present and _is_radar(d, column_names):
                    continue  # debiasing is an astrometric (ra/dec) correction
                d["ra"], d["dec"] = debias(
                    ra=d["ra"],
                    dec=d["dec"],
                    epoch_jd_tdb=convert_tdb_date_to_julian_date(d["obsTime"], cache_dir),
                    catalog=d["astCat"] if astcat_column_present else None,
                    bias_dict=bias_dict,
                )

        # Convert the astrometry data to a list of Observations
        # Reminder to label the units.  Within an Observation struct,
        # and internal to the C++ code in general, we are using
        # radians.
        observations = []
        for d in data:
            if radar_columns_present and _is_radar(d, column_names):
                o = _radar_observation(
                    d[primary_id_column_name],
                    d,
                    convert_tdb_date_to_julian_date(d["obsTime"], cache_dir),  # JD TDB
                    column_names,
                )
                observations.append(o)
                continue
            if position_rates_columns_present and (not np.isnan(d["raRate"]) and not np.isnan(d["decRate"])):
                # Rate uncertainties (rmsRArate/rmsDecrate) share raRate's
                # arcsec/hour units; convert to rad/day. Absent -> C++ default.
                streak_rate_unc = {}
                if (
                    rate_unc_columns_present
                    and not np.isnan(d["rmsRArate"])
                    and not np.isnan(d["rmsDecrate"])
                ):
                    streak_rate_unc["ra_rate_unc"] = abs(d["rmsRArate"]) * ARCSEC_PER_HOUR_TO_RAD_PER_DAY
                    streak_rate_unc["dec_rate_unc"] = abs(d["rmsDecrate"]) * ARCSEC_PER_HOUR_TO_RAD_PER_DAY
                o = Observation.from_streak_with_id(
                    str(d[primary_id_column_name]),
                    d["ra"] * np.pi / 180.0,
                    d["dec"] * np.pi / 180.0,
                    # arcsec/hour (great-circle) -> rad/day; raRate already
                    # carries the cos(Dec) factor (see module constant above).
                    d["raRate"] * ARCSEC_PER_HOUR_TO_RAD_PER_DAY,
                    d["decRate"] * ARCSEC_PER_HOUR_TO_RAD_PER_DAY,
                    convert_tdb_date_to_julian_date(d["obsTime"], cache_dir),  # Convert obstime to JD TDB
                    [d["x"], d["y"], d["z"]],  # Barycentric position
                    [d["vx"], d["vy"], d["vz"]],  # Barycentric velocity
                    **streak_rate_unc,
                )
            else:
                o = Observation.from_astrometry_with_id(
                    str(d[primary_id_column_name]),
                    d["ra"] * np.pi / 180.0,
                    d["dec"] * np.pi / 180.0,
                    convert_tdb_date_to_julian_date(d["obsTime"], cache_dir),  # Convert obstime to JD TDB
                    [d["x"], d["y"], d["z"]],  # Barycentric position
                    [d["vx"], d["vy"], d["vz"]],  # Barycentric velocity
                )

            if weight_data:
                # astrometric_uncertainty_Veres2017 returns the astrometric uncertainty in
                # ARCSECONDS (per its docstring), but Observation.ra_unc /
                # dec_unc are stored in RADIANS.  Convert at the assignment.
                sigma_arcsec = astrometric_uncertainty_Veres2017(
                    obsCode=d["stn"],
                    jd_tdb=convert_tdb_date_to_julian_date(d["obsTime"], cache_dir),
                    catalog=d["astCat"] if astcat_column_present else None,
                    program=d["program"] if program_column_present else None,
                )
                sigma_rad = sigma_arcsec * np.pi / (180.0 * 3600.0)

                o.ra_unc = sigma_rad
                o.dec_unc = sigma_rad

            observations.append(o)

        # Radar delay/Doppler rows use the variable-row packing, which only the
        # Cartesian engine supports; the BK-native engine assumes 2 rows per
        # observation and would silently drop them.
        if radar_columns_present and engine != "cartesian":
            logger.warning(
                "Radar (delay/Doppler) observations require engine='cartesian'; overriding %r.",
                engine,
            )
            engine = "cartesian"

        # if cache_dir is not provided, use the default os_cache
        if cache_dir is None:
            kernels_loc = str(pooch.os_cache("layup"))
        else:
            kernels_loc = str(cache_dir)

        jds = convert_tdb_date_to_julian_date(data["obsTime"])
        sequence = _build_sequence(jds, sep_dt=90.0)

        # Perform the orbit fitting
        if initial_guess is None or initial_guess["flag"] != 0:
            if iod.lower() in ["gauss", "auto"]:
                res = do_fit(
                    observations=observations,
                    seq=sequence,
                    cache_dir=kernels_loc,
                    iod=iod.lower(),
                    engine=engine,
                )
            else:
                res = do_other_fit(iod=iod.lower())
        else:
            guess_to_use = parse_fit_result(initial_guess)
            res = run_from_vector_with_initial_guess(get_ephem(kernels_loc), guess_to_use, observations)

        # Non-gravitational params (issue #351): once the 6-parameter orbit has
        # converged, refine it jointly with the requested non-grav params, seeded
        # from that solution. They are weakly constrained on short arcs, so if the
        # joint fit is degenerate (flag 6) or fails to converge we keep the
        # 6-parameter result and report the params as NaN (graceful guard).
        if auto_nongrav and res.flag == 0:
            # Adopt the most parsimonious statistically-warranted non-grav model,
            # or keep the gravity-only fit (issue #357).
            res = _select_nongrav_auto(
                get_ephem(kernels_loc),
                res,
                observations,
                nongrav_auto_thresholds or _AUTO_DEFAULT_THRESHOLDS,
            )
        elif nongrav_mask and res.flag == 0:
            res_ng = run_from_vector_with_initial_guess(
                get_ephem(kernels_loc), res, observations, nongrav_mask=nongrav_mask
            )
            if res_ng.flag == 0:
                res = res_ng
            else:
                logger.debug("Non-grav refinement did not converge; reporting non-grav params as NaN.")

        # Populate our output structured array with the orbit fit results
        success = res.flag == 0
        if not success:
            _warn_if_short_arc(jds, data[primary_id_column_name][0])
        cov_matrix = tuple(res.cov[i] for i in range(36)) if success else (np.nan,) * 36
        nongrav_cols = ()
        if nongrav_names:
            # Report each parameter only if it was actually adopted (its bit is set
            # in the fit's nongrav_mask); for 'auto' that is the selected subset.
            fitted_mask = getattr(res, "nongrav_mask", 0) if success else 0
            nongrav_cols = tuple(
                v
                for n in nongrav_names
                for v in (
                    (getattr(res, n.lower()) if (fitted_mask & _NONGRAV_BITS[n]) else np.nan),
                    (getattr(res, n.lower() + "_unc") if (fitted_mask & _NONGRAV_BITS[n]) else np.nan),
                )
            )
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
                + nongrav_cols  # non-grav params + uncertainties (issue #351), when fit_nongrav
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
    iod="gauss",
    engine="cartesian",
    fit_nongrav=False,
    nongrav_auto_thresholds=None,
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
    iod : str
        The IOD used to generate an initial guess orbit. Supports 'gauss'
        (default) and 'auto' (Gauss with BK-IOD fallback).
        Default is 'gauss'.
    fit_nongrav : bool | str | iterable of str
        Which non-gravitational Marsden parameters to fit after the 6-parameter
        orbit converges. ``False`` (default) fits none; ``True`` fits A2 (the
        transverse Yarkovsky term, the common asteroid case); a string or iterable
        naming params -- e.g. ``"A2"``, ``"A1A2A3"``, ``["A1", "A3"]`` -- selects a
        subset of A1 (radial), A2 (transverse), A3 (normal). ``"auto"`` selects the
        model adaptively per object (issue #357): the gravity-only fit is kept
        unless its reduced chi-square is unacceptable, in which case the most
        parsimonious non-grav model that is well-conditioned and statistically
        significant is adopted (the A1/A2/A3 columns are all present, with only the
        adopted params filled and the rest NaN). For each fitted param an ``a{n}``
        value and ``a{n}_unc`` 1-sigma column (au/day^2) are added to the result.
        Cartesian engine only; params weakly constrained on short arcs are reported
        as NaN (issue #351).
    nongrav_auto_thresholds : NongravAutoThresholds, optional
        Decision thresholds used when ``fit_nongrav="auto"`` -- how unacceptable the
        gravity-only fit must be before a non-grav is tried, and how large the
        chi-square drop and per-parameter significance must be to adopt one. Default
        (``None``) uses the standard thresholds; pass a customized
        ``NongravAutoThresholds`` to tune. Ignored unless ``fit_nongrav="auto"``.
    """

    layup_observatory = LayupObservatory(cache_dir=cache_dir)

    # The units of et are seconds (from J2000). This new column is used by
    # data_processing_utilities.obscodes_to_barycentric.
    et_col = np.array([spice.str2et(row["obsTime"]) for row in data], dtype="<f8")
    data = rfn.append_fields(data, "et", et_col, usemask=False, asrecarray=True)

    pos_vel = layup_observatory.obscodes_to_barycentric(data)
    data = rfn.merge_arrays([data, pos_vel], flatten=True, asrecarray=True, usemask=False)

    # Radar (delay/Doppler) observations need the barycentric observer
    # acceleration for the two-leg light-time model; compute it only when radar
    # columns are present so optical/streak fits are unaffected.
    if any(col in data.dtype.names for col in ("delay", "doppler")):
        data = _append_observer_acceleration(data, layup_observatory)

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
        iod=iod,
        engine=engine,
        fit_nongrav=fit_nongrav,
        nongrav_auto_thresholds=nongrav_auto_thresholds,
    )


def _observations_for_update(data, cache_dir, weight_data=False, bias_dict=None):
    """Augment one object's observations with the observer barycentric state and
    build the C++ ``Observation`` list, mirroring ``orbitfit()``'s preprocessing.

    Supports optical astrometry and streak (rate) rows -- the observation kinds a
    steady-state catalog update sees. (Radar/occultation are not yet handled by
    the sequential path; the driver's full-refit fallback covers them.)
    """
    DEG = np.pi / 180.0
    kernels_loc = str(pooch.os_cache("layup")) if cache_dir is None else str(cache_dir)
    observatory = LayupObservatory(cache_dir=cache_dir)

    et = np.array([spice.str2et(row["obsTime"]) for row in data], dtype="<f8")
    data = rfn.append_fields(data, "et", et, usemask=False, asrecarray=True)
    pos_vel = observatory.obscodes_to_barycentric(data)
    data = rfn.merge_arrays([data, pos_vel], flatten=True, asrecarray=True, usemask=False)

    column_names = data.dtype.names
    astcat = "astCat" in column_names
    program = "program" in column_names
    rates_present = all(c in column_names for c in ("raRate", "decRate"))
    rate_unc_present = all(c in column_names for c in ("rmsRArate", "rmsDecrate"))

    if bias_dict is not None:
        for d in data:
            d["ra"], d["dec"] = debias(
                ra=d["ra"],
                dec=d["dec"],
                epoch_jd_tdb=convert_tdb_date_to_julian_date(d["obsTime"], kernels_loc),
                catalog=d["astCat"] if astcat else None,
                bias_dict=bias_dict,
            )

    observations = []
    for d in data:
        jd = convert_tdb_date_to_julian_date(d["obsTime"], kernels_loc)
        if rates_present and not np.isnan(d["raRate"]) and not np.isnan(d["decRate"]):
            streak_rate_unc = {}
            if rate_unc_present and not np.isnan(d["rmsRArate"]) and not np.isnan(d["rmsDecrate"]):
                streak_rate_unc["ra_rate_unc"] = abs(d["rmsRArate"]) * ARCSEC_PER_HOUR_TO_RAD_PER_DAY
                streak_rate_unc["dec_rate_unc"] = abs(d["rmsDecrate"]) * ARCSEC_PER_HOUR_TO_RAD_PER_DAY
            o = Observation.from_streak_with_id(
                str(d["provID"]),
                d["ra"] * DEG,
                d["dec"] * DEG,
                d["raRate"] * ARCSEC_PER_HOUR_TO_RAD_PER_DAY,
                d["decRate"] * ARCSEC_PER_HOUR_TO_RAD_PER_DAY,
                jd,
                [d["x"], d["y"], d["z"]],
                [d["vx"], d["vy"], d["vz"]],
                **streak_rate_unc,
            )
        else:
            o = Observation.from_astrometry_with_id(
                str(d["provID"]),
                d["ra"] * DEG,
                d["dec"] * DEG,
                jd,
                [d["x"], d["y"], d["z"]],
                [d["vx"], d["vy"], d["vz"]],
            )
        if weight_data:
            sigma_arcsec = astrometric_uncertainty_Veres2017(
                obsCode=d["stn"],
                jd_tdb=jd,
                catalog=d["astCat"] if astcat else None,
                program=d["program"] if program else None,
            )
            sigma_rad = sigma_arcsec * np.pi / (180.0 * 3600.0)
            o.ra_unc = sigma_rad
            o.dec_unc = sigma_rad
        observations.append(o)
    return observations


def _update_mahalanobis(prior_fit, updated_fit):
    """Mahalanobis distance of the state update in prior standard deviations:
    sqrt((x1 - x0)^T P0^-1 (x1 - x0)). This is the dimensionless size of the move
    the new observations induced, and the natural nonlinearity gate -- a large
    move means the linearization of the summarized old observations (which is
    anchored at the prior mean x0) is no longer trustworthy."""
    dx = np.array(updated_fit.state) - np.array(prior_fit.state)
    P0 = np.array(list(prior_fit.cov)).reshape(6, 6)
    try:
        return float(np.sqrt(dx @ np.linalg.solve(P0, dx)))
    except np.linalg.LinAlgError:
        return np.inf


def sequential_update(
    prior,
    new_data,
    cache_dir,
    *,
    all_data=None,
    weight_data=False,
    debias_data=False,
    max_update_sigma=4.0,
    iter_max=100,
):
    """Sequential / information-filter update of a prior orbit fit (issue #419).

    Refines ``prior`` using ONLY ``new_data`` (the newly reported observations of
    one object); the previously-fit observations enter through the prior's state
    and 6x6 covariance, which becomes the information matrix Lambda0 = P0^-1 added
    to the normal equations. To the extent the prior is locally Gaussian this
    equals a full batch refit over all observations, at the cost of integrating
    only the new ones -- the throughput win for steady-state catalog maintenance.

    Parameters
    ----------
    prior : FitResult or numpy structured array (one row)
        The prior fit (state, 6x6 covariance, epoch). A result-catalog row is
        accepted and parsed via ``parse_fit_result``.
    new_data : numpy structured array
        The new observations of this object (optical astrometry and/or streaks).
    cache_dir : str or None
        Kernel/ephemeris cache directory (None -> the default layup os_cache).
    all_data : numpy structured array, optional
        The full observation set (old + new). Required for the nonlinearity
        fallback: when the update is too large the driver refits over all_data.
    weight_data, debias_data : bool
        Apply Veres (2017) weighting / MPC debiasing to the new observations,
        matching the corresponding ``orbitfit`` options.
    max_update_sigma : float
        Nonlinearity gate. If the update moves the state more than this many prior
        standard deviations (Mahalanobis), fall back to a full refit over
        ``all_data`` when provided, else flag the result (flag=8).
    iter_max : int
        LM iteration cap.

    Returns
    -------
    FitResult
        The updated fit. ``method`` is ``"sequential_update"`` for an accepted
        information-filter update, or ``"orbit_fit"`` when the fallback refit ran.
    """
    prior_fit = prior if isinstance(prior, FitResult) else parse_fit_result(prior)
    kernels_loc = str(pooch.os_cache("layup")) if cache_dir is None else str(cache_dir)
    ephem = get_ephem(kernels_loc)
    bias_dict = generate_bias_dict(cache_dir) if debias_data else None

    new_obs = _observations_for_update(new_data, cache_dir, weight_data, bias_dict)
    seq = run_sequential_update(ephem, prior_fit, new_obs, iter_max)

    def _full_refit():
        if all_data is None:
            return None
        all_obs = _observations_for_update(all_data, cache_dir, weight_data, bias_dict)
        return run_from_vector_with_initial_guess(ephem, prior_fit, all_obs, iter_max)

    # The information update did not converge (e.g. a non-positive-definite prior,
    # flag 7): fall back to a full refit if we can, else surface the failure.
    if seq.flag != 0:
        fallback = _full_refit()
        return fallback if fallback is not None else seq

    # Nonlinearity gate: a large move means the linearization is untrustworthy.
    if _update_mahalanobis(prior_fit, seq) > max_update_sigma:
        fallback = _full_refit()
        if fallback is not None:
            return fallback
        seq.flag = 8  # nonlinear update, no full-obs set supplied to refit
    return seq


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
        iod = cli_args.iod
        engine = getattr(cli_args, "engine", "cartesian")
    else:
        cache_dir = None
        debias = False
        guess_file = None
        weight_data = False
        output_orbit_format = "COM"  # Default output orbit format.
        iod = "gauss"
        engine = "cartesian"

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

    reader = reader_class(
        input_file,
        primary_id_column_name=_primary_id_column_name,
        sep=separator,
        required_columns=REQUIRED_INPUT_OBSERVATIONS_COLUMN_NAMES,
    )
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
        guess_reader = reader_class(
            guess_file,
            primary_id_column_name=_primary_id_column_name,
            sep=separator,
            required_columns=REQUIRED_INPUT_GUESS_COLUMN_NAMES,
        )

    chunks = create_chunks(reader, chunk_size)

    # Output is written one chunk at a time. The first write to each file
    # overwrites any stale file left by a previous run (so re-runs are
    # idempotent); later chunks append. Without this a re-run, or a re-merge onto
    # an existing file, would duplicate every row.
    _written_files = set()

    def _emit(arr, path):
        first = path not in _written_files
        _written_files.add(path)
        if output_file_format == "hdf5":
            (write_hdf5 if first else append_hdf5)(arr, path, key="data")
        else:  # csv: write_csv appends when the file already exists
            if first and os.path.exists(path):
                os.remove(path)
            write_csv(arr, path)

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
            iod=iod,
            engine=engine,
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

            if len(fit_orbits_success) > 0:
                _emit(fit_orbits_success, output_file)

            if len(fit_orbits_failed) > 0:
                _emit(fit_orbits_failed[[_primary_id_column_name, "method", "flag"]], output_file_flagged)

        else:  # All results go to a single output file
            _emit(fit_orbits, output_file)

    logger.info(f"Data has been written to {output_file}")


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
    column_names = data.dtype.names
    valid_conditions = [
        len(data) >= 3,
        np.all(
            data["et"] >= -6279962400.00
        ),  # excludes all datasets before 1801, data["et"] = 0 is j2000, 6279962400.00 is seconds between 1801 and j2000
        np.all(is_numeric(data["x"])),
        np.all(is_numeric(data["y"])),
        np.all(is_numeric(data["z"])),
        np.all(is_numeric(data["vx"])),
        np.all(is_numeric(data["vy"])),
        np.all(is_numeric(data["vz"])),
    ]

    # Each row must carry a usable observable: ra/dec for optical (and streak),
    # or a delay/Doppler for radar. Validate per row so a radar file -- which has
    # no ra/dec columns -- is not rejected, while optical rows still require
    # numeric ra/dec.
    if any(c in column_names for c in ["delay", "doppler"]):
        have_radec = "ra" in column_names and "dec" in column_names
        valid_conditions.append(
            all(
                _is_radar(d, column_names) or (have_radec and is_numeric(d["ra"]) and is_numeric(d["dec"]))
                for d in data
            )
        )
    else:
        valid_conditions.append(np.all(is_numeric(data["ra"])))
        valid_conditions.append(np.all(is_numeric(data["dec"])))

    return all(valid_conditions)


def is_numeric(obj):  # checks object is numeric by checking object has all required attributes
    attrs = ["__add__", "__sub__", "__mul__", "__truediv__", "__pow__"]
    return all(hasattr(obj, attr) for attr in attrs)
