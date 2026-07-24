import logging
import os
from pathlib import Path
from typing import Literal

import jax
import numpy as np
from scipy.linalg import block_diag
from sorcha.ephemeris.simulation_geometry import EQ_TO_ECL_ROTATION_MATRIX, equatorial_to_ecliptic
from sorcha.ephemeris.simulation_parsing import parse_orbit_row
from sorcha.ephemeris.simulation_setup import _create_assist_ephemeris

from layup.utilities.data_processing_utilities import (
    get_cov_columns,
    get_format,
    has_cov_columns,
    process_data,
)
from layup.utilities.file_io import CSVDataReader, HDF5DataReader
from layup.utilities.file_io.file_output import write_csv, write_hdf5
from layup.utilities.layup_configs import LayupConfigs
from layup.utilities.orbit_conversion import (
    ECL_TO_EQ_ROTATION_MATRIX,
    covariance_cometary_xyz,
    covariance_eq_to_ecl,
    covariance_keplerian_xyz,
    parse_covariance_row_to_CART,
    universal_cartesian,
    universal_cometary,
    universal_keplerian,
)

logger = logging.getLogger(__name__)

# Columns which may be added to the output data by the orbit fitting process
ORBIT_FIT_COLS = [
    ("csq", "f8"),  # Chi-square value
    ("ndof", "i4"),  # Number of degrees of freedom
    ("niter", "i4"),  # Number of iterations
    ("method", "O"),  # Method used for orbit fitting
    ("flag", "i4"),  # Single-character flag indicating success of the fit
]

# Columns which use degrees as units in each orbit format
degree_columns = {
    "BCOM": ["inc", "node", "argPeri"],
    "COM": ["inc", "node", "argPeri"],
    "BKEP": ["inc", "node", "argPeri", "ma"],
    "KEP": ["inc", "node", "argPeri", "ma"],
}

# Ordered 6-element state for each format; the index of an element here is also
# its row/column index in the 6x6 covariance. This ordering MUST match the basis
# of the covariance produced by the conversion routines (covariance_*_xyz /
# parse_covariance_row_to_CART); _scale_degree_cov relies on it, and
# test_element_order_matches_degree_columns guards the dict from drifting.
element_order = {
    "BCART": ["x", "y", "z", "xdot", "ydot", "zdot"],
    "BCART_EQ": ["x", "y", "z", "xdot", "ydot", "zdot"],
    "CART": ["x", "y", "z", "xdot", "ydot", "zdot"],
    "BCOM": ["q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB"],
    "COM": ["q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB"],
    "BKEP": ["a", "e", "inc", "node", "argPeri", "ma"],
    "KEP": ["a", "e", "inc", "node", "argPeri", "ma"],
}


def _scale_degree_cov(arr, fmt, factor, mask=None):
    """Congruence-scale the flattened 6x6 covariance of a structured array.

    Applies ``diag(s) C diag(s)`` where ``s`` is ``factor`` on each of
    ``fmt``'s degree columns and 1 elsewhere -- i.e. the covariance transform
    for rescaling those state elements by ``factor``. Pass ``factor = 180/pi``
    when the matching state values were just converted radians->degrees, or
    ``pi/180`` for the inverse. Restricted to rows in ``mask`` when given,
    else applied to the whole column.

    Relies on the invariant that ``element_order[fmt]`` indexes the ``cov_i_j``
    rows/cols (see element_order; guarded by
    test_element_order_matches_degree_columns).
    """
    s = np.ones(6)
    for col in degree_columns[fmt]:
        s[element_order[fmt].index(col)] = factor
    sel = slice(None) if mask is None else mask
    for i in range(6):
        for j in range(6):
            arr[f"cov_{i}_{j}"][sel] *= s[i] * s[j]


# Add this to MJD to convert to JD
MJD_TO_JD = 2400000.5


def _subtract_sun(coords, vels, sun):
    """Shift a barycentric equatorial state to heliocentric by subtracting the Sun."""
    return (
        coords - np.array((sun.x, sun.y, sun.z)),
        vels - np.array((sun.vx, sun.vy, sun.vz)),
    )


def _to_ecliptic(coords, vels):
    """Rotate an equatorial (coords, vels) state pair into ecliptic coordinates."""
    return (
        np.array(equatorial_to_ecliptic(coords)),
        np.array(equatorial_to_ecliptic(vels)),
    )


INPUT_READERS = {
    "csv": CSVDataReader,
    "hdf5": HDF5DataReader,
}


def get_output_column_names_and_types(primary_id_column_name, has_covariance, extra_cols_to_keep):
    """
    Get the output column names and types for the converted data.

    Parameters
    ----------
    primary_id_column_name : str
        The name of the column in the data that contains the primary ID of the object.
    has_covariance : bool
        Whether the data has covariance information.
    extra_cols_to_keep : list
        List of tuples containing extra column names and dtypes to keep in the output data.

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
        "BCART_EQ": [primary_id_column_name, "FORMAT", "x", "y", "z", "xdot", "ydot", "zdot", "epochMJD_TDB"],
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
    default_column_dtypes = ["O", "<U8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8"]
    default_column_dtypes.extend([dtype for _, dtype in extra_cols_to_keep])
    if has_covariance:
        # Flattened 6x6 covariance matrix
        default_column_dtypes += ["f8"] * 36
    for format in required_column_names:
        for col_name, _ in extra_cols_to_keep:
            # Add the column name and dtype to the default column dtypes
            required_column_names[format].append(col_name)
        if has_covariance:
            # Add the covariance columns to the required column names
            required_column_names[format] += get_cov_columns()
    return required_column_names, default_column_dtypes


# Input formats whose elements are heliocentric; parse_orbit_row adds the Sun
# for these to reach the barycentric frame.
_HELIOCENTRIC_INPUT = ("COM", "KEP", "CART")


def _rotate_batch(rot_mat, states):
    """Column-stacked (3, N) form of sorcha's per-vector rotations.

    ``equatorial_to_ecliptic``/``ecliptic_to_equatorial`` use the ``v @ rot_mat``
    convention (row vectors); for a state stacked as columns that is
    ``rot_mat.T @ states``.
    """
    return rot_mat.T @ states


def _sun_states_by_epoch(ephem, epochs_mjd):
    """Sun equatorial state as a (6, N) array, evaluating the ephemeris once per
    unique epoch (mirrors parse_orbit_row's per-epoch ``sun_dict`` cache)."""
    uniq, inv = np.unique(epochs_mjd, return_inverse=True)
    sun = np.empty((6, len(uniq)))
    for k, epoch in enumerate(uniq):
        p = ephem.get_particle("Sun", epoch + MJD_TO_JD - ephem.jd_ref)
        sun[:, k] = (p.x, p.y, p.z, p.vx, p.vy, p.vz)
    return sun[:, inv]


def _parse_to_bcart_eq(data, gm_sun, gm_total, sun):
    """Vectorized ``parse_orbit_row``: any input FORMAT -> equatorial barycentric
    Cartesian ``(coords, vels)``, each a (3, N) array.

    Mirrors ``sorcha.ephemeris.simulation_parsing.parse_orbit_row`` element for
    element. ``universal_cartesian`` solves Kepler's equation with a numba (non-
    jax) Halley iteration, so element inputs are converted per row; Cartesian and
    BCART_EQ inputs are fully vectorized.
    """
    n = len(data)
    fmt = np.asarray(data["FORMAT"])
    epoch_jd = np.asarray(data["epochMJD_TDB"], dtype=float) + MJD_TO_JD
    deg = np.pi / 180.0
    ecl = np.full((6, n), np.nan)

    for f in ("COM", "BCOM", "KEP", "BKEP"):
        m = fmt == f
        if not m.any():
            continue
        barycentric = f in ("BCOM", "BKEP")
        mu = gm_total if barycentric else gm_sun
        e = np.asarray(data["e"][m], dtype=float)
        if f in ("COM", "BCOM"):
            q = np.asarray(data["q"][m], dtype=float)
            tp = np.asarray(data["t_p_MJD_TDB"][m], dtype=float) + MJD_TO_JD
        else:
            a = np.asarray(data["a"][m], dtype=float)
            q = a * (1 - e)
            tp = epoch_jd[m] - (np.asarray(data["ma"][m], dtype=float) * deg) * np.sqrt(a**3 / mu)
        inc = np.asarray(data["inc"][m], dtype=float) * deg
        node = np.asarray(data["node"][m], dtype=float) * deg
        argperi = np.asarray(data["argPeri"][m], dtype=float) * deg
        ep = epoch_jd[m]
        ecl[:, m] = np.array(
            [
                universal_cartesian(mu, q[i], e[i], inc[i], node[i], argperi[i], tp[i], ep[i])
                for i in range(int(m.sum()))
            ]
        ).T

    for f in ("CART", "BCART"):
        m = fmt == f
        if m.any():
            ecl[:, m] = np.stack(
                [np.asarray(data[c][m], dtype=float) for c in ("x", "y", "z", "xdot", "ydot", "zdot")]
            )

    coords = _rotate_batch(ECL_TO_EQ_ROTATION_MATRIX, ecl[:3])
    vels = _rotate_batch(ECL_TO_EQ_ROTATION_MATRIX, ecl[3:])
    helio = np.isin(fmt, _HELIOCENTRIC_INPUT)
    coords[:, helio] += sun[:3, helio]
    vels[:, helio] += sun[3:, helio]

    m = fmt == "BCART_EQ"  # already equatorial barycentric; no rotation/Sun shift
    if m.any():
        coords[:, m] = np.stack([np.asarray(data[c][m], dtype=float) for c in ("x", "y", "z")])
        vels[:, m] = np.stack([np.asarray(data[c][m], dtype=float) for c in ("xdot", "ydot", "zdot")])
    return coords, vels


def _bcart_eq_to_elements(mu, coords, vels, epoch_mjd, cometary):
    """Vectorized ecliptic Cartesian -> Keplerian/cometary elements.

    Closed-form counterpart of ``universal_keplerian`` / ``universal_cometary``
    (validated to reproduce them to ~1e-11; see
    ``test_convert_vectorized_matches_rowwise``). The closed form covers elliptic
    orbits; the rare ``e >= 1`` rows are recomputed with the exact routine, which
    also carries the parabolic/hyperbolic branches.
    """
    x, y, z = coords
    vx, vy, vz = vels
    hx = y * vz - z * vy
    hy = z * vx - x * vz
    hz = x * vy - y * vx
    hs = hx * hx + hy * hy + hz * hz
    h = np.sqrt(hs)
    r = np.sqrt(x * x + y * y + z * z)
    rdot = (x * vx + y * vy + z * vz) / r
    p = hs / mu
    incl = np.arccos(hz / h)
    node = np.arctan2(hx, -hy)
    ecos = p / r - 1.0
    esin = rdot * h / mu
    e = np.sqrt(ecos * ecos + esin * esin)
    q = p / (1 + e)
    a = q / (1 - e)
    trueanom = np.arctan2(esin, ecos)
    cn, sn = np.cos(node), np.sin(node)
    arglat = np.arctan2((y * cn - x * sn) / np.cos(incl), x * cn + y * sn)
    argperi = arglat - trueanom
    with np.errstate(invalid="ignore"):  # sqrt(1 - e) is NaN for the e >= 1 rows, fixed below
        eccanom = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(trueanom / 2), np.sqrt(1 + e) * np.cos(trueanom / 2))
        mean_anom = eccanom - e * np.sin(eccanom)
    # |a| keeps the mean motion real for a < 0 (hyperbolic), matching universal_keplerian.
    tp = epoch_mjd - mean_anom / np.sqrt(mu / np.abs(a) ** 3)
    first = q if cometary else a
    sixth = tp if cometary else mean_anom

    nonelliptic = ~(e < 1.0)
    if nonelliptic.any():
        fn = universal_cometary if cometary else universal_keplerian
        for i in np.nonzero(nonelliptic)[0]:
            vals = [float(t) for t in fn(mu, x[i], y[i], z[i], vx[i], vy[i], vz[i], epoch_mjd[i])]
            first[i], e[i], incl[i], node[i], argperi[i], sixth[i] = vals
    return np.stack([first, e, incl, node, argperi, sixth])


# Block-diagonal 6x6 rotation congruences for covariance, matching
# orbit_conversion.covariance_eq_to_ecl / covariance_ecl_to_eq.
_COV_EQ_TO_ECL = block_diag(EQ_TO_ECL_ROTATION_MATRIX.T, EQ_TO_ECL_ROTATION_MATRIX.T)
_COV_ECL_TO_EQ = block_diag(ECL_TO_EQ_ROTATION_MATRIX.T, ECL_TO_EQ_ROTATION_MATRIX.T)

# Batched (vmapped) forms of the pure-jax output covariance transforms.
_covariance_keplerian_xyz_batch = jax.vmap(covariance_keplerian_xyz, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))
_covariance_cometary_xyz_batch = jax.vmap(covariance_cometary_xyz, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))


def _parse_cov_to_bcart_eq(data, gm_sun, gm_total):
    """Vectorized ``parse_covariance_row_to_CART``: each row's flattened input
    covariance -> equatorial Cartesian, as an (N, 6, 6) array.

    Cartesian inputs are a batched rotation congruence; element inputs use the
    per-row (numba) ``parse_covariance_row_to_CART`` (its Jacobian goes through
    ``universal_cartesian``, which is not jax-traceable). Assumes any degree-format
    input covariance has already been rescaled to radians by the caller.
    """
    fmt = np.asarray(data["FORMAT"])
    cov = np.empty((len(data), 6, 6))
    for i in range(6):
        for j in range(6):
            cov[:, i, j] = np.asarray(data[f"cov_{i}_{j}"], dtype=float)

    out = cov.copy()
    m = np.isin(fmt, ("CART", "BCART"))  # ecliptic -> equatorial rotation congruence
    if m.any():
        out[m] = _COV_ECL_TO_EQ @ cov[m] @ _COV_ECL_TO_EQ.T
    for i in np.nonzero(np.isin(fmt, ("COM", "BCOM", "KEP", "BKEP")))[0]:
        out[i] = parse_covariance_row_to_CART(data[i], gm_total, gm_sun)
    # BCART_EQ rows are already equatorial Cartesian (unchanged).
    return out


def _apply_convert_vectorized(
    data,
    convert_to,
    ephem,
    gm_sun,
    gm_total,
    primary_id_column_name,
    has_covariance,
    output_dtype,
    cols_to_keep,
):
    """Vectorized ``_apply_convert`` for all input/output formats.

    Numerically reproduces the per-row path (``_apply_convert_rowwise``; pinned by
    ``test_convert_vectorized_matches_rowwise``) while avoiding the per-row jax
    dispatch, per-row ephemeris query and per-row struct-array build. ~250x for
    Cartesian/BCART_EQ inputs (the bulk case), ~30x for element inputs (bounded by
    the per-row ``universal_cartesian`` Kepler solve; element-input covariances
    likewise fall back to the per-row Jacobian).
    """
    if has_covariance:
        # Copy, float the covariance columns (a CSV of all-zero placeholders may be
        # read as int), and rescale degree-format input covariances to radians so
        # the conversion Jacobians -- which work in radians -- stay consistent.
        cov_names = set(get_cov_columns())
        data = data.astype([(nm, "f8" if nm in cov_names else data.dtype[nm]) for nm in data.dtype.names])
        for f in degree_columns:
            m = data["FORMAT"] == f
            if m.any():
                _scale_degree_cov(data, f, np.pi / 180, mask=m)

    n = len(data)
    fmt = np.asarray(data["FORMAT"])
    epoch_mjd = np.asarray(data["epochMJD_TDB"], dtype=float)

    out = np.zeros(n, dtype=output_dtype)
    out[primary_id_column_name] = data[primary_id_column_name]
    convertible = fmt != "NONE"
    out["FORMAT"] = np.where(convertible, convert_to, "NONE")
    out["epochMJD_TDB"] = epoch_mjd
    for col, _ in cols_to_keep:
        out[col] = data[col]

    value_cols = element_order[convert_to]
    d = data[convertible]
    if len(d):
        sun = _sun_states_by_epoch(ephem, np.asarray(d["epochMJD_TDB"], dtype=float))
        coords, vels = _parse_to_bcart_eq(d, gm_sun, gm_total, sun)
        epoch = np.asarray(d["epochMJD_TDB"], dtype=float)
        cov = _parse_cov_to_bcart_eq(d, gm_sun, gm_total) if has_covariance else None

        if convert_to == "BCART_EQ":
            values = np.vstack([coords, vels])
            cov_out = cov
        elif convert_to in ("BCART", "CART"):
            if convert_to == "CART":  # heliocentric ecliptic Cartesian
                coords, vels = coords - sun[:3], vels - sun[3:]
            values = np.vstack(
                [
                    _rotate_batch(EQ_TO_ECL_ROTATION_MATRIX, coords),
                    _rotate_batch(EQ_TO_ECL_ROTATION_MATRIX, vels),
                ]
            )
            cov_out = _COV_EQ_TO_ECL @ cov @ _COV_EQ_TO_ECL.T if has_covariance else None
        else:  # BCOM / COM / BKEP / KEP
            barycentric = convert_to in ("BCOM", "BKEP")
            cometary = convert_to in ("BCOM", "COM")
            mu = gm_total if barycentric else gm_sun
            if not barycentric:  # heliocentric: subtract the Sun (equatorial)
                coords, vels = coords - sun[:3], vels - sun[3:]
            if has_covariance:
                # The output covariance is built from the equatorial state (the basis
                # covariance_*_xyz expect; they rotate to ecliptic internally).
                batch = _covariance_cometary_xyz_batch if cometary else _covariance_keplerian_xyz_batch
                cov_out = np.asarray(
                    batch(mu, coords[0], coords[1], coords[2], vels[0], vels[1], vels[2], epoch, cov)
                )
            else:
                cov_out = None
            ecl_coords = _rotate_batch(EQ_TO_ECL_ROTATION_MATRIX, coords)
            ecl_vels = _rotate_batch(EQ_TO_ECL_ROTATION_MATRIX, vels)
            values = _bcart_eq_to_elements(mu, ecl_coords, ecl_vels, epoch, cometary)

        for k, col in enumerate(value_cols):
            out[col][convertible] = values[k]
        if has_covariance:
            for i in range(6):
                for j in range(6):
                    out[f"cov_{i}_{j}"][convertible] = cov_out[:, i, j]

    for col in value_cols:  # non-convertible (FORMAT == "NONE") rows carry NaN state
        out[col][~convertible] = np.nan
    if has_covariance:
        for i in range(6):
            for j in range(6):
                out[f"cov_{i}_{j}"][~convertible] = np.nan

    for col in degree_columns.get(convert_to, []):
        out[col] = (out[col] * 180 / np.pi) % 360
    if has_covariance and convert_to in degree_columns:
        _scale_degree_cov(out, convert_to, 180 / np.pi)
    return out


def _apply_convert(data, convert_to, cache_dir=None, primary_id_column_name=None, extra_cols_to_keep=None):
    """
    Apply the appropriate conversion function to the data

    Parameters
    ----------
    data : numpy structured array
        The data to convert.
    convert_to : str
        The orbital format to convert the data to. Must be one of: "BCART", "BCART_EQ", "BCOM", "BKEP", "CART", "COM", "KEP"
    cache_dir : str, optional
        The base directory for downloaded files.
    primary_id_column_name : str, optional
        The name of the column in the data that contains the primary ID of the object.
    extra_cols_to_keep : list, optional
        List of tuples containing extra column names and dtypes to keep in the output data.

    Returns
    -------
    data : numpy structured array
        The converted data
    """
    if len(data) == 0:
        return data

    if extra_cols_to_keep is None:
        extra_cols_to_keep = []

    expected_formats = ["BCART", "BCART_EQ", "BCOM", "BKEP", "CART", "COM", "KEP"]
    if convert_to not in expected_formats:
        raise ValueError(f"Invalid conversion type {convert_to}. Must be one of: {expected_formats}")
    has_covariance = has_cov_columns(data)
    logger.debug(f"Data has covariance: {has_covariance}")

    # In addition to columns requested by the user, to be kept in the output, add in extra columns
    # that may have been added by the orbit fitting process.
    cols_to_keep = extra_cols_to_keep + ORBIT_FIT_COLS

    # Filter the extra columns to keep to only those that are present in the data
    cols_to_keep = [(col, dtype) for col, dtype in cols_to_keep if col in data.dtype.names]
    logger.debug(f"Columns to keep: {cols_to_keep}")

    # Now generate the required and columns names and dtypes for the output data
    required_colum_names, default_column_dtypes = get_output_column_names_and_types(
        primary_id_column_name, has_covariance, cols_to_keep
    )

    # Fetch layup configs to get the necessary auxiliary data
    config = LayupConfigs()
    ephem, gm_sun, gm_total = _create_assist_ephemeris(config.auxiliary, cache_dir)

    # Construct the output dtype for the converted data
    output_dtype = [
        (col, dtype)
        for col, dtype in zip(required_colum_names[convert_to], default_column_dtypes, strict=False)
    ]

    return _apply_convert_vectorized(
        data,
        convert_to,
        ephem,
        gm_sun,
        gm_total,
        primary_id_column_name,
        has_covariance,
        output_dtype,
        cols_to_keep,
    )


def _apply_convert_rowwise(
    data,
    convert_to,
    ephem,
    gm_sun,
    gm_total,
    primary_id_column_name,
    has_covariance,
    output_dtype,
    cols_to_keep,
):
    """Reference per-row implementation of ``_apply_convert``.

    Superseded in production by ``_apply_convert_vectorized`` (which reproduces it
    to ~1e-11); retained as the equivalence oracle for
    ``test_convert_vectorized_matches_rowwise``.
    """
    # A stored degree-format orbit carries its covariance in degrees (consistent
    # with its degree-valued angles). The conversion math below works in radians
    # (parse_covariance_row_to_CART and the element-value parsing use radians), so
    # scale the angle rows/cols of the input covariance back to radians first.
    # This mirrors the radians->degrees output scaling and keeps round-trips exact.
    if has_covariance:
        # Copy, and ensure the covariance columns are float: a CSV of all-zero
        # placeholder covariances may be read as int, which an in-place float
        # multiply cannot cast into under numpy's same-kind rule.
        cov_cols = set(get_cov_columns())
        data = data.astype([(n, "f8" if n in cov_cols else data.dtype[n]) for n in data.dtype.names])
        for fmt in degree_columns:
            mask = data["FORMAT"] == fmt
            if not mask.any():
                continue
            # Input values for this format are in degrees; rescale the
            # covariance of those elements back to radians (pi/180) for the
            # rows of this format before conversion.
            _scale_degree_cov(data, fmt, np.pi / 180, mask=mask)

    # For each row in the data, convert the orbit to the desired format
    results = []
    for d in data:
        # Note that the format value may be "NONE" for rows which were from layup orbit fit failures
        # For these we simply output NaNs rather than try to convert.
        row = (np.nan,) * 6
        cov = np.full((6, 6), np.nan)
        if d["FORMAT"] != "NONE":
            if not isinstance(d["FORMAT"], str):
                raise ValueError(f"FORMAT column must be a string {d['FORMAT']}.")

            epoch = d["epochMJD_TDB"]
            sun = ephem.get_particle("Sun", epoch + MJD_TO_JD - ephem.jd_ref)

            # First convert the input into equatorial barycentric Cartesian
            # coordinates regardless of input format, so the per-output-format
            # logic below only has to convert *out* of BCART_EQ.
            if d["FORMAT"] == "BCART_EQ":
                # Already BCART_EQ; parse_orbit_row not needed.
                x, y, z, xdot, ydot, zdot = d["x"], d["y"], d["z"], d["xdot"], d["ydot"], d["zdot"]
            else:
                x, y, z, xdot, ydot, zdot = parse_orbit_row(d, epoch + MJD_TO_JD, ephem, {}, gm_sun, gm_total)
            # Parse the 6x6 equatorial Cartesian covariance (basis-independent of
            # the eventual output format; handled per-format below).
            if has_covariance:
                cov = parse_covariance_row_to_CART(d, gm_total, gm_sun)

            eq_coords = np.array((x, y, z))
            eq_vels = np.array((xdot, ydot, zdot))

            if convert_to == "BCART_EQ":
                # Already equatorial barycentric Cartesian.
                row = x, y, z, xdot, ydot, zdot
            elif convert_to in ("BCART", "CART"):
                # Both are ecliptic Cartesian; CART is additionally heliocentric.
                cov = covariance_eq_to_ecl(cov)
                coords, vels = eq_coords, eq_vels
                if convert_to == "CART":
                    coords, vels = _subtract_sun(coords, vels, sun)
                ecl_coords, ecl_vels = _to_ecliptic(coords, vels)
                row = tuple(np.concatenate([ecl_coords, ecl_vels]))
            else:
                # Cometary (BCOM/COM) and Keplerian (BKEP/KEP). These differ only
                # in: central mass (barycentric gm_total vs heliocentric gm_sun),
                # whether the Sun is subtracted, and which element/covariance
                # routine is used. The covariance is built from the *equatorial*
                # state (the basis those routines expect); the element conversion
                # uses the *ecliptic* state.
                barycentric = convert_to in ("BCOM", "BKEP")
                cometary = convert_to in ("BCOM", "COM")
                mu = gm_total if barycentric else gm_sun

                coords, vels = eq_coords, eq_vels
                if not barycentric:
                    coords, vels = _subtract_sun(coords, vels, sun)

                if has_covariance:
                    cov_fn = covariance_cometary_xyz if cometary else covariance_keplerian_xyz
                    cov = cov_fn(mu, coords[0], coords[1], coords[2], vels[0], vels[1], vels[2], epoch, cov)

                ecl_coords, ecl_vels = _to_ecliptic(coords, vels)
                transform = universal_cometary if cometary else universal_keplerian
                row = transform(
                    mu,
                    ecl_coords[0],
                    ecl_coords[1],
                    ecl_coords[2],
                    ecl_vels[0],
                    ecl_vels[1],
                    ecl_vels[2],
                    epoch,
                )

        row += (d["epochMJD_TDB"],)
        row += tuple(d[col] for col, _ in cols_to_keep)

        # If the covariance matrix is present, convert it to a flattened tuple for output.
        cov_res = tuple(val for val in cov.flatten()) if has_covariance else tuple()

        # Turn our converted row into a structured array
        output_format = convert_to if d["FORMAT"] != "NONE" else "NONE"
        result_struct_array = np.array(
            [(d[primary_id_column_name], output_format) + row + cov_res],
            dtype=output_dtype,
        )
        results.append(result_struct_array)

    # Convert the list of results to a numpy structured array
    output = np.squeeze(np.array(results)) if len(results) > 1 else results[0]

    # The outputs of the sorcha orbit conversion utilities are always in radians, so convert to degrees for any such columns.
    for col in degree_columns.get(convert_to, []):
        # Convert from radians to degrees and wrap to [0, 360)
        output[col] = (output[col] * 180 / np.pi) % 360

    # The covariance is propagated in radians, so the rows/columns of any
    # element whose value was just converted to degrees must be scaled by
    # 180/pi to stay consistent with the (now degree-valued) state. Without
    # this, the reported angular uncertainties are too small by 180/pi (their
    # variances by (180/pi)^2). Round-trip conversions hide this because the
    # inverse conversion undoes both the value and the covariance scaling.
    if has_covariance and convert_to in degree_columns:
        # Mirror the value conversion directly above (radians->degrees), which
        # operates on the whole column, so scale the covariance whole-column
        # too (no row mask) to keep values and covariance consistent.
        _scale_degree_cov(output, convert_to, 180 / np.pi)

    return output


def convert(
    data,
    convert_to,
    num_workers=1,
    cache_dir=None,
    primary_id_column_name="ObjID",
    extra_cols_to_keep=None,
):
    """
    Convert a structured numpy array to a different orbital format with support for parallel processing.

    Parameters
    ----------
    data : numpy structured array
        The data to convert.
    convert_to : str
        The format to convert the data to. Must be one of: "BCART_EQ", "BCOM", "BKEP", "CART", "COM", "KEP"
    num_workers : int, optional (default=1)
        The number of workers to use for parallel processing.
    cache_dir : str, optional
        The base directory for downloaded files.
    primary_id_column_name : str, optional (default="ObjID")
        The name of the column in the data that contains the primary ID of the object.
    extra_cols_to_keep : list, optional
        List of tuples containing additional column names and dtypes to keep in the output data.

    Returns
    -------
    data : numpy structured array
        The converted data
    """
    if num_workers == 1:
        return _apply_convert(
            data,
            convert_to,
            cache_dir=cache_dir,
            primary_id_column_name=primary_id_column_name,
            extra_cols_to_keep=extra_cols_to_keep,
        )
    # Parallelize the conversion of the data across the requested number of workers
    return process_data(
        data,
        num_workers,
        _apply_convert,
        convert_to=convert_to,
        cache_dir=cache_dir,
        primary_id_column_name=primary_id_column_name,
        extra_cols_to_keep=extra_cols_to_keep,
    )


def convert_cli(
    input: str,
    output_file_stem: str,
    convert_to: Literal["BCART", "BCART_EQ", "BCOM", "BKEP", "CART", "COM", "KEP"],
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
        The format to convert the input file to. Must be one of: "BCART", "BCART_EQ", "BCOM", "BKEP", "CART", "COM", "KEP"
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

    logger.info(f"Reading the first line of {input_file} as header:\n{sample_data.dtype.names}\n")
    # Check orbit format in the file
    input_format = get_format(sample_data)

    # Check that the input format is not already the desired format
    if convert_to == input_format:
        logger.error("Input file is already in the desired format")

    # Reopen the file now that we know the input format and can validate the column names
    required_columns_names, _ = get_output_column_names_and_types(
        primary_id_column_name,
        False,  # We don't need to know if the input data has covariance columns for basic validation.
        [],  # No additional columns to keep
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

    logger.info(f"Data has been written to {output_file}")
