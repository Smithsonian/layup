"""Move fitted orbits to a different reference epoch (issue #578).

``convert`` changes the *parameterization* of an orbit at a fixed epoch;
this changes the epoch. The two are complementary and neither substitutes for
the other.

The state is integrated with ASSIST, the same dynamics the fitter uses, and the
covariance is carried with it as ``C' = Phi C Phi^T`` where ``Phi`` is the
state-transition matrix over the interval.

No light-time correction is applied. Propagation to an epoch is a dynamical
operation; :mod:`layup.predict` applies light time because it computes an
observable, and conflating the two would be a subtle and expensive error.

Why this exists: Layup already emits two epoch conventions and could not
reconcile them. A cold-started fit is referenced to the middle observation of
its IOD triplet, so it sits inside its own arc; a warm-started fit inherits the
epoch of its initial guess, so it sits whereever the catalogue that seeded it
does. Comparing the two, or comparing either against a reference solution
published at a third epoch, requires moving one of them.
"""

import logging

import numpy as np

from layup.routines import get_ephem, propagate_state
from layup.utilities.data_processing_utilities import parse_fit_result
from layup.utilities.cache_location import default_cache_dir

logger = logging.getLogger(__name__)

__all__ = ["propagate", "propagate_row"]

MJD_TO_JD = 2400000.5


def _cache(cache_dir):
    return str(default_cache_dir()) if cache_dir is None else str(cache_dir)


def propagate_row(fit_row, target_epoch_mjd_tdb, cache_dir=None, return_stm=False):
    """Propagate one fit result to ``target_epoch_mjd_tdb``.

    Parameters
    ----------
    fit_row : numpy structured array
        One row of ``orbitfit`` output. Its covariance columns are carried if
        present; a row without them propagates the state alone.
    target_epoch_mjd_tdb : float
        Target epoch, MJD TDB -- the same units as the ``epochMJD_TDB`` column.
    cache_dir : str, optional
        Kernel cache. Defaults to Layup's.
    return_stm : bool
        Also return the 6x6 state-transition matrix.

    Returns
    -------
    numpy structured array
        A copy of ``fit_row`` at the new epoch. ⚠️ **Check the flag.** If the
        propagation fails -- most often a target outside the ephemeris
        coverage, 1550 to 2650 -- the row comes back with ``flag != 0`` at its
        ORIGINAL epoch and state, so a caller that ignores the flag gets an
        unchanged orbit rather than a plausible wrong one.
    """
    row = np.atleast_1d(fit_row).copy()
    if len(row) != 1:
        raise ValueError(f"propagate_row takes a single row, got {len(row)}")

    fit = parse_fit_result(row)
    out, stm = propagate_state(get_ephem(_cache(cache_dir)), fit, float(target_epoch_mjd_tdb) + MJD_TO_JD)

    if out.flag == 0:
        for i, c in enumerate(("x", "y", "z", "xdot", "ydot", "zdot")):
            if c in row.dtype.names:
                row[c] = out.state[i]
        if "epochMJD_TDB" in row.dtype.names:
            row["epochMJD_TDB"] = out.epoch - MJD_TO_JD
        cov = list(out.cov)
        for i in range(6):
            for j in range(6):
                name = f"cov_{i}_{j}"
                if name in row.dtype.names:
                    row[name] = cov[i * 6 + j]
    else:
        if "flag" in row.dtype.names:
            row["flag"] = out.flag
        logger.warning(
            "Propagation to MJD %.6f failed; returning the orbit unchanged at MJD %.6f. "
            "The most common cause is a target outside the ephemeris coverage (1550-2650).",
            float(target_epoch_mjd_tdb),
            float(row["epochMJD_TDB"][0]) if "epochMJD_TDB" in row.dtype.names else float("nan"),
        )

    if return_stm:
        return row, np.asarray(list(stm), dtype=float).reshape(6, 6)
    return row


def propagate(fit_results, target_epoch_mjd_tdb, cache_dir=None):
    """Propagate every row of ``fit_results`` to a common epoch.

    The point of a *common* epoch: two orbits cannot be compared, differenced or
    averaged unless they are referenced to the same time. Differencing states at
    epochs that differ by even a minute is a mistake that leaves no trace in the
    output -- 69 seconds is about 1,000 km for a main-belt object and 2,000 km
    for a fast near-Earth one.

    Rows that fail to propagate are returned unchanged with ``flag != 0``; they
    are counted in a warning rather than dropped, so the output has the same
    length and order as the input.
    """
    data = np.atleast_1d(fit_results)
    out = data.copy()
    failed = 0
    for i in range(len(data)):
        row = propagate_row(data[i : i + 1], target_epoch_mjd_tdb, cache_dir=cache_dir)
        out[i] = row[0]
        if "flag" in out.dtype.names and out[i]["flag"] != 0:
            failed += 1
    if failed:
        logger.warning(
            "%d of %d orbits could not be propagated to MJD %.6f and are unchanged.",
            failed,
            len(data),
            float(target_epoch_mjd_tdb),
        )
    return out
