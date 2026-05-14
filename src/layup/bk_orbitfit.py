"""Adapter from layup's Observation/FitResult types to the liborbfit
Python wrapper (https://github.com/matthewholman/liborbfit).

The heavy lifting — ctypes bindings, the αβγ→Cartesian transform,
and the do_bk_fit entry point — lives in liborbfit's own repo at
`python/liborbfit/bk_orbitfit.py`.  This file is a thin adapter:

  * `_build_obs(...)` converts a sequence of layup Observation
    objects into liborbfit's Observation dataclass.
  * `_to_layup_result(...)` packs a liborbfit BKFitResult into a
    layup FitResult (constructing the Cartesian state from BK via
    `BKFitResult.to_cartesian()`).
  * `set_ephem`, `do_bk_fit`, `compute_r_au` are the public entry
    points consumed by orbitfit.py; their surface is unchanged.

To use, clone and build liborbfit (`./setup_deps.sh && make` in the
liborbfit repo), then export two env vars so layup can load both the
C library (via ctypes) and the Python wrapper:

    export LIBORBFIT_PATH=/path/to/liborbfit/liborbfit.so
    export PYTHONPATH=/path/to/liborbfit/python:$PYTHONPATH

When the env vars are not set the BK engine is unavailable; layup's
auto-dispatch falls back to the Cartesian engine and explicit
`engine="bk"` calls return a FitResult with flag=3.

Failure modes are surfaced through FitResult.flag:
  0 = success
  1 = liborbfit.do_bk_fit returned a non-zero rc (singular, too few
      obs, etc. — see orbfit_lib.h)
  2 = ephemeris not loaded
  3 = liborbfit module unavailable (not on PYTHONPATH, library
      missing, ABI symbol mismatch, …)
"""

from __future__ import annotations

import os
from typing import Sequence, Any

import numpy as np

# liborbfit (the standalone Python wrapper around liborbfit.so).
# Imported lazily so this module remains importable even on
# machines without the BK build; callers see the failure when they
# try to use the API.
try:
    from liborbfit import (
        Observation as _LiborbfitObs,
        set_ephem as _set_ephem,
        do_bk_fit as _do_bk_fit,
    )
    _LIBORBFIT_OK = True
except ImportError:  # pragma: no cover — let callers see the failure
    _LiborbfitObs = None  # type: ignore
    _set_ephem    = None  # type: ignore
    _do_bk_fit    = None  # type: ignore
    _LIBORBFIT_OK = False

# layup's pybind11-bound FitResult, the same type returned by the
# Cartesian-LM path so orbitfit.do_fit can hand it to its caller
# without knowing which engine produced it.
try:
    from _layup_cpp._core import FitResult  # type: ignore
except ImportError:  # pragma: no cover
    FitResult = None  # type: ignore


ARCSEC = 1.0 / 206265.0


# --------------------------------------------------------------------------- #
# Public entry points                                                         #
# --------------------------------------------------------------------------- #

def set_ephem(planets_path: str | os.PathLike,
              asteroids_path: str | os.PathLike) -> None:
    """Load the ASSIST kernels once per process; pass-through to
    liborbfit.set_ephem."""
    if not _LIBORBFIT_OK:
        raise RuntimeError(
            "liborbfit Python package not importable; install it and "
            "ensure both LIBORBFIT_PATH and PYTHONPATH point at the build.")
    _set_ephem(planets_path, asteroids_path)


def do_bk_fit(observations: Sequence[Any]) -> "FitResult":
    """Fit a sequence of layup Observations with BK and return a
    layup FitResult.  See module docstring for the FitResult.flag
    convention."""
    if FitResult is None:
        raise RuntimeError(
            "_layup_cpp._core.FitResult is not importable; build layup first.")
    if not _LIBORBFIT_OK:
        return _failed_result(flag=3)
    obs = _build_obs(observations)
    r = _do_bk_fit(obs)
    return _to_layup_result(r)


def compute_r_au(observations: Sequence[Any]) -> float | None:
    """Cheap heliocentric-distance estimate via Gauss IOD on the
    first / middle / last obs.  Used by orbitfit.do_fit's auto
    dispatch (route to BK when r ≥ threshold)."""
    try:
        from _layup_cpp._core import gauss as _gauss  # type: ignore
    except ImportError:
        return None
    if len(observations) < 3:
        return None
    o0, o1, o2 = (observations[0],
                  observations[len(observations) // 2],
                  observations[-1])
    try:
        solns = _gauss(o0, o1, o2)
    except Exception:
        return None
    if not solns:
        return None
    state = np.array(solns[0].state)
    return float(np.linalg.norm(state[:3]))


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #

def _radec_from_rho(rho: np.ndarray) -> tuple[float, float]:
    """Unit direction vector (ICRS-equatorial) -> (ra, dec) radians."""
    rx, ry, rz = rho[0], rho[1], rho[2]
    ra = np.arctan2(ry, rx) % (2 * np.pi)
    dec = np.arctan2(rz, np.hypot(rx, ry))
    return float(ra), float(dec)


def _build_obs(observations: Sequence[Any]) -> list:
    """Convert layup Observation list to liborbfit Observation list."""
    out = []
    for o in observations:
        rho = (o.rho_hat if isinstance(o.rho_hat, np.ndarray)
               else np.asarray(o.rho_hat))
        ra, dec = _radec_from_rho(rho.reshape(-1))
        out.append(_LiborbfitObs(
            jd_tdb=float(o.epoch),
            ra=ra,
            dec=dec,
            sigma_ra =float(o.ra_unc  if o.ra_unc  is not None else ARCSEC),
            sigma_dec=float(o.dec_unc if o.dec_unc is not None else ARCSEC),
            xe=float(o.observer_position[0]),
            ye=float(o.observer_position[1]),
            ze=float(o.observer_position[2]),
        ))
    return out


def _failed_result(flag: int) -> "FitResult":
    res = FitResult()
    res.method = "BK"
    res.niter  = 0
    res.flag   = flag
    res.csq    = float("nan")
    res.ndof   = 0
    res.epoch  = 0.0
    res.state  = [0.0] * 6
    res.cov    = [0.0] * 36
    return res


def _to_layup_result(r) -> "FitResult":
    """Pack a liborbfit BKFitResult into a layup FitResult."""
    res = FitResult()
    res.method = "BK"
    res.niter  = 0  # liborbfit doesn't currently report iteration count

    if r.flag != 0:
        res.flag  = 1
        res.csq   = float("nan")
        res.ndof  = 0
        res.epoch = 0.0
        res.state = [0.0] * 6
        res.cov   = [0.0] * 36
        return res

    state, _J, cov_cart = r.to_cartesian()
    res.flag  = 0
    res.csq   = float(r.chisq)
    res.ndof  = int(r.dof)
    res.epoch = float(r.jd0)
    res.state = [float(x) for x in state]
    res.cov   = [float(x) for x in cov_cart.reshape(-1)]
    return res
