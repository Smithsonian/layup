"""Bernstein-Khushalani orbit-fit path for layup.

Wraps the C library `liborbfit.so` (built in ~/Dropbox/claude_tests; see
create_lib_links.sh for the symlink). BK is well-suited to fitting
distant solar-system bodies from short-arc astrometry — it
parameterizes in a tangent-plane frame so the linearized geometry is
much less degenerate than barycentric Cartesian, and as of 2026-05-11
the library uses variational particles for the Jacobian by default, so
the gravity-curvature-vs-params dependence Bernstein originally dropped
is now picked up properly.

Public entry points:
  - `set_ephem(planets_path, asteroids_path)`: load the ASSIST kernels
    once per process. Optional but recommended; subsequent fits reuse
    the cached ephem.
  - `do_bk_fit(observations) -> FitResult`: fit a sequence of layup
    Observation objects via BK. Returns a FitResult-compatible object
    whose state is barycentric ICRS-equatorial Cartesian at the fit's
    chosen reference epoch, matching the convention of the Cartesian-
    LM path in orbitfit.py.
  - `compute_r_au(observations)`: r estimate from Gauss IOD on the
    first/middle/last observations. Cheap. Use as a dispatch criterion
    (route to BK when r exceeds a threshold).

Failure modes are surfaced through FitResult.flag:
  0 = success
  1 = orbfit_fit returned a non-zero rc (singular, too few obs, etc.)
  2 = ephemeris not set / could not be loaded
  3 = library missing or symbols not exported
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
from typing import Sequence, Any

import numpy as np

# FitResult class is provided by the pybind11 core (orbit_fit::FitResult).
# We construct one and populate its fields with our BK results.
try:
    from _layup_cpp._core import FitResult  # type: ignore
except ImportError:  # pragma: no cover — let callers see the error
    FitResult = None  # type: ignore


# --------------------------------------------------------------------------- #
# Library load                                                                #
# --------------------------------------------------------------------------- #

_lib: C.CDLL | None = None
_ephem_loaded: bool = False

ARCSEC = 1.0 / 206265.0
ECL_RAD = (84381.448 / 3600.0) * np.pi / 180.0  # J2000 obliquity (matches orbfit.h)
KM_PER_AU = 149597870.7


def _candidate_lib_paths() -> list[Path]:
    """Search order for liborbfit.so."""
    env = os.environ.get("LIBORBFIT_PATH")
    if env:
        yield_env = [Path(env)]
    else:
        yield_env = []
    here = Path(__file__).resolve().parent
    return [
        *yield_env,
        # layup repo-root symlink (create_lib_links.sh)
        here.parent.parent / "liborbfit.so",
        here.parent / "liborbfit.so",
        # Direct path to the build out of claude_tests.
        Path.home() / "Dropbox" / "claude_tests" / "liborbfit.so",
    ]


class ObsInput(C.Structure):
    _fields_ = [
        ("jd_tdb",    C.c_double),
        ("ra",        C.c_double),
        ("dec",       C.c_double),
        ("sigma_ra",  C.c_double),
        ("sigma_dec", C.c_double),
        ("xe",        C.c_double),
        ("ye",        C.c_double),
        ("ze",        C.c_double),
    ]


class _OrbitFit(C.Structure):
    """ctypes mirror of struct OrbitFit (orbfit_lib.h)."""
    _fields_ = [
        ("a",     C.c_double), ("adot",  C.c_double),
        ("b",     C.c_double), ("bdot",  C.c_double),
        ("g",     C.c_double), ("gdot",  C.c_double),
        ("covar", C.c_double * 36),
        ("jd0",   C.c_double),
        ("lat0",  C.c_double), ("lon0",  C.c_double),
        ("xBary", C.c_double), ("yBary", C.c_double), ("zBary", C.c_double),
        ("chisq", C.c_double),
        ("dof",   C.c_int),
        ("fitparms", C.c_int),
    ]


def _load_lib() -> C.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    last_err: OSError | None = None
    for p in _candidate_lib_paths():
        if not p.exists():
            continue
        try:
            _lib = C.CDLL(str(p))
            break
        except OSError as e:
            last_err = e
            continue
    if _lib is None:
        raise OSError(
            "Could not load liborbfit.so. Set LIBORBFIT_PATH or run "
            "layup/create_lib_links.sh."
            + (f" Last error: {last_err}" if last_err else "")
        )

    _lib.orbfit_set_ephem.argtypes = [C.c_char_p, C.c_char_p]
    _lib.orbfit_set_ephem.restype  = C.c_int
    _lib.orbfit_fit.argtypes = [
        C.POINTER(ObsInput), C.c_int,
        C.c_char_p, C.c_char_p,
        C.POINTER(_OrbitFit),
    ]
    _lib.orbfit_fit.restype = C.c_int
    # Jacobian mode is now variational by default in the library; setters
    # exposed for callers that need to override.
    if hasattr(_lib, "kbo3d_assist_set_jacobian_mode"):
        _lib.kbo3d_assist_set_jacobian_mode.argtypes = [C.c_int]
        _lib.kbo3d_assist_set_jacobian_mode.restype  = None
    return _lib


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def set_ephem(planets_path: str | os.PathLike,
              asteroids_path: str | os.PathLike) -> None:
    """Load ASSIST kernels once for the process. Idempotent."""
    global _ephem_loaded
    lib = _load_lib()
    rc = lib.orbfit_set_ephem(str(planets_path).encode(),
                              str(asteroids_path).encode())
    if rc != 0:
        raise RuntimeError(f"orbfit_set_ephem returned {rc} for "
                           f"{planets_path!s}, {asteroids_path!s}")
    _ephem_loaded = True


def _radec_from_rho(rho: np.ndarray) -> tuple[float, float]:
    """Unit direction vector (ICRS-equatorial) -> (ra, dec) in radians."""
    rx, ry, rz = rho[0], rho[1], rho[2]
    ra = np.arctan2(ry, rx) % (2 * np.pi)
    dec = np.arctan2(rz, np.hypot(rx, ry))
    return float(ra), float(dec)


def _build_obs_array(observations: Sequence[Any]) -> tuple[Any, int]:
    """Translate layup Observation objects to a ctypes ObsInput array."""
    n = len(observations)
    arr = (ObsInput * n)()
    for i, o in enumerate(observations):
        rho = np.asarray(o.rho_hat) if not isinstance(o.rho_hat, np.ndarray) \
              else o.rho_hat
        ra, dec = _radec_from_rho(rho.reshape(-1))
        # Observation.ra_unc / dec_unc are already in radians (e.g.
        # Observation.from_astrometry stores 1.0/206265). Pass straight through.
        sigma_ra  = float(o.ra_unc  if o.ra_unc  is not None else ARCSEC)
        sigma_dec = float(o.dec_unc if o.dec_unc is not None else ARCSEC)
        arr[i].jd_tdb    = float(o.epoch)  # caller is responsible for TDB
        arr[i].ra        = ra
        arr[i].dec       = dec
        arr[i].sigma_ra  = sigma_ra
        arr[i].sigma_dec = sigma_dec
        arr[i].xe        = float(o.observer_position[0])
        arr[i].ye        = float(o.observer_position[1])
        arr[i].ze        = float(o.observer_position[2])
    return arr, n


# --- BK -> Cartesian (state + Jacobian) ----------------------------------- #
# Mirror of pbasis_to_bary_eq in orbfit1.c so the 6x6 cov can be transformed
# without adding a new C export. Pure rotations; verified against bk_test.py.

def _proj_to_ec(xp: np.ndarray, yp: np.ndarray, zp: np.ndarray,
                lat0: float, lon0: float):
    slat, clat = np.sin(lat0), np.cos(lat0)
    slon, clon = np.sin(lon0), np.cos(lon0)
    x_ec = -slon * xp - clon * slat * yp + clon * clat * zp
    y_ec =  clon * xp - slon * slat * yp + slon * clat * zp
    z_ec =              clat * yp        + slat * zp
    return x_ec, y_ec, z_ec


def _R_ec_from_proj(lat0: float, lon0: float) -> np.ndarray:
    slat, clat = np.sin(lat0), np.cos(lat0)
    slon, clon = np.sin(lon0), np.cos(lon0)
    return np.array([
        [-slon, -clon * slat, clon * clat],
        [ clon, -slon * slat, slon * clat],
        [ 0.0,         clat,        slat ],
    ])


def _R_eq_from_ec() -> np.ndarray:
    se, ce = np.sin(ECL_RAD), np.cos(ECL_RAD)
    # orbfit's xyz_eq_to_ec is the inverse of this. ec->eq = transpose.
    return np.array([
        [1.0,  0.0,  0.0],
        [0.0,  ce,  -se ],
        [0.0,  se,   ce ],
    ])


def _bk_to_cartesian(orb: _OrbitFit) -> tuple[np.ndarray, np.ndarray]:
    """Returns (state_eq[6], J[6,6]) where J = d(state_eq)/d(BK)."""
    g = orb.g
    z0 = 1.0 / g

    # Projection-frame state.
    x_p = orb.a * z0;   y_p = orb.b * z0;   z_p = z0
    vx_p = orb.adot * z0; vy_p = orb.bdot * z0; vz_p = orb.gdot * z0

    # Subtract barycenter shift (position only).
    x_p -= orb.xBary; y_p -= orb.yBary; z_p -= orb.zBary

    # 6x6 dProj/dBK (row = proj component, col = BK param 0..5 = a,adot,b,bdot,g,gdot).
    # Layout matches orbfit1.c pbasis_to_bary_eq's dProj_dp (after 1-indexed -> 0).
    dProj_dp = np.zeros((6, 6))
    dProj_dp[0, 0] = z0                          # d x_p / d a
    dProj_dp[0, 4] = -orb.a    * z0 * z0         # d x_p / d g
    dProj_dp[1, 2] = z0                          # d y_p / d b
    dProj_dp[1, 4] = -orb.b    * z0 * z0         # d y_p / d g
    dProj_dp[2, 4] = -z0 * z0                    # d z_p / d g
    dProj_dp[3, 1] = z0                          # d vx_p / d adot
    dProj_dp[3, 4] = -orb.adot * z0 * z0         # d vx_p / d g
    dProj_dp[4, 3] = z0                          # d vy_p / d bdot
    dProj_dp[4, 4] = -orb.bdot * z0 * z0         # d vy_p / d g
    dProj_dp[5, 5] = z0                          # d vz_p / d gdot
    dProj_dp[5, 4] = -orb.gdot * z0 * z0         # d vz_p / d g

    R_ec_proj = _R_ec_from_proj(orb.lat0, orb.lon0)
    R_eq_ec   = _R_eq_from_ec()
    R         = R_eq_ec @ R_ec_proj              # 3x3

    # Apply R to position and velocity rows separately; the proj->eq
    # transform is block-diag(R, R).
    R6 = np.zeros((6, 6))
    R6[:3, :3] = R
    R6[3:, 3:] = R
    J = R6 @ dProj_dp

    # Rotate the projection-frame state to barycentric ICRS-equatorial.
    state_proj = np.array([x_p, y_p, z_p, vx_p, vy_p, vz_p])
    state_eq = np.empty(6)
    state_eq[:3] = R @ state_proj[:3]
    state_eq[3:] = R @ state_proj[3:]
    return state_eq, J


def _build_fit_result(orb: _OrbitFit, rc: int, n_obs: int) -> "FitResult":
    """Pack a liborbfit OrbitFit into a layup FitResult."""
    if FitResult is None:
        raise RuntimeError("layup FitResult binding not importable")
    res = FitResult()
    res.method = "BK"
    res.niter  = 0  # liborbfit doesn't report iter count today
    if rc != 0:
        res.flag = 1
        res.csq  = float("nan")
        res.ndof = 0
        res.epoch = 0.0
        res.state = [0.0] * 6
        res.cov   = [0.0] * 36
        return res

    state_eq, J = _bk_to_cartesian(orb)
    cov_bk = np.array(list(orb.covar)).reshape(6, 6)
    cov_cart = J @ cov_bk @ J.T

    res.flag  = 0
    res.csq   = float(orb.chisq)
    res.ndof  = int(orb.dof)
    res.epoch = float(orb.jd0)
    res.state = [float(x) for x in state_eq]
    res.cov   = [float(x) for x in cov_cart.reshape(-1)]
    return res


def do_bk_fit(observations: Sequence[Any]) -> "FitResult":
    """Fit `observations` with BK. Returns a layup FitResult.

    The library must already have an ephem loaded (call set_ephem once
    per process before fitting).
    """
    if FitResult is None:
        raise RuntimeError(
            "_layup_cpp._core.FitResult is not importable; build layup first.")
    lib = _load_lib()
    if not _ephem_loaded:
        raise RuntimeError(
            "ephem not loaded; call bk_orbitfit.set_ephem(...) first.")

    obs_arr, n = _build_obs_array(observations)
    out = _OrbitFit()
    rc = lib.orbfit_fit(obs_arr, n, None, None, C.byref(out))
    return _build_fit_result(out, rc, n)


def compute_r_au(observations: Sequence[Any]) -> float | None:
    """Cheap r estimate via Gauss IOD on first/middle/last obs.

    Returns the heliocentric distance in AU implied by the chosen Gauss
    root, or None if Gauss failed. Used by orbitfit.do_fit's dispatch
    decision: r above the threshold -> route to BK.
    """
    # We just delegate to layup's Gauss bindings if available — this avoids
    # duplicating the polynomial-root machinery here. Imported lazily so
    # bk_orbitfit.py is usable in environments where the C++ core isn't built.
    try:
        from _layup_cpp._core import gauss as _gauss  # type: ignore
    except ImportError:
        return None

    if len(observations) < 3:
        return None
    o0, o1, o2 = observations[0], observations[len(observations) // 2], \
                 observations[-1]
    try:
        solns = _gauss(o0, o1, o2)
    except Exception:
        return None
    if not solns:
        return None
    state = np.array(solns[0].state)
    return float(np.linalg.norm(state[:3]))
