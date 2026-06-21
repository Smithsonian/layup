"""Pluggable initial orbit determination (IOD) layer for layup.

An IOD method is a callable that proposes one or more seed orbits for
the Marquardt fitter to refine. The expected signature is

    iod(observations, seq) -> list[FitResult] | None

where `observations` is the time-ordered list of layup `Observation`s,
`seq` is the per-segment index list (`seq[0]` is the primary segment
used for the IOD), and the return is either a list of candidate seed
orbits (each a `FitResult` with at least `state` and `epoch`
populated) or `None` / empty if no candidate could be produced.

Multiple IOD candidates are returned when the underlying method is
multi-valued (Gauss's polynomial in r₂ has up to eight real roots);
`do_fit` runs LM from each candidate and picks the best converged fit
(smallest χ² subject to a sanity bound on heliocentric distance).

Methods register themselves at import time via the module-level
registry; use `register_iod(name, callable)` to add new methods,
`get_iod(name)` to look one up, and `iod_methods()` to list all
available names.

Why a registry instead of subclassing: IOD methods are stateless
strategies whose entire interface fits on one line. A function-pointer
registry is the smallest abstraction that supports drop-in
replacements (e.g. a Lambert-based method, a motion-rate prior, a
prelim from BK's tangent-plane linear fit) without forcing each
implementation through a class hierarchy.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Sequence

from layup.routines import FitResult, Observation, gauss

logger = logging.getLogger(__name__)

GMtotal = 0.0002963092748799319
AU_M = 149597870700
SPEED_OF_LIGHT = 2.99792458e8 * 86400.0 / AU_M

# Default bounds for the cheap physical-feasibility filter. We
# deliberately *don't* check bound-orbit energy here: Gauss's velocity
# component can be wildly wrong (5-10× circular) for a seed whose
# *position* is correct, and LM walks those into convergence reliably.
# Throwing them out would silently kill the right root in cases like
# the 41 AU classical KBO where Gauss returns a hyperbolic-looking
# velocity but the right geometry.
_MIN_R_AU = 0.05  # interior to Mercury — almost certainly unphysical
_MAX_R_AU = 1000.0  # well past the Kuiper-belt range we typically care about

# Geocentric distance below which we treat a candidate as a close-Earth-
# approach risk: full ASSIST integration will get stuck on the close
# encounter (and a 2-body workaround would silently mishandle the real
# physics of NEO close passes — those exist and need their own solution
# eventually). For now the prefilter just passes such candidates through
# to LM unchanged; LM may also be slow on them, which is a known issue.
_CLOSE_EARTH_AU = 0.1


# An IOD method takes (observations, seq) and returns either a list of
# candidate seed orbits or None.
IODCallable = Callable[[Sequence[Observation], Sequence[Sequence[int]]], Optional[list]]


# Module-level registry, name -> callable. Populated below.
_REGISTRY: dict[str, IODCallable] = {}


def register_iod(name: str, func: IODCallable) -> None:
    """Register an IOD method under `name`. Overwrites an existing entry."""
    _REGISTRY[name.lower()] = func


def get_iod(name: str) -> IODCallable:
    """Look up an IOD method by name. Raises ValueError if unknown."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown IOD method {name!r}. " f"Registered methods: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def iod_methods() -> list[str]:
    """Return the sorted list of registered IOD method names."""
    return sorted(_REGISTRY)


# ----------------------------------------------------------------------- #
# Built-in: Gauss's method.                                               #
# ----------------------------------------------------------------------- #


def gauss_iod(observations, seq):
    """Gauss's method on the first/middle/last observation of seq[0].

    The C++ `gauss` binding returns up to eight candidate seed orbits
    (corresponding to the real roots of the 8th-degree polynomial in
    r₂); we pass them all upstream so the picker can pick the right
    one rather than committing to `solns[0]` blindly.
    """
    idx0 = seq[0][0]
    idx1 = seq[0][len(seq[0]) // 2]
    idx2 = seq[0][-1]
    logger.debug(f"gauss_iod: indices {idx0}, {idx1}, {idx2}")
    solns = gauss(GMtotal, observations[idx0], observations[idx1], observations[idx2], 0.0001, SPEED_OF_LIGHT)
    return solns


register_iod("gauss", gauss_iod)


# ----------------------------------------------------------------------- #
# Candidate filter (held-out angular residual).                           #
# ----------------------------------------------------------------------- #


def _passes_physical_bounds(candidate, min_r_au: float = _MIN_R_AU, max_r_au: float = _MAX_R_AU) -> bool:
    """Cheap algebraic feasibility check on an IOD candidate state.

    Rejects candidates with non-positive r² or |r| outside [min, max]
    AU. Deliberately does *not* reject hyperbolic-looking velocities:
    Gauss's velocity can be wildly wrong even for the correct
    geometric root, and LM walks those to convergence routinely.
    """
    sx, sy, sz = candidate.state[0], candidate.state[1], candidate.state[2]
    r2 = sx * sx + sy * sy + sz * sz
    if r2 <= 0.0:
        return False
    r = math.sqrt(r2)
    if r < min_r_au or r > max_r_au:
        return False
    return True


def _predict_rho_hat(ephem, state, state_epoch, obs):
    """Propagate `state` to `obs.epoch` via full ASSIST and return the
    predicted apparent unit direction (no light-time correction; coarse
    filter only).
    """
    import rebound, assist
    import numpy as np

    sim = rebound.Simulation()
    sim.t = float(state_epoch) - ephem.jd_ref
    sim.add(x=state[0], y=state[1], z=state[2], vx=state[3], vy=state[4], vz=state[5])
    extras = assist.Extras(sim, ephem)
    extras.integrate_or_interpolate(float(obs.epoch) - ephem.jd_ref)
    p = sim.particles[0]
    rx = p.x - obs.observer_position[0]
    ry = p.y - obs.observer_position[1]
    rz = p.z - obs.observer_position[2]
    rho = math.sqrt(rx * rx + ry * ry + rz * rz)
    return np.array([rx / rho, ry / rho, rz / rho])


def _inertial_min_geocentric_AU(state, state_epoch, observations) -> float:
    """Smallest |candidate - observer| over the observation arc, treating
    candidate motion as inertial (position + velocity·Δt).

    Used to detect candidates whose trajectory passes close to Earth (or
    the ground observer); full ASSIST integration would then spend most
    of its time resolving the close encounter. Inertial-extrapolation is
    OK for the detection (we just need an order of magnitude); the actual
    close approach with gravity could be different.
    """
    min_d2 = float("inf")
    sx, sy, sz, vx, vy, vz = (state[i] for i in range(6))
    for obs in observations:
        dt = float(obs.epoch) - float(state_epoch)
        px = sx + vx * dt
        py = sy + vy * dt
        pz = sz + vz * dt
        ox, oy, oz = obs.observer_position
        dx = px - ox
        dy = py - oy
        dz = pz - oz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < min_d2:
            min_d2 = d2
    return math.sqrt(min_d2)


def filter_candidates_by_residual(
    candidates,
    observations,
    ephem,
    threshold_sigma: float = 1000.0,
    min_obs_for_filter: int = 4,
    close_earth_AU: float = _CLOSE_EARTH_AU,
):
    """Drop IOD candidates whose predicted positions miss the observations
    by more than `threshold_sigma` times the per-axis astrometric σ.

    The right Gauss root predicts the observations within a few σ; phantom
    roots are typically off by 10⁵-10⁶ σ. A loose threshold (1000σ
    default) keeps the right root in every realistic case while throwing
    out the obviously-wrong ones before LM ever runs on them.

    Candidates whose inertial trajectory passes within `close_earth_AU`
    of the observer at any obs time are passed through unfiltered. Full
    ASSIST integration gets stuck on close Earth encounters (tens of
    seconds per propagation), and replacing it with a 2-body
    approximation would silently mishandle the real physics of NEO close
    passes — those are valid science targets that need a different
    solution. Until that solution exists, we just skip the filter for
    such candidates and let LM handle them (slowly, on the same close
    encounters, but that's a separate known issue).

    Parameters
    ----------
    candidates : list[FitResult]
        Output of an IOD method (states + epoch filled in).
    observations : sequence[Observation]
        Full observation list; we evaluate the candidate against every
        one of them.
    ephem : assist.Ephem
        The Python ASSIST ephemeris handle (e.g.
        `assist.Ephem(planets_path, sb_path)`). Not the C struct from
        layup.routines.get_ephem.
    threshold_sigma : float
        Reject candidates whose angular residual exceeds this multiple
        of the per-observation σ on any observation.
    min_obs_for_filter : int
        Bypass the filter (return all candidates that pass the physical
        bounds) when there are fewer than this many observations.
    close_earth_AU : float
        Pass-through threshold for the close-Earth-approach check
        described above.

    Returns
    -------
    list[FitResult]
        Filtered list. If every candidate fails the residual test, the
        list of physical-bound-passing candidates is returned instead —
        we'd rather hand a bad seed to LM than no seed at all.
    """
    import numpy as np

    physical = [c for c in candidates if _passes_physical_bounds(c)]
    if not physical:
        # Nothing passes even the cheap test — surface the original
        # list so the picker decides.
        return list(candidates)

    if len(observations) < min_obs_for_filter:
        # Not enough leverage; rely on physical bounds only.
        return physical

    survivors = []
    best = None  # (max_resid_sigma, candidate) over residual-evaluated candidates
    for c in physical:
        # Close-Earth-approach pass-through: avoid both the integrator
        # blowup and a silently-wrong 2-body shortcut.
        min_geo = _inertial_min_geocentric_AU(c.state, c.epoch, observations)
        if min_geo < close_earth_AU:
            survivors.append(c)
            continue

        max_resid_sigma = 0.0
        integrable = True
        for obs in observations:
            try:
                pred = _predict_rho_hat(ephem, c.state, c.epoch, obs)
            except Exception:
                # ASSIST refused to integrate (e.g. state walked outside
                # the kernel time range). Treat as a failed candidate.
                integrable = False
                break
            actual = np.asarray(obs.rho_hat).flatten()
            cos_sep = float(np.clip(pred @ actual, -1.0, 1.0))
            sep_rad = math.acos(cos_sep)
            sigma_ra = float(obs.ra_unc if obs.ra_unc is not None else 1.0 / 206265)
            sigma_dec = float(obs.dec_unc if obs.dec_unc is not None else 1.0 / 206265)
            sigma = max(sigma_ra, sigma_dec)
            max_resid_sigma = max(max_resid_sigma, sep_rad / sigma)

        if not integrable:
            continue
        # Track the single best-fitting integrable candidate so we can
        # guarantee it is never discarded (see the invariant below).
        if best is None or max_resid_sigma < best[0]:
            best = (max_resid_sigma, c)
        if max_resid_sigma <= threshold_sigma:
            survivors.append(c)

    # Correctness invariant: the prefilter is a performance optimization
    # (skip LM on obvious garbage) and must never discard the single
    # best-fitting seed. A *valid* Gauss root can have a large raw-seed
    # residual — a rough 3-point seed propagated across a long arc drifts
    # well past the threshold (e.g. ~9500σ for a 52-day main-belt arc)
    # even though LM converges cleanly from it. Without this guard the
    # threshold silently rejects the right root and leaves only phantoms.
    if best is not None and not any(c is best[1] for c in survivors):
        survivors.append(best[1])

    if not survivors:
        # Don't reject everything — fall back to physically-OK candidates.
        return physical
    return survivors
