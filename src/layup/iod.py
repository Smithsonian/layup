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

import bisect
import logging
import math
from typing import Callable, Optional, Sequence

from layup.constants import GMtotal, SPEED_OF_LIGHT
from layup.routines import FitResult, Observation, gauss, get_ephem
from layup.utilities.herget_iod import herget_with_assist

from layup.orbit_maths import build_ephem_and_mus

logger = logging.getLogger(__name__)

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


# Gauss triplet selection (issue #509).
#
# Gauss truncates the Lagrange f and g series, so the interval must be short
# against the orbital period -- classically well under 60 degrees of mean
# anomaly between the outer two observations. Taking the first, middle and last
# observation of seq[0] ignores that, and seq[0] is by construction the
# LONGEST-span chunk, so on a long arc the triplet is as wide as it can be.
# Measured on a 39-object flag-3 residue, those objects span a median 53 deg of
# mean anomaly against 29 deg for objects that fit.
#
# Choosing the triplet to sit near a target span instead converges 34 of those
# 39. The target is in mean anomaly, which needs a period, which needs the orbit
# we are trying to find -- so it is converted to days using an ASSUMED semimajor
# axis rather than the object's own, which keeps it a prior and not an oracle.
# Using each object's published a is slightly WORSE (28/39), so there is no
# chicken-and-egg problem here.
_GAUSS_TARGET_A_AU = 2.5  # assumed a for the period; main belt
_GAUSS_TARGET_DEG = 15.0  # target outer span, degrees of mean anomaly at that a
_GAUSS_MIN_BALANCE = 0.10  # each sub-interval, as a fraction of the outer span


def _gauss_target_days(a_au=_GAUSS_TARGET_A_AU, deg=_GAUSS_TARGET_DEG):
    """Target outer span in days: `deg` of mean anomaly at an assumed `a_au`."""
    return 365.25 * a_au**1.5 * deg / 360.0


def _select_gauss_triplet(epochs, idx0, target_days=None):
    """Indices of the triplet from `idx0` whose outer span is nearest the target.

    `epochs` is indexable by the entries of `idx0`. Returns a (first, middle,
    last) tuple, or None when no triplet satisfies the balance guard.

    The middle observation must be genuinely between the other two, not merely
    indexed between them: on a sparse chunk several observations can share an
    epoch, and a midpoint that coincides with an endpoint gives a zero-length
    sub-interval and no usable root. `_GAUSS_MIN_BALANCE` enforces that.
    """
    if target_days is None:
        target_days = _gauss_target_days()
    n = len(idx0)
    if n < 3:
        return None
    t = [float(epochs[i]) for i in idx0]
    # Nothing to shorten. When the whole segment is already inside the target
    # span the widest triplet is the best available, which is what
    # first/middle/last takes anyway -- so this selection could only change the
    # MIDDLE observation, which is not what it is for. On a short arc that is
    # pure downside: the 3I/ATLAS fixture spans 19 days against a ~60 day
    # target, both selections take the same outer pair, and moving the middle
    # point alone shifted the epoch by 8 days and cost 1% in position on a
    # weakly-constrained hyperbolic orbit. Defer to the caller's fallback.
    if t[-1] - t[0] <= target_days:
        return None
    best, cost = None, float("inf")
    for a in range(n - 2):
        # Nearest outer partner to the target span. t is time-ordered within a
        # chunk, so a bisect finds it; check the neighbour on each side too.
        lo = bisect.bisect_left(t, t[a] + target_days, a + 2, n)
        for c in (lo - 1, lo, lo + 1):
            if c <= a + 1 or c >= n:
                continue
            span = t[c] - t[a]
            if span <= 0:
                continue
            this = abs(span - target_days)
            if this >= cost:
                continue
            floor = _GAUSS_MIN_BALANCE * span
            mid = 0.5 * (t[a] + t[c])
            # most central observation that keeps both gaps substantial
            j = bisect.bisect_left(t, mid, a + 1, c)
            pick = None
            for b in sorted(range(a + 1, c), key=lambda k: (abs(k - j), k)):
                if min(t[b] - t[a], t[c] - t[b]) >= floor:
                    pick = b
                    break
            if pick is None:
                continue
            best, cost = (idx0[a], idx0[pick], idx0[c]), this
    return best


def gauss_iod(observations, seq):
    """Gauss's method on a span-targeted triplet drawn from seq[0].

    The C++ `gauss` binding returns up to eight candidate seed orbits
    (corresponding to the real roots of the 8th-degree polynomial in
    r₂); we pass them all upstream so the picker can pick the right
    one rather than committing to `solns[0]` blindly.
    """
    idx0 = list(seq[0])
    trip = _select_gauss_triplet([o.epoch for o in observations], idx0)
    if trip is None:
        # Degenerate chunk (fewer than three observations, or every candidate
        # middle collapses against an endpoint). Fall back to the original
        # first/middle/last so behaviour is never worse than before.
        trip = (idx0[0], idx0[len(idx0) // 2], idx0[-1])
        logger.debug("gauss_iod: span selection found no triplet; using first/middle/last")
    idx0_, idx1, idx2 = trip
    logger.debug(f"gauss_iod: indices {idx0_}, {idx1}, {idx2}")
    solns = gauss(
        GMtotal, observations[idx0_], observations[idx1], observations[idx2], 0.0001, SPEED_OF_LIGHT
    )
    return solns


register_iod("gauss", gauss_iod)


def herget_iod(observations, seq):
    """"""
    ephem, _, _ = build_ephem_and_mus()
    solns = herget_with_assist(observations, seq, ephem, tolerance=0.0001, max_iterations=100)
    return solns


register_iod("herget", herget_iod)


# ----------------------------------------------------------------------- #
# Candidate filter (held-out angular residual).                           #
# ----------------------------------------------------------------------- #


def _passes_physical_bounds(candidate, min_r_au: float = _MIN_R_AU, max_r_au: float = _MAX_R_AU) -> bool:
    """Cheap algebraic feasibility check on an IOD candidate state.

    Rejects candidates with non-positive r² or ``|r|`` outside [min, max]
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
    """Smallest ``|candidate - observer|`` over the observation arc, treating
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


def partition_close_approach(observations, candidates, close_earth_AU: float = _CLOSE_EARTH_AU):
    """Split IOD candidates into those safe to integrate and those to defer.

    A Gauss root whose trajectory passes very close to Earth is expensive
    rather than wrong-looking: the integrator resolves the encounter step by
    step and a single candidate can consume the whole fit budget. On short
    main-belt arcs the offender is usually the degenerate branch that
    collapses onto Earth's own orbit -- near-zero topocentric distance,
    co-moving with the observer -- which is spurious for the objects that
    produce it and is rejected at 10^4-10^6 sigma when it is actually
    evaluated (Smithsonian/layup#465).

    These candidates cannot simply be dropped: for a genuine near-Earth
    object with a real close approach, the close root is the *correct*
    orbit. So they are deferred, not discarded. The caller fits the safe
    candidates first and only falls back to the deferred ones if none of
    them converged.

    The same ``close_earth_AU`` threshold and the same inertial
    approximation are used as in :func:`filter_candidates_by_residual`,
    which passes these candidates through its residual test unfiltered for
    the same reason.

    Returns
    -------
    (safe, deferred) : tuple[list, list]
        Both preserve the input ordering. ``deferred`` is empty in the
        common case.
    """
    safe, deferred = [], []
    for c in candidates:
        try:
            min_geo = _inertial_min_geocentric_AU(c.state, c.epoch, observations)
        except Exception:
            # Unreadable candidate: treat as safe and let the picker judge it.
            safe.append(c)
            continue
        (deferred if min_geo < close_earth_AU else safe).append(c)
    return safe, deferred


def filter_candidates_by_residual(
    candidates,
    observations,
    ephem,
    threshold_sigma: float = 1000.0,
    residual_percentile: float = 80.0,
    min_obs_for_filter: int = 4,
    close_earth_AU: float = _CLOSE_EARTH_AU,
):
    """Drop IOD candidates whose predicted positions miss the observations
    by more than `threshold_sigma` times the per-axis astrometric σ.

    The right Gauss root predicts the bulk of the observations within a few
    σ; phantom roots are typically off by 10⁵-10⁶ σ on essentially every
    observation. A loose threshold (1000σ default) keeps the right root in
    every realistic case while throwing out the obviously-wrong ones before
    LM ever runs on them.

    The per-candidate metric is the `residual_percentile`-th percentile of
    the per-observation residuals (in σ), *not* the worst single point. A
    max-residual criterion is brittle: one contaminating observation (or a
    rough multi-point seed propagated across a long arc) can throw a single
    large residual that exceeds the threshold even for the correct root.
    Using a high percentile (80th by default) tolerates a minority of bad
    points — which the downstream robust LM fit cleans up — while still
    rejecting candidates that miss the *majority* of observations. Keeping
    it a percentile rather than the median means a candidate must still fit
    most of the arc, so a seed that only matches one of two mis-linked
    tracklet groups is not waved through.

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
        Reject candidates whose `residual_percentile`-th-percentile
        angular residual exceeds this multiple of the per-observation σ.
    residual_percentile : float
        Percentile (0-100) of the per-observation residuals used as the
        per-candidate rejection metric. 80.0 by default; 50.0 gives the
        median (tolerant of up to half the points being outliers), 100.0
        recovers the legacy worst-point behavior.
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
    best = None  # (resid_metric_sigma, candidate) over residual-evaluated candidates
    for c in physical:
        # Close-Earth-approach pass-through: avoid both the integrator
        # blowup and a silently-wrong 2-body shortcut.
        min_geo = _inertial_min_geocentric_AU(c.state, c.epoch, observations)
        if min_geo < close_earth_AU:
            survivors.append(c)
            continue

        resids_sigma = []
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
            resids_sigma.append(sep_rad / sigma)

        if not integrable or not resids_sigma:
            continue
        # Robust per-candidate metric: a high percentile of the per-obs
        # residuals rather than the single worst point, so a minority of
        # contaminating observations can't reject an otherwise-good root
        # (the downstream robust LM cleans those up). See the docstring.
        resid_metric_sigma = float(np.percentile(resids_sigma, residual_percentile))
        # Track the single best-fitting integrable candidate so we can
        # guarantee it is never discarded (see the invariant below).
        if best is None or resid_metric_sigma < best[0]:
            best = (resid_metric_sigma, c)
        if resid_metric_sigma <= threshold_sigma:
            survivors.append(c)

    # Correctness invariant: the prefilter is a performance optimization
    # (skip LM on obvious garbage) and must never discard the single
    # best-fitting seed. A *valid* Gauss root can still land above the
    # threshold — a rough multi-point seed propagated across a long arc
    # drifts on part of the arc even where LM later converges cleanly.
    # Without this guard the threshold could silently reject the right
    # root and leave only phantoms.
    if best is not None and not any(c is best[1] for c in survivors):
        survivors.append(best[1])

    if not survivors:
        # Don't reject everything — fall back to physically-OK candidates.
        return physical
    return survivors
