"""Universal-variable Kepler propagator with variational partials.

Python port of ``universal-kepler.c`` in this directory (Danby 1988, p.178),
intended as the numerical core of a Herget initial-orbit-determination
prototype.

WHAT THIS IS
------------
The *initial*-value problem: given a state (r0, v0) and an interval dt,
return the state at t0 + dt, optionally propagating a 6-vector deviation
alongside it.  The deviation is the state-transition matrix applied to a
direction -- which is what Herget actually needs, since only two directions
matter (d/d rho_1 and d/d rho_3), not the full 6x6.

WHAT THIS IS NOT
----------------
A Lambert solver.  It does not solve the two-point boundary-value problem
(r1, t1, r3, t3 -> v1).  It is, however, the natural engine for a *shooting*
Lambert: guess v1, propagate here, compare to r3, and correct using the
variational output as the Jacobian d r(t3) / d v1.

Handles elliptic, hyperbolic and near-parabolic motion without branching on
a bound-orbit assumption -- the sign of ``alpha = gm/a`` selects the regime.

UNITS are whatever ``gm`` is expressed in; for AU and days use layup's
``constants.MU_SUN`` (heliocentric) or ``constants.GMtotal`` (barycentric),
matching the frame the state is expressed in.  The two differ by 0.13% and
layup uses both deliberately, so pick consciously.

DIFFERENCES FROM THE C
----------------------
1.  **Multi-revolution fix.**  The C is silently wrong for ``|dt|`` longer
    than one orbital period: its elliptic branch reduces the local ``dt`` by
    whole revolutions to build the initial guess, but the Newton iteration
    then solves against that *reduced* value while ``universal_step`` forms
    ``g = dt - gm*g3`` from the *unreduced* one.  The Lagrange g comes out
    too large by exactly ``n_rev * P``, the state runs away along the
    velocity direction, and the routine still reports success.

    Here the reduction is used for the guess *only*; ``n_rev`` revolutions
    are added back as ``n_rev * 2*pi / sqrt(alpha)`` (one revolution advances
    the universal anomaly by 2*pi) and the Newton solves the FULL dt.  That
    is deliberately not just a guard: reducing dt and keeping the result
    would give the right *state* but the wrong *partials*, because
    neighbouring orbits have different periods and the deviation grows
    secularly with every revolution.  A state-only fix would silently break
    exactly the thing this module exists to provide.

2.  **Relative convergence tolerance.**  The C tests ``|ds| > EPS`` with
    ``EPS = 1e-13`` as an *absolute* tolerance on ``s``, which carries units
    of time/length.  Over one revolution ``s`` spans ``P/a``: about 577 for
    a mainbelt orbit but **2395 for a TNO at 43 AU**, so the test demands
    4e-17 in relative terms -- below double precision, and unreachable.
    Newton then exhausts its six iterations, Laguerre-Conway its fifteen,
    and ``kepler()`` returns ``KEPLER_FLAG`` -- which ``universal_step``
    ignores, computing a state from the unconverged ``s`` regardless.  The
    failure therefore gets worse the more distant the object, which is
    precisely backwards for this project.  (The C's author evidently saw
    it: both loops carry a commented-out alternative on ``f/dt``, a
    scale-invariant relative residual.)  Here the test is relative to
    ``max(1, |s|)``.

3.  ``r`` is recomputed from the converged g-functions rather than reused
    from the last Newton evaluation (the C's ``*rx = fp`` is one iteration
    stale -- negligible once converged, but free to do properly).

4.  ``gdot`` falls back to ``1 - (gm/r)*g2`` when ``|f|`` is too small for
    the Wronskian form ``(1 + g*fdot)/f`` the C uses unconditionally.

5.  Non-convergence raises rather than returning a flag the caller may
    ignore -- as ``universal_step`` itself does in the C.

The unused ``stumpff()`` in the C (dead code -- both call sites use
``cfun()``) is not ported.  Note that layup's own
``utilities.orbit_conversion.stumpff`` returns c0..c3 only; the variational
path here needs c4 and c5 via g1a/g2a/g3a, which is why ``cfun`` is ported
rather than reused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "KeplerConvergenceError",
    "KeplerStep",
    "stumpff_c",
    "universal_step",
    "state_transition_matrix",
]

# Iteration budgets, matching the C.
_NEWTON_MAX = 6
_LAGCON_MAX = 15
_TWO_PI = 2.0 * math.pi

# Convergence tolerance on the universal variable s, applied RELATIVE to |s|
# (see note 2 in the module docstring).  The C uses this as an absolute
# tolerance, which is unreachable for distant orbits.
_TOL = 1e-13


def _converged(ds: float, s: float) -> bool:
    return abs(ds) <= _TOL * max(1.0, abs(s))


# Below this |f| the Wronskian form of gdot loses precision; see note 3 above.
_F_FLOOR = 1e-8


class KeplerConvergenceError(RuntimeError):
    """Neither Newton nor Laguerre-Conway reached the tolerance."""


@dataclass
class KeplerStep:
    """Result of one propagation.

    Attributes
    ----------
    state : (6,) ndarray
        Propagated (x, y, z, vx, vy, vz).
    variation : (6,) ndarray or None
        The input deviation propagated to the same epoch, i.e. the
        state-transition matrix applied to it.  None if none was supplied.
    s : float
        Converged universal variable.
    r : float
        Heliocentric (or barycentric) distance at the end of the step.
    n_rev : int
        Whole revolutions spanned by dt.  Nonzero means the multi-revolution
        path was exercised -- the regime the C got wrong.
    n_iter : int
        Iterations used by whichever solver converged.
    solver : str
        "newton" or "laguerre-conway".
    """

    state: np.ndarray
    variation: np.ndarray | None
    s: float
    r: float
    n_rev: int
    n_iter: int
    solver: str


def stumpff_c(z: float) -> tuple[float, float, float, float, float, float]:
    """Stumpff functions c0..c5 by Mikkola's argument four-folding.

    Port of the C ``cfun()``.  The four-folding keeps the series arguments
    small (|h| < 0.1) so the truncated rational approximations for c4 and c5
    stay accurate, then unfolds with the duplication identities.
    """
    h = z
    k = 0
    while abs(h) >= 0.1:
        h *= 0.25
        k += 1

    c4 = (1.0 - h * (1.0 - h * (1.0 - h / 90.0 / (1.0 + h / 132.0)) / 56.0) / 30.0) / 24.0
    c5 = (1.0 - h * (1.0 - h * (1.0 - h / 110.0 / (1.0 + h / 156.0)) / 72.0) / 42.0) / 120.0

    for _ in range(k):
        c3 = 1.0 / 6.0 - h * c5
        c2 = 0.5 - h * c4
        c5 = (c5 + c4 + c2 * c3) / 16.0
        c4 = c3 * (2.0 - h * c3) / 8.0
        h *= 4.0

    c3 = 1.0 / 6.0 - z * c5
    c2 = 0.5 - z * c4
    c1 = 1.0 - z * c3
    c0 = 1.0 - z * c2
    return c0, c1, c2, c3, c4, c5


def _initial_guess(gm: float, dt: float, r0: float, alpha: float, u: float) -> tuple[float, int]:
    """Starting value for the universal variable, and the revolution count.

    Returns (s_guess, n_rev).  n_rev is nonzero only on the elliptic branch,
    and the returned guess already includes the whole revolutions -- callers
    solve against the full dt.
    """
    # Short step relative to the current distance: the series guess is good
    # and there is no revolution structure to unwrap.
    if abs(dt / r0) <= 0.2:
        return dt / r0 - (dt * dt * u) / (2.0 * r0**3), 0

    if alpha <= 0.0:
        # Hyperbolic.  a < 0 here, so the sqrt arguments below are positive.
        a = gm / alpha
        en = math.sqrt(-gm / (a * a * a))
        ch = 1.0 - r0 / a
        sh = u / math.sqrt(-a * gm)
        e = math.sqrt(ch * ch - sh * sh)
        dm = en * dt
        if dm < 0.0:
            return -math.log((-2.0 * dm + 1.8 * e) / (ch - sh)) / math.sqrt(-alpha), 0
        return math.log((2.0 * dm + 1.8 * e) / (ch + sh)) / math.sqrt(-alpha), 0

    # Elliptic.
    a = gm / alpha
    en = math.sqrt(gm / (a * a * a))
    ec = 1.0 - r0 / a
    es = u / (en * a * a)

    # Whole revolutions are stripped to keep the RK4 guess inside one orbit,
    # then added back below.  Truncation toward zero (not floor) so the
    # remainder keeps dt's sign, matching the C's (int) cast -- but in Python
    # the int is arbitrary precision, so the C's overflow at ~2e9 revolutions
    # does not apply.
    n_rev = int(en * dt / _TWO_PI)
    dt_red = dt - n_rev * _TWO_PI / en

    y = en * dt_red - es

    # One RK4 step of the (ec, es) rotation, as a cheap high-order guess for
    # the eccentric-anomaly increment.  (The C also computes the eccentricity
    # here for Danby's alternative guess, which is commented out; omitted.)
    xx, yy = ec, es
    h = en * dt_red
    omx = h / (1.0 - xx)
    k0x, k0y = -yy * omx, xx * omx
    xx1, yy1 = xx + k0x / 2.0, yy + k0y / 2.0
    omx = h / (1.0 - xx1)
    k1x, k1y = -yy1 * omx, xx1 * omx
    xx1, yy1 = xx + k1x / 2.0, yy + k1y / 2.0
    omx = h / (1.0 - xx1)
    k2x, k2y = -yy1 * omx, xx1 * omx
    xx1 = xx + k2x
    omx = h / (1.0 - xx1)
    k3y = xx1 * omx
    yy += (k0y + 2.0 * (k1y + k2y) + k3y) / 6.0

    root_alpha = math.sqrt(alpha)
    # sqrt(alpha)*s is the eccentric-anomaly increment, so a revolution is
    # worth 2*pi/sqrt(alpha) in s.
    s = (y + yy) / root_alpha + n_rev * _TWO_PI / root_alpha
    return s, n_rev


def _solve_kepler(gm, dt, r0, alpha, u, zeta):
    """Solve the universal Kepler equation for s; return the g-functions.

    Returns (g0..g5, r, s, n_rev, n_iter, solver).
    """
    s_guess, n_rev = _initial_guess(gm, dt, r0, alpha, u)

    def _fvals(s):
        c0, c1, c2, c3, _, _ = stumpff_c(s * s * alpha)
        c1 *= s
        c2 *= s * s
        c3 *= s * s * s
        f = r0 * c1 + u * c2 + gm * c3 - dt
        fp = r0 * c0 + u * c1 + gm * c2  # == r at this s, hence always > 0
        fpp = zeta * c1 + u * c0
        fppp = zeta * c0 - u * alpha * c1
        return f, fp, fpp, fppp

    # Newton with Danby's cubic correction: three nested refinements of the
    # same step reuse one function evaluation.
    s = s_guess
    ds = math.inf
    solver, n_iter = "newton", 0
    for n_iter in range(1, _NEWTON_MAX + 1):
        f, fp, fpp, fppp = _fvals(s)
        ds = -f / fp
        ds = -f / (fp + ds * fpp / 2.0)
        ds = -f / (fp + ds * fpp / 2.0 + ds * ds * fppp / 6.0)
        s += ds
        if _converged(ds, s):
            break

    if not _converged(ds, s):
        # Laguerre-Conway from the original guess: larger convergence basin,
        # at the cost of a square root per iteration.
        solver = "laguerre-conway"
        s = s_guess
        ln = 5.0
        for n_iter in range(1, _LAGCON_MAX + 1):
            f, fp, fpp, _ = _fvals(s)
            disc = (ln - 1.0) ** 2 * fp * fp - (ln - 1.0) * ln * f * fpp
            ds = -ln * f / (fp + math.copysign(math.sqrt(abs(disc)), fp))
            s += ds
            if _converged(ds, s):
                break
        if not _converged(ds, s):
            raise KeplerConvergenceError(
                f"Kepler equation did not converge: dt={dt!r} r0={r0!r} "
                f"alpha={alpha!r} u={u!r} last |ds|={abs(ds):.3e} s={s!r}"
            )

    c0, c1, c2, c3, c4, c5 = stumpff_c(s * s * alpha)
    g0 = c0
    g1 = c1 * s
    g2 = c2 * s**2
    g3 = c3 * s**3
    g4 = c4 * s**4
    g5 = c5 * s**5
    r = r0 * g0 + u * g1 + gm * g2
    return g0, g1, g2, g3, g4, g5, r, s, n_rev, n_iter, solver


def universal_step(gm, dt, state, variation=None) -> KeplerStep:
    """Propagate `state` by `dt`, optionally carrying a deviation along.

    Parameters
    ----------
    gm : float
        Gravitational parameter, in units consistent with `state` and `dt`.
    dt : float
        Interval.  May be negative.
    state : array_like, shape (6,)
        (x, y, z, vx, vy, vz) at the start of the step.
    variation : array_like, shape (6,), optional
        A deviation in the *initial* state.  Propagated exactly (to the
        two-body model) to the end of the step, i.e. the state-transition
        matrix applied to this vector.

    Returns
    -------
    KeplerStep
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (6,):
        raise ValueError(f"state must have shape (6,), got {state.shape}")

    r0vec, v0vec = state[:3], state[3:]
    r0 = float(np.linalg.norm(r0vec))
    if r0 == 0.0:
        raise ValueError("state has zero position; the two-body problem is singular there")

    v0s = float(v0vec @ v0vec)
    u = float(r0vec @ v0vec)
    alpha = 2.0 * gm / r0 - v0s  # = gm/a; sign selects ellipse vs hyperbola
    zeta = gm - alpha * r0

    g0, g1, g2, g3, g4, g5, r, s, n_rev, n_iter, solver = _solve_kepler(gm, dt, r0, alpha, u, zeta)

    f = 1.0 - (gm / r0) * g2
    g = dt - gm * g3
    fdot = -(gm / (r * r0)) * g1
    # The Wronskian form f*gdot - fdot*g = 1 is better conditioned than
    # 1 - (gm/r)*g2 except where f itself is near zero.
    gdot = (1.0 + g * fdot) / f if abs(f) > _F_FLOOR else 1.0 - (gm / r) * g2

    out = np.empty(6)
    out[:3] = f * r0vec + g * v0vec
    out[3:] = fdot * r0vec + gdot * v0vec

    var_out = None
    if variation is not None:
        dvar = np.asarray(variation, dtype=float)
        if dvar.shape != (6,):
            raise ValueError(f"variation must have shape (6,), got {dvar.shape}")
        dr, dv = dvar[:3], dvar[3:]

        # Derivatives of the scalars the solve depends on, along the deviation.
        r0pr = float(r0vec @ dr) / r0
        alphapr = -(2.0 * gm / (r0 * r0)) * r0pr - 2.0 * float(v0vec @ dv)
        upr = float(r0vec @ dv) + float(v0vec @ dr)
        zetapr = -alpha * r0pr - r0 * alphapr

        # d g_k / d alpha at fixed s (Stumpff recurrences).
        g1a = 0.5 * (g3 - s * g2)
        g2a = 0.5 * (2.0 * g4 - s * g3)
        g3a = 0.5 * (3.0 * g5 - s * g4)

        # d s / d(deviation) from differentiating the Kepler equation at fixed dt.
        spr = -(s * r0pr + g3 * zetapr + g2 * upr + (g3a * zeta + u * g2a) * alphapr) / r

        g1pr = g0 * spr + g1a * alphapr
        g2pr = g1 * spr + g2a * alphapr
        g3pr = g2 * spr + g3a * alphapr
        rpr = r0pr + g1 * upr + g2 * zetapr + u * g1pr + zeta * g2pr

        fpr = (gm * g2 / (r0 * r0)) * r0pr - (gm / r0) * g2pr
        gpr = -gm * g3pr
        fdotpr = (gm / (r * r * r0)) * g1 * rpr + (gm / (r * r0 * r0)) * g1 * r0pr - (gm / (r * r0)) * g1pr
        gdotpr = (gm / (r * r)) * g2 * rpr - (gm / r) * g2pr

        var_out = np.empty(6)
        var_out[:3] = f * dr + g * dv + fpr * r0vec + gpr * v0vec
        var_out[3:] = fdot * dr + gdot * dv + fdotpr * r0vec + gdotpr * v0vec

    return KeplerStep(state=out, variation=var_out, s=s, r=r, n_rev=n_rev, n_iter=n_iter, solver=solver)


def state_transition_matrix(gm, dt, state) -> np.ndarray:
    """Full 6x6 d state(t0+dt) / d state(t0), by six unit variations.

    Herget needs only two columns of this (d/d rho_1 and d/d rho_3), so
    prefer calling `universal_step` directly with the deviation you care
    about; this exists for testing and for callers that want the whole thing.
    """
    stm = np.empty((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1.0
        stm[:, j] = universal_step(gm, dt, state, variation=e).variation
    return stm
