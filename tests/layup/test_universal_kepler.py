"""Tests for the universal-variable Kepler propagator port.

Run with:  pytest kernels/universal-kepler/

Four things are checked, in increasing order of what they would catch:

1.  The Stumpff functions against their series/closed forms -- in particular
    c4 and c5, which layup's own stumpff() does not provide and which only
    the *partials* depend on, so an error there is invisible in the state.
2.  The propagated state against an independent classical-elements
    propagator (different formulation, so a shared bug is unlikely), plus
    the conserved quantities.
3.  The variational output against central finite differences, and against
    symplecticity -- this is the test that matters, since the partials are
    the reason this module exists.
4.  Faithfulness to the original C, where the C is correct (dt < one
    period), by building it and calling through ctypes.

Multi-revolution behaviour gets its own tests because that is where the C is
wrong; see the module docstring of universal_kepler.py.
"""

from __future__ import annotations

import ctypes
import math
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from universal_kepler import (
    KeplerConvergenceError,
    state_transition_matrix,
    stumpff_c,
    universal_step,
)

GM = 2.9591220828559104e-4  # AU^3/day^2, layup's MU_SUN

HERE = pathlib.Path(__file__).parent


# --------------------------------------------------------------------------
# Test states.  Named so a failure report says which regime broke.
# --------------------------------------------------------------------------

def _state(a, e, gm=GM):
    """A planar state at aphelion for semimajor axis a and eccentricity e.

    At an apse u = r.v = 0, which makes the expected radial range trivially
    checkable: r stays within [a(1-e), a(1+e)].
    """
    r = a * (1.0 + e)
    v = math.sqrt(gm * (2.0 / r - 1.0 / a))
    return np.array([r, 0.0, 0.0, 0.0, v, 0.0])


def _tilted(state, angle=0.4):
    """Rotate out of the plane so z-components are exercised too."""
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    out = np.empty(6)
    out[:3] = rot @ state[:3]
    out[3:] = rot @ state[3:]
    return out


STATES = {
    "mainbelt": _tilted(_state(2.5, 0.15)),
    "tno": _tilted(_state(43.0, 0.08)),
    "eccentric": _tilted(_state(20.0, 0.85)),
    "near_circular": _tilted(_state(1.0, 0.001)),
}


def _hyperbolic_state(gm=GM):
    r = 3.0
    v_esc = math.sqrt(2.0 * gm / r)
    return _tilted(np.array([r, 0.0, 0.0, 0.0, 1.6 * v_esc, 0.0]))


def _period(state, gm=GM):
    r0 = np.linalg.norm(state[:3])
    alpha = 2.0 * gm / r0 - state[3:] @ state[3:]
    a = gm / alpha
    return 2.0 * math.pi * math.sqrt(a**3 / gm)


# --------------------------------------------------------------------------
# 1. Stumpff functions
# --------------------------------------------------------------------------

def _stumpff_series(k, z, nterms=40):
    """c_k(z) = sum_n (-z)^n / (2n+k)!  -- direct summation, independent
    of the four-folding recurrences under test."""
    total = 0.0
    for n in range(nterms):
        total += (-z) ** n / math.factorial(2 * n + k)
    return total


@pytest.mark.parametrize("z", [-8.0, -2.0, -0.5, -0.05, 0.0, 0.05, 0.5, 2.0, 8.0, 30.0])
def test_stumpff_matches_series(z):
    got = stumpff_c(z)
    for k in range(6):
        assert got[k] == pytest.approx(_stumpff_series(k, z), rel=1e-13, abs=1e-15)


@pytest.mark.parametrize("z", [0.3, 2.0, 25.0])
def test_stumpff_closed_form_elliptic(z):
    """For z > 0 the low-order Stumpff functions are trigonometric."""
    c0, c1, c2, c3, _, _ = stumpff_c(z)
    sq = math.sqrt(z)
    assert c0 == pytest.approx(math.cos(sq), rel=1e-13)
    assert c1 == pytest.approx(math.sin(sq) / sq, rel=1e-13)
    assert c2 == pytest.approx((1.0 - math.cos(sq)) / z, rel=1e-13)
    assert c3 == pytest.approx((sq - math.sin(sq)) / z**1.5, rel=1e-13)


@pytest.mark.parametrize("z", [-0.3, -2.0, -25.0])
def test_stumpff_closed_form_hyperbolic(z):
    c0, c1, c2, c3, _, _ = stumpff_c(z)
    sq = math.sqrt(-z)
    assert c0 == pytest.approx(math.cosh(sq), rel=1e-13)
    assert c1 == pytest.approx(math.sinh(sq) / sq, rel=1e-13)
    assert c2 == pytest.approx((math.cosh(sq) - 1.0) / (-z), rel=1e-13)
    assert c3 == pytest.approx((math.sinh(sq) - sq) / (-z) ** 1.5, rel=1e-13)


@pytest.mark.parametrize("z", [-5.0, -0.2, 0.0, 0.2, 5.0, 40.0])
def test_stumpff_recurrence(z):
    """c_k(z) = 1/k! - z*c_{k+2}(z) ties the high orders to the low ones."""
    c0, c1, c2, c3, c4, c5 = stumpff_c(z)
    assert c0 == pytest.approx(1.0 - z * c2, rel=1e-14, abs=1e-16)
    assert c1 == pytest.approx(1.0 - z * c3, rel=1e-14, abs=1e-16)
    assert c2 == pytest.approx(0.5 - z * c4, rel=1e-14, abs=1e-16)
    assert c3 == pytest.approx(1.0 / 6.0 - z * c5, rel=1e-14, abs=1e-16)


# --------------------------------------------------------------------------
# 2. The propagated state
# --------------------------------------------------------------------------

def _propagate_via_elements(gm, dt, state):
    """Independent two-body propagation through classical elements.

    Deliberately a different formulation from the universal-variable solver
    under test: elements -> Kepler's equation in E -> back to Cartesian.
    Elliptic orbits only.
    """
    r0v, v0v = state[:3], state[3:]
    r0 = np.linalg.norm(r0v)
    a = 1.0 / (2.0 / r0 - (v0v @ v0v) / gm)
    assert a > 0, "elements reference handles ellipses only"
    n = math.sqrt(gm / a**3)

    # Eccentricity vector and the eccentric anomaly at t0.
    h = np.cross(r0v, v0v)
    evec = np.cross(v0v, h) / gm - r0v / r0
    e = np.linalg.norm(evec)

    cosE0 = (1.0 - r0 / a) / e
    sinE0 = (r0v @ v0v) / (e * math.sqrt(gm * a))
    E0 = math.atan2(sinE0, cosE0)
    M = E0 - e * math.sin(E0) + n * dt

    # Newton on Kepler's equation.
    E = M
    for _ in range(200):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < 1e-15:
            break

    # Lagrange f/g in terms of the eccentric-anomaly increment.
    dE_ = E - E0
    r = a * (1.0 - e * math.cos(E))
    f = 1.0 - a / r0 * (1.0 - math.cos(dE_))
    g = dt + (math.sin(dE_) - dE_) / n
    fdot = -math.sqrt(gm * a) / (r * r0) * math.sin(dE_)
    gdot = 1.0 - a / r * (1.0 - math.cos(dE_))

    out = np.empty(6)
    out[:3] = f * r0v + g * v0v
    out[3:] = fdot * r0v + gdot * v0v
    return out


@pytest.mark.parametrize("name", sorted(STATES))
@pytest.mark.parametrize("frac", [-0.4, -0.05, 0.001, 0.05, 0.3, 0.75, 0.99])
def test_state_matches_classical_elements(name, frac):
    state = STATES[name]
    dt = frac * _period(state)
    got = universal_step(GM, dt, state).state
    want = _propagate_via_elements(GM, dt, state)
    assert got == pytest.approx(want, rel=1e-11, abs=1e-13)


@pytest.mark.parametrize("name", sorted(STATES))
def test_energy_and_angular_momentum_conserved(name):
    state = STATES[name]
    P = _period(state)
    e0 = 0.5 * state[3:] @ state[3:] - GM / np.linalg.norm(state[:3])
    h0 = np.cross(state[:3], state[3:])
    for frac in (0.13, 0.4, 0.87, 2.6):
        s = universal_step(GM, frac * P, state).state
        e1 = 0.5 * s[3:] @ s[3:] - GM / np.linalg.norm(s[:3])
        assert e1 == pytest.approx(e0, rel=1e-12)
        assert np.cross(s[:3], s[3:]) == pytest.approx(h0, rel=1e-12)


@pytest.mark.parametrize("name", sorted(STATES))
def test_round_trip(name):
    """Forward then back returns the state and the deviation."""
    state = STATES[name]
    dt = 0.37 * _period(state)
    var = np.array([0.3, -0.2, 0.11, 1e-3, 2e-3, -5e-4])
    fwd = universal_step(GM, dt, state, variation=var)
    back = universal_step(GM, -dt, fwd.state, variation=fwd.variation)
    assert back.state == pytest.approx(state, rel=1e-11, abs=1e-13)
    assert back.variation == pytest.approx(var, rel=1e-9, abs=1e-13)


def test_hyperbolic_propagation():
    state = _hyperbolic_state()
    r0 = np.linalg.norm(state[:3])
    alpha = 2.0 * GM / r0 - state[3:] @ state[3:]
    assert alpha < 0, "test state should be unbound"
    e0 = 0.5 * state[3:] @ state[3:] - GM / r0
    for dt in (-400.0, -20.0, 5.0, 200.0, 3000.0):
        out = universal_step(GM, dt, state)
        e1 = 0.5 * out.state[3:] @ out.state[3:] - GM / np.linalg.norm(out.state[:3])
        assert e1 == pytest.approx(e0, rel=1e-11)
    # and it should come back
    dt = 1500.0
    there = universal_step(GM, dt, state).state
    back = universal_step(GM, -dt, there).state
    assert back == pytest.approx(state, rel=1e-10, abs=1e-13)


def test_near_parabolic_branch_is_exercised():
    """|dt/r0| <= 0.2 takes the series guess; make sure it is right there."""
    state = STATES["mainbelt"]
    r0 = np.linalg.norm(state[:3])
    dt = 0.1 * r0  # inside the branch
    got = universal_step(GM, dt, state).state
    want = _propagate_via_elements(GM, dt, state)
    assert got == pytest.approx(want, rel=1e-11, abs=1e-13)


def test_zero_dt_is_identity():
    state = STATES["tno"]
    var = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out = universal_step(GM, 0.0, state, variation=var)
    assert out.state == pytest.approx(state, abs=1e-14)
    assert out.variation == pytest.approx(var, abs=1e-14)


# --------------------------------------------------------------------------
# 3. The variational output -- the reason this module exists
# --------------------------------------------------------------------------

def _fd_stm(gm, dt, state, rel=1e-7):
    """Central-difference d state(t+dt) / d state(t)."""
    stm = np.empty((6, 6))
    scale = np.array([np.linalg.norm(state[:3])] * 3 + [np.linalg.norm(state[3:])] * 3)
    for j in range(6):
        h = rel * scale[j]
        plus, minus = state.copy(), state.copy()
        plus[j] += h
        minus[j] -= h
        stm[:, j] = (
            universal_step(gm, dt, plus).state - universal_step(gm, dt, minus).state
        ) / (2.0 * h)
    return stm


@pytest.mark.parametrize("name", sorted(STATES))
@pytest.mark.parametrize("frac", [-0.3, 0.08, 0.45, 0.9])
def test_variational_matches_finite_differences(name, frac):
    state = STATES[name]
    dt = frac * _period(state)
    analytic = state_transition_matrix(GM, dt, state)
    numeric = _fd_stm(GM, dt, state)
    # Compare column-wise against that column's own magnitude: the position
    # and velocity blocks differ by many orders of magnitude, so a single
    # global tolerance would be meaningless.
    for j in range(6):
        scale = max(np.max(np.abs(numeric[:, j])), 1e-12)
        assert np.max(np.abs(analytic[:, j] - numeric[:, j])) / scale < 2e-6


def test_variational_matches_finite_differences_hyperbolic():
    state = _hyperbolic_state()
    for dt in (-300.0, 50.0, 900.0):
        analytic = state_transition_matrix(GM, dt, state)
        numeric = _fd_stm(GM, dt, state)
        for j in range(6):
            scale = max(np.max(np.abs(numeric[:, j])), 1e-12)
            assert np.max(np.abs(analytic[:, j] - numeric[:, j])) / scale < 2e-6


@pytest.mark.parametrize("name", sorted(STATES))
def test_stm_is_symplectic(name):
    """M^T J M = J for a Hamiltonian flow.  Stronger than det(M) = 1, and it
    catches sign or index errors in the variational algebra that a
    finite-difference check with loose tolerance could let through.

    The tolerance has to scale as ||M||^2 * eps: the position-vs-velocity
    block grows like dt (order 1e5 for a TNO over a period), so forming
    M^T J M cancels large numbers down to O(1).  Measured worst case is
    about 0.6 * ||M||^2 * eps, so 20x that is a real test rather than a
    rubber stamp.
    """
    state = STATES[name]
    J = np.block(
        [[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]]
    )
    for frac in (0.11, 0.5, 1.7):
        M = state_transition_matrix(GM, frac * _period(state), state)
        floor = 20.0 * np.linalg.norm(M) ** 2 * np.finfo(float).eps
        assert M.T @ J @ M == pytest.approx(J, abs=floor)
        assert np.linalg.det(M) == pytest.approx(1.0, abs=max(1e-13, floor))


def test_variation_is_linear():
    """The propagated deviation must be linear in the input deviation."""
    state = STATES["tno"]
    dt = 0.3 * _period(state)
    a = np.array([0.5, -0.25, 0.1, 1e-4, -2e-4, 3e-5])
    b = np.array([-0.1, 0.4, 0.7, 5e-5, 1e-4, -1e-4])
    va = universal_step(GM, dt, state, variation=a).variation
    vb = universal_step(GM, dt, state, variation=b).variation
    vab = universal_step(GM, dt, state, variation=2.0 * a - 3.0 * b).variation
    assert vab == pytest.approx(2.0 * va - 3.0 * vb, rel=1e-10, abs=1e-14)


# --------------------------------------------------------------------------
# Multi-revolution: where the C is wrong
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2, 5, 40])
def test_state_is_periodic_across_revolutions(n):
    """Adding whole periods must not move the state.

    The C fails this from n=1: its g is too large by n*P*|v|, so |r| grows
    without bound while the routine still reports success.
    """
    state = STATES["mainbelt"]
    P = _period(state)
    ref = universal_step(GM, 0.3 * P, state)
    got = universal_step(GM, (n + 0.3) * P, state)
    assert got.n_rev == n
    # Rounding in a single solve at large s accumulates with revolution
    # count: measured 2.7e-14 AU at n=1 rising to 6.2e-10 AU at n=40.
    # 1e-8 AU (about 1.5 m) is loose enough to be stable and still four
    # orders tighter than any real effect.
    assert got.state == pytest.approx(ref.state, abs=1e-8)


@pytest.mark.parametrize("n", [1, 3])
def test_variational_still_matches_fd_across_revolutions(n):
    """The partials must be right in the multi-revolution regime too.

    This is why the fix solves the full dt rather than simply reducing it:
    reducing would give the correct state but the wrong partials, since
    neighbouring orbits have different periods and the deviation grows
    secularly with every revolution.
    """
    state = STATES["mainbelt"]
    dt = (n + 0.3) * _period(state)
    analytic = state_transition_matrix(GM, dt, state)
    # Larger FD step than the single-revolution case: the STM grows with
    # revolution count (that is the point of this test), so the differencing
    # cancels more and a smaller step is noisier, not better.
    numeric = _fd_stm(GM, dt, state, rel=1e-6)
    for j in range(6):
        scale = max(np.max(np.abs(numeric[:, j])), 1e-12)
        assert np.max(np.abs(analytic[:, j] - numeric[:, j])) / scale < 1e-5


def test_partials_grow_secularly_with_revolutions():
    """The along-track deviation grows with revolution count -- the concrete
    reason a state-only multi-rev fix would have been wrong."""
    state = STATES["mainbelt"]
    P = _period(state)
    norms = [
        np.linalg.norm(state_transition_matrix(GM, (n + 0.3) * P, state))
        for n in (0, 1, 2, 5)
    ]
    assert norms == sorted(norms), f"STM norm should increase with revolutions: {norms}"
    assert norms[-1] > 3.0 * norms[0]


# --------------------------------------------------------------------------
# 4. Faithfulness to the original C
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def c_universal_step(tmp_path_factory):
    """Build universal-kepler.c as a shared library and expose universal_step.

    Skipped when there is no compiler, or when the C source has moved.
    """
    src = HERE / "universal-kepler.c"
    if not src.exists():
        pytest.skip(f"C source not found at {src}")
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler available")

    build = tmp_path_factory.mktemp("uk_c")
    # The C declares `extern double machine_epsilon` but never uses it; give
    # it a definition so the link is clean.
    shim = build / "shim.c"
    shim.write_text("double machine_epsilon = 0.0;\n")
    so = build / "libuk.so"
    proc = subprocess.run(
        [cc, "-std=gnu99", "-w", "-O2", "-fPIC", "-shared", str(src), str(shim), "-lm", "-o", str(so)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"could not build the C reference: {proc.stderr[-400:]}")

    lib = ctypes.CDLL(str(so))
    arr6 = ctypes.c_double * 6
    lib.universal_step.restype = ctypes.c_int
    lib.universal_step.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(arr6),
        ctypes.POINTER(arr6),
        ctypes.POINTER(arr6),
    ]

    def call(gm, dt, state, variation):
        s0 = arr6(*[float(x) for x in state])
        out = arr6()
        var = arr6(*[float(x) for x in variation])
        flag = lib.universal_step(
            ctypes.c_double(gm),
            ctypes.c_double(dt),
            ctypes.byref(s0),
            ctypes.byref(out),
            ctypes.byref(var),
        )
        return flag, np.array(out[:]), np.array(var[:])

    return call


@pytest.mark.parametrize("name", sorted(STATES))
@pytest.mark.parametrize("frac", [-0.4, 0.02, 0.25, 0.6, 0.95])
def test_matches_c_within_one_period(c_universal_step, name, frac):
    """Inside one revolution -- where the C is correct -- the port must
    reproduce it, state and partials alike."""
    state = STATES[name]
    dt = frac * _period(state)
    var = np.array([0.3, -0.2, 0.11, 1e-3, 2e-3, -5e-4])

    flag, c_state, c_var = c_universal_step(GM, dt, state, var)
    if flag != 0:
        # The C's absolute tolerance is unreachable for some orbits; that is
        # pinned by test_c_tolerance_fails_on_distant_orbits below, and there
        # is nothing to compare against here.
        pytest.skip(f"C did not converge for {name} at {frac}P (its own defect)")

    py = universal_step(GM, dt, state, variation=var)
    assert py.state == pytest.approx(c_state, rel=1e-11, abs=1e-13)
    assert py.variation == pytest.approx(c_var, rel=1e-9, abs=1e-13)


def test_matches_c_hyperbolic(c_universal_step):
    state = _hyperbolic_state()
    var = np.array([0.2, 0.1, -0.3, 1e-3, -1e-3, 2e-4])
    for dt in (-250.0, 40.0, 800.0):
        flag, c_state, c_var = c_universal_step(GM, dt, state, var)
        assert flag == 0
        py = universal_step(GM, dt, state, variation=var)
        assert py.state == pytest.approx(c_state, rel=1e-11, abs=1e-13)
        assert py.variation == pytest.approx(c_var, rel=1e-9, abs=1e-13)


def test_c_tolerance_fails_on_distant_orbits(c_universal_step):
    """Pins the C's scale-dependent convergence test.

    EPS = 1e-13 is applied as an absolute tolerance on the universal
    variable s, which spans P/a over one revolution -- 577 for a mainbelt
    orbit but 2395 for a TNO at 43 AU.  The demand is then below double
    precision, so convergence becomes a coin flip that the C loses more
    often the more distant the object.  The port's relative test converges
    everywhere in this sweep.
    """
    fracs = (-0.4, 0.02, 0.25, 0.5, 0.6, 0.75, 0.95)
    c_failures, distant_failures = [], []
    for name in sorted(STATES):
        state = STATES[name]
        P = _period(state)
        for frac in fracs:
            dt = frac * P
            flag, _, _ = c_universal_step(GM, dt, state, np.zeros(6))
            if flag != 0:
                c_failures.append((name, frac))
                if name in ("tno", "eccentric"):
                    distant_failures.append((name, frac))
            # the port must converge regardless
            universal_step(GM, dt, state)

    assert c_failures, "expected the C to fail somewhere in this sweep"
    assert distant_failures, f"expected distant-orbit failures, got {c_failures}"


def test_c_is_wrong_past_one_period_and_the_port_is_not(c_universal_step):
    """Pins the defect this port fixes, so nobody 'restores' the C behaviour.

    If this ever fails because the C now agrees, the C was fixed upstream and
    this test should become an equality check.
    """
    state = STATES["mainbelt"]
    P = _period(state)
    var = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ref = universal_step(GM, 0.3 * P, state).state
    dt = 5.3 * P
    flag, c_state, _ = c_universal_step(GM, dt, state, var)

    assert flag == 0, "the C reports success -- that is what makes it dangerous"
    c_err = np.linalg.norm(c_state[:3] - ref[:3])
    assert c_err > 10.0, f"expected the C to be far off, got {c_err} AU"

    py = universal_step(GM, dt, state).state
    assert py == pytest.approx(ref, rel=1e-9, abs=1e-11)


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        universal_step(GM, 1.0, np.zeros(5))
    with pytest.raises(ValueError):
        universal_step(GM, 1.0, STATES["tno"], variation=np.zeros(3))


def test_rejects_zero_position():
    with pytest.raises(ValueError):
        universal_step(GM, 1.0, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
