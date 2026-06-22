"""Layer 1 tests for the BK basis primitives.

These tests exercise the pure-math layer of the universal BK fitter
(`src/lib/orbit_fit/bk_basis.cpp`) via the pybind11 bindings exposed
through `layup.routines`.  No ASSIST/REBOUND/ephemeris setup is
required -- the math primitives operate on barycentric Cartesian
states and BK parameters directly.

Coverage:
  * round-trip Cartesian <-> BK transforms
  * analytic dcart_dbk vs central-difference finite-differences
  * mixed-partial symmetry of the second derivatives that show up
    in the bottom-left cross-term block
  * fiducial-direction gauge invariance (different choices of n0
    must give the same Cartesian orbit)
  * special-case forms at alpha = beta = 0 (the fiducial direction)
  * sigma_gdot_sq agreement with the bound-orbit energy bound
    computed independently in Cartesian
"""

from __future__ import annotations

import numpy as np
import pytest

from layup.routines import (
    BKState,
    BKFiducial,
    bk_choose_fiducial,
    bk_to_cartesian,
    cartesian_to_bk,
    dcart_dbk,
    sigma_gdot_sq,
)

# GM_sun in AU^3 / day^2 (Gaussian gravitational constant squared).
MU_SUN = 0.00029591220828559104


# A spread of test BK states covering mainbelt / NEO / TNO regimes.
# Format: (alpha, beta, gamma, adot, bdot, gdot) with gamma = 1/r_helio in 1/AU.
_BK_CASES = [
    # mainbelt ~3 AU, near-circular
    (0.1, -0.05, 1.0 / 3.0, 1e-3, -8e-4, 3e-6),
    # NEO ~1.2 AU, modest rates
    (-0.2, 0.15, 1.0 / 1.2, 5e-3, 4e-3, -1e-5),
    # TNO ~42 AU, slow
    (0.02, 0.01, 1.0 / 42.0, 4e-5, -3e-5, 1e-8),
    # near the fiducial direction (small alpha, beta) -- a common case
    (1e-4, 2e-4, 0.025, 6e-5, 5e-5, -2e-7),
    # off the fiducial direction
    (0.5, -0.4, 0.05, 2e-4, 1e-4, -3e-8),
]


def _make_fiducial(rng: np.random.Generator) -> BKFiducial:
    """Pick a reproducible fiducial direction not aligned with an ICRS axis."""
    n0 = rng.normal(size=3)
    n0 /= np.linalg.norm(n0)
    return bk_choose_fiducial([n0])


def _bk_from_tuple(t):
    return BKState(*t)


# ---------------------------------------------------------------------------
# Round-trip Cartesian <-> BK
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _BK_CASES)
def test_round_trip_bk_to_cart_to_bk(case):
    """BK -> Cartesian -> BK recovers the input to machine precision."""
    rng = np.random.default_rng(seed=12345)
    fid = _make_fiducial(rng)
    bk = _bk_from_tuple(case)
    cart = bk_to_cartesian(bk, fid)
    bk_back = cartesian_to_bk(cart, fid)
    for name in ("alpha", "beta", "gamma", "adot", "bdot", "gdot"):
        original = getattr(bk, name)
        recovered = getattr(bk_back, name)
        np.testing.assert_allclose(
            recovered, original, rtol=1e-12, atol=1e-15, err_msg=f"BK.{name} not recovered through round-trip"
        )


@pytest.mark.parametrize("case", _BK_CASES)
def test_round_trip_cart_to_bk_to_cart(case):
    """Cartesian -> BK -> Cartesian recovers the input to machine precision."""
    rng = np.random.default_rng(seed=67890)
    fid = _make_fiducial(rng)
    bk = _bk_from_tuple(case)
    cart_in = np.asarray(bk_to_cartesian(bk, fid)).flatten()
    bk_round = cartesian_to_bk(cart_in, fid)
    cart_out = np.asarray(bk_to_cartesian(bk_round, fid)).flatten()
    np.testing.assert_allclose(cart_out, cart_in, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Cartesian velocity is the exact time-derivative of the Cartesian position
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _BK_CASES)
def test_velocity_is_time_derivative_of_position(case):
    """The Cartesian velocity from bk_to_cartesian is dr/dt along the BK motion.

    Along the instantaneous BK trajectory the angular coordinates advance at
    their rates: alpha(t) = alpha + adot*t, beta(t) = beta + bdot*t,
    gamma(t) = gamma + gdot*t. Central-differencing the *position* part of
    bk_to_cartesian along that motion must reproduce the *velocity* part, since
    v is by construction d/dt of r = rho_hat(alpha, beta) / gamma.

    This pins down that v is a genuine velocity (AU/day) and not a quantity
    with anomalous units -- independent of the round-trip and Jacobian tests.
    The unusual-looking magnitudes of the intermediate rho_hat_alpha/beta terms
    are expected: those tangent vectors are deliberately not unit length.
    """
    rng = np.random.default_rng(seed=2024)
    fid = _make_fiducial(rng)
    alpha, beta, gamma, adot, bdot, gdot = case
    v_analytic = np.asarray(bk_to_cartesian(_bk_from_tuple(case), fid)).flatten()[3:]

    def position(t):
        state = BKState(alpha + adot * t, beta + bdot * t, gamma + gdot * t, adot, bdot, gdot)
        return np.asarray(bk_to_cartesian(state, fid)).flatten()[:3]

    dt = 1e-4
    v_fd = (position(dt) - position(-dt)) / (2.0 * dt)
    np.testing.assert_allclose(
        v_analytic,
        v_fd,
        rtol=1e-6,
        atol=1e-12,
        err_msg="Cartesian velocity is not the time-derivative of the Cartesian position",
    )


# ---------------------------------------------------------------------------
# Analytic dcart_dbk vs finite-difference
# ---------------------------------------------------------------------------


def _bk_perturb(bk: BKState, idx: int, delta: float) -> BKState:
    names = ("alpha", "beta", "gamma", "adot", "bdot", "gdot")
    vals = [getattr(bk, n) for n in names]
    vals[idx] += delta
    return BKState(*vals)


@pytest.mark.parametrize("case", _BK_CASES)
def test_dcart_dbk_matches_finite_difference(case):
    """Analytic 6x6 Jacobian agrees with central-difference per element."""
    rng = np.random.default_rng(seed=11111)
    fid = _make_fiducial(rng)
    bk = _bk_from_tuple(case)
    J_analytic = np.asarray(dcart_dbk(bk, fid))

    # Step sizes scaled by parameter magnitude so we get a sensible FD
    # for both the O(1) (alpha, beta) and O(1e-5) (rates) axes.
    param_vals = (bk.alpha, bk.beta, bk.gamma, bk.adot, bk.bdot, bk.gdot)
    eps = np.array([max(abs(v), 1.0) * 1e-6 for v in param_vals])

    J_fd = np.zeros((6, 6))
    for i in range(6):
        cart_plus = np.asarray(bk_to_cartesian(_bk_perturb(bk, i, eps[i]), fid)).flatten()
        cart_minus = np.asarray(bk_to_cartesian(_bk_perturb(bk, i, -eps[i]), fid)).flatten()
        J_fd[:, i] = (cart_plus - cart_minus) / (2.0 * eps[i])

    # Relative tolerance covering the dynamic range of J entries.
    scale = np.maximum(np.abs(J_analytic), np.abs(J_fd))
    scale = np.where(scale > 0, scale, 1.0)
    np.testing.assert_array_less(
        np.abs(J_analytic - J_fd) / scale,
        1e-5,
        err_msg="Analytic dcart_dbk disagrees with finite-difference",
    )


# ---------------------------------------------------------------------------
# Mixed-partial symmetry of the second-derivative cross-terms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _BK_CASES)
def test_mixed_partial_symmetry_via_finite_difference(case):
    """d(r,v)/(d alpha d beta) -- approached via FD of dcart_dbk -- is symmetric."""
    rng = np.random.default_rng(seed=22222)
    fid = _make_fiducial(rng)
    bk = _bk_from_tuple(case)
    eps_a = max(abs(bk.alpha), 1.0) * 1e-6
    eps_b = max(abs(bk.beta), 1.0) * 1e-6

    # FD of dcart_dbk's alpha column with respect to beta
    J_plus_b = np.asarray(dcart_dbk(_bk_perturb(bk, 1, eps_b), fid))
    J_minus_b = np.asarray(dcart_dbk(_bk_perturb(bk, 1, -eps_b), fid))
    d2r_dadb = (J_plus_b[:, 0] - J_minus_b[:, 0]) / (2.0 * eps_b)

    # FD of dcart_dbk's beta column with respect to alpha
    J_plus_a = np.asarray(dcart_dbk(_bk_perturb(bk, 0, eps_a), fid))
    J_minus_a = np.asarray(dcart_dbk(_bk_perturb(bk, 0, -eps_a), fid))
    d2r_dbda = (J_plus_a[:, 1] - J_minus_a[:, 1]) / (2.0 * eps_a)

    np.testing.assert_allclose(d2r_dadb, d2r_dbda, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Fiducial-direction gauge invariance
# ---------------------------------------------------------------------------


def test_fiducial_gauge_invariance():
    """Two valid n0 choices give the same physical Cartesian orbit."""
    rng = np.random.default_rng(seed=33333)
    # An arbitrary Cartesian state (40 AU object, small velocity)
    r = np.array([20.0, 30.0, 5.0])
    v = np.array([-2e-4, 1e-4, 5e-5])
    cart = np.concatenate([r, v])

    # Two different fiducial frames: one nearly aligned with r_hat, one tilted.
    r_hat = r / np.linalg.norm(r)
    fid_aligned = bk_choose_fiducial([r_hat])

    tilt = np.array([0.0, 0.0, 1.0])
    fid_tilted = bk_choose_fiducial([r_hat + 0.3 * tilt])

    bk_aligned = cartesian_to_bk(cart, fid_aligned)
    bk_tilted = cartesian_to_bk(cart, fid_tilted)

    cart_back_aligned = np.asarray(bk_to_cartesian(bk_aligned, fid_aligned)).flatten()
    cart_back_tilted = np.asarray(bk_to_cartesian(bk_tilted, fid_tilted)).flatten()

    np.testing.assert_allclose(cart_back_aligned, cart, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(cart_back_tilted, cart, rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------------------
# Special-case forms at alpha = beta = 0 (the fiducial direction)
# ---------------------------------------------------------------------------


def test_special_case_at_fiducial():
    """At alpha = beta = 0, rho_hat = n0, rho_hat_alpha = a, rho_hat_beta = b."""
    rng = np.random.default_rng(seed=44444)
    fid = _make_fiducial(rng)
    bk = BKState(alpha=0.0, beta=0.0, gamma=0.05, adot=0.0, bdot=0.0, gdot=0.0)
    cart = np.asarray(bk_to_cartesian(bk, fid)).flatten()

    # Position at (alpha, beta) = (0, 0) is exactly (1/gamma) * n0.
    expected_r = (1.0 / bk.gamma) * np.asarray(fid.n0)
    np.testing.assert_allclose(cart[:3], expected_r, rtol=1e-13, atol=1e-13)

    # With all rates zero, velocity should be zero.
    np.testing.assert_allclose(cart[3:], np.zeros(3), atol=1e-15)


def test_jacobian_at_fiducial():
    """At the fiducial direction, the Jacobian's top-left 3x3 block is
    [a | b | -(1/gamma) * n0] (each as a column)."""
    rng = np.random.default_rng(seed=55555)
    fid = _make_fiducial(rng)
    gamma = 0.025
    bk = BKState(alpha=0.0, beta=0.0, gamma=gamma, adot=0.0, bdot=0.0, gdot=0.0)
    J = np.asarray(dcart_dbk(bk, fid))

    expected_col_alpha = (1.0 / gamma) * np.asarray(fid.a)
    expected_col_beta = (1.0 / gamma) * np.asarray(fid.b)
    expected_col_gamma = -(1.0 / gamma**2) * np.asarray(fid.n0)

    np.testing.assert_allclose(J[:3, 0], expected_col_alpha, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(J[:3, 1], expected_col_beta, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(J[:3, 2], expected_col_gamma, rtol=1e-13, atol=1e-15)

    # Bottom-right block should equal the top-left block (same shape).
    np.testing.assert_allclose(J[3:, 3:], J[:3, :3], rtol=1e-13, atol=1e-15)

    # Bottom-left block should be zero when adot = bdot = gdot = 0.
    np.testing.assert_allclose(J[3:, :3], np.zeros((3, 3)), atol=1e-15)


# ---------------------------------------------------------------------------
# sigma_gdot_sq vs Cartesian-side energy-bound calculation
# ---------------------------------------------------------------------------


def test_sigma_gdot_sq_consistent_with_energy_bound():
    """sigma_gdot_sq matches the Cartesian-side bound on |gdot|^2 for a bound orbit.

    Derivation: the bound-orbit energy constraint 0.5 |v|^2 <= mu / |r|
    rearranges, in BK at fixed (alpha, beta, gamma, adot, bdot), to
    gdot^2 <= gamma^2 * (2 * mu * gamma^3 - adot^2 - bdot^2),
    which is exactly the formula sigma_gdot_sq returns.
    """
    rng = np.random.default_rng(seed=66666)
    fid = _make_fiducial(rng)
    # Pick (alpha, beta, gamma, adot, bdot) such that the orbit is bound for
    # at least some gdot range.
    bk = BKState(alpha=0.05, beta=-0.03, gamma=1.0 / 40.0, adot=4e-5, bdot=-3e-5, gdot=0.0)
    sigma_sq = sigma_gdot_sq(bk, MU_SUN)

    # The bound-orbit constraint -> |v|^2 <= 2 * mu * gamma.  At the BK state
    # with gdot pinned to +sqrt(sigma_sq), the orbit should be exactly at the
    # boundary (|v|^2 == 2 * mu * gamma).
    assert sigma_sq > 0.0 and np.isfinite(sigma_sq)
    bk_at_boundary = BKState(
        alpha=bk.alpha, beta=bk.beta, gamma=bk.gamma, adot=bk.adot, bdot=bk.bdot, gdot=np.sqrt(sigma_sq)
    )
    cart = np.asarray(bk_to_cartesian(bk_at_boundary, fid)).flatten()
    r_norm = np.linalg.norm(cart[:3])
    v_norm_sq = np.dot(cart[3:], cart[3:])
    energy = 0.5 * v_norm_sq - MU_SUN / r_norm
    # At the boundary, total energy = 0 (parabolic orbit).
    np.testing.assert_allclose(energy, 0.0, atol=1e-12)


def test_sigma_gdot_sq_returns_inf_for_hyperbolic_tangentials():
    """When tangential rates already exceed escape velocity, sigma_gdot_sq
    signals 'no prior' by returning +infinity."""
    bk = BKState(alpha=0.0, beta=0.0, gamma=1.0 / 50.0, adot=1e-3, bdot=1e-3, gdot=0.0)  # huge rates at 50 AU
    assert np.isinf(sigma_gdot_sq(bk, MU_SUN))
