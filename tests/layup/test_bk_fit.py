"""Layer 2 tests for the universal BK fitter (`run_bk_native_fit`).

These tests cover the LM driver itself.  They reuse the same Gauss IOD +
observation setup as the existing Cartesian fit so the only difference
between the two engines is the parameterization + the energy prior on
gdot, isolating any disagreement to the BK-specific code path.

Tests skip when the ASSIST ephemeris files aren't available, so CI on
machines without `~/Library/Caches/layup/{linux_p1550p2650.440,
sb441-n16.bsp}` is unaffected.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from layup.routines import (
    FitResult,
    Observation,
    get_ephem,
    predict_sequence,
    run_bk_native_fit,
    run_from_vector_with_initial_guess,
)

CACHE = os.path.expanduser("~/Library/Caches/layup")
EPHEM_PLANETS = os.path.join(CACHE, "linux_p1550p2650.440")
EPHEM_SMALLBODIES = os.path.join(CACHE, "sb441-n16.bsp")
EPHEM_AVAILABLE = os.path.exists(EPHEM_PLANETS) and os.path.exists(EPHEM_SMALLBODIES)

# GM_sun in AU^3 / day^2.
MU_SUN = 0.00029591220828559104

pytestmark = pytest.mark.skipif(
    not EPHEM_AVAILABLE,
    reason=f"ASSIST ephemeris missing at {CACHE}; skipping BK-fit Layer 2 tests.",
)


# ---------------------------------------------------------------------------
# Tests that don't need real observations -- exercise the API + early-exit
# guards.
# ---------------------------------------------------------------------------


def test_run_bk_native_fit_returns_fitresult_for_empty_obs():
    """With zero observations, the fit returns a FitResult with flag != 0 and
    does not crash."""
    ephem = get_ephem(CACHE)
    ig = FitResult()
    ig.state = [40.0, 10.0, 5.0, -8e-4, 9e-4, 1e-4]
    ig.epoch = 2460000.5
    result = run_bk_native_fit(ephem, ig, [], MU_SUN)
    assert result.method == "bk_native"
    assert result.flag != 0


def test_run_bk_native_fit_returns_fitresult_for_too_few_obs():
    """With <3 observations the early-exit guard fires; no crash, flag != 0."""
    ephem = get_ephem(CACHE)
    ig = FitResult()
    ig.state = [40.0, 10.0, 5.0, -8e-4, 9e-4, 1e-4]
    ig.epoch = 2460000.5
    obs = [
        Observation.from_astrometry(
            ra=1.57,
            dec=0.1,
            epoch=2459995.5,
            observer_position=[-0.5, 0.8, 0.0],
            observer_velocity=[-0.018, -0.009, 0.0],
        ),
        Observation.from_astrometry(
            ra=1.57,
            dec=0.1,
            epoch=2460005.5,
            observer_position=[-0.5, 0.8, 0.0],
            observer_velocity=[-0.018, -0.009, 0.0],
        ),
    ]
    result = run_bk_native_fit(ephem, ig, obs, MU_SUN)
    assert result.method == "bk_native"
    assert result.flag != 0


# ---------------------------------------------------------------------------
# Synthetic-orbit convergence tests.
#
# We pick a known Cartesian state, generate synthetic observations from it
# via the layup C++ predict path, then feed those observations back into
# both run_bk_native_fit and run_from_vector_with_initial_guess and check
# that (a) BK converges, (b) BK recovers the input state, and (c) the BK
# and Cartesian fits agree at convergence.
# ---------------------------------------------------------------------------


def _generate_synthetic_observations(ephem, truth_state, truth_epoch, obs_times):
    """Generate synthetic Observation objects consistent with `truth_state`
    at `truth_epoch`, observed from a fixed point (Sun in barycentric coords)
    at each of `obs_times`.

    Returns a list of Observation objects whose epochs and rho_hat directions
    match what the truth orbit predicts.
    """
    # Use a fixed observer at the solar system barycenter so the only
    # dynamical content in the synthetic data is the orbit itself.  The
    # observer-velocity is zero -- consistent with a barycenter "observer."
    observer_position = [0.0, 0.0, 0.0]
    observer_velocity = [0.0, 0.0, 0.0]

    # Template observations at the desired times with dummy (ra, dec)
    template = [
        Observation.from_astrometry(
            ra=0.0,
            dec=0.0,
            epoch=float(t),
            observer_position=observer_position,
            observer_velocity=observer_velocity,
        )
        for t in obs_times
    ]

    # Run predict on each template; the FitResult holds the truth state at epoch.
    truth_fit = FitResult()
    truth_fit.state = list(map(float, truth_state))
    truth_fit.epoch = float(truth_epoch)
    cov = np.zeros((6, 6))
    preds = predict_sequence(ephem, truth_fit, template, cov)

    # Build real Observations from the predicted rho unit vectors.
    synth = []
    for t, pr in zip(obs_times, preds):
        rho = np.asarray(pr.rho)
        # Defensive normalization (predict returns a unit vector already).
        rho = rho / np.linalg.norm(rho)
        ra = np.arctan2(rho[1], rho[0])
        dec = np.arcsin(np.clip(rho[2], -1.0, 1.0))
        synth.append(
            Observation.from_astrometry(
                ra=float(ra),
                dec=float(dec),
                epoch=float(t),
                observer_position=observer_position,
                observer_velocity=observer_velocity,
            )
        )
    return synth


def _seed_from_state(state, epoch):
    fit = FitResult()
    fit.state = list(map(float, state))
    fit.epoch = float(epoch)
    return fit


@pytest.mark.parametrize(
    "name, state, arc_days, nobs",
    [
        # ~3 AU mainbelt-ish (well-constrained)
        ("mainbelt_3au_60d", [3.0, 0.0, 0.0, 0.0, 0.0102, 0.001], 60.0, 12),
        # ~40 AU TNO, longer arc
        ("tno_40au_300d", [40.0, 0.0, 5.0, 0.0, 0.00125, 0.0], 300.0, 12),
    ],
)
def test_bk_native_fit_recovers_known_state(name, state, arc_days, nobs):
    """Synthetic obs from a known state, fitted with BK from a perfect seed,
    should recover the input state and produce a tiny chi-square."""
    ephem = get_ephem(CACHE)
    truth_epoch = 2460000.5
    obs_times = np.linspace(truth_epoch - 0.5 * arc_days, truth_epoch + 0.5 * arc_days, nobs)

    obs = _generate_synthetic_observations(ephem, state, truth_epoch, obs_times)
    seed = _seed_from_state(state, truth_epoch)

    result = run_bk_native_fit(ephem, seed, obs, MU_SUN)
    assert result.flag == 0, f"[{name}] BK fit did not converge (flag={result.flag})"
    np.testing.assert_allclose(
        np.asarray(result.state),
        np.asarray(state),
        rtol=1e-6,
        atol=1e-9,
        err_msg=f"[{name}] BK fit did not recover the truth state",
    )
    # 2N residuals, 6 free params, noise-free obs -> chi2 essentially zero.
    assert result.csq < 1e-12, f"[{name}] BK fit chi-square unexpectedly large: {result.csq}"


@pytest.mark.parametrize(
    "name, state, arc_days, nobs, rel_perturb",
    [
        # Modest perturbation -- exercises the LM loop without falling out of basin.
        ("mainbelt_3au_60d_pert", [3.0, 0.0, 0.0, 0.0, 0.0102, 0.001], 60.0, 12, 1e-3),
        ("tno_40au_300d_pert", [40.0, 0.0, 5.0, 0.0, 0.00125, 0.0], 300.0, 12, 1e-3),
    ],
)
def test_bk_native_fit_recovers_from_perturbed_seed(name, state, arc_days, nobs, rel_perturb):
    """With a 0.1%-perturbed seed, the LM loop still converges to the truth state."""
    ephem = get_ephem(CACHE)
    truth_epoch = 2460000.5
    obs_times = np.linspace(truth_epoch - 0.5 * arc_days, truth_epoch + 0.5 * arc_days, nobs)

    obs = _generate_synthetic_observations(ephem, state, truth_epoch, obs_times)

    # Perturb each component by rel_perturb * |component| (deterministic, no RNG).
    perturbed = np.asarray(state) * (1.0 + rel_perturb)
    seed = _seed_from_state(perturbed.tolist(), truth_epoch)

    result = run_bk_native_fit(ephem, seed, obs, MU_SUN)
    assert result.flag == 0, f"[{name}] BK fit did not converge (flag={result.flag})"
    np.testing.assert_allclose(
        np.asarray(result.state),
        np.asarray(state),
        rtol=1e-6,
        atol=1e-9,
        err_msg=f"[{name}] BK fit did not recover truth from perturbed seed",
    )
    # niter should be > 1 since we actually had to iterate.
    assert result.niter >= 1, f"[{name}] niter={result.niter} -- expected at least 1"


@pytest.mark.parametrize(
    "name, state, arc_days, nobs",
    [
        ("mainbelt_3au_60d", [3.0, 0.0, 0.0, 0.0, 0.0102, 0.001], 60.0, 12),
    ],
)
def test_bk_and_cartesian_fits_agree(name, state, arc_days, nobs):
    """For well-constrained synthetic observations, the BK and Cartesian
    engines should converge to states that match to within numerical noise."""
    ephem = get_ephem(CACHE)
    truth_epoch = 2460000.5
    obs_times = np.linspace(truth_epoch - 0.5 * arc_days, truth_epoch + 0.5 * arc_days, nobs)

    obs = _generate_synthetic_observations(ephem, state, truth_epoch, obs_times)
    seed = _seed_from_state(state, truth_epoch)

    bk_result = run_bk_native_fit(ephem, seed, obs, MU_SUN)
    cart_result = run_from_vector_with_initial_guess(ephem, seed, obs)

    assert bk_result.flag == 0, f"[{name}] BK fit failed: {bk_result.flag}"
    assert cart_result.flag == 0, f"[{name}] Cartesian fit failed: {cart_result.flag}"
    np.testing.assert_allclose(
        np.asarray(bk_result.state),
        np.asarray(cart_result.state),
        rtol=1e-6,
        atol=1e-9,
        err_msg=f"[{name}] BK and Cartesian fits disagree at convergence",
    )


# ---------------------------------------------------------------------------
# Engine dispatch through orbitfit._run_fit
# ---------------------------------------------------------------------------


def test_run_fit_dispatch_cartesian():
    """orbitfit._run_fit(engine='cartesian') matches direct
    run_from_vector_with_initial_guess."""
    from layup.orbitfit import _run_fit

    ephem = get_ephem(CACHE)
    state = [3.0, 0.0, 0.0, 0.0, 0.0102, 0.001]
    epoch = 2460000.5
    obs = _generate_synthetic_observations(ephem, state, epoch, np.linspace(epoch - 30, epoch + 30, 12))
    seed = _seed_from_state(state, epoch)

    via_dispatch = _run_fit(ephem, seed, obs, "cartesian")
    direct = run_from_vector_with_initial_guess(ephem, seed, obs)
    np.testing.assert_array_equal(via_dispatch.state, direct.state)
    assert via_dispatch.method == direct.method


def test_run_fit_dispatch_bk_native():
    """orbitfit._run_fit(engine='bk_native') matches direct
    run_bk_native_fit with MU_SUN."""
    from layup.orbitfit import _MU_SUN, _run_fit

    ephem = get_ephem(CACHE)
    state = [3.0, 0.0, 0.0, 0.0, 0.0102, 0.001]
    epoch = 2460000.5
    obs = _generate_synthetic_observations(ephem, state, epoch, np.linspace(epoch - 30, epoch + 30, 12))
    seed = _seed_from_state(state, epoch)

    via_dispatch = _run_fit(ephem, seed, obs, "bk_native")
    direct = run_bk_native_fit(ephem, seed, obs, _MU_SUN)
    np.testing.assert_array_equal(via_dispatch.state, direct.state)
    assert via_dispatch.method == "bk_native"


def test_run_fit_dispatch_unknown_engine_raises():
    """An unrecognized engine name raises ValueError."""
    from layup.orbitfit import _run_fit

    ephem = get_ephem(CACHE)
    seed = _seed_from_state([3.0, 0.0, 0.0, 0.0, 0.01, 0.0], 2460000.5)
    with pytest.raises(ValueError, match="Unknown engine"):
        _run_fit(ephem, seed, [], "not_an_engine")
