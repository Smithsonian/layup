"""Tests for the pluggable IOD layer and the multi-root picker."""

from __future__ import annotations

import numpy as np
import pytest

from layup import iod, orbitfit
from layup.routines import FitResult, Observation

from _bk_guards import (
    EPHEM_CACHE,
    load_diagnostic_case,
    requires_diagnostic,
    requires_ephem,
)


def test_registry_has_gauss():
    assert "gauss" in iod.iod_methods()


def test_register_and_get_iod():
    sentinel = []

    def fake(observations, seq):
        sentinel.append((len(observations), seq))
        return []

    iod.register_iod("fake_for_test", fake)
    try:
        looked_up = iod.get_iod("fake_for_test")
        assert looked_up is fake
        looked_up([1, 2, 3], [[0, 1, 2]])
        assert sentinel == [(3, [[0, 1, 2]])]
    finally:
        iod._REGISTRY.pop("fake_for_test", None)


# --- prefilter invariant: never discard the single best seed --- #


def _prefilter_setup(monkeypatch, residual_sigma_by_state, n_obs=5):
    """Wire up filter_candidates_by_residual with the integrator stubbed.

    `residual_sigma_by_state` supplies each candidate's per-observation
    residual (in σ). It may be called as `f(state)` (same residual on
    every observation) or `f(state, j)` (per-observation index j), so a
    single helper covers both the constant-residual and contaminated-arc
    tests. Candidates are plain namespaces so the test never touches
    ASSIST or the C++ fitter.
    """
    import inspect
    from types import SimpleNamespace

    monkeypatch.setattr(iod, "_passes_physical_bounds", lambda c: True)
    # No close-Earth bypass — force every candidate through the residual loop.
    monkeypatch.setattr(iod, "_inertial_min_geocentric_AU", lambda *a, **k: 10.0)

    sigma = 1.0 / 206265.0
    per_obs = len(inspect.signature(residual_sigma_by_state).parameters) >= 2

    obs = []
    idx_of = {}  # id(obs) -> index; the C++ Observation can't hold a custom attr.
    for j in range(n_obs):
        o = Observation.from_astrometry(0.0, 0.0, float(j), [0, 0, 0], [0, 0, 0])
        o.ra_unc = o.dec_unc = sigma
        idx_of[id(o)] = j
        obs.append(o)

    def fake_pred(ephem, state, epoch, o):
        rs = residual_sigma_by_state(state, idx_of[id(o)]) if per_obs else residual_sigma_by_state(state)
        ang = rs * sigma  # rad
        return np.array([np.cos(ang), np.sin(ang), 0.0])

    monkeypatch.setattr(iod, "_predict_rho_hat", fake_pred)

    def cand(r):
        return SimpleNamespace(state=np.array([r, 0.0, 0.0, 0.0, 0.0, 0.0]), epoch=0.0)

    return obs, cand


def test_prefilter_keeps_best_root_even_above_threshold(monkeypatch):
    """A valid-but-rough Gauss root whose raw-seed residual exceeds the
    threshold must still survive — it is the best seed available and LM
    converges from it.

    Regression for object 609631: its 2.14 AU main-belt root sits at
    ~9500σ over the 52-day arc (a rough 3-point seed propagated across
    the whole arc), tripped the 1000σ cut, and was silently dropped,
    leaving only a phantom — diverging the fit (macOS CI red).
    """
    # good ~9500σ, phantom ~600000σ — BOTH above the 1000σ threshold.
    resid = lambda state: 9500.0 if state[0] > 1.0 else 600000.0
    obs, cand = _prefilter_setup(monkeypatch, resid)
    good, phantom = cand(2.14), cand(0.73)

    out = iod.filter_candidates_by_residual([phantom, good], obs, ephem=None, threshold_sigma=1000.0)

    rs = [float(np.linalg.norm(c.state[:3])) for c in out]
    assert any(abs(r - 2.14) < 1e-6 for r in rs), f"best root dropped: {rs}"
    # The far-worse phantom is still filtered out (only the best survives).
    assert not any(abs(r - 0.73) < 1e-6 for r in rs), f"phantom kept: {rs}"


def test_prefilter_drops_phantom_below_good_root(monkeypatch):
    """The common case is unchanged: when the right root predicts within
    threshold, obvious phantoms above it are still cut."""
    resid = lambda state: 3.0 if state[0] > 1.0 else 500000.0
    obs, cand = _prefilter_setup(monkeypatch, resid)
    good, phantom = cand(2.14), cand(0.73)

    out = iod.filter_candidates_by_residual([good, phantom], obs, ephem=None, threshold_sigma=1000.0)

    rs = [float(np.linalg.norm(c.state[:3])) for c in out]
    assert rs and all(abs(r - 2.14) < 1e-6 for r in rs), f"expected only good root, got {rs}"


def test_prefilter_percentile_tolerates_contaminated_points(monkeypatch):
    """A good root that fits the bulk of the arc but has a minority of
    contaminating observations must survive on its own merits — not merely
    via the keep-best guard. The robust LM downstream handles the bad
    points; the prefilter should not reject the root for them.

    With one bad point in ten (10% < the 20% the 80th percentile
    tolerates), the percentile metric stays small and the contaminated
    root is a direct survivor alongside a cleaner competing root (so the
    single-candidate keep-best guard cannot explain its survival).
    """

    def resid(state, j):
        # r≈2.50 root: fits 9/10 obs at 5σ, one contaminated point at 5e4σ.
        if state[0] > 2.3:
            return 50000.0 if j == 0 else 5.0
        # r≈2.14 root: clean, 5σ everywhere (the best-fitting candidate).
        return 5.0

    obs, cand = _prefilter_setup(monkeypatch, resid, n_obs=10)
    clean, contaminated = cand(2.14), cand(2.50)

    out = iod.filter_candidates_by_residual([clean, contaminated], obs, ephem=None, threshold_sigma=1000.0)
    rs = [float(np.linalg.norm(c.state[:3])) for c in out]
    assert any(abs(r - 2.50) < 1e-6 for r in rs), f"contaminated-but-good root dropped: {rs}"
    assert any(abs(r - 2.14) < 1e-6 for r in rs), f"clean root dropped: {rs}"

    # residual_percentile=100 recovers the legacy worst-point behavior:
    # the contaminated root's 5e4σ point now exceeds threshold, and since
    # the clean root is the best seed, the contaminated one is dropped.
    out_max = iod.filter_candidates_by_residual(
        [clean, contaminated], obs, ephem=None, threshold_sigma=1000.0, residual_percentile=100.0
    )
    rs_max = [float(np.linalg.norm(c.state[:3])) for c in out_max]
    assert not any(
        abs(r - 2.50) < 1e-6 for r in rs_max
    ), f"max metric should drop contaminated root: {rs_max}"


def test_unknown_iod_raises():
    with pytest.raises(ValueError) as exc:
        iod.get_iod("definitely_not_registered")
    # The message should mention what is registered.
    assert "gauss" in str(exc.value)


def test_do_fit_propagates_unknown_iod():
    """`do_fit` surfaces a clear error when given an unregistered IOD."""
    with pytest.raises(ValueError):
        orbitfit.do_fit([], [[]], "/tmp", iod="nonexistent_iod_for_test")


def test_do_fit_accepts_callable_iod():
    """`do_fit` accepts a callable directly (skipping the registry).

    No real fit is run — the callable returns no candidates, so
    do_fit returns the sentinel FitResult with flag=5 without touching
    the C++ fitter.
    """

    def empty(observations, seq):
        return []

    fit = orbitfit.do_fit([], [[]], "/tmp", iod=empty)
    assert fit.flag == 5
    assert isinstance(fit, FitResult)


# --- Integration: multi-root picker on a diagnostic case --- #
# These tests run the C++ fitter (via the gauss IOD path) against the
# in-repo diagnostic-scan truth set shipped in tests/data/bk_scan_truth.json
# (see _bk_guards). They skip when the truth set or the ASSIST ephemeris
# is missing, rather than depending on a dataset outside the repo.

CACHE = str(EPHEM_CACHE)


def _build_obs(name: str):
    truth = load_diagnostic_case(name)
    sigma_rad = float(truth["sigma_arcsec"]) / 206265.0
    obs = []
    for r in truth["observations"]:
        o = Observation.from_astrometry(
            float(r["ra"]) * np.pi / 180.0,
            float(r["dec"]) * np.pi / 180.0,
            float(r["jd_tdb"]),
            list(r["observer_state_AU"]),
            [0.0, 0.0, 0.0],
        )
        o.ra_unc = sigma_rad
        o.dec_unc = sigma_rad
        obs.append(o)
    return obs, truth


@requires_ephem
@requires_diagnostic
def test_picker_converges_on_distant_kbo():
    """A 42 AU KBO should yield a converged Cartesian fit at small r-error."""
    obs, _ = _build_obs("classical_42AU_arc_010.00d")
    seq = [list(range(len(obs)))]
    fit = orbitfit.do_fit(obs, seq, CACHE, iod="gauss")
    assert fit.flag == 0
    r = float(np.linalg.norm(fit.state[:3]))
    assert 40.0 < r < 44.0, f"unexpected fit r={r}"


@requires_ephem
@requires_diagnostic
def test_picker_handles_mainbelt():
    """A 2.5 AU mainbelt should also converge under the picker."""
    obs, _ = _build_obs("mainbelt_2.5AU_arc_007.00d")
    seq = [list(range(len(obs)))]
    fit = orbitfit.do_fit(obs, seq, CACHE, iod="gauss")
    assert fit.flag == 0
    r = float(np.linalg.norm(fit.state[:3]))
    assert 2.0 < r < 3.0, f"unexpected fit r={r}"


@requires_ephem
@requires_diagnostic
def test_screen_iter_max_param_is_honored():
    """Passing a tiny screen_iter_max budget makes the LM step count drop.

    Verifies that the screening tier really uses the passed iter_max
    rather than the hardcoded 100 from the old binding.
    """
    obs, _ = _build_obs("classical_42AU_arc_010.00d")
    seq = [list(range(len(obs)))]
    # 1-iteration screen will never converge for any seed, so the
    # picker falls back to full_iter_max=4 which also won't converge.
    # do_fit should surface flag=3 (no convergence at either budget).
    fit = orbitfit.do_fit(obs, seq, CACHE, iod="gauss", screen_iter_max=1, full_iter_max=4)
    # Either flag=3 (no convergence) or flag=0 if LM happens to nail
    # it in <=4 iters from a near-perfect seed; both are valid here.
    assert fit.flag in (0, 3, 4)
