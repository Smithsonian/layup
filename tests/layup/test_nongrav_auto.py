"""Adaptive non-grav selection logic for fit_nongrav="auto" (issue #357).

These unit-test the decision layer directly with crafted fit results, so they run
without the ASSIST ephemeris: the gravity-acceptance gate, the chi-square +
per-parameter significance test, and the parsimony ladder in _select_nongrav_auto.
The end-to-end orbitfit() paths are covered in test_nongrav_a2.py.
"""

import types

from layup import orbitfit as O


def _res(**kw):
    """A minimal stand-in for a C++ FitResult (only the fields the auto logic reads)."""
    d = dict(
        csq=0.0,
        ndof=100,
        flag=0,
        nongrav_mask=0,
        a1=0.0,
        a1_unc=1.0,
        a2=0.0,
        a2_unc=1.0,
        a3=0.0,
        a3_unc=1.0,
    )
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_gravity_acceptable_gate():
    assert O._gravity_fit_acceptable(csq=50.0, ndof=100)  # reduced chi2 0.5 -> acceptable
    assert not O._gravity_fit_acceptable(csq=300.0, ndof=100)  # reduced chi2 3.0 -> not
    assert O._gravity_fit_acceptable(csq=1e9, ndof=0)  # no dof to judge -> acceptable


def test_nongrav_warranted_requires_chi2_drop_and_significance():
    # Big chi2 drop and a strongly-determined A2 -> warranted.
    good = _res(csq=10.0, a2=1e-13, a2_unc=1e-15)  # |A2|/sigma = 100
    assert O._nongrav_warranted(csq_gravity=1000.0, res_ng=good, names=("A2",))

    # Chi2 drop below the per-parameter threshold -> not warranted.
    small_drop = _res(csq=995.0, a2=1e-13, a2_unc=1e-15)  # drop 5 < 9
    assert not O._nongrav_warranted(1000.0, small_drop, ("A2",))

    # Large chi2 drop but the parameter itself is not significant -> not warranted.
    insignificant = _res(csq=10.0, a2=1e-15, a2_unc=1e-15)  # |A2|/sigma = 1
    assert not O._nongrav_warranted(1000.0, insignificant, ("A2",))


def test_nongrav_warranted_multiparam_needs_every_param_significant():
    both = _res(csq=10.0, a1=1e-13, a1_unc=1e-15, a2=1e-13, a2_unc=1e-15)
    assert O._nongrav_warranted(1000.0, both, ("A1", "A2"))  # drop 990 > 18, both sig
    a1_weak = _res(csq=10.0, a1=1e-15, a1_unc=1e-15, a2=1e-13, a2_unc=1e-15)
    assert not O._nongrav_warranted(1000.0, a1_weak, ("A1", "A2"))  # A1 not significant


def test_select_auto_keeps_gravity_when_acceptable(monkeypatch):
    tried = []
    monkeypatch.setattr(O, "run_from_vector_with_initial_guess", lambda *a, **k: tried.append(k))
    grav = _res(csq=50.0, ndof=100)  # reduced chi2 0.5
    out = O._select_nongrav_auto(assist_ephem=None, res_grav=grav, observations=None)
    assert out is grav and not tried  # never attempted a non-grav fit


def test_select_auto_adopts_most_parsimonious(monkeypatch):
    grav = _res(csq=1000.0, ndof=100)  # reduced chi2 10 -> try non-gravs

    def fake_fit(ephem, res, obs, nongrav_mask=0, **kwargs):
        # A2-only already resolves it well and A2 is significant.
        return _res(csq=10.0, flag=0, nongrav_mask=nongrav_mask, a2=1e-13, a2_unc=1e-15)

    monkeypatch.setattr(O, "run_from_vector_with_initial_guess", fake_fit)
    out = O._select_nongrav_auto(None, grav, None)
    assert out.csq == 10.0
    assert out.nongrav_mask == O._NONGRAV_BITS["A2"]  # stopped at the parsimonious A2 rung


def test_select_auto_escalates_when_a2_alone_insufficient(monkeypatch):
    grav = _res(csq=1000.0, ndof=100)

    def fake_fit(ephem, res, obs, nongrav_mask=0, **kwargs):
        # No SINGLE parameter is warranted on its own -- each is insignificant --
        # but A1+A2 together land significant with a big drop, so the selector
        # has to escalate past the whole one-parameter tier to find it.
        singles = {O._NONGRAV_BITS[n] for n in ("A1", "A2", "A3")}
        if nongrav_mask in singles:
            return _res(
                csq=995.0,
                flag=0,
                nongrav_mask=nongrav_mask,
                a1=1e-15,
                a1_unc=1e-15,
                a2=1e-15,
                a2_unc=1e-15,
                a3=1e-15,
                a3_unc=1e-15,
            )
        return _res(
            csq=10.0, flag=0, nongrav_mask=nongrav_mask, a1=1e-13, a1_unc=1e-15, a2=1e-13, a2_unc=1e-15
        )

    monkeypatch.setattr(O, "run_from_vector_with_initial_guess", fake_fit)
    out = O._select_nongrav_auto(None, grav, None)
    assert out.nongrav_mask == (O._NONGRAV_BITS["A1"] | O._NONGRAV_BITS["A2"])


def test_select_auto_falls_back_when_all_illconditioned(monkeypatch):
    grav = _res(csq=1000.0, ndof=100)
    # Every non-grav fit trips the weak-constraint / conditioning guard (flag 6).
    monkeypatch.setattr(O, "run_from_vector_with_initial_guess", lambda *a, **k: _res(flag=6, csq=1.0))
    out = O._select_nongrav_auto(None, grav, None)
    assert out is grav  # keep the gravity-only fit


def test_thresholds_default_values():
    t = O.NongravAutoThresholds()
    assert (t.accept_reduced_chi2, t.delta_chi2_per_param, t.nsigma) == (1.5, 9.0, 3.0)


def test_gravity_acceptable_gate_is_tunable():
    # reduced chi2 = 2.0: acceptable under the default 1.5? no. Under a raised 3.0? yes.
    assert not O._gravity_fit_acceptable(csq=200.0, ndof=100)
    lenient = O.NongravAutoThresholds(accept_reduced_chi2=3.0)
    assert O._gravity_fit_acceptable(csq=200.0, ndof=100, thresholds=lenient)
    strict = O.NongravAutoThresholds(accept_reduced_chi2=0.1)
    assert not O._gravity_fit_acceptable(csq=50.0, ndof=100, thresholds=strict)  # 0.5 > 0.1


def test_nongrav_warranted_is_tunable():
    # chi2 drop 5 per one param: not warranted at default 9.0, warranted at 4.0.
    res_ng = _res(csq=995.0, a2=1e-13, a2_unc=1e-15)  # drop 5, A2 strongly significant
    assert not O._nongrav_warranted(1000.0, res_ng, ("A2",))
    assert O._nongrav_warranted(
        1000.0, res_ng, ("A2",), thresholds=O.NongravAutoThresholds(delta_chi2_per_param=4.0)
    )
    # |A2|/sigma = 5: significant at nsigma=3 (default), not at nsigma=10.
    res_sig = _res(csq=10.0, a2=5e-15, a2_unc=1e-15)
    assert O._nongrav_warranted(1000.0, res_sig, ("A2",))
    assert not O._nongrav_warranted(1000.0, res_sig, ("A2",), thresholds=O.NongravAutoThresholds(nsigma=10.0))


def test_select_auto_threshold_suppresses_nongrav(monkeypatch):
    # A case that adopts A2 under the default thresholds...
    grav = _res(csq=1000.0, ndof=100)  # reduced chi2 10 -> non-grav tried by default

    def fake_fit(ephem, res, obs, nongrav_mask=0, **kwargs):
        return _res(csq=10.0, flag=0, nongrav_mask=nongrav_mask, a2=1e-13, a2_unc=1e-15)

    monkeypatch.setattr(O, "run_from_vector_with_initial_guess", fake_fit)
    assert O._select_nongrav_auto(None, grav, None).nongrav_mask == O._NONGRAV_BITS["A2"]
    # ...is kept gravity-only when the acceptance threshold is raised above its reduced chi2.
    tried = []
    monkeypatch.setattr(O, "run_from_vector_with_initial_guess", lambda *a, **k: tried.append(k))
    lenient = O.NongravAutoThresholds(accept_reduced_chi2=20.0)
    out = O._select_nongrav_auto(None, grav, None, thresholds=lenient)
    assert out is grav and not tried


# --- #544: single-parameter rungs, and best-in-tier selection --- #


def _significant(mask, csq, **kw):
    """A converged, well-conditioned fit with every requested parameter at 100 sigma."""
    vals = {}
    for name, bit in O._NONGRAV_BITS.items():
        if mask & bit:
            vals[name.lower()] = 1e-13
            vals[name.lower() + "_unc"] = 1e-15
    vals.update(kw)
    return _res(csq=csq, flag=0, nongrav_mask=mask, **vals)


def test_a1_alone_is_reachable(monkeypatch):
    """The ladder ran A2 -> A1A2 -> A1A2A3, so a radial-only acceleration could
    only be found through a rung already fitting A2 beside it. On 1I/'Oumuamua
    that is fatal: A1 alone is 10.9 sigma and A1+A2+A3 puts it at 0.9."""
    grav = _res(csq=1000.0, ndof=100)
    A1, A2, A3 = (O._NONGRAV_BITS[n] for n in ("A1", "A2", "A3"))

    def fake_fit(ephem, res, obs, nongrav_mask=0, **kwargs):
        if nongrav_mask == A1:
            return _significant(A1, csq=100.0)
        return _res(
            csq=999.0,
            flag=0,
            nongrav_mask=nongrav_mask,
            a1=1e-15,
            a1_unc=1e-15,
            a2=1e-15,
            a2_unc=1e-15,
            a3=1e-15,
            a3_unc=1e-15,
        )

    monkeypatch.setattr(O, "run_from_vector_with_initial_guess", fake_fit)
    assert O._select_nongrav_auto(None, grav, None).nongrav_mask == A1


def test_a3_alone_is_reachable(monkeypatch):
    """The dark comets are the A3 case -- A3 significant, A2 not."""
    grav = _res(csq=1000.0, ndof=100)
    A3 = O._NONGRAV_BITS["A3"]

    def fake_fit(ephem, res, obs, nongrav_mask=0, **kwargs):
        if nongrav_mask == A3:
            return _significant(A3, csq=100.0)
        return _res(
            csq=999.0,
            flag=0,
            nongrav_mask=nongrav_mask,
            a1=1e-15,
            a1_unc=1e-15,
            a2=1e-15,
            a2_unc=1e-15,
            a3=1e-15,
            a3_unc=1e-15,
        )

    monkeypatch.setattr(O, "run_from_vector_with_initial_guess", fake_fit)
    assert O._select_nongrav_auto(None, grav, None).nongrav_mask == A3


def test_best_in_tier_not_first_in_tier(monkeypatch):
    """Two warranted one-parameter models must be separated by fit quality, not
    by the order the tier is written in.

    This is the 1I case: A2 is warranted at 3.7 sigma with dchi2 13.9, A1 at
    10.9 sigma with dchi2 119.0, and A2 is tried first. Returning the first
    warranted model makes the answer an artifact of the tuple's order.
    """
    grav = _res(csq=1000.0, ndof=100)
    A1, A2 = O._NONGRAV_BITS["A1"], O._NONGRAV_BITS["A2"]

    def fake_fit(ephem, res, obs, nongrav_mask=0, **kwargs):
        if nongrav_mask == A2:
            return _significant(A2, csq=986.0)  # warranted, but barely
        if nongrav_mask == A1:
            return _significant(A1, csq=881.0)  # warranted, and far better
        return _res(csq=1000.0, flag=1, nongrav_mask=nongrav_mask)

    monkeypatch.setattr(O, "run_from_vector_with_initial_guess", fake_fit)
    out = O._select_nongrav_auto(None, grav, None)
    assert out.nongrav_mask == A1, "should take the better warranted single, not the first"


def test_parsimony_still_beats_quality_across_tiers(monkeypatch):
    """A warranted single is preferred to a warranted pair even when the pair
    fits better -- tiers are searched in order, so parsimony is not traded away."""
    grav = _res(csq=1000.0, ndof=100)
    A1, A2 = O._NONGRAV_BITS["A1"], O._NONGRAV_BITS["A2"]

    def fake_fit(ephem, res, obs, nongrav_mask=0, **kwargs):
        if nongrav_mask == A1:
            return _significant(A1, csq=900.0)
        if nongrav_mask == (A1 | A2):
            return _significant(A1 | A2, csq=10.0)  # much better, but two parameters
        return _res(csq=1000.0, flag=1, nongrav_mask=nongrav_mask)

    monkeypatch.setattr(O, "run_from_vector_with_initial_guess", fake_fit)
    assert O._select_nongrav_auto(None, grav, None).nongrav_mask == A1
