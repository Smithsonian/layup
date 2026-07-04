"""Incremental / steady-state orbit-fit tests (issue #419).

Covers the observation fingerprint, the skip-unchanged pre-filter, and the
warm-start path (Lever 1 + Lever 2). The pure-fingerprint/schema tests need no
ephemeris; the end-to-end tests perform real fits of the shared micro fixtures,
matching the style of ``test_orbit_fit.py``.
"""

import numpy as np
import pytest

from layup.orbitfit import (
    _carry_forward_result,
    _get_result_dtypes,
    _obs_fingerprint,
    create_empty_result,
    orbitfit,
)
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader

# --------------------------------------------------------------------------- #
# Fingerprint + schema unit tests (no ephemeris)                              #
# --------------------------------------------------------------------------- #


def _tiny_obs():
    dt = np.dtype([("obsTime", "O"), ("ra", "f8"), ("dec", "f8"), ("stn", "O")])
    rows = [
        ("2020-01-01T00:00:00", 10.0, 5.0, "568"),
        ("2020-01-02T00:00:00", 10.1, 5.1, "F51"),
        ("2020-01-03T00:00:00", 10.2, 5.2, "T05"),
    ]
    return np.array(rows, dtype=dt)


def test_result_schema_has_fingerprint_columns():
    dt = _get_result_dtypes("provID")
    assert dt.names[-2:] == ("obs_hash", "nobs_fit")
    # non-grav columns still precede the fingerprint columns
    dt_ng = _get_result_dtypes("provID", ("A2",))
    assert dt_ng.names[-4:] == ("a2", "a2_unc", "obs_hash", "nobs_fit")


def test_fingerprint_deterministic_and_order_independent():
    a = _tiny_obs()
    n1, h1 = _obs_fingerprint(a, a.dtype.names)
    # Same physical observations in a different row order -> identical fingerprint.
    n2, h2 = _obs_fingerprint(a[[2, 0, 1]], a.dtype.names)
    assert (n1, h1) == (n2, h2)
    assert len(h1) == 16 and n1 == 3


def test_fingerprint_sensitive_to_change_and_count():
    a = _tiny_obs()
    _, h = _obs_fingerprint(a, a.dtype.names)
    # A sub-microarcsecond change to one measurement changes the hash.
    nudged = a.copy()
    nudged["dec"][1] += 1e-9
    _, h_nudged = _obs_fingerprint(nudged, a.dtype.names)
    assert h_nudged != h
    # Appending an observation changes both count and hash.
    appended = np.concatenate([a, a[:1]])
    n_app, h_app = _obs_fingerprint(appended, a.dtype.names)
    assert n_app == 4 and h_app != h


def test_fingerprint_ignores_absent_columns():
    # Only columns present in the data are hashed; a data set without any of the
    # fingerprint columns still yields a stable (count-only) fingerprint.
    dt = np.dtype([("mag", "f8")])
    a = np.array([(21.0,), (22.0,)], dtype=dt)
    n, h = _obs_fingerprint(a, a.dtype.names)
    assert n == 2 and isinstance(h, str)


def test_carry_forward_result_round_trips():
    dt = _get_result_dtypes("provID")
    prior = create_empty_result("X", dt)
    prior["flag"], prior["csq"], prior["x"] = 0, 1.23, 0.5
    prior["obs_hash"], prior["nobs_fit"] = "abc123", 7
    out = _carry_forward_result(prior, dt)
    assert out["csq"][0] == 1.23 and out["x"][0] == 0.5
    assert str(out["obs_hash"][0]) == "abc123" and out["nobs_fit"][0] == 7


# --------------------------------------------------------------------------- #
# End-to-end incremental fit tests (real fits of the micro fixture)          #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def micro_fit():
    """A single-object micro fixture and its cold fit (shared across tests)."""
    data = CSVDataReader(
        get_test_filepath("1_random_mpc_ADES_provIDs_no_sats_micro.csv"),
        "csv",
        primary_id_column_name="provID",
    ).read_rows()
    cold = orbitfit(data, cache_dir=None)
    assert cold[0]["flag"] == 0
    return data, cold


def test_cold_fit_populates_fingerprint(micro_fit):
    data, cold = micro_fit
    r = cold[0]
    assert str(r["obs_hash"]) != "" and r["nobs_fit"] == len(data)


def test_skip_unchanged_carries_forward_without_refitting(micro_fit):
    data, cold = micro_fit
    # Perturb the prior state to a value the fitter would never return, but keep
    # the fingerprint intact. If the object is truly skipped, the perturbed value
    # is carried through verbatim; if it were re-fit, it would be corrected.
    prior = cold.copy()
    prior["x"][0] += 123.456
    out = orbitfit(data, cache_dir=None, initial_guess=prior, skip_unchanged=True)
    assert out[0]["x"] == prior["x"][0], "object was re-fit instead of carried forward"
    # Every field is the (perturbed) prior, i.e. a verbatim carry-forward.
    assert all(np.array_equal(np.nan_to_num(prior[c]), np.nan_to_num(out[c])) for c in cold.dtype.names)


def test_changed_obs_is_refit_not_skipped(micro_fit):
    data, cold = micro_fit
    changed = data.copy()
    changed["dec"][0] += 0.02 / 3600.0  # move one observation by 20 mas
    out = orbitfit(changed, cache_dir=None, initial_guess=cold, skip_unchanged=True)
    r = out[0]
    assert r["flag"] == 0
    # A carry-forward would retain the prior fingerprint; a refit stores the new
    # one. The changed observation therefore yields a different obs_hash.
    assert str(r["obs_hash"]) != str(cold[0]["obs_hash"]), "changed object was wrongly skipped"


def test_warm_start_matches_cold(micro_fit):
    data, cold = micro_fit
    # Lever 2: warm-start from the prior fit (no skip) reproduces the cold answer.
    warm = orbitfit(data, cache_dir=None, initial_guess=cold, skip_unchanged=False)
    w, r = warm[0], cold[0]
    assert w["flag"] == 0
    dstate = max(abs(r[c] - w[c]) for c in ("x", "y", "z", "xdot", "ydot", "zdot"))
    assert dstate < 1e-6


def test_skip_without_fingerprint_columns_falls_back_to_fit(micro_fit):
    data, cold = micro_fit
    # A prior catalog lacking the fingerprint columns must not crash and must not
    # skip -- it should simply fit (drop the obs_hash/nobs_fit columns to simulate
    # an older catalog).
    from numpy.lib import recfunctions as rfn

    legacy = rfn.drop_fields(cold, ["obs_hash", "nobs_fit"])
    out = orbitfit(data, cache_dir=None, initial_guess=legacy, skip_unchanged=True)
    assert out[0]["flag"] == 0 and str(out[0]["obs_hash"]) != ""
