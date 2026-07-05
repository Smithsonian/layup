"""Steady-state incremental driver tests (issue #419 capstone).

Exercises the per-object routing of ``incremental_orbitfit``: skip (unchanged),
sequential update (append-only new obs), full refit (a modified old obs), and
cold fit (no prior). Priors are warm-started from a full-arc seed to avoid the
cold short-arc IAS15 grind, matching the sequential-update tests.
"""

import numpy as np
import pytest

from layup.orbitfit import incremental_orbitfit, orbitfit
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader


@pytest.fixture(scope="module")
def cyc():
    data = CSVDataReader(
        get_test_filepath("1_random_mpc_ADES_provIDs_no_sats_micro.csv"),
        "csv",
        primary_id_column_name="provID",
    ).read_rows()
    data = np.sort(data, order="obsTime", kind="mergesort")
    split = (2 * len(data)) // 3
    old, new = data[:split], data[split:]
    seed = orbitfit(data, cache_dir=None)
    assert seed[0]["flag"] == 0
    prior_old = orbitfit(old, cache_dir=None, initial_guess=seed)  # prior over OLD obs
    prior_all = orbitfit(data, cache_dir=None, initial_guess=seed)  # fit over ALL (reference + skip prior)
    assert prior_old[0]["flag"] == 0 and prior_all[0]["flag"] == 0
    return {"data": data, "old": old, "new": new, "prior_old": prior_old, "prior_all": prior_all}


def test_skip_route_carries_forward(cyc):
    # Prior was fit over the same (all) obs -> fingerprint matches -> skip.
    out, routing = incremental_orbitfit(cyc["data"], None, cyc["prior_all"], prior_obs=cyc["data"])
    assert routing == {"skip": 1}
    # The carried row is byte-identical to the prior.
    assert all(
        np.array_equal(np.nan_to_num(cyc["prior_all"][c]), np.nan_to_num(out[c])) for c in out.dtype.names
    )


def test_sequential_route_matches_full_refit(cyc):
    out, routing = incremental_orbitfit(cyc["data"], None, cyc["prior_old"], prior_obs=cyc["old"])
    assert routing == {"sequential": 1}
    assert out[0]["flag"] == 0 and str(out[0]["method"]) == "sequential_update"
    # The sequential update reproduces the full refit over all obs.
    ref = cyc["prior_all"][0]
    fs = np.array([ref[c] for c in ("x", "y", "z", "xdot", "ydot", "zdot")])
    ss = np.array([out[0][c] for c in ("x", "y", "z", "xdot", "ydot", "zdot")])
    assert np.linalg.norm(ss - fs) / np.linalg.norm(fs) < 1e-4
    # The updated row carries the current-obs fingerprint (for next cycle's skip).
    assert str(out[0]["obs_hash"]) != "" and out[0]["nobs_fit"] == len(cyc["data"])


def test_modified_old_obs_forces_full_refit(cyc):
    # Change one OLD observation -> not append-only -> full refit route.
    changed = cyc["data"].copy()
    changed["dec"][0] += 0.02 / 3600.0
    out, routing = incremental_orbitfit(changed, None, cyc["prior_old"], prior_obs=cyc["old"])
    assert routing == {"full": 1}
    assert out[0]["flag"] == 0


def test_cold_route_when_no_prior(cyc):
    out, routing = incremental_orbitfit(cyc["data"], None, None)
    assert routing == {"cold": 1}
    assert out[0]["flag"] == 0 and str(out[0]["obs_hash"]) != ""


def test_no_prior_obs_falls_back_to_full_refit(cyc):
    # A changed object with a prior but no prior_obs cannot be diffed -> full refit.
    out, routing = incremental_orbitfit(cyc["data"], None, cyc["prior_old"], prior_obs=None)
    assert routing == {"full": 1}
    assert out[0]["flag"] == 0
