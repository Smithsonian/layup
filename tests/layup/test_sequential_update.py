"""Sequential / information-filter update tests (issue #419, Lever 3).

Validates that ``sequential_update`` (prior + new observations only) reproduces a
full batch refit over all observations, tightens the covariance, and falls back
to a full refit when the update is too nonlinear.

The prior is built by warm-starting from a full-arc seed rather than a cold fit
of the short "old" sub-arc: a cold angles-only fit of a truncated arc is
degenerate (see the opposition-degeneracy note in the IOD code) and its wild seed
state sends the integrator into the IAS15 close-approach grind. A steady-state
update always has a real prior in hand, so warm-starting is both faster and more
representative.
"""

import numpy as np
import pytest

from layup.orbitfit import orbitfit, sequential_update
from layup.utilities.data_processing_utilities import parse_fit_result
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader


def _cov6(fit):
    return np.array(list(fit.cov)).reshape(6, 6)


@pytest.fixture(scope="module")
def fits():
    """Time-split one object's obs and produce: a prior fit over the OLD obs, and
    a full batch refit over ALL obs (both at the same epoch), computed once."""
    data = CSVDataReader(
        get_test_filepath("1_random_mpc_ADES_provIDs_no_sats_micro.csv"),
        "csv",
        primary_id_column_name="provID",
    ).read_rows()
    data = np.sort(data, order="obsTime", kind="mergesort")
    split = (2 * len(data)) // 3
    old, new = data[:split], data[split:]

    seed = orbitfit(data, cache_dir=None)  # well-constrained full-arc seed
    assert seed[0]["flag"] == 0, "seed fit did not converge"
    prior = orbitfit(old, cache_dir=None, initial_guess=seed)  # old-obs fit (warm)
    assert prior[0]["flag"] == 0, "prior (old-only) fit did not converge"
    full = orbitfit(data, cache_dir=None, initial_guess=seed)  # reference (same epoch)
    assert full[0]["flag"] == 0, "full refit did not converge"
    return {"data": data, "old": old, "new": new, "prior": prior, "full": full}


def test_sequential_update_matches_full_refit(fits):
    seq = sequential_update(fits["prior"], fits["new"], cache_dir=None, all_data=fits["data"])
    assert seq.flag == 0 and seq.method == "sequential_update"

    fs = np.array([fits["full"][0][c] for c in ("x", "y", "z", "xdot", "ydot", "zdot")])
    ss = np.array(seq.state)
    assert np.linalg.norm(ss - fs) / np.linalg.norm(fs) < 1e-4, "state disagrees with full refit"

    sf = np.sqrt(np.diag(_cov6(parse_fit_result(fits["full"]))))
    sq = np.sqrt(np.diag(_cov6(seq)))
    assert np.max(np.abs(sq - sf) / sf) < 1e-2, "covariance disagrees with full refit"


def test_sequential_update_tightens_covariance(fits):
    seq = sequential_update(fits["prior"], fits["new"], cache_dir=None, all_data=fits["data"])
    assert seq.flag == 0
    # Adding observations adds information, so no posterior variance can exceed the
    # prior (Lambda0 + B^T W B >= Lambda0 => posterior cov <= prior cov).
    sp = np.sqrt(np.diag(_cov6(parse_fit_result(fits["prior"]))))
    sq = np.sqrt(np.diag(_cov6(seq)))
    assert np.all(sq <= sp * (1 + 1e-9)), "new observations must not loosen the covariance"
    assert np.any(sq < sp), "new observations should tighten at least one component"


def test_nonlinearity_gate_falls_back_to_full_refit(fits):
    # An unreachably tight gate forces the fallback for any nonzero update.
    res = sequential_update(
        fits["prior"], fits["new"], cache_dir=None, all_data=fits["data"], max_update_sigma=1e-12
    )
    assert res.flag == 0 and res.method == "orbit_fit", "should have fallen back to a full refit"


def test_nonlinearity_gate_flags_when_no_fallback_data(fits):
    # Same tight gate but no all_data to refit -> the result is flagged (8), not skipped.
    res = sequential_update(
        fits["prior"], fits["new"], cache_dir=None, all_data=None, max_update_sigma=1e-12
    )
    assert res.flag == 8
