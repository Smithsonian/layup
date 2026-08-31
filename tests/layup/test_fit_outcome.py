"""Outcome facts are reported side by side, not collapsed into one flag (#499).

`do_fit` marks where a fit gave up -- 3 when no IOD root converged on the primary
interval, 4 when the incremental build-up stopped. Both assignments overwrite
whatever the fitter returned, so a candidate that *converged* and was then
rejected by a post-convergence gate (2 = reduced chi-square above threshold,
6 = degenerate covariance) used to be reported as one that never converged.

Rather than arbitrate between the two facts in a single integer, each is now
reported in its own column. `flag` is unchanged and stays the summary.
"""

import numpy as np
import pytest

import layup.orbitfit as orbitfit
from layup.constants import (
    FLAG_BUILDUP_FAILED,
    FLAG_CONVERGED,
    FLAG_CSQ_TOO_LARGE,
    FLAG_DEGENERATE_COV,
    FLAG_DID_NOT_CONVERGE,
    FLAG_INCREMENTAL_NO_FULL_OBS,
    FLAG_NO_ROOT_CONVERGED,
    FLAG_NO_SOLUTION,
    FLAG_NOT_ATTEMPTED,
    OUTCOME_COLUMNS,
    STAGE_BUILDUP,
    STAGE_COMPLETE,
    STAGE_INCREMENTAL,
    STAGE_NO_CANDIDATES,
    STAGE_NOT_ATTEMPTED,
    STAGE_PRIMARY,
)
from layup.orbitfit import FitOutcome


class _Fit:
    """Stand-in for the C++ FitResult, carrying only what do_fit reads."""

    def __init__(self, flag, csq=1.0):
        self.flag = flag
        self.csq = csq
        self.ndof = 1
        self.state = [1.0, 0.0, 0.0, 0.0, 0.017, 0.0]
        self.epoch = 2460000.5
        self.cov = [0.0] * 36
        self.method = "test"
        self.niter = 1


def _drive(monkeypatch, flags, n_obs=6):
    """Run do_fit with _run_fit returning `flags` in turn, and no real IOD."""
    seq = iter(flags)
    monkeypatch.setattr(orbitfit, "_run_fit", lambda *a, **k: _Fit(next(seq)))
    monkeypatch.setattr(orbitfit, "get_iod", lambda name: (lambda obs, s: [_Fit(0)]))
    monkeypatch.setattr(orbitfit, "filter_candidates_by_residual", lambda cands, *a, **k: (list(cands), None))
    monkeypatch.setattr(orbitfit, "get_ephem", lambda *a, **k: None)
    outcome = FitOutcome()
    observations = [object()] * n_obs
    result = orbitfit.do_fit(
        observations, [list(range(n_obs))], cache_dir="/tmp", iod="gauss", outcome=outcome
    )
    return result, outcome


# --------------------------------------------------------------------------
# The collision the separate columns remove
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gate_flag,column", [(FLAG_CSQ_TOO_LARGE, "failed_csq"), (FLAG_DEGENERATE_COV, "failed_cov")]
)
def test_gate_verdict_survives_the_no_root_marker(monkeypatch, gate_flag, column):
    """flag still reports where it stopped; the gate is reported separately."""
    result, outcome = _drive(monkeypatch, [gate_flag] * 3)
    assert result.flag == FLAG_NO_ROOT_CONVERGED, "the summary flag still marks where the fit stopped"
    assert getattr(outcome, column) is True, "the gate verdict is no longer lost"
    assert outcome.converged is True, "flags 2 and 6 are set after convergence"
    assert outcome.stage == STAGE_PRIMARY


@pytest.mark.parametrize(
    "gate_flag,column", [(FLAG_CSQ_TOO_LARGE, "failed_csq"), (FLAG_DEGENERATE_COV, "failed_cov")]
)
def test_gate_verdict_survives_the_buildup_marker(monkeypatch, gate_flag, column):
    """First fit converges, the full-set refit is gated, build-up then fails."""
    result, outcome = _drive(monkeypatch, [0, gate_flag, gate_flag])
    assert result.flag == FLAG_BUILDUP_FAILED
    assert getattr(outcome, column) is True
    assert outcome.converged is True
    assert outcome.stage == STAGE_BUILDUP


def test_plain_nonconvergence_reports_no_gate(monkeypatch):
    """A fit that never converged must not look like a gated one."""
    result, outcome = _drive(monkeypatch, [FLAG_DID_NOT_CONVERGE] * 3)
    assert result.flag == FLAG_NO_ROOT_CONVERGED
    assert outcome.converged is False
    assert outcome.failed_csq is False and outcome.failed_cov is False
    assert outcome.stage == STAGE_PRIMARY


def test_clean_fit_is_complete_and_ungated(monkeypatch):
    result, outcome = _drive(monkeypatch, [0, 0, 0])
    assert result.flag == FLAG_CONVERGED
    assert outcome.converged is True
    assert outcome.stage == STAGE_COMPLETE
    assert not any((outcome.failed_csq, outcome.failed_cov, outcome.failed_physical))


def test_no_iod_candidates_reports_that_stage(monkeypatch):
    monkeypatch.setattr(orbitfit, "get_iod", lambda name: (lambda obs, s: []))
    monkeypatch.setattr(orbitfit, "get_ephem", lambda *a, **k: None)
    outcome = FitOutcome()
    result = orbitfit.do_fit([object()] * 6, [list(range(6))], "/tmp", iod="gauss", outcome=outcome)
    assert result.flag == FLAG_NO_SOLUTION
    assert outcome.stage == STAGE_NO_CANDIDATES
    assert outcome.converged is False


# --------------------------------------------------------------------------
# The record itself
# --------------------------------------------------------------------------


def test_outcome_is_optional_so_existing_callers_are_unaffected(monkeypatch):
    """do_fit's contract is unchanged when no record is supplied."""
    seq = iter([0, 0, 0])
    monkeypatch.setattr(orbitfit, "_run_fit", lambda *a, **k: _Fit(next(seq)))
    monkeypatch.setattr(orbitfit, "get_iod", lambda name: (lambda obs, s: [_Fit(0)]))
    monkeypatch.setattr(orbitfit, "filter_candidates_by_residual", lambda cands, *a, **k: (list(cands), None))
    monkeypatch.setattr(orbitfit, "get_ephem", lambda *a, **k: None)
    result = orbitfit.do_fit([object()] * 6, [list(range(6))], "/tmp", iod="gauss")
    assert result.flag == FLAG_CONVERGED


@pytest.mark.parametrize(
    "flag,converged,stage",
    [
        (FLAG_CONVERGED, True, STAGE_COMPLETE),
        (FLAG_CSQ_TOO_LARGE, True, STAGE_COMPLETE),
        (FLAG_DEGENERATE_COV, True, STAGE_COMPLETE),
        (FLAG_DID_NOT_CONVERGE, False, STAGE_NOT_ATTEMPTED),
        (FLAG_NO_ROOT_CONVERGED, False, STAGE_PRIMARY),
        (FLAG_BUILDUP_FAILED, False, STAGE_BUILDUP),
        (FLAG_NO_SOLUTION, False, STAGE_NO_CANDIDATES),
        (FLAG_INCREMENTAL_NO_FULL_OBS, False, STAGE_INCREMENTAL),
    ],
)
def test_from_flag_reconstructs_what_the_summary_still_permits(flag, converged, stage):
    outcome = FitOutcome.from_flag(flag)
    assert outcome.converged is converged
    assert outcome.stage == stage


def test_as_row_is_ints_in_column_order():
    outcome = FitOutcome(converged=True, stage=STAGE_COMPLETE, failed_csq=True)
    row = outcome.as_row()
    assert len(row) == len(OUTCOME_COLUMNS)
    assert all(isinstance(v, int) for v in row)
    assert row == (1, STAGE_COMPLETE, 1, 0, 0)


# --------------------------------------------------------------------------
# The output schema
# --------------------------------------------------------------------------


def test_columns_respect_the_existing_layout_invariants():
    """Two layouts are already pinned by test_incremental_fit: the fingerprint
    columns are last, and the optional non-grav columns sit immediately before
    them. The outcome columns are unconditional, so they group with the other
    unconditional columns ahead of both."""
    dt = orbitfit._get_result_dtypes("ObjID", [])
    assert dt.names[-2:] == ("obs_hash", "nobs_fit")
    assert set(OUTCOME_COLUMNS) <= set(dt.names)
    assert dt.names.index("flag") < dt.names.index("converged")

    dt_ng = orbitfit._get_result_dtypes("ObjID", ("A2",))
    assert dt_ng.names[-4:] == ("a2", "a2_unc", "obs_hash", "nobs_fit")
    assert dt_ng.names.index("failed_physical") < dt_ng.names.index("a2")


def test_empty_result_reports_not_attempted():
    dt = orbitfit._get_result_dtypes("ObjID", [])
    row = orbitfit.create_empty_result("x", dt)
    assert row["flag"][0] == FLAG_NOT_ATTEMPTED
    assert row["converged"][0] == 0
    assert row["stage"][0] == STAGE_NOT_ATTEMPTED
