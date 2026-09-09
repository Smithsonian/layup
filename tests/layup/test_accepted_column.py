"""One column that says whether a fit is usable, without reading the taxonomy.

Reviewers on #495 asked for a status column that reads as a predicate --
``if (!error)`` or ``if (success)`` -- rather than an integer to be compared
against a documented list. The two are not interchangeable: ``0`` means
converged today, so ``!error`` works with the values unchanged while ``success``
is false for every good fit unless the values also invert. And renaming ``flag``
is breaking: it appears across three modules, sixteen test files and seven
fixture CSVs that carry it as a column header, so output already written stops
loading.

``accepted`` is the alternative from #498: keep ``flag``, add a derived boolean.
It reads the same, inverts nothing, and does not label ``-1`` (never attempted)
or ``8`` (incremental bookkeeping) as errors, which they are not.

It is derived from the flag the row actually carries, not tracked beside the
other outcome facts, so the two cannot disagree -- and it is read *after* any
stage marker has overwritten the fitter's own verdict, because the overwritten
value is what a reader sees.
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
    FLAG_IMPLAUSIBLE_ORBIT,
    FLAG_INCREMENTAL_NO_FULL_OBS,
    FLAG_NO_ROOT_CONVERGED,
    FLAG_NO_SOLUTION,
    FLAG_NOT_ATTEMPTED,
    OUTCOME_COLUMNS,
    STAGE_COMPLETE,
)
from layup.orbitfit import FitOutcome

ACCEPTED = OUTCOME_COLUMNS.index("accepted")

# Every documented flag other than the accepting one.
REJECTING_FLAGS = [
    FLAG_NOT_ATTEMPTED,
    FLAG_DID_NOT_CONVERGE,
    FLAG_CSQ_TOO_LARGE,
    FLAG_NO_ROOT_CONVERGED,
    FLAG_BUILDUP_FAILED,
    FLAG_NO_SOLUTION,
    FLAG_DEGENERATE_COV,
    FLAG_INCREMENTAL_NO_FULL_OBS,
    FLAG_IMPLAUSIBLE_ORBIT,
]


def test_accepted_is_the_first_outcome_column():
    """It is the one to filter on, so it leads the group rather than sitting
    among the individual verdicts."""
    assert OUTCOME_COLUMNS[0] == "accepted"


def test_accepted_is_one_exactly_for_the_converged_flag():
    row = FitOutcome(converged=True, stage=STAGE_COMPLETE).as_row(FLAG_CONVERGED)
    assert row[ACCEPTED] == 1


@pytest.mark.parametrize("flag", REJECTING_FLAGS)
def test_every_other_flag_is_not_accepted(flag):
    """A single value means usable and it is the only one. Anything else -- a
    rejection, a stage marker, or the never-attempted sentinel -- is not."""
    row = FitOutcome.from_flag(flag).as_row(flag)
    assert row[ACCEPTED] == 0, f"flag {flag} should not read as accepted"


def test_accepted_follows_the_flag_the_row_carries_not_the_fitters_verdict():
    """The case #499 is about: a fit converged, was rejected on chi-square, and
    then had its flag overwritten by a stage marker. The row carries 4, so it is
    not accepted -- even though the outcome facts still record that it converged.
    Deriving from the flag is what keeps the two consistent."""
    outcome = FitOutcome(converged=True, stage=STAGE_COMPLETE, failed_csq=True)
    row = outcome.as_row(FLAG_BUILDUP_FAILED)
    assert row[ACCEPTED] == 0
    assert row[OUTCOME_COLUMNS.index("converged")] == 1
    assert row[OUTCOME_COLUMNS.index("failed_csq")] == 1


def test_accepted_is_a_plain_int_like_the_other_columns():
    """The row goes straight into an i1 column; a bool would be stored without
    complaint."""
    row = FitOutcome(converged=True).as_row(FLAG_CONVERGED)
    assert all(isinstance(v, int) for v in row)
    assert len(row) == len(OUTCOME_COLUMNS)


def test_never_attempted_is_not_accepted_and_not_a_failure():
    """The reason #498 preferred this over renaming flag to ``error``: a row that
    was never fit is not an error, it is an absence. It reads as not accepted,
    with no check marked failed."""
    row = orbitfit.create_empty_result("x", orbitfit._get_result_dtypes("ObjID", []))
    assert row["accepted"][0] == 0
    assert row["flag"][0] == FLAG_NOT_ATTEMPTED
    assert row["failed_csq"][0] == 0
    assert row["failed_cov"][0] == 0
    assert row["failed_physical"][0] == 0


def test_accepted_agrees_with_the_flag_across_the_whole_taxonomy():
    """The column exists so that callers stop writing ``flag == 0`` by hand. If
    the two ever disagree, the column is worse than nothing."""
    for flag in [FLAG_CONVERGED] + REJECTING_FLAGS:
        row = FitOutcome.from_flag(flag).as_row(flag)
        assert row[ACCEPTED] == int(flag == FLAG_CONVERGED), f"disagreement at flag {flag}"


def test_the_column_is_in_the_output_dtype():
    dt = orbitfit._get_result_dtypes("ObjID", [])
    assert "accepted" in dt.names
    assert np.dtype(dt["accepted"]).kind == "i"
    # It stays inside the outcome group, ahead of the fingerprint columns.
    assert dt.names.index("flag") < dt.names.index("accepted")
    assert dt.names[-2:] == ("obs_hash", "nobs_fit")
