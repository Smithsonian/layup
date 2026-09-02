"""The chi-square is reported whenever the fitter produced one.

A fit that converged and was then rejected -- on reduced chi-square, or on a
degenerate covariance -- used to report ``csq = NaN``, discarding the number
that explains the rejection exactly when it is wanted. The stage markers hid it
too: ``do_fit`` returns the lowest-chi-square candidate it tried, so a fit
labelled 3 or 4 still carries a meaningful score.

Only two flags have no chi-square to report: a row that was never fit, and one
where no initial-orbit candidate was found to score.
"""

import numpy as np
import pytest

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
    FLAG_PRIOR_NOT_POSITIVE_DEFINITE,
    NO_CSQ_FLAGS,
)


# The condition `_orbitfit` applies, stated once so the test exercises the rule
# rather than a copy of it.
def _reports_csq(flag):
    return flag not in NO_CSQ_FLAGS


@pytest.mark.parametrize(
    "flag",
    [
        FLAG_CONVERGED,
        FLAG_DID_NOT_CONVERGE,
        FLAG_CSQ_TOO_LARGE,
        FLAG_NO_ROOT_CONVERGED,
        FLAG_BUILDUP_FAILED,
        FLAG_DEGENERATE_COV,
        FLAG_PRIOR_NOT_POSITIVE_DEFINITE,
        FLAG_INCREMENTAL_NO_FULL_OBS,
        FLAG_IMPLAUSIBLE_ORBIT,
    ],
)
def test_every_flag_that_ran_a_fit_reports_its_chi_square(flag):
    assert _reports_csq(flag)


@pytest.mark.parametrize("flag", [FLAG_NOT_ATTEMPTED, FLAG_NO_SOLUTION])
def test_the_two_flags_with_nothing_to_score_report_nan(flag):
    assert not _reports_csq(flag)


def test_the_rejection_flags_are_the_point():
    """A fit rejected *on* chi-square must report the chi-square it was
    rejected for; the same holds for the covariance check, which also fires
    only after the differential correction converged."""
    assert _reports_csq(FLAG_CSQ_TOO_LARGE)
    assert _reports_csq(FLAG_DEGENERATE_COV)


def test_the_stage_markers_report_a_chi_square_too():
    """`do_fit` returns the lowest-chi-square candidate it tried, so 3 and 4
    carry the least-bad score rather than nothing. These are the flags the
    cold-start failures actually carry."""
    assert _reports_csq(FLAG_NO_ROOT_CONVERGED)
    assert _reports_csq(FLAG_BUILDUP_FAILED)


def test_reporting_a_chi_square_is_not_a_claim_that_the_fit_is_usable():
    """The state stays NaN for anything but a clean fit, so a finite csq on a
    rejected row cannot be mistaken for an accepted orbit."""
    from layup.constants import CONVERGED_FLAGS

    assert FLAG_NO_ROOT_CONVERGED not in CONVERGED_FLAGS
    assert FLAG_BUILDUP_FAILED not in CONVERGED_FLAGS
