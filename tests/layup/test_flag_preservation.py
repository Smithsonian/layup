"""Gate verdicts survive the driver's own flag assignments (issue #499).

`do_fit` marks where a fit gave up: 3 when no IOD root converged on the primary
interval, 4 when the incremental build-up stopped at a segment. Both used to be
assigned unconditionally, overwriting whatever the fitter returned -- so a
candidate that *converged* and was then rejected by a post-convergence gate
(2 = chi-square per degree of freedom above threshold, 6 = degenerate
covariance) was reported as one that never converged, and the reason was lost.

These pin the distinction: a plain non-convergence still gets the driver's
marker, a gate verdict is kept.
"""

import numpy as np
import pytest

import layup.orbitfit as orbitfit
from layup.orbitfit import (
    FLAG_BUILDUP_FAILED,
    FLAG_DID_NOT_CONVERGE,
    FLAG_NO_ROOT_CONVERGED,
)


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
    observations = [object()] * n_obs
    return orbitfit.do_fit(observations, [list(range(n_obs))], cache_dir="/tmp", iod="gauss")


@pytest.mark.parametrize("gate_flag", [2, 6])
def test_gate_verdict_survives_the_no_root_marker(monkeypatch, gate_flag):
    """A converged-then-rejected candidate keeps its reason, not flag 3."""
    result = _drive(monkeypatch, [gate_flag, gate_flag, gate_flag])
    assert result.flag == gate_flag, f"gate verdict {gate_flag} was overwritten with {result.flag}"


def test_plain_nonconvergence_still_gets_the_no_root_marker(monkeypatch):
    result = _drive(monkeypatch, [FLAG_DID_NOT_CONVERGE] * 3)
    assert result.flag == FLAG_NO_ROOT_CONVERGED


def test_constants_match_the_documented_values():
    """The values are an output format; they must not drift silently."""
    assert (FLAG_DID_NOT_CONVERGE, FLAG_NO_ROOT_CONVERGED, FLAG_BUILDUP_FAILED) == (1, 3, 4)
