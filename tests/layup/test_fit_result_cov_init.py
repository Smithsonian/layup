"""A flagged fit must return a deterministic covariance (#547).

``orbit_fit`` fills ``FitResult::cov`` only inside ``if (flag == 0)``, and
``run_sequential_update`` returns early at the non-positive-definite prior check
without reaching its fill either. ``FitResult::cov`` had no initialiser, so
those paths handed back whatever was on the stack: measured on the flag-7 path
before the fix, 31 to 32 of the 36 entries were non-zero denormals around
1.9e-313, and **the count differed between identical calls** -- so the same
object could score differently on two machines, and a consumer could not tell an
absent covariance from a real one.

``bk_fit`` already did the right thing (``result.cov.fill(0.0)``), but a
per-site fill only covers the sites someone remembered. The fix is a default
member initialiser on the declaration in ``fit_result.cpp``, which covers every
construction site including ones added later.

Zero rather than NaN, matching the convention ``bk_fit`` established. The flag,
not the covariance, is what says whether a fit is usable.

⚠️ These must run against the C++ path. A Python-constructed ``FitResult()`` is
value-initialised by pybind11's ``py::init<>()`` and reads as all-zero **with or
without** the fix, so a test built on one passes either way and proves nothing.
"""

import numpy as np
import pytest

from layup.routines import FitResult, get_ephem, run_sequential_update

from _bk_guards import EPHEM_CACHE, requires_ephem

pytestmark = requires_ephem

CACHE = str(EPHEM_CACHE)
STATE = [40.0, 10.0, 5.0, -8e-4, 9e-4, 1e-4]
EPOCH = 2460000.5

# Prior covariance not positive-definite, so the LLT check fails and
# run_sequential_update returns before it would populate `cov`. The docstring on
# the binding states this contract: "Fails (flag != 0) if the prior covariance is
# not positive-definite."
FLAG_PRIOR_NOT_POSITIVE_DEFINITE = 7


def _degenerate_prior():
    prior = FitResult()
    prior.state = STATE
    prior.epoch = EPOCH
    prior.cov = [0.0] * 36
    return prior


def test_early_return_path_yields_a_zeroed_covariance():
    result = run_sequential_update(get_ephem(CACHE), _degenerate_prior(), [])
    assert result.flag == FLAG_PRIOR_NOT_POSITIVE_DEFINITE
    cov = np.asarray(list(result.cov), dtype=float)
    assert np.all(cov == 0.0), f"{int((cov != 0).sum())} of 36 entries are non-zero"


def test_the_zeroed_covariance_is_reproducible():
    """The symptom was non-determinism, not merely non-zero values: repeated
    identical calls returned different numbers of dirty entries."""
    ephem = get_ephem(CACHE)
    seen = {tuple(run_sequential_update(ephem, _degenerate_prior(), []).cov) for _ in range(8)}
    assert seen == {(0.0,) * 36}, "covariance differs between identical calls"


def test_the_returned_covariance_is_finite():
    """A consumer that inverts or averages a returned covariance must not be
    handed a trap value."""
    result = run_sequential_update(get_ephem(CACHE), _degenerate_prior(), [])
    assert np.all(np.isfinite(np.asarray(list(result.cov), dtype=float)))
