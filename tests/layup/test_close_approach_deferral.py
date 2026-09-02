"""Close-approach IOD candidates are deferred, not discarded (layup#465).

A Gauss root whose trajectory grazes the Earth is expensive to integrate
rather than obviously wrong, and on short main-belt arcs it is the
degenerate branch that collapses onto Earth's own orbit. The picker fits
the other roots first and only returns to it if none of them converged.
"""

import numpy as np
import pytest

from layup.iod import _CLOSE_EARTH_AU, partition_close_approach


class _Obs:
    """Minimal stand-in for the C++ Observation the partition reads."""

    def __init__(self, epoch, observer_position):
        self.epoch = epoch
        self.observer_position = observer_position


class _Cand:
    """Minimal stand-in for an IOD candidate."""

    def __init__(self, state, epoch=0.0):
        self.state = list(state)
        self.epoch = epoch


# Observer sitting at 1 au along x at three epochs, near enough to static
# over the arc that the geometry below is easy to reason about.
_OBS = [_Obs(float(t), (1.0, 0.0, 0.0)) for t in (0.0, 1.0, 2.0)]

# A main-belt-like root: 2.5 au away from the observer, never close.
_FAR = _Cand((3.5, 0.0, 0.0, 0.0, 0.0, 0.0))

# The degenerate branch: sitting essentially on the observer and co-moving,
# so its minimum geocentric distance over the arc is ~0.
_CLOSE = _Cand((1.001, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_far_candidate_is_safe():
    safe, deferred = partition_close_approach(_OBS, [_FAR])
    assert safe == [_FAR]
    assert deferred == []


def test_close_candidate_is_deferred_not_dropped():
    safe, deferred = partition_close_approach(_OBS, [_CLOSE])
    assert safe == []
    assert deferred == [_CLOSE], "close candidates must be kept for fallback"


def test_partition_preserves_every_candidate_and_its_order():
    cands = [_FAR, _CLOSE, _Cand((-4.0, 1.0, 0.0, 0.0, 0.0, 0.0))]
    safe, deferred = partition_close_approach(_OBS, cands)
    assert len(safe) + len(deferred) == len(cands), "no candidate may be lost"
    assert safe == [cands[0], cands[2]]
    assert deferred == [cands[1]]


def test_threshold_is_the_shared_close_earth_constant():
    """Just inside and just outside _CLOSE_EARTH_AU land on opposite sides."""
    inside = _Cand((1.0 + 0.5 * _CLOSE_EARTH_AU, 0.0, 0.0, 0.0, 0.0, 0.0))
    outside = _Cand((1.0 + 2.0 * _CLOSE_EARTH_AU, 0.0, 0.0, 0.0, 0.0, 0.0))
    safe, deferred = partition_close_approach(_OBS, [inside, outside])
    assert deferred == [inside]
    assert safe == [outside]


def test_motion_towards_the_observer_is_detected():
    """The test is over the whole arc, not just the candidate's epoch."""
    # Starts 1 au beyond the observer, arrives at it by the last epoch.
    approaching = _Cand((2.0, 0.0, 0.0, -0.5, 0.0, 0.0))
    safe, deferred = partition_close_approach(_OBS, [approaching])
    assert deferred == [approaching]


def test_unreadable_candidate_is_treated_as_safe():
    class _Bad:
        state = property(lambda self: (_ for _ in ()).throw(RuntimeError("no state")))
        epoch = 0.0

    bad = _Bad()
    safe, deferred = partition_close_approach(_OBS, [bad])
    assert safe == [bad] and deferred == []
