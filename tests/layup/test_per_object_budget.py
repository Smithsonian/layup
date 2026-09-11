"""One object that never returns must not withhold every other result (#492).

``_run_pool`` used ``starmap``, which returns only once the slowest task has
finished. Fitting 108 short arcs in a single call ran over an hour with no
output while the same 108 fitted individually took under a minute, because two
of them ground indefinitely and nothing interrupted them. The measured
pathological rate on short MPC arcs is 1.9 per cent at seven days and 7.2 at
fourteen, so at catalogue scale this is the difference between a run finishing
and not.

Nothing here interrupts a running task -- a fit inside the C integrator cannot
be preempted from Python, since a signal handler runs only at a bytecode
boundary. The budget abandons the task and terminates the pool instead.

The module-level helpers exist because the spawn start method pickles the
callable by reference, so a function defined inside a test cannot be sent to a
worker.
"""

import time

import numpy as np
import pytest

from layup.utilities.data_processing_utilities import _run_pool, process_data_by_id


def _quick(data, **kwargs):
    return np.asarray([len(data)])


def _slow(data, **kwargs):
    time.sleep(30)
    return np.asarray([len(data)])


def _quick_by_id(data, primary_id_column_name=None, **kwargs):
    return np.asarray([len(data)])


def _slow_for_b(data, primary_id_column_name=None, **kwargs):
    """Grind forever on one particular object, return promptly for the rest."""
    if str(data[primary_id_column_name][0]) == "b":
        time.sleep(30)
    return np.asarray([len(data)])


ROWS = np.array([("a",), ("b",), ("c",)], dtype=[("provID", "U4")])


def test_without_a_budget_the_behaviour_is_unchanged():
    out = _run_pool([(_quick, np.zeros(3), {}), (_quick, np.zeros(5), {})], 2)
    assert sorted(out.tolist()) == [3, 5]


def test_a_hung_task_no_longer_withholds_the_others():
    """The point of the issue: the fast results come back."""
    tasks = [(_quick, np.zeros(3), {}), (_slow, np.zeros(9), {}), (_quick, np.zeros(5), {})]
    t0 = time.monotonic()
    out = _run_pool(tasks, 3, per_task_budget_s=2.0)
    elapsed = time.monotonic() - t0
    assert sorted(out.tolist()) == [3, 5], "the two quick tasks must survive"
    assert elapsed < 25, f"returned in {elapsed:.1f}s; must not wait out the hung task"


def test_the_abandoned_object_is_named(caplog):
    """It must say which object was dropped, not silently omit it."""
    tasks = [(_quick, np.zeros(3), {}), (_slow, np.zeros(9), {})]
    with caplog.at_level("WARNING"):
        _run_pool(tasks, 2, per_task_budget_s=2.0, task_labels=["alpha", "beta"])
    assert "Abandoned 1 of 2" in caplog.text
    assert "beta" in caplog.text, "the abandoned task must be identified"


def test_a_generous_budget_abandons_nothing():
    tasks = [(_quick, np.zeros(3), {}), (_quick, np.zeros(5), {})]
    out = _run_pool(tasks, 2, per_task_budget_s=60.0)
    assert sorted(out.tolist()) == [3, 5]


def test_process_data_by_id_threads_the_budget():
    """The per-object entry point: one object grinds, the other two return."""
    t0 = time.monotonic()
    out = process_data_by_id(ROWS, 3, _slow_for_b, "provID", per_object_budget_s=2.0)
    elapsed = time.monotonic() - t0
    assert out.tolist() == [1, 1], "objects a and c must come back"
    assert elapsed < 25, f"returned in {elapsed:.1f}s"


def test_process_data_by_id_is_unchanged_without_a_budget():
    out = process_data_by_id(ROWS, 3, _quick_by_id, "provID")
    assert sorted(out.tolist()) == [1, 1, 1]
