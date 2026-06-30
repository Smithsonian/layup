"""Unit tests for the IAS15 integrator-tuning knobs exposed by the C++ core.

These cover the four process-global getter/setter bindings
(``set/get_ias15_min_dt`` and ``set/get_ias15_adaptive_mode``) that previously
had zero test coverage. They are pure global state -- no ASSIST/REBOUND
ephemeris is touched -- so they run everywhere, including CI without the
ephemeris cache.

The settings are process-wide (file-scope globals in orbit_fit.cpp), so each
test saves and restores the prior value to avoid leaking state into other
tests in the same process.
"""

from __future__ import annotations

import contextlib

import pytest

from layup.routines import (
    get_ias15_adaptive_mode,
    get_ias15_min_dt,
    set_ias15_adaptive_mode,
    set_ias15_min_dt,
)


@contextlib.contextmanager
def _restore_min_dt():
    saved = get_ias15_min_dt()
    try:
        yield
    finally:
        set_ias15_min_dt(saved)


@contextlib.contextmanager
def _restore_adaptive_mode():
    saved = get_ias15_adaptive_mode()
    try:
        yield
    finally:
        set_ias15_adaptive_mode(saved)


def test_min_dt_default_is_unset():
    """By default min_dt is 0.0, the sentinel meaning 'do not override'."""
    assert get_ias15_min_dt() == 0.0


@pytest.mark.parametrize("value", [1e-6, 1e-3, 0.25, 5.0])
def test_min_dt_round_trip(value):
    """set_ias15_min_dt is read back exactly by get_ias15_min_dt."""
    with _restore_min_dt():
        set_ias15_min_dt(value)
        assert get_ias15_min_dt() == value


def test_min_dt_reset_to_zero():
    """Setting back to 0.0 restores the 'unset' sentinel."""
    with _restore_min_dt():
        set_ias15_min_dt(0.1)
        assert get_ias15_min_dt() == 0.1
        set_ias15_min_dt(0.0)
        assert get_ias15_min_dt() == 0.0


def test_adaptive_mode_default_is_unset():
    """By default adaptive_mode is -1, the sentinel meaning 'do not override'."""
    assert get_ias15_adaptive_mode() == -1


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_adaptive_mode_round_trip(mode):
    """set_ias15_adaptive_mode is read back exactly by get_ias15_adaptive_mode."""
    with _restore_adaptive_mode():
        set_ias15_adaptive_mode(mode)
        assert get_ias15_adaptive_mode() == mode


def test_adaptive_mode_reset_to_sentinel():
    """Setting back to -1 restores the 'unset' sentinel."""
    with _restore_adaptive_mode():
        set_ias15_adaptive_mode(2)
        assert get_ias15_adaptive_mode() == 2
        set_ias15_adaptive_mode(-1)
        assert get_ias15_adaptive_mode() == -1
