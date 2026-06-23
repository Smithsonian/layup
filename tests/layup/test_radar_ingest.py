"""Unit tests for the radar (delay/Doppler) ingest layer (issue #146, P2).

These exercise the Python-side conversion from JPL units (round-trip delay in us,
Doppler in Hz at a transmit frequency) to the fitter's internal units (delay in
days, round-trip range-rate in au/day) and the row dispatch / validation -- all
without ASSIST, so they always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from layup.orbitfit import (
    SPEED_OF_LIGHT,
    US_TO_DAYS,
    _is_radar,
    _is_valid_data,
    _radar_observation,
)

# A radar row carries observer barycentric state (added by orbitfit()) plus the
# JPL observables. freqTx ~ Goldstone X-band.
_F_TX = 8.56e9  # Hz
_DTYPE = [
    ("provID", "U8"),
    ("obsTime", "U32"),
    ("stn", "U4"),
    ("et", "f8"),
    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),
    ("vx", "f8"),
    ("vy", "f8"),
    ("vz", "f8"),
    ("delay", "f8"),
    ("rmsDelay", "f8"),
    ("doppler", "f8"),
    ("rmsDoppler", "f8"),
    ("freqTx", "f8"),
]


def _row(delay=1.0e4, doppler=5.0e3, rmsDelay=1.0, rmsDoppler=0.1, freqTx=_F_TX):
    a = np.array(
        [
            (
                "synth",
                "2021-11-19",
                "251",
                0.0,
                0.42,
                0.81,
                0.35,
                -0.015,
                0.0068,
                0.0029,
                delay,
                rmsDelay,
                doppler,
                rmsDoppler,
                freqTx,
            )
        ],
        dtype=_DTYPE,
    )
    return a, a.dtype.names


def test_unit_constants():
    """Lock the JPL->internal unit conventions."""
    assert US_TO_DAYS == pytest.approx(1.0e-6 / 86400.0)
    # SPEED_OF_LIGHT is au/day (matches predict.cpp).
    assert SPEED_OF_LIGHT == pytest.approx(2.99792458e8 * 86400.0 / 149597870700.0)


def test_radar_observation_unit_conversion():
    """delay us->days; doppler Hz->au/day = -c*Hz/freqTx; uncertainties likewise."""
    a, cols = _row(delay=1.0e4, doppler=5.0e3, rmsDelay=2.0, rmsDoppler=0.5)
    d = a[0]
    o = _radar_observation("synth", d, 2459545.0, cols)
    rd = o.observation_type

    assert type(rd).__name__ == "RadarObservation"
    assert rd.has_delay and rd.has_doppler
    assert rd.delay == pytest.approx(1.0e4 * US_TO_DAYS)
    assert rd.doppler == pytest.approx(-SPEED_OF_LIGHT * 5.0e3 / _F_TX)
    assert o.delay_unc == pytest.approx(2.0 * US_TO_DAYS)
    assert o.doppler_unc == pytest.approx(abs(SPEED_OF_LIGHT * 0.5 / _F_TX))
    assert o.epoch == 2459545.0


def test_radar_observation_delay_only_and_doppler_only():
    """A NaN observable is absent; the corresponding has_* flag is False."""
    a, cols = _row(doppler=np.nan, rmsDoppler=np.nan)
    o = _radar_observation("s", a[0], 2459545.0, cols)
    assert o.observation_type.has_delay and not o.observation_type.has_doppler

    a, cols = _row(delay=np.nan, rmsDelay=np.nan)
    o = _radar_observation("s", a[0], 2459545.0, cols)
    assert o.observation_type.has_doppler and not o.observation_type.has_delay


def test_radar_doppler_requires_freqTx():
    """A Doppler row without a usable freqTx is an error, not a silent NaN."""
    a, cols = _row(freqTx=np.nan)
    with pytest.raises(ValueError, match="freqTx"):
        _radar_observation("s", a[0], 2459545.0, cols)


def test_is_radar_dispatch():
    a, cols = _row()
    assert _is_radar(a[0], cols)
    # A purely optical row (no delay/doppler columns) is not radar.
    opt = np.array([(1.0, 2.0)], dtype=[("ra", "f8"), ("dec", "f8")])
    assert not _is_radar(opt[0], opt.dtype.names)


def test_is_valid_data_accepts_radar_without_radec():
    """A radar-only array (no ra/dec columns) must validate, while a missing
    observer state must not."""
    rows = np.array(
        [
            (
                "s",
                "2021-11-19",
                "251",
                0.0,
                0.42,
                0.81,
                0.35,
                -0.015,
                0.0068,
                0.0029,
                1.0e4,
                1.0,
                5.0e3,
                0.1,
                _F_TX,
            )
        ]
        * 3,
        dtype=_DTYPE,
    )
    assert _is_valid_data(rows)

    # A row with neither a radar observable nor ra/dec is not a usable datum.
    bad = rows.copy()
    bad["delay"][0] = np.nan
    bad["doppler"][0] = np.nan
    assert not _is_valid_data(bad)
