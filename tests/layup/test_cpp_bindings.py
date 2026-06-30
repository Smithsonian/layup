"""Unit tests for the small C++ <-> Python binding surface of the layup core.

These cover bindings that previously had no direct coverage and need no ASSIST
ephemeris, so they run anywhere (including CI without the ephemeris cache):

  * ``numpy_to_eigen`` (the flat-array -> Eigen matrix helper used by predict),
  * the ``FitResult`` and ``PredictResult`` result structs (field round-trips),
  * the ``Observation`` streak/radar factories and their variant payloads.

The Gauss IOD, IAS15 settings and BK-basis bindings are covered separately in
test_gauss.py / test_ias15_settings.py / test_bk_basis.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from layup.routines import (
    FitResult,
    Observation,
    PredictResult,
    RadarObservation,
    StreakObservation,
    numpy_to_eigen,
)

# ---------------------------------------------------------------------------
# numpy_to_eigen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m,n", [(1, 1), (2, 3), (6, 6), (3, 1), (1, 4)])
def test_numpy_to_eigen_round_trip(m, n):
    """A flat (row-major) array becomes the matching m x n matrix."""
    flat = list(np.arange(m * n, dtype=float))
    mat = np.asarray(numpy_to_eigen(flat, m, n))
    assert mat.shape == (m, n)
    np.testing.assert_array_equal(mat, np.array(flat).reshape(m, n))


def test_numpy_to_eigen_size_mismatch_raises():
    """A flat array whose length != m*n is rejected."""
    with pytest.raises(Exception):
        numpy_to_eigen([1.0, 2.0, 3.0], 2, 2)


# ---------------------------------------------------------------------------
# FitResult / PredictResult field round-trips
# ---------------------------------------------------------------------------


def test_fitresult_field_round_trip():
    r = FitResult()
    r.csq = 12.5
    r.ndof = 7
    r.epoch = 2460000.5
    r.niter = 4
    r.method = "orbit_fit"
    r.flag = 0
    r.state = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    r.cov = [float(i) for i in range(36)]
    assert r.csq == 12.5
    assert r.ndof == 7
    assert r.epoch == 2460000.5
    assert r.niter == 4
    assert r.method == "orbit_fit"
    assert r.flag == 0
    assert list(r.state) == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    assert list(r.cov) == [float(i) for i in range(36)]


def test_fitresult_nongrav_fields():
    """The A1/A2/A3 non-grav fields (issue #351) round-trip and default to zero."""
    r = FitResult()
    assert (r.nongrav_mask, r.a1, r.a2, r.a3) == (0, 0.0, 0.0, 0.0)
    r.nongrav_mask = 2  # A2 bit
    r.a2 = 1.5e-14
    r.a2_unc = 3.0e-15
    assert r.nongrav_mask == 2
    assert r.a2 == 1.5e-14
    assert r.a2_unc == 3.0e-15


def test_predictresult_field_round_trip():
    p = PredictResult()
    p.epoch = 2460123.5
    assert p.epoch == 2460123.5
    # rho and obs_cov are exposed; confirm they are accessible.
    assert p.rho is not None
    assert p.obs_cov is not None


# ---------------------------------------------------------------------------
# Observation factories (streak + radar variants)
# ---------------------------------------------------------------------------

_OBS_POS = [0.5, -0.8, -0.3]
_OBS_VEL = [0.01, 0.005, 0.002]


def _unit_los(ra, dec):
    return np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])


def test_observation_from_streak():
    ra, dec, ra_rate, dec_rate = 1.0, -0.3, 2.5e-3, -1.1e-3
    obs = Observation.from_streak(ra, dec, ra_rate, dec_rate, 2460000.5, _OBS_POS, _OBS_VEL)
    assert isinstance(obs.observation_type, StreakObservation)
    assert obs.observation_type.ra_rate == ra_rate
    assert obs.observation_type.dec_rate == dec_rate
    np.testing.assert_allclose(np.asarray(obs.rho_hat), _unit_los(ra, dec), atol=1e-12)
    np.testing.assert_array_equal(obs.observer_position, _OBS_POS)


def test_observation_from_radar():
    delay, doppler = 1.2e-4, -3.4e-5
    obs = Observation.from_radar(delay, doppler, True, False, 2460000.5, _OBS_POS, _OBS_VEL)
    rd = obs.observation_type
    assert isinstance(rd, RadarObservation)
    assert rd.delay == delay
    assert rd.doppler == doppler
    assert rd.has_delay is True
    assert rd.has_doppler is False
