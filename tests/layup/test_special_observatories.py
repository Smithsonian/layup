"""Tests for JPL-Horizons resolution of space-based observatories (issue #55).

The pure-logic tests (the obscode map, the JD conversion, the Horizons
response parser, and the HTTP client with the network mocked) run everywhere.
The integration tests construct a ``LayupObservatory``, which furnishes SPICE
kernels, so they are gated behind ``requires_ephem``; they still mock the HTTP
layer so they never touch the network.
"""

import numpy as np
import pytest

from layup.utilities import special_observatories as so
from layup.utilities.data_processing_utilities import AU_KM, LayupObservatory
from layup.utilities.special_observatories import (
    SPACE_OBSERVATORIES,
    _parse_vectors,
    et_to_jd_tdb,
    is_space_observatory,
    query_horizons_geocentric,
)

from _bk_guards import requires_ephem

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _vector_block(rows):
    """Build a CSV Horizons ``result`` payload for (jd, pos_km, vel_km_s) rows.

    Mirrors the format produced by CSV_FORMAT=YES, VEC_TABLE=2, VEC_LABELS=NO:
    ``JDTDB, CalendarDate, X, Y, Z, VX, VY, VZ,`` between the $$SOE/$$EOE marks.
    """
    lines = ["$$SOE"]
    for jd, pos, vel in rows:
        nums = ", ".join(f"{v:.16E}" for v in (*pos, *vel))
        lines.append(f"{jd:.9f}, A.D. 2000-Jan-01 00:00:00.0000, {nums},")
    lines.append("$$EOE")
    return "\n".join(["JPL Horizons preamble...", *lines, "trailing"])


class _FakeResponse:
    def __init__(self, text):
        self._text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"result": self._text}


# ---------------------------------------------------------------------------
# pure logic: map + conversions
# ---------------------------------------------------------------------------


def test_space_observatory_map():
    # A few anchor mappings ported from spacerocks.
    assert SPACE_OBSERVATORIES["250"][0] == "-48"  # Hubble
    assert SPACE_OBSERVATORIES["274"][0] == "-170"  # JWST
    assert SPACE_OBSERVATORIES["C51"][0] == "-163"  # WISE
    # Every NAIF id is a negative integer (spacecraft convention) + a name.
    for code, (naif, name) in SPACE_OBSERVATORIES.items():
        assert int(naif) < 0, code
        assert isinstance(name, str) and name


def test_is_space_observatory():
    assert is_space_observatory("250")
    assert is_space_observatory("C51")
    assert not is_space_observatory("247")  # roving observer
    assert not is_space_observatory("500")  # geocenter
    assert not is_space_observatory("568")  # Mauna Kea (ground station)


def test_et_to_jd_tdb():
    assert et_to_jd_tdb(0.0) == 2451545.0
    assert et_to_jd_tdb(86400.0) == 2451546.0
    assert et_to_jd_tdb(-43200.0) == 2451544.5


# ---------------------------------------------------------------------------
# pure logic: response parsing
# ---------------------------------------------------------------------------


def test_parse_vectors_extracts_state():
    pos = np.array([1.0e3, 2.0e3, 3.0e3])
    vel = np.array([1.0, -2.0, 3.0])
    text = _vector_block([(2451545.0, pos, vel)])

    out = _parse_vectors(text, [2451545.0], "-48")
    got_pos, got_vel = out[2451545.0]
    np.testing.assert_allclose(got_pos, pos)
    np.testing.assert_allclose(got_vel, vel)


def test_parse_vectors_no_data_raises():
    with pytest.raises(RuntimeError, match="no vector data"):
        _parse_vectors("some error message, no markers here", [2451545.0], "-48")


def test_parse_vectors_missing_epoch_raises():
    # Requested two epochs; response only carries one.
    text = _vector_block([(2451545.0, np.zeros(3), np.zeros(3))])
    with pytest.raises(RuntimeError, match="did not return a state"):
        _parse_vectors(text, [2451545.0, 2451546.0], "-48")


# ---------------------------------------------------------------------------
# HTTP client (network mocked)
# ---------------------------------------------------------------------------


def test_query_horizons_geocentric_mocked(monkeypatch):
    pos = np.array([10.0, 20.0, 30.0])
    vel = np.array([0.1, 0.2, 0.3])
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse(_vector_block([(2451545.0, pos, vel)]))

    monkeypatch.setattr(so.requests, "get", fake_get)

    out = query_horizons_geocentric("-48", [2451545.0])
    np.testing.assert_allclose(out[2451545.0][0], pos)
    np.testing.assert_allclose(out[2451545.0][1], vel)

    # The request must ask for a geometric, geocentric, ICRF state in km / km-s.
    p = captured["params"]
    assert captured["url"] == so.HORIZONS_API
    assert p["COMMAND"] == "'-48'"
    assert p["CENTER"] == "'@399'"
    assert p["VEC_CORR"] == "NONE"
    assert p["OUT_UNITS"] == "KM-S"
    assert p["REF_PLANE"] == "FRAME"
    assert "2451545.0" in p["TLIST"]


def test_query_horizons_batches_in_chunks(monkeypatch):
    # More epochs than one chunk -> multiple requests, all results returned.
    n = so._TLIST_CHUNK + 5
    jds = [2451545.0 + i for i in range(n)]
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        rows = [(float(jd), np.array([float(jd), 0.0, 0.0]), np.zeros(3)) for jd in params["TLIST"].split()]
        return _FakeResponse(_vector_block(rows))

    monkeypatch.setattr(so.requests, "get", fake_get)

    out = query_horizons_geocentric("-48", jds)
    assert calls["n"] == 2  # ceil(45 / 40)
    assert len(out) == n
    np.testing.assert_allclose(out[jds[0]][0], [jds[0], 0, 0])


# ---------------------------------------------------------------------------
# integration with LayupObservatory (ephem-gated, HTTP mocked)
# ---------------------------------------------------------------------------


def _row(dtype_fields, values):
    return np.array([tuple(values)], dtype=dtype_fields)[0]


@requires_ephem
def test_populate_space_observatory_uses_horizons(monkeypatch):
    obs = LayupObservatory()
    pos_km = np.array([-6905.9, -673.9, -353.1])
    vel_km_s = np.array([1.0, -2.0, 3.0])

    monkeypatch.setattr(
        obs, "_fetch_space_observatory_states", lambda naif, jds: {jds[0]: (pos_km, vel_km_s)}
    )

    et = 1.0e8
    row = _row([("stn", "U4"), ("et", "<f8")], ("250", et))
    key = obs.populate_observatory("250", et, row)

    assert key == f"250_{et}"
    np.testing.assert_allclose(obs.ObservatoryXYZ[key], pos_km)
    np.testing.assert_allclose(obs.ObservatoryVel[key], vel_km_s)


@requires_ephem
def test_ades_position_overrides_horizons(monkeypatch):
    """A user-supplied ADES position for a spacecraft code must win over Horizons."""
    obs = LayupObservatory()

    def explode(naif, jds):  # pragma: no cover - must not be called
        raise AssertionError("Horizons must not be queried when a position is supplied")

    monkeypatch.setattr(obs, "_fetch_space_observatory_states", explode)

    et = 1.0e8
    dtype = [
        ("stn", "U4"),
        ("et", "<f8"),
        ("sys", "U7"),
        ("ctr", "i4"),
        ("pos1", "<f8"),
        ("pos2", "<f8"),
        ("pos3", "<f8"),
    ]
    row = _row(dtype, ("250", et, "ICRF_KM", 399, 1000.0, 2000.0, 3000.0))
    key = obs.populate_observatory("250", et, row)

    np.testing.assert_allclose(obs.ObservatoryXYZ[key], [1000.0, 2000.0, 3000.0])


@requires_ephem
def test_prefetch_batches_one_request_per_spacecraft(monkeypatch):
    obs = LayupObservatory()
    calls = []

    def fake_fetch(naif, jds):
        calls.append((naif, tuple(jds)))
        return {jd: (np.array([jd, 0.0, 0.0]), np.zeros(3)) for jd in jds}

    monkeypatch.setattr(obs, "_fetch_space_observatory_states", fake_fetch)

    ets = [1.0e8, 2.0e8, 3.0e8]
    dtype = [("stn", "U4"), ("et", "<f8")]
    data = np.array(
        [("250", ets[0]), ("250", ets[1]), ("C51", ets[2])],
        dtype=dtype,
    )
    obs._prefetch_space_observatories(data)

    # One request per spacecraft (250 and C51), not one per observation.
    assert len(calls) == 2
    assert {c[0] for c in calls} == {"-48", "-163"}
    # HST's request batched both of its epochs.
    hst = next(c for c in calls if c[0] == "-48")
    assert len(hst[1]) == 2
    # Cache populated for every epoch.
    for et in ets[:2]:
        assert f"250_{et}" in obs.ObservatoryXYZ
    assert f"C51_{ets[2]}" in obs.ObservatoryXYZ


@requires_ephem
def test_obscodes_to_barycentric_space_observatory(monkeypatch):
    """End-to-end: a space-based obs is placed at Earth + its geocentric state."""
    import spiceypy as spice

    obs = LayupObservatory()
    et = spice.str2et("2003-01-26T00:24:24.480Z")
    geocentric_km = np.array([-6905.9, -673.9, -353.1])  # HST-like LEO offset
    geocentric_vel = np.array([1.0, -2.0, 3.0])

    monkeypatch.setattr(
        obs,
        "_fetch_space_observatory_states",
        lambda naif, jds: {jds[0]: (geocentric_km, geocentric_vel)},
    )

    row = np.array([("250", et)], dtype=[("stn", "U4"), ("et", "<f8")])
    bary = np.atleast_1d(obs.obscodes_to_barycentric(row))

    obs_pos_km = np.array([bary["x"][0], bary["y"][0], bary["z"][0]]) * AU_KM
    earth_state, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "SSB")
    offset_km = obs_pos_km - np.array(earth_state[:3])
    np.testing.assert_allclose(offset_km, geocentric_km, atol=1.0)
