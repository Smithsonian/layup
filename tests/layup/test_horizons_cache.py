"""Persistent (naif_id, jd) -> state cache for space-observatory Horizons lookups.

At MPC full-catalog scale the space observatories (WISE, HST, JWST, ...) resolve
their geocentric state from JPL Horizons per (naif_id, epoch); with only an
in-memory cache, every parallel worker re-queries cold and JPL rate-limits to
HTTP 503. These tests exercise the on-disk cache that makes it one query per
(spacecraft, epoch) ever. The network is mocked, so no JPL calls are made.
"""

import numpy as np
import pytest

import layup.utilities.special_observatories as so


@pytest.fixture
def fake_horizons(monkeypatch):
    """Replace the network chunk-query with a deterministic stub that records the
    epochs it was asked for."""
    calls = []

    def _fake_chunk(naif_id, jd_chunk, timeout):
        calls.append(list(jd_chunk))
        return {jd: (np.array([jd, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])) for jd in jd_chunk}

    monkeypatch.setattr(so, "_query_chunk", _fake_chunk)
    return calls


def test_cache_writes_then_serves_from_disk(tmp_path, fake_horizons):
    jds = [2459000.5, 2459001.5]

    out1 = so.query_horizons_geocentric("-48", jds, cache_dir=str(tmp_path))
    assert len(fake_horizons) == 1 and sorted(fake_horizons[0]) == jds
    assert (tmp_path / "horizons" / "-48").is_dir()

    # Second identical call: served entirely from disk, no new network query.
    out2 = so.query_horizons_geocentric("-48", jds, cache_dir=str(tmp_path))
    assert len(fake_horizons) == 1
    for jd in jds:
        assert np.allclose(out2[jd][0], out1[jd][0])
        assert np.allclose(out2[jd][1], out1[jd][1])


def test_partial_overlap_fetches_only_misses(tmp_path, fake_horizons):
    so.query_horizons_geocentric("-48", [2459000.5], cache_dir=str(tmp_path))
    assert len(fake_horizons) == 1

    # One cached epoch + one new one -> only the new epoch is queried.
    out = so.query_horizons_geocentric("-48", [2459000.5, 2459002.5], cache_dir=str(tmp_path))
    assert len(fake_horizons) == 2 and fake_horizons[1] == [2459002.5]
    assert set(out) == {2459000.5, 2459002.5}


def test_cache_is_per_spacecraft(tmp_path, fake_horizons):
    so.query_horizons_geocentric("-48", [2459000.5], cache_dir=str(tmp_path))
    # Same epoch, different spacecraft -> a real query (not a cross-craft cache hit).
    so.query_horizons_geocentric("-163", [2459000.5], cache_dir=str(tmp_path))
    assert len(fake_horizons) == 2
    assert (tmp_path / "horizons" / "-48").is_dir()
    assert (tmp_path / "horizons" / "-163").is_dir()


def test_cache_disabled_always_queries(tmp_path, fake_horizons):
    so.query_horizons_geocentric("-48", [2459000.5], cache_dir=False)
    so.query_horizons_geocentric("-48", [2459000.5], cache_dir=False)
    assert len(fake_horizons) == 2  # no caching -> queried both times


def test_env_var_disables_cache(tmp_path, fake_horizons, monkeypatch):
    monkeypatch.setenv("LAYUP_HORIZONS_CACHE", "0")
    so.query_horizons_geocentric("-48", [2459000.5], cache_dir=str(tmp_path))
    so.query_horizons_geocentric("-48", [2459000.5], cache_dir=str(tmp_path))
    assert len(fake_horizons) == 2


def test_corrupt_cache_file_is_refetched(tmp_path, fake_horizons):
    so.query_horizons_geocentric("-48", [2459000.5], cache_dir=str(tmp_path))
    # Corrupt the cached file; the next call must treat it as a miss and re-fetch.
    cache_file = next((tmp_path / "horizons" / "-48").glob("*.npz"))
    cache_file.write_bytes(b"not an npz")
    so.query_horizons_geocentric("-48", [2459000.5], cache_dir=str(tmp_path))
    assert len(fake_horizons) == 2
