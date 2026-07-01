"""Regression tests for the observatory-codes fallback (issue #388).

When the MPC observatory-codes file is not already cached, ``LayupObservatory``
must use the copy bundled with layup immediately, rather than letting Sorcha's
``Observatory`` attempt a 25-retry MPC download that can wedge CI for over an
hour (and, at pytest collection time, escape pytest-timeout).

These tests stub SPICE furnishing and Sorcha's ``Observatory.__init__`` so they
need neither the ephemeris nor the network.
"""

import os

import layup.utilities.data_processing_utilities as dpu
from layup.utilities.data_processing_utilities import LayupObservatory
from layup.utilities.layup_configs import LayupConfigs


def test_cached_obscodes_present(tmp_path):
    aux = LayupConfigs().auxiliary
    # Empty cache dir -> the codes are not present.
    assert LayupObservatory._cached_obscodes_present(str(tmp_path), aux) is False
    # Once the decompressed file exists, they are present.
    (tmp_path / aux.observatory_codes).write_text("{}")
    assert LayupObservatory._cached_obscodes_present(str(tmp_path), aux) is True


def _capture_oc_file(monkeypatch):
    """Stub SPICE furnishing + Sorcha's Observatory.__init__; capture oc_file."""
    recorded = {}

    def fake_super_init(self, args, auxconfigs, oc_file=None):
        recorded["oc_file"] = oc_file

    monkeypatch.setattr(dpu, "layup_furnish_spiceypy", lambda cache_dir: None)
    monkeypatch.setattr(dpu.SorchaObservatory, "__init__", fake_super_init)
    return recorded


def test_uncached_obscodes_uses_bundled(tmp_path, monkeypatch):
    """No cached codes -> hand Sorcha the bundled copy (never a blocking download)."""
    recorded = _capture_oc_file(monkeypatch)

    LayupObservatory(cache_dir=str(tmp_path))

    # oc_file must be a real file (the bundled copy), NOT None -- None would let
    # Sorcha attempt the retry-heavy MPC download.
    assert recorded["oc_file"] is not None
    assert os.path.isfile(recorded["oc_file"])


def test_cached_obscodes_uses_cache(tmp_path, monkeypatch):
    """Cached codes -> let Sorcha read the local file (oc_file=None), no download."""
    aux = LayupConfigs().auxiliary
    (tmp_path / aux.observatory_codes).write_text("{}")  # pretend `layup bootstrap` cached it
    recorded = _capture_oc_file(monkeypatch)

    LayupObservatory(cache_dir=str(tmp_path))

    assert recorded["oc_file"] is None
