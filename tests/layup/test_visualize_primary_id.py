"""Tests for visualize's configurable primary-id column (issue #383).

`visualize` previously hard-coded an ``ObjID`` column, so an orbit-fit result
(which uses ``provID``) could not be visualized without renaming the column.
These tests cover the new ``primary_id_column_name`` argument (CLI ``-pid``),
which defaults to ``provID``. The ephemeris-backed figure build and the blocking
Dash server are stubbed so only the read path is exercised (no ephemeris).
"""

import pytest

from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.visualize import visualize_cli

# An orbit-fit-style output file: keyed by provID, FORMAT=BCART_EQ.
PROVID_FILE = "predict_chunk_BCART_EQ.csv"


@pytest.fixture
def _stub_render(monkeypatch):
    """Stub the ephemeris/figure build and the Dash server; record if reached."""
    calls = {}
    monkeypatch.setattr("layup.visualize.build_fig_caches", lambda **kwargs: ({}, {}, []))
    monkeypatch.setattr(
        "layup.visualize.run_dash_app",
        lambda *args, **kwargs: calls.__setitem__("ran", True),
    )
    return calls


def test_visualize_defaults_to_provid(_stub_render):
    """A provID-keyed orbit-fit output visualizes with no flag (default provID)."""
    visualize_cli(get_test_filepath(PROVID_FILE))
    assert _stub_render.get("ran"), "default provID should read an orbit-fit output file"


def test_visualize_accepts_objid_via_pid(tmp_path, _stub_render):
    """Files keyed by ObjID still work when primary_id_column_name='ObjID'."""
    with open(get_test_filepath(PROVID_FILE)) as fh:
        header, rest = fh.read().split("\n", 1)
    objid_file = tmp_path / "objid.csv"
    objid_file.write_text(header.replace("provID", "ObjID", 1) + "\n" + rest)

    visualize_cli(str(objid_file), primary_id_column_name="ObjID")
    assert _stub_render.get("ran")


def test_visualize_wrong_primary_id_errors(_stub_render):
    """Asking for a column the file doesn't have is a clear error, not a crash."""
    with pytest.raises(SystemExit):
        visualize_cli(get_test_filepath(PROVID_FILE), primary_id_column_name="ObjID")
    assert not _stub_render.get("ran")
