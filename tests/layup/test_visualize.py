"""Regression tests for ``layup visualize`` (issue #384 / PR #385)."""

from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.visualize import visualize_cli


def _bcart_eq_file_with_objid(src_name, tmp_path):
    """Copy a shipped BCART_EQ fixture, renaming its ``provID`` header to ``ObjID``.

    ``visualize``'s readers key on an ``ObjID`` column, while the shipped
    orbit-fit fixtures use ``provID``; only the header needs adjusting.
    """
    with open(get_test_filepath(src_name)) as fh:
        header, rest = fh.read().split("\n", 1)
    out = tmp_path / "bcart_eq_objid.csv"
    out.write_text(header.replace("provID", "ObjID", 1) + "\n" + rest)
    return out


def test_visualize_cli_accepts_bcart_eq(tmp_path, monkeypatch):
    """Regression for #384: BCART_EQ is orbit-fit's default output format and must
    be visualizable.

    Before #385, ``visualize_cli`` raised ``KeyError`` looking up
    ``REQUIRED_COLUMN_NAMES['BCART_EQ']`` (that dict only has the base ``BCART``
    key). The fix normalizes the inferred format to ``BCART`` (barycentric,
    equatorial). We stub the ephemeris-backed figure build and the blocking Dash
    server so the test exercises only the read / format-inference path the fix
    touches, and assert the call reaches the render step.
    """
    orbit_file = _bcart_eq_file_with_objid("predict_chunk_BCART_EQ.csv", tmp_path)

    calls = {}
    monkeypatch.setattr("layup.visualize.build_fig_caches", lambda **kwargs: ({}, {}, []))
    monkeypatch.setattr(
        "layup.visualize.run_dash_app",
        lambda *args, **kwargs: calls.__setitem__("ran", True),
    )

    visualize_cli(str(orbit_file))

    assert calls.get("ran"), "visualize_cli should reach run_dash_app for a BCART_EQ input"
