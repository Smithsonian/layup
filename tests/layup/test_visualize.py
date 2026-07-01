"""Regression tests for ``layup visualize`` (issue #384 / PR #385)."""

from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.visualize import visualize_cli


def test_visualize_cli_accepts_bcart_eq(monkeypatch):
    """Regression for #384: BCART_EQ is orbit-fit's default output format and must
    be visualizable.

    Before #385, ``visualize_cli`` raised ``KeyError`` looking up
    ``REQUIRED_COLUMN_NAMES['BCART_EQ']`` (that dict only has the base ``BCART``
    key). The fix normalizes the inferred format to ``BCART`` (barycentric,
    equatorial). We stub the ephemeris-backed figure build and the blocking Dash
    server so the test exercises only the read / format-inference path, and
    assert the call reaches the render step.

    The fixture is keyed by ``provID`` -- visualize's default primary-id column
    since #383 -- so a fresh orbit-fit output (provID + BCART_EQ) visualizes with
    no extra flag.
    """
    orbit_file = get_test_filepath("predict_chunk_BCART_EQ.csv")

    calls = {}
    monkeypatch.setattr("layup.visualize.build_fig_caches", lambda **kwargs: ({}, {}, []))
    monkeypatch.setattr(
        "layup.visualize.run_dash_app",
        lambda *args, **kwargs: calls.__setitem__("ran", True),
    )

    visualize_cli(str(orbit_file))

    assert calls.get("ran"), "visualize_cli should reach run_dash_app for a BCART_EQ input"
