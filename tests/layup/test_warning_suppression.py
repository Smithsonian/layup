"""Regression test for the sbpy TestRunner warning suppression (issue #376)."""

import subprocess
import sys


def test_no_sbpy_testrunner_warning_on_import():
    """Importing a layup module that pulls in sbpy must not surface the cosmetic
    ``AstropyDeprecationWarning`` about the deprecated ``TestRunner``.

    Run in a fresh subprocess so the check isn't affected by warning state (or
    imports) from the rest of the test session.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import layup.orbitfit"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "TestRunner" not in result.stderr, result.stderr
