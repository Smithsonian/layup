"""Tests for LayupLogger's file handlers.

The .err file is created lazily (issue #481): following the documented
quickstart used to leave an empty layup-<timestamp>.err beside every command,
which reads as though something had gone wrong on a run that succeeded.
"""

import logging

import pytest

from layup.utilities.layup_logging import LayupLogger


@pytest.fixture
def clean_layup_logger():
    """Remove any handlers LayupLogger has already attached, and restore them after.

    LayupLogger configures the shared top-level ``layup`` logger, so without this
    a second instantiation in the same process stacks handlers on top of the
    first and the tests see each other's output.
    """
    top = logging.getLogger("layup")
    saved = top.handlers[:]
    top.handlers = []
    yield
    for h in top.handlers:
        h.close()
    top.handlers = saved


def test_no_err_file_when_nothing_is_logged_at_error(tmp_path, clean_layup_logger):
    """A run that logs nothing at ERROR leaves no .err file behind."""
    logger = LayupLogger(log_directory=str(tmp_path)).get_logger("layup.test")
    logger.info("an ordinary run")
    logger.warning("a warning is not an error")

    assert list(tmp_path.glob("*.log")), "the .log file should always be written"
    assert not list(tmp_path.glob("*.err")), (
        "an .err file was created for a run that logged nothing at ERROR: "
        f"{[p.name for p in tmp_path.glob('*.err')]}"
    )


def test_err_file_is_written_when_an_error_is_logged(tmp_path, clean_layup_logger):
    """An ERROR still produces an .err file, containing the message."""
    logger = LayupLogger(log_directory=str(tmp_path)).get_logger("layup.test")
    logger.info("starting")
    logger.error("something actually went wrong")

    err = list(tmp_path.glob("*.err"))
    assert len(err) == 1, f"expected exactly one .err file, got {[p.name for p in err]}"
    contents = err[0].read_text()
    assert "something actually went wrong" in contents
    assert "starting" not in contents, "the .err file should carry only ERROR and above"
