"""Fail-fast download policy (issue #388).

A stalled MPC/JPL connection was previously retried 25 times with no request
timeout, wedging a fit or a CI run for 20-76 minutes. These tests pin the
bounded-retry + fail-fast-timeout policy so it cannot silently regress. All are
network-free (they only inspect the pooch objects layup constructs).
"""

import pooch

from layup.utilities.bootstrap_utilties.download_utilities import (
    make_retriever,
    layup_downloader,
    _RETRY_IF_FAILED,
    _CONNECT_TIMEOUT,
    _READ_TIMEOUT,
)
from layup.utilities.layup_configs import AuxiliaryConfigs


def test_make_retriever_uses_bounded_retries(tmp_path):
    retriever = make_retriever(AuxiliaryConfigs(), str(tmp_path))
    assert retriever.retry_if_failed == _RETRY_IF_FAILED
    # The whole point of #388: far fewer than the old, unbounded-feeling 25.
    assert retriever.retry_if_failed < 25


def test_layup_downloader_has_fail_fast_timeout():
    dl = layup_downloader()
    assert isinstance(dl, pooch.HTTPDownloader)
    # timeout is forwarded to requests.get as (connect, read) seconds.
    assert dl.kwargs.get("timeout") == (_CONNECT_TIMEOUT, _READ_TIMEOUT)


def test_layup_downloader_progressbar_flag():
    assert layup_downloader(progressbar=True).progressbar is True
    assert layup_downloader().progressbar is False
