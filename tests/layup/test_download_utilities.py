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
    _decompress,
    _check_for_existing_files,
    _remove_extracted_archives,
    _EXTRACTED_ARCHIVE_EXTENSIONS,
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


def test_decompress_removes_the_tar_archive(tmp_path, monkeypatch):
    """`layup bootstrap` must not leave the extracted archive behind (issue #436).

    The debiasing tarball is ~156 MB and nothing reads it again after extraction,
    so keeping it alongside the extracted data roughly doubles that part of the
    cache. Network-free: builds a small tarball locally and runs the same
    `_decompress` hook pooch calls.
    """
    import os
    import tarfile

    monkeypatch.chdir(tmp_path)
    payload = tmp_path / "bias.dat"
    payload.write_text("x" * 1000)
    archive = tmp_path / "debias_hires2018.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(payload, arcname="bias.dat")
    payload.unlink()

    _decompress(str(archive), "download", None)

    assert not archive.exists(), "the .tgz archive should be removed after extraction"
    assert (tmp_path / "bias.dat").exists(), "the extracted contents must survive"


def test_decompress_survives_an_unremovable_archive(tmp_path, monkeypatch):
    """Failing to delete the archive must not abort a bootstrap.

    Removal is best-effort: a read-only cache or a platform that holds the handle
    should cost disk space, not the whole download.
    """
    import tarfile

    monkeypatch.chdir(tmp_path)
    payload = tmp_path / "bias.dat"
    payload.write_text("y" * 100)
    archive = tmp_path / "debias_hires2018.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(payload, arcname="bias.dat")
    payload.unlink()

    import os as _os

    real_remove = _os.remove

    def refuse(path, *a, **kw):
        if str(path).endswith(".tgz"):
            raise OSError("simulated read-only cache")
        return real_remove(path, *a, **kw)

    monkeypatch.setattr(_os, "remove", refuse)
    _decompress(str(archive), "download", None)  # must not raise

    assert (tmp_path / "bias.dat").exists(), "extraction must still have happened"


def test_extracted_archive_is_not_reported_missing(tmp_path):
    """A cache with no `.tgz` is complete, not missing a file (issue #482).

    `data_file_list` names the debiasing tarball, and issue #472 deletes that
    tarball as soon as it is unpacked. Counting it as missing made every
    subsequent bootstrap re-download ~156 MB from JPL, unpack it, delete it, and
    do the same again next time.
    """
    aux = AuxiliaryConfigs()
    assert aux.debiasing_data_compressed.endswith(_EXTRACTED_ARCHIVE_EXTENSIONS)

    # A cache as a post-#472 bootstrap leaves it: everything but the archive.
    for file_name in aux.data_file_list:
        if file_name != aux.debiasing_data_compressed:
            (tmp_path / file_name).touch()

    retriever = make_retriever(aux, str(tmp_path))
    assert _check_for_existing_files(aux, retriever) is True


def test_a_genuinely_missing_file_is_still_reported(tmp_path):
    """The skip is narrow: only extracted archives, not the data they contain."""
    aux = AuxiliaryConfigs()
    for file_name in aux.data_file_list:
        if file_name not in (aux.debiasing_data_compressed, aux.debiasing_data):
            (tmp_path / file_name).touch()

    retriever = make_retriever(aux, str(tmp_path))
    # bias.dat, the tarball's extracted payload, is absent and must be noticed.
    assert _check_for_existing_files(aux, retriever) is False


def test_leftover_archive_is_swept_from_an_old_cache(tmp_path, capsys):
    """Caches populated before #472 keep the tarball; sweep it (issue #482)."""
    aux = AuxiliaryConfigs()
    for file_name in aux.data_file_list:
        (tmp_path / file_name).touch()
    archive = tmp_path / aux.debiasing_data_compressed
    archive.write_bytes(b"x" * 2048)

    retriever = make_retriever(aux, str(tmp_path))
    _remove_extracted_archives(aux, retriever)

    assert not archive.exists()
    assert "MB reclaimed" in capsys.readouterr().out
    # The extracted payload it came from must survive.
    assert (tmp_path / aux.debiasing_data).exists()


def test_sweep_is_quiet_when_there_is_nothing_to_remove(tmp_path, capsys):
    aux = AuxiliaryConfigs()
    retriever = make_retriever(aux, str(tmp_path))
    _remove_extracted_archives(aux, retriever)  # must not raise
    assert capsys.readouterr().out == ""
