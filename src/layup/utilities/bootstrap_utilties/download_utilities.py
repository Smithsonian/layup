from argparse import Namespace
import concurrent.futures
import os
import pooch
from typing import Optional
from layup.utilities.layup_configs import AuxiliaryConfigs
from layup.utilities.bootstrap_utilties.create_meta_kernel import build_meta_kernel_file

# Fail-fast network policy for all layup data downloads (issue #388).
# Previously a stalled MPC/JPL connection was retried 25 times with no request
# timeout, wedging a fit or a CI run for 20-76 minutes (and escaping
# pytest-timeout when it happened at fixture time). Now every request times out
# quickly and is retried a small, bounded number of times, so a flaky host fails
# in about a minute rather than hours.
_CONNECT_TIMEOUT = 15  # seconds to establish the connection
_READ_TIMEOUT = 30  # max seconds between received bytes (bounds a *stall*, not total)
_RETRY_IF_FAILED = 3  # bounded retries (was 25)

# Archives that `_decompress` unpacks and then deletes (issue #436). They are
# tracked in the pooch registry so they can still be fetched, but they are not
# expected to survive in the cache, so their absence is not a missing file.
_EXTRACTED_ARCHIVE_EXTENSIONS = (".tar.gz", ".tgz")


def layup_downloader(progressbar: bool = False) -> pooch.HTTPDownloader:
    """A pooch HTTP downloader with a fail-fast timeout (issue #388).

    ``timeout`` is forwarded to ``requests.get`` as ``(connect, read)``. The read
    timeout is the maximum gap between received bytes, so it bounds a *stalled*
    connection without interrupting a slow-but-progressing large download (e.g.
    the ~600 MB JPL ephemeris keeps flowing, so it is never falsely cut off).
    """
    return pooch.HTTPDownloader(timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), progressbar=progressbar)


def make_retriever(aux_config: AuxiliaryConfigs, directory_path: Optional[str] = None) -> pooch.Pooch:
    """Create a Pooch object to track and retrieve ephemeris files.

    Parameters
    ----------
    aux_config: AuxiliaryConfigs
        Dataclass of auxiliary configuration file arguments.
    directory_path : string, optional
        The base directory to place all downloaded files. Default = None

    Returns
    -------
    : pooch.Pooch
        The instance of a Pooch object used to track and retrieve files.
    """
    dir_path = directory_path if directory_path else pooch.os_cache("layup")

    return pooch.create(
        path=dir_path,
        base_url="",
        urls=aux_config.urls,
        registry=aux_config.registry,
        retry_if_failed=_RETRY_IF_FAILED,
    )


def download_files_if_missing(aux_config: AuxiliaryConfigs, args: Namespace) -> None:
    """This function partially encapsulates the functionality of the
    `layup.bootstrap` command line tool. While `layup.bootstrap` allows for forced
    re-downloading of files, this function will only download files that are not
    already present in the local cache.

    Parameters
    ----------
    aux_configs : AuxiliaryConfigs
        Dataclass of configuration file arguments.
        This includes the AuxiliaryConfigs dataclass that contains the
        configuration for the bootstrap files.
    args : Namespace
        The command line arguments. This includes the `--cache` argument
        that specifies the directory to store the downloaded files.
    """

    # create the Pooch retriever that tracks and retrieves the requested files
    retriever = make_retriever(aux_config, args.ar_data_file_path)

    print("Checking cache for existing files.")
    found_all_files = _check_for_existing_files(aux_config, retriever)

    if found_all_files:
        # Everything the cache is expected to hold is present, so any archive
        # still sitting beside it is a leftover from a pre-#472 bootstrap.
        _remove_extracted_archives(aux_config, retriever)

    if not found_all_files:
        # Fetch each file with its own fail-fast downloader (issue #388); a fresh
        # downloader per call keeps the per-file progress bars independent across
        # the parallel threads.
        def _fetch(file_name):
            return retriever.fetch(
                file_name, processor=_decompress, downloader=layup_downloader(progressbar=True)
            )

        # download the data files in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_fetch, aux_config.data_files_to_download)

        # build the meta_kernel.txt file
        build_meta_kernel_file(aux_config, retriever, downloader=layup_downloader())

        print("Checking cache after attempting to download and create files.")
        _check_for_existing_files(aux_config, retriever)


def _remove_extracted_archives(aux_config: AuxiliaryConfigs, retriever: pooch.Pooch) -> None:
    """Delete download archives that an earlier bootstrap left in the cache.

    Issue #472 removes each archive as soon as it is unpacked, but only for
    bootstraps run since. A cache populated before that keeps the ~156 MB
    debiasing tarball indefinitely, and never re-extracts, so it never reaches
    that cleanup. Sweeping here makes the fix retroactive without a re-download.

    Only call this once the extracted contents are known to be present.

    Parameters
    ------------
    aux_config: AuxiliaryConfigs
        Dataclass of auxiliary configuration file arguments.
    retriever : pooch.Pooch
        Pooch object that maintains the registry of files to download.

    Returns
    ----------
    None
    """
    for file_name in aux_config.data_file_list:
        if not file_name.endswith(_EXTRACTED_ARCHIVE_EXTENSIONS):
            continue
        file_path = os.path.join(retriever.abspath, file_name)
        if not os.path.exists(file_path):
            continue
        # Best-effort, as in `_decompress`: reclaiming space is never worth
        # failing a bootstrap over.
        try:
            reclaimed_mb = os.path.getsize(file_path) / (1024 * 1024)
            os.remove(file_path)
            print(f"Removed the leftover archive {file_path} ({reclaimed_mb:.0f} MB reclaimed).")
        except OSError as exc:  # pragma: no cover - filesystem-dependent
            print(f"Could not remove the leftover archive {file_path}: {exc}")


def _check_for_existing_files(aux_config: AuxiliaryConfigs, retriever: pooch.Pooch) -> bool:
    """Will check for existing local files, any file not found will be printed
    to the terminal.

    Parameters
    -------------
    aux_configs: AuxiliaryConfigs
        Dataclass of auxiliary configuration file arguments.
    retriever : pooch.Pooch
        Pooch object that maintains the registry of files to download.

    Returns
    ----------
    :  bool
        Returns True if all files are found in the local cache, False otherwise.
    """

    file_list = aux_config.data_file_list

    found_all_files = True
    missing_files = []
    for file_name in file_list:
        # An archive that was unpacked and removed is not missing, it is done
        # with. Counting it as missing made every bootstrap after issue #472
        # re-download the ~156 MB debiasing tarball (issue #482).
        if file_name.endswith(_EXTRACTED_ARCHIVE_EXTENSIONS):
            continue
        if not os.path.exists(os.path.join(retriever.abspath, file_name)):
            missing_files.append(file_name)
            found_all_files = False

    if found_all_files:
        print(f"All expected files were found in the local cache: {retriever.abspath}/")
    else:
        print(f"The following file(s) were not found in the local cache: {retriever.abspath}/")
        for file_name in missing_files:
            print(f"- {file_name}")

    return found_all_files


def _decompress(fname: str, action: str, pup: pooch.Pooch) -> None:  # pragma: no cover
    """Override the functionality of Pooch's `Decompress` class so that the resulting
    decompressed file uses the original file name without the compression extension.
    For instance `filename.json.bz` will be decompressed and saved as `filename.json`.

    Parameters
    ------------
    fname : str
        Original filename
    action : str
        One of ["download", "update", "fetch"]
    pup : pooch.Pooch
        The Pooch object that defines the location of the file.

    Returns
    ----------
    None
    """
    known_extentions = [".gz", ".bz2", ".xz"]
    if os.path.splitext(fname)[-1] in known_extentions:
        pooch.Decompress(method="auto", name=os.path.splitext(fname)[0]).__call__(fname, action, pup)

    tar_extentions = [".tar.gz", ".tgz"]
    if os.path.splitext(fname)[-1] in tar_extentions:
        print(f"Trying to decompress tar file: {fname}")
        pooch.Untar(extract_dir=".").__call__(fname, action, pup)
        # Remove the archive once its contents are extracted (issue #436).  The
        # debiasing tarball is ~156 MB and pooch keeps it alongside the extracted
        # data, so leaving it roughly doubles that part of the cache for no benefit
        # -- nothing reads the archive again.  Failure to remove it is not worth
        # aborting a bootstrap over, so this is best-effort.
        try:
            os.remove(fname)
        except OSError as exc:  # pragma: no cover - filesystem-dependent
            print(f"Could not remove the extracted archive {fname}: {exc}")


def _remove_files(aux_config: AuxiliaryConfigs, retriever: pooch.Pooch) -> None:
    """Utility to remove all the files tracked by the pooch retriever. This includes
    the decompressed ObservatoryCodes.json file as well as the META_KERNEL file
    that are created after downloading the files in the DATA_FILES_TO_DOWNLOAD
    list.

    Parameters
    ------------
    aux_config: AuxiliaryConfigs
        Dataclass of auxiliary configuration file arguments.
    retriever : pooch.Pooch
        Pooch object that maintains the registry of files to download.

    Returns
    ----------
    None
    """

    for file_name in aux_config.data_file_list:
        file_path = os.path.join(retriever.abspath, file_name)
        if os.path.exists(file_path):
            print(f"Deleting file: {file_path}")
            os.remove(file_path)
