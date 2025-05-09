import concurrent.futures
import os
import pooch
from functools import partial
from typing import Optional
from layup.utilities.layup_configs import AuxiliaryConfigs
from layup.utilities.bootstrap_utilties.create_meta_kernel import build_meta_kernel_file


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
        retry_if_failed=25,
    )


def download_files_if_missing(configs, args) -> None:
    """This function partially encapsulates the functionality of the
    `layup.bootstrap` command line tool. While `layup.bootstrap` allows for forced
    re-downloading of files, this function will only download files that are not
    already present in the local cache.

    Parameters
    ----------
    configs : LayupConfigs
        Dataclass of configuration file arguments.
        This includes the AuxiliaryConfigs dataclass that contains the
        configuration for the bootstrap files.
    args : Namespace
        The command line arguments. This includes the `--cache` argument
        that specifies the directory to store the downloaded files.
    """

    aux_config = configs.auxiliary

    # create the Pooch retriever that tracks and retrieves the requested files
    retriever = make_retriever(aux_config, args.cache)

    print("Checking cache for existing files.")
    found_all_files = _check_for_existing_files(aux_config, retriever)

    if not found_all_files:
        # create a partial function of `Pooch.fetch` including the `_decompress` method
        fetch_partial = partial(retriever.fetch, processor=_decompress, progressbar=True)

        # download the data files in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(fetch_partial, aux_config.data_files_to_download)

        # build the meta_kernel.txt file
        build_meta_kernel_file(aux_config, retriever)

        print("Checking cache after attempting to download and create files.")
        _check_for_existing_files(aux_config, retriever)


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
