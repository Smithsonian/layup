import os
import pooch
from typing import Optional
from layup.utilities.layup_configs import AuxiliaryConfigs


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
        file_path = retriever.fetch(file_name)
        print(f"Deleting file: {file_path}")
        os.remove(file_path)
