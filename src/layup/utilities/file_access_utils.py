import logging
import os
import sys
from pathlib import Path


logger = logging.getLogger(__name__)

def find_file_or_exit(arg_fn, argname):
    """Checks to see if a file given by a filename exists. If it doesn't,
    this fails gracefully and exits to the command line.

    Parameters
    -----------
    arg_fn : string
        The filepath/name of the file to be checked.

    argname : string
        The name of the argument being checked. Used for error message.

    Returns
    ----------
    arg_fn : string
        The filepath/name of the file to be checked.

    """

    if os.path.exists(arg_fn):
        return arg_fn
    else:
        logger.error(f"Filename {arg_fn} supplied for {argname} argument does not exist.")
        sys.exit(f"ERROR: filename {arg_fn} supplied for {argname} argument does not exist.")


def find_directory_or_exit(arg_fn, argname):
    """Checks to see if a directory given by a filepath exists. If it doesn't,
    this fails gracefully and exits to the command line.

    Parameters
    -----------
    arg_fn : string
        The filepath of the directory to be checked.

    argname : string
        The name of the argument being checked. Used for error message.

    Returns
    ----------
    arg_fn : string
        The filepath of the directory to be checked.
    """

    file_path = Path(f"{arg_fn}")
    file_path = file_path.parent.resolve()

    if not file_path.is_dir():
        logger.error(f"Filepath {arg_fn} supplied for {argname} argument does not exist.")
        sys.exit(f"ERROR: filepath {arg_fn} supplied for {argname} argument does not exist.")
