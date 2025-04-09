# for checking/ parsing things in the cli command line
import logging
import os
from pathlib import Path
import sys

logger = logging.getLogger(__name__)


def warn_or_remove_file(filepath, force_remove, pplogger):
    """Given a path to a file(s), first determine if the file exists. If it does not
    exist, pass through.

    If the file does exist check if the user has set `--force` on the command line.
    If the user set --force, log that the existing file will be removed.
    Otherwise, warn the user that the file exists and exit the program.

    Parameters
    ----------
    filepath : string
        The full file path to a given file. i.e. /home/data/output.csv
    force_remove : boolean
        Whether to remove the file if it exists.
    pplogger : Logger
        Used to log the output.
    """
    file_exists = Path(filepath).exists()

    if file_exists and force_remove:
        pplogger.info(f"Existing file found at {filepath}. -f flag set: deleting existing file.")
        os.remove(filepath)
    elif file_exists and not force_remove:
        pplogger.error(
            f"ERROR: existing file found at output location {filepath}. Set -f flag to overwrite this file."
        )
        sys.exit(
            f"ERROR: existing file found at output location {filepath}. Set -f flag to overwrite this file."
        )
