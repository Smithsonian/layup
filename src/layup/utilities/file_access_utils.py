import os
import sys


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
        sys.exit("ERROR: filename {} supplied for {} argument does not exist.".format(arg_fn, argname))


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

    if os.path.isdir(arg_fn):
        return arg_fn
    else:
        sys.exit("ERROR: filepath {} supplied for {} argument does not exist.".format(arg_fn, argname))
