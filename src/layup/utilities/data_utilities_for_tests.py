import os
from pathlib import Path


def get_config_setups_filepath(filename):
    """Return the full path to a test file in the ``.../config_setups`` directory.

    Parameters
    ----------
    filename : string
        The name of the file inside the ``config_setups`` directory.

    Returns
    -------
    : string
        The full path to the file.
    """

    # This file's path: `<base_directory>/src/layup/utilities/dataUtilitiesForTests.py
    # THIS_DIR = `<base_directory>/`
    THIS_DIR = Path(__file__).parent.parent.parent.parent

    # Returned path: `<base_directory>/src/layup/config_setups
    return os.path.join(THIS_DIR, "src/layup/config_setups", filename)


def get_test_filepath(filename):
    """Return the full path to a test file in the ``.../tests/data`` directory.

    Parameters
    ----------
    filename : string
        The name of the file inside the ``tests/data`` directory.

    Returns
    -------
    : string
        The full path to the file.
    """

    # This file's path: `<base_directory>/src/layup/utilities/test_data_utilities.py`
    # THIS_DIR = `<base_directory>/`
    THIS_DIR = Path(__file__).parent.parent.parent.parent

    # Returned path: `<base_directory>/tests/data/filename`
    return os.path.join(THIS_DIR, "tests/data", filename)


def layup_cli(*args):
    """Build an argv that runs this environment's ``layup``, not ``PATH``'s.

    ``subprocess.run(["layup", ...])`` resolves the name against ``PATH``, so
    the test runs whichever layup comes first on the machine instead of the one
    being tested. On a machine with an older layup installed the test fails on
    arguments the current code added, and -- worse -- when the older one happens
    to accept them, it passes without testing anything (issue #500).

    Console scripts are installed next to the interpreter, so resolving from
    ``sys.executable`` pins the call to this environment while still going
    through the real entry point and its verb dispatch.
    """
    import sys
    from pathlib import Path

    exe = Path(sys.executable).parent / "layup"
    return [str(exe) if exe.exists() else "layup", *args]
