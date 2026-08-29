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
    """argv prefix that runs *this* environment's ``layup``, not ``PATH``'s.

    ``subprocess.run(["layup", ...])`` resolves the name against ``PATH``, so a
    test exercises whichever installation comes first on the machine rather than
    the one under test. That fails confusingly when another layup is installed
    (a conda environment with a broken assist/rebound link, say) and, worse,
    passes for the wrong reason when the stale installation happens to work
    (issue #500).

    Console scripts are installed alongside the interpreter running the tests,
    so resolving from ``sys.executable`` pins the invocation to this environment
    while still exercising the real entry point and its subcommand dispatch.
    """
    import sys
    from pathlib import Path

    exe = Path(sys.executable).parent / "layup"
    return [str(exe) if exe.exists() else "layup", *args]
