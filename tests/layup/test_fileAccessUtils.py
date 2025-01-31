#!/bin/python
import os
import shutil
import configparser
import pytest
import glob

from layup.utilities.dataUtilitiesForTests import get_config_setups_filepath


def test_FindFileOrExit():
    from layup.utilities.fileAccessUtils import FindFileOrExit

    test_file = FindFileOrExit(
        get_config_setups_filepath("Default_config_file.ini"), "Default_config_file.ini"
    )

    with pytest.raises(SystemExit) as e:
        FindFileOrExit("totally_fake_file.txt", "test")

    assert test_file == get_config_setups_filepath("Default_config_file.ini")
    assert e.type == SystemExit
    assert e.value.code == "ERROR: filename totally_fake_file.txt supplied for test argument does not exist."

    return


def test_FindDirectoryOrExit():
    from layup.utilities.fileAccessUtils import FindDirectoryOrExit

    test_dir = FindDirectoryOrExit("./", "test")

    with pytest.raises(SystemExit) as e:
        FindDirectoryOrExit("./fake_dir/", "test")

    assert test_dir == "./"
    assert e.type == SystemExit
    assert e.value.code == "ERROR: filepath ./fake_dir/ supplied for test argument does not exist."

    return
