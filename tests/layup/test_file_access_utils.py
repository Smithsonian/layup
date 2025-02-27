#!/bin/python
import os
import shutil
import configparser
import pytest
import glob

from layup.utilities.data_utilities_for_tests import get_config_setups_filepath


def test_find_file_or_exit():
    from layup.utilities.file_access_utils import find_file_or_exit

    test_file = find_file_or_exit(get_config_setups_filepath("default_config.ini"), "default_config.ini")

    with pytest.raises(SystemExit) as e:
        find_file_or_exit("totally_fake_file.txt", "test")

    assert test_file == get_config_setups_filepath("default_config.ini")
    assert e.type == SystemExit
    assert e.value.code == "ERROR: filename totally_fake_file.txt supplied for test argument does not exist."

    return


def test_find_directory_or_exit():
    from layup.utilities.file_access_utils import find_directory_or_exit

    test_dir = find_directory_or_exit("./", "test")

    with pytest.raises(SystemExit) as e:
        find_directory_or_exit("./fake_dir/", "test")

    assert test_dir == "./"
    assert e.type == SystemExit
    assert e.value.code == "ERROR: filepath ./fake_dir/ supplied for test argument does not exist."

    return
