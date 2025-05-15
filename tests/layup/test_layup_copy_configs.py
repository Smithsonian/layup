import os

import pytest

# Test simple comment
def test_layupCopyConfigs(tmp_path):
    from layup.utilities.layup_copy_configs import copy_demo_configs

    # test that the config files are successfully copied
    copy_demo_configs(tmp_path, "Default", False)

    assert os.path.isfile(os.path.join(tmp_path, "default_config.ini"))

    # test that files are successfully overwritten if -f flag used
    copy_demo_configs(tmp_path, "Default", True)

    # test the error message if user supplies non-existent directory
    dummy_folder = os.path.join(tmp_path, "dummy_folder/file.csv")
    with pytest.raises(SystemExit) as e:
        copy_demo_configs(dummy_folder, "all", False)

    assert e.value.code == f"ERROR: filepath {dummy_folder} supplied for filepath argument does not exist."

    # test the error message if user supplies unrecognised keyword for which_configs variable
    with pytest.raises(SystemExit) as e2:
        copy_demo_configs(tmp_path, "laphroaig", True)

    assert e2.value.code == "String 'laphroaig' not recognised for 'configs' variable. Must be 'Default'."

    # Test simple comment
    # test the error message if file exists and overwrite isn't forced

    with pytest.raises(SystemExit) as e3:
        copy_demo_configs(tmp_path, "Default", False)

    assert (
        e3.value.code
        == "Identically named file exists at location. Re-run with -f or --force to force overwrite."
    )
