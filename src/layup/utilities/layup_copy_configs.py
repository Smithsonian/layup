import os
import argparse
from pathlib import Path
import shutil
import sys
from importlib.resources import files

from layup.utilities.file_access_utils import find_directory_or_exit


def copy_demo_configs(copy_location, which_configs, force_overwrite):
    """
    Copies the example layup configuration files to a user-specified location.

    Parameters
    -----------
    copy_location : string
        String containing the filepath of the location to which the configuration files should be copied.

    which_configs : string
        String indicating which configuration files to retrieve. Should be "rubin", "demo" or "all".

    force_overwrite: boolean
        Flag for determining whether existing files should be overwritten.

    Returns
    -----------
    None

    """
    # For the future multiple config file formats can be stored in config_setups and be called

    find_directory_or_exit(copy_location, "filepath")

    config_data_root = files("layup.config_setups")

    configs = {
        "Default": ["default_config.ini"],
    }

    if which_configs in configs:
        config_locations = configs[which_configs]
    # elif which_configs == "all":
    #     config_locations = [fn for fns in configs.values() for fn in fns]
    else:
        sys.exit(
            "String '{}' not recognised for 'configs' variable. Must be 'Default'.".format(which_configs)
        )

    for fn in config_locations:
        if not force_overwrite and os.path.isfile(os.path.join(copy_location, fn)):
            sys.exit(
                "Identically named file exists at location. Re-run with -f or --force to force overwrite."
            )

        config_path = config_data_root.joinpath(fn)
        shutil.copy(config_path, copy_location)

    print("Example configuration files {} copied to {}.".format(config_locations, copy_location))
