import os
import argparse
from pathlib import Path
import shutil
import sys
from importlib.resources import files

from layup.utilities.file_access_utils import find_directory_or_exit
from layup.utilities.layup_demo_commands import print_demo_command


def copy_demo_files(verb,copy_location, force_overwrite):
    """
    Copies the files needed to run the Sorcha demo to a user-specified location.

    Parameters
    -----------
    copy_location : string
        String containing the filepath of the location to which the configuration files should be copied.

    force_overwrite : boolean
        Flag for determining whether existing files should be overwritten.

    Returns
    -----------
    None

    """

    _ = find_directory_or_exit(copy_location, "filepath")

    # add verb demo specific files here and create the directorys 
    if verb == "orbitfit":
        demo_data_root = files("layup.data.demo.orbitfit")

        demo_files = [
            "holman_data_working.csv",
        ]
    elif verb == "convert":
        demo_data_root = files("layup.data.demo.convert")

        demo_files = [
            
        ]
       
    elif verb == "predict":
        demo_data_root = files("layup.data.demo.predict")

        demo_files = [
           
        ]
       
    elif verb == "comet":
        demo_data_root = files("layup.data.demo.comet")

        demo_files = [
            
        ]
        
    elif verb == "visualize":
        demo_data_root = files("layup.data.demo.visualise")

        demo_files = [
            
        ]
      


    for fn in demo_files:
        if not force_overwrite and os.path.isfile(os.path.join(copy_location, fn)):
            sys.exit(
                "Identically named file exists at location. Re-run with -f or --force to force overwrite."
            )

        demo_path = demo_data_root.joinpath(fn)
        shutil.copy(demo_path, copy_location)

    print("Demo files {} copied to {}.".format(demo_files, copy_location))

    print_demo_command(verb,printall=False)