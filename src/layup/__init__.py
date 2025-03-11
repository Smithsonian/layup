import os
from sys import platform

# create symlinks for the assist and rebound shared objects
# in the main directory, ensuring that they are discoverable
# to cpython at runtime.
if not os.path.isfile("./libassist.so"):
    os.symlink("./include/assist/src/libassist.so", "./libassist.so")

if not os.path.isfile("./librebound.so"):
    os.symlink("./include/rebound/src/librebound.so", "./librebound.so")

# make libraries discoverable for linux
if platform == "linux" or platform == "linux2":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if "LD_LIBRARY_PATH" in os.environ.keys():
        ld_lib_path = os.environ["LD_LIBRARY_PATH"]
        os.environ["LD_LIBRARY_PATH"] = ld_lib_path + ":" + root_dir
    else:
        os.environ["LD_LIBRARY_PATH"] = root_dir
