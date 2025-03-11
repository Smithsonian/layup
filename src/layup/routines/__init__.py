from __future__ import annotations

from _layup_cpp._core import __doc__, __version__, hello_world

# make libraries discoverable for linux
from sys import platform

if platform == "linux" or platform == "linux2":
    import os

    root_dir = os.path.dirname(os.path.abspath(__file__))
    if "LD_LIBRARY_PATH" in os.environ.keys():
        ld_lib_path = os.environ["LD_LIBRARY_PATH"]
        os.environ["LD_LIBRARY_PATH"] = ld_lib_path + ":" + root_dir
    else:
        os.environ["LD_LIBRARY_PATH"] = root_dir

__all__ = ["__doc__", "__version__", "hello_world"]
