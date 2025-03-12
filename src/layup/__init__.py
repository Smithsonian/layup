import os
from sys import platform
from pathlib import Path

# create symlinks for the assist and rebound shared objects
# in the main directory, ensuring that they are discoverable
# to cpython at runtime.

# root_dir = Path(__file__).parent.parent.parent
# assist_lib_path = os.path.join(root_dir, "include/assist/src/libassist.so")
# assist_sym_path_linux = os.path.join(root_dir, "src/libassist.so")
# assist_sym_path_osx = os.path.join(root_dir, "libassist.so")
# if not os.path.isfile(assist_sym_path_linux):
#     os.symlink(assist_lib_path, assist_sym_path_linux)
# if not os.path.isfile(assist_sym_path_osx):
#     os.symlink(assist_lib_path, assist_sym_path_osx)

# rebound_lib_path = os.path.join(root_dir, "include/rebound/src/librebound.so")
# rebound_sym_path_linux = os.path.join(root_dir, "src/librebound.so")
# rebound_sym_patgh_osx = os.path.join(root_dir, "librebound.so")
# if not os.path.isfile(rebound_sym_path_linux):
#     os.symlink(rebound_lib_path, rebound_sym_path_linux)
# if not os.path.isfile(rebound_sym_path_osx):
#     os.symlink(rebound_lib_path, rebound_sym_path_osx)
try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")
