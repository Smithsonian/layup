from __future__ import annotations
import os
from pathlib import Path

# temporarily change the working directory to help MacOS find
# the library objects, if running in a different directory.
cwd = os.getcwd()
os.chdir(Path(__file__).parent.parent.parent.parent)
from _layup_cpp._core import *

os.chdir(cwd)
