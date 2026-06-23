from __future__ import annotations

# The compiled extension is linked with an RPATH (``$ORIGIN/..`` on Linux,
# ``@loader_path/..`` on macOS, set in CMakeLists.txt) so the dynamic loader
# finds librebound/libassist in site-packages regardless of the working
# directory. The previous os.chdir() dance is therefore no longer needed (#75).
from _layup_cpp._core import *
