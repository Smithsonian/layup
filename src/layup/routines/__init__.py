from __future__ import annotations

import sys

# On Linux the compiled extension is linked against librebound/libassist with an
# RPATH (``$ORIGIN/..``, set in CMakeLists.txt), so the dynamic loader pulls them
# in and the previous os.chdir() dance is not needed (#75).
#
# On macOS we cannot link against them at all. assist and rebound ship their C
# libraries as Python extension modules, and whether those come out as dynamic
# libraries (linkable) or bundles (loadable only) depends on the toolchain that
# built them; Xcode 26 / AppleClang 21 produces bundles, which ``ld`` refuses to
# link against ("unsupported mach-o filetype") -- issue #457. The extension is
# therefore built unlinked on macOS, with ``-undefined dynamic_lookup``, and its
# rebound/assist symbols are resolved from the flat namespace at load time.
#
# Importing ``assist`` and ``rebound`` is NOT enough to populate that namespace:
# both load their libraries through ctypes, whose default mode is ``RTLD_LOCAL``,
# so the symbols stay private to those handles. We reopen the very same files
# with ``RTLD_GLOBAL``, which promotes the already-loaded images rather than
# mapping second copies -- important, because a second copy would give the C++
# side its own library-global state, separate from the one the Python-level
# ``assist``/``rebound`` calls use. Order matters: libassist depends on librebound.
if sys.platform == "darwin":
    import ctypes

    import assist
    import rebound

    for _lib in (rebound, assist):
        _libpath = getattr(_lib, "__libpath__", "")
        if not _libpath:
            raise ImportError(
                f"{_lib.__name__} does not expose __libpath__, so layup cannot promote its "
                "library to the global symbol namespace. layup's compiled extension resolves "
                f"its symbols there on macOS; check the installed {_lib.__name__} version."
            )
        ctypes.CDLL(_libpath, mode=ctypes.RTLD_GLOBAL)

from _layup_cpp._core import *
