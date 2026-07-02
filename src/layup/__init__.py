import warnings as _warnings

# Silence a cosmetic AstropyDeprecationWarning emitted at import time by sbpy
# (sbpy/_astropy_init.py builds a deprecated astropy TestRunner). sbpy is a
# transitive dependency of layup, so the warning is pure noise on the layup CLI
# and in notebooks. The filter is narrow -- it matches only the TestRunner
# message, so unrelated astropy deprecation warnings are unaffected -- and is
# installed here, before any layup submodule imports sbpy. See issue #376.
_warnings.filterwarnings("ignore", message=r".*TestRunner.*")

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")
