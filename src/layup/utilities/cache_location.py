"""Where layup keeps its downloaded ephemeris and reference data.

layup writes roughly 1.6 GB of SPICE kernels, observatory codes and debiasing
tables to a cache directory. The default comes from ``pooch.os_cache("layup")``,
which is under the user's home directory. On a cluster the home directory is
often a different, smaller partition than the one layup itself is installed on,
so that default is the wrong place (issue #448, raised in #443).

Individual entry points already accept a ``cache_dir`` argument, but that has to
be threaded through every call. This module makes the *default* settable once,
through the ``LAYUP_CACHE_DIR`` environment variable::

    export LAYUP_CACHE_DIR=/data/shared/layup-cache

Every place in layup that needs the default calls :func:`default_cache_dir`, so
setting the variable moves all of it. An explicit ``cache_dir`` argument still
wins over the environment variable, which in turn wins over the OS cache.
"""

import os
from pathlib import Path

import pooch

#: Environment variable that overrides the default cache location.
LAYUP_CACHE_ENV_VAR = "LAYUP_CACHE_DIR"


def default_cache_dir() -> Path:
    """Return the directory layup should use for downloaded data by default.

    Returns
    -------
    pathlib.Path
        ``$LAYUP_CACHE_DIR`` if that variable is set to a non-empty value, with
        a leading ``~`` expanded; otherwise the platform cache directory that
        ``pooch`` chooses, which is what layup used before this was
        configurable.

    Notes
    -----
    The directory is not created here and is not required to exist. Callers
    hand it to ``pooch``, which creates it on download; returning a path for a
    directory that does not exist yet is the same behaviour as before.
    """
    override = os.environ.get(LAYUP_CACHE_ENV_VAR)
    if override and override.strip():
        return Path(override.strip()).expanduser()
    return Path(pooch.os_cache("layup"))
