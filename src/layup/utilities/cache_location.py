"""Where layup keeps its downloaded ephemeris and reference data.

layup downloads roughly 1.6 GB of SPICE kernels, observatory codes and debiasing
tables. By default these go to the platform cache directory under the user's
home. On a cluster the home directory is often a smaller partition than the one
layup runs from, so the default can be the wrong place.

Set ``LAYUP_CACHE_DIR`` to move it::

    export LAYUP_CACHE_DIR=/data/shared/layup-cache

Precedence: an explicit ``cache_dir`` argument, then ``LAYUP_CACHE_DIR``, then
the platform cache.

The variable is read on each call rather than at import, so it can be changed
within a running process. Note that it is a *process* setting: two jobs started
from the same shell share whatever that shell exports, so to give them separate
caches, set it per job rather than exporting it once.
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
    hand it to ``pooch``, which creates it on download.
    """
    override = os.environ.get(LAYUP_CACHE_ENV_VAR)
    if override and override.strip():
        return Path(override.strip()).expanduser()
    return Path(pooch.os_cache("layup"))
