"""Pluggable initial orbit determination (IOD) layer for layup.

An IOD method is a callable that proposes one or more seed orbits for
the Marquardt fitter to refine. The expected signature is

    iod(observations, seq) -> list[FitResult] | None

where `observations` is the time-ordered list of layup `Observation`s,
`seq` is the per-segment index list (`seq[0]` is the primary segment
used for the IOD), and the return is either a list of candidate seed
orbits (each a `FitResult` with at least `state` and `epoch`
populated) or `None` / empty if no candidate could be produced.

Multiple IOD candidates are returned when the underlying method is
multi-valued (Gauss's polynomial in r₂ has up to eight real roots);
`do_fit` runs LM from each candidate and picks the best converged fit
(smallest χ² subject to a sanity bound on heliocentric distance).

Methods register themselves at import time via the module-level
registry; use `register_iod(name, callable)` to add new methods,
`get_iod(name)` to look one up, and `iod_methods()` to list all
available names.

Why a registry instead of subclassing: IOD methods are stateless
strategies whose entire interface fits on one line. A function-pointer
registry is the smallest abstraction that supports drop-in
replacements (e.g. a Lambert-based method, a motion-rate prior, a
prelim from BK's tangent-plane linear fit) without forcing each
implementation through a class hierarchy.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

from layup.routines import FitResult, Observation, gauss

logger = logging.getLogger(__name__)

GMtotal = 0.0002963092748799319
AU_M = 149597870700
SPEED_OF_LIGHT = 2.99792458e8 * 86400.0 / AU_M


# An IOD method takes (observations, seq) and returns either a list of
# candidate seed orbits or None.
IODCallable = Callable[[Sequence[Observation], Sequence[Sequence[int]]],
                       Optional[list]]


# Module-level registry, name -> callable. Populated below.
_REGISTRY: dict[str, IODCallable] = {}


def register_iod(name: str, func: IODCallable) -> None:
    """Register an IOD method under `name`. Overwrites an existing entry."""
    _REGISTRY[name.lower()] = func


def get_iod(name: str) -> IODCallable:
    """Look up an IOD method by name. Raises ValueError if unknown."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown IOD method {name!r}. "
            f"Registered methods: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def iod_methods() -> list[str]:
    """Return the sorted list of registered IOD method names."""
    return sorted(_REGISTRY)


# ----------------------------------------------------------------------- #
# Built-in: Gauss's method.                                               #
# ----------------------------------------------------------------------- #

def gauss_iod(observations, seq):
    """Gauss's method on the first/middle/last observation of seq[0].

    The C++ `gauss` binding returns up to eight candidate seed orbits
    (corresponding to the real roots of the 8th-degree polynomial in
    r₂); we pass them all upstream so the picker can pick the right
    one rather than committing to `solns[0]` blindly.
    """
    idx0 = seq[0][0]
    idx1 = seq[0][len(seq[0]) // 2]
    idx2 = seq[0][-1]
    logger.debug(f"gauss_iod: indices {idx0}, {idx1}, {idx2}")
    solns = gauss(GMtotal,
                  observations[idx0], observations[idx1], observations[idx2],
                  0.0001, SPEED_OF_LIGHT)
    return solns


register_iod("gauss", gauss_iod)
