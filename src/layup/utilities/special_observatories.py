"""Resolve space-based ("special case") MPC observatory codes via JPL Horizons.

A handful of MPC observatory codes denote *spacecraft* rather than fixed ground
stations (issue #55). They have no parallax constants, so layup cannot compute
their position geometrically, and -- unlike a roving ground observer -- the user
usually does not carry an explicit per-observation position for them. For these
codes we query the JPL Horizons vector API for the spacecraft's geocentric state
at each observation epoch and feed it into the same moving-observer path that an
ADES-supplied position uses (issue #147).

The obscode -> NAIF-id map is ported from the Rust ``spacerocks`` project
(``~/Dropbox/roman-footprint/spacerocks``). Spacecraft carry negative NAIF ids
by JPL convention.

This is intentionally a *live* HTTP lookup (the case is rare); a user-supplied
ADES position always takes priority, so fits never depend on Horizons being
reachable when a position is provided.
"""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

#: JPL Horizons REST endpoint.
HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"

#: MPC observatory code -> (JPL/NAIF id, human-readable name). Ported from the
#: Rust ``spacerocks`` ``SPACEOBSERVATORIES`` map.
SPACE_OBSERVATORIES = {
    "245": ("-79", "Spitzer Space Telescope"),
    "250": ("-48", "Hubble Space Telescope"),
    "274": ("-170", "James Webb Space Telescope"),
    "C49": ("-234", "STEREO-A"),
    "C50": ("-235", "STEREO-B"),
    "C51": ("-163", "WISE"),
    "C54": ("-98", "New Horizons"),
    "C55": ("-227", "Kepler"),
    "C58": ("-33", "NEO Surveyor"),
}

#: Julian Date of the J2000 epoch.
J2000_JD = 2451545.0
SECONDS_PER_DAY = 86400.0

#: Max epochs per Horizons request. Keeps the GET URL well within limits while
#: still collapsing a many-observation fit into a few requests per spacecraft.
_TLIST_CHUNK = 40


def et_to_jd_tdb(et):
    """SPICE ephemeris time (seconds past J2000 TDB) -> Julian Date (TDB)."""
    return J2000_JD + et / SECONDS_PER_DAY


def is_space_observatory(obscode):
    """Whether ``obscode`` is a spacecraft we resolve via JPL Horizons."""
    return obscode in SPACE_OBSERVATORIES


def query_horizons_geocentric(naif_id, jd_tdb_list, timeout=30, cache_dir=None):
    """Geocentric ICRF state of a spacecraft at the given epochs, via JPL Horizons.

    Parameters
    ----------
    naif_id : str
        JPL/NAIF id of the spacecraft (e.g. ``"-48"`` for Hubble).
    jd_tdb_list : sequence of float
        Observation epochs as Julian Date (TDB).
    timeout : float, optional
        Per-request HTTP timeout in seconds.
    cache_dir : str or path or None or False, optional
        Location of the persistent ``(naif_id, jd) -> state`` disk cache. The
        states are geometric (``VEC_CORR='NONE'``) and therefore deterministic and
        safe to cache across objects, processes, and runs -- one Horizons call per
        (spacecraft, epoch), ever, instead of once per object per process (which
        rate-limits to HTTP 503 at catalog scale). Precedence: an explicit
        ``cache_dir`` (uses a ``horizons/`` subdir under it) > the
        ``LAYUP_HORIZONS_CACHE`` env var > the default layup pooch cache. Pass
        ``cache_dir=False`` (or ``LAYUP_HORIZONS_CACHE=0``) to disable.

    Returns
    -------
    dict
        ``{jd_tdb: (pos_km, vel_km_s)}``; each value is a length-3 ``numpy``
        array. Positions are geocentric (center = Earth body center), ICRF /
        equatorial-J2000, in km, and velocities in km/s. No light-time or
        aberration correction is applied (``VEC_CORR='NONE'``): the *geometric*
        state at the requested TDB epoch is returned, matching how layup
        supplies every other observer position (light time is handled later, in
        the integrator).

    Notes
    -----
    We deliberately query the **geocentric** spacecraft state (center = Earth's
    body center) rather than the barycentric one, so this plugs into the same
    "geocentric observer offset + Earth's barycentric state" path the ground
    stations use: ``_barycentric_moving_observatory`` adds Earth's barycentric
    state (from layup's own SPICE kernel) to this offset. Referencing the offset
    against a physical body (Earth) is more reproducible than referencing against
    the solar-system barycenter, whose definition shifts between ephemeris
    versions.

    Caveat for high-precision work: Earth's geocenter as realized by layup's
    SPICE kernel may differ slightly from the geocenter implied by the
    spacecraft's Horizons reference ephemeris. The geocentric offset cancels
    Earth's own position to first order, but a small residual inconsistency
    (~meters) can remain. That is negligible for normal astrometry, but could
    matter for very high precision applications such as space-based stellar
    occultations -- flagged here as a known limitation (see PR #377 discussion).
    """
    jd_list = list(jd_tdb_list)
    root = _horizons_cache_root(cache_dir)
    if root is None:
        return _query_horizons_uncached(naif_id, jd_list, timeout)

    cdir = root / str(naif_id)
    out, misses = {}, []
    for jd in jd_list:
        state = _cache_load(cdir, jd)
        if state is None:
            misses.append(jd)
        else:
            out[jd] = state
    if misses:
        fetched = _query_horizons_uncached(naif_id, misses, timeout)
        for jd, state in fetched.items():
            _cache_store(cdir, jd, state)
            out[jd] = state
    return out


def _query_horizons_uncached(naif_id, jd_list, timeout):
    """The raw chunked Horizons query, with no disk cache (one network round-trip
    per ``_TLIST_CHUNK`` epochs)."""
    out = {}
    for start in range(0, len(jd_list), _TLIST_CHUNK):
        out.update(_query_chunk(naif_id, jd_list[start : start + _TLIST_CHUNK], timeout))
    return out


def _horizons_cache_root(cache_dir):
    """Resolve the persistent-cache root directory, or ``None`` to disable caching.

    See ``query_horizons_geocentric`` for the precedence. Returns a ``Path`` whose
    per-spacecraft subdirectories hold one ``<jd>.npz`` per cached epoch.
    """
    if cache_dir is False:
        return None
    env = os.environ.get("LAYUP_HORIZONS_CACHE")
    if env is not None and env.lower() in ("0", "off", "false", "none", ""):
        return None
    if cache_dir:
        return Path(cache_dir) / "horizons"
    if env:
        return Path(env)
    # No explicit cache_dir and no env override -> caching off, preserving the
    # historical behavior for direct callers. LayupObservatory threads its own
    # cache_dir, so the observatory path (and the catalog run) caches by default.
    return None


def _cache_file(cdir, jd):
    # Round to 1e-6 day (~0.09 s) to match the parser's jd matching, so two obs at
    # effectively the same epoch share one cache entry.
    return cdir / f"{round(jd, 6):.6f}.npz"


def _cache_load(cdir, jd):
    path = _cache_file(cdir, jd)
    if not path.exists():
        return None
    try:
        with np.load(path) as d:
            return (d["pos"].copy(), d["vel"].copy())
    except Exception:
        return None  # corrupt/partial file -> treat as a miss and re-fetch/overwrite


def _cache_store(cdir, jd, state):
    """Best-effort atomic write of one (pos, vel) state. Distinct epochs are
    distinct files, so this is lock-free and safe on a shared/parallel filesystem;
    a temp-file-plus-rename keeps a concurrent reader from seeing a partial file.
    A cache write must never break a fit, so failures are swallowed."""
    pos, vel = state
    try:
        cdir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cdir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, pos=np.asarray(pos), vel=np.asarray(vel))
            os.replace(tmp, _cache_file(cdir, jd))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception:
        pass


def _query_chunk(naif_id, jd_chunk, timeout):
    params = {
        "format": "json",
        "COMMAND": f"'{naif_id}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": "'@399'",  # geocentric offset (see precision caveat in the docstring)
        "REF_PLANE": "FRAME",  # ICRF equatorial (not the ecliptic)
        "REF_SYSTEM": "ICRF",
        "VEC_CORR": "NONE",  # geometric state at the epoch
        "OUT_UNITS": "KM-S",  # km and km/s -> layup's internal moving-observer units
        "VEC_TABLE": "2",  # position + velocity
        "VEC_LABELS": "NO",
        "VEC_DELTA_T": "NO",
        "CSV_FORMAT": "YES",
        "TLIST_TYPE": "JD",
        "TLIST": " ".join(f"{jd:.9f}" for jd in jd_chunk),
    }
    resp = requests.get(HORIZONS_API, params=params, timeout=timeout)
    resp.raise_for_status()
    text = resp.json().get("result", "")
    return _parse_vectors(text, jd_chunk, naif_id)


def _parse_vectors(text, jd_chunk, naif_id):
    """Parse the CSV vector table between the ``$$SOE``/``$$EOE`` markers.

    With ``CSV_FORMAT='YES'``, ``VEC_TABLE='2'`` and ``VEC_LABELS='NO'`` each
    data row is ``JDTDB, CalendarDate, X, Y, Z, VX, VY, VZ`` (trailing comma).
    """
    lines = text.splitlines()
    try:
        soe = lines.index("$$SOE")
        eoe = lines.index("$$EOE")
    except ValueError:
        raise RuntimeError(
            f"JPL Horizons returned no vector data for target {naif_id}. " f"Response begins:\n{text[:500]}"
        )

    parsed = {}
    for line in lines[soe + 1 : eoe]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        jd = float(parts[0])
        state = np.array([float(p) for p in parts[2:8]], dtype=float)
        parsed[round(jd, 6)] = (state[0:3], state[3:6])

    result = {}
    for jd in jd_chunk:
        match = parsed.get(round(jd, 6))
        if match is None:
            raise RuntimeError(
                f"JPL Horizons did not return a state for target {naif_id} at "
                f"JD(TDB) {jd:.9f} (outside the spacecraft's SPK coverage?)."
            )
        result[jd] = match
    return result
