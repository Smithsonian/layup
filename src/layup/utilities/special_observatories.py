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


def query_horizons_geocentric(naif_id, jd_tdb_list, timeout=30):
    """Geocentric ICRF state of a spacecraft at the given epochs, via JPL Horizons.

    Parameters
    ----------
    naif_id : str
        JPL/NAIF id of the spacecraft (e.g. ``"-48"`` for Hubble).
    jd_tdb_list : sequence of float
        Observation epochs as Julian Date (TDB).
    timeout : float, optional
        Per-request HTTP timeout in seconds.

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
    out = {}
    jd_list = list(jd_tdb_list)
    for start in range(0, len(jd_list), _TLIST_CHUNK):
        out.update(_query_chunk(naif_id, jd_list[start : start + _TLIST_CHUNK], timeout))
    return out


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
