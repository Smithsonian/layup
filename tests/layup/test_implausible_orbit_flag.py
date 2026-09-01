"""A fit can converge, satisfy every statistical check, and still describe an
orbit no object could occupy.

A short arc does not constrain the velocity, so the differential correction can
settle on an impossibly fast state with an excellent reduced chi-square. The
chi-square check cannot catch it: the less the object appears to move across the
arc, the better that fit looks.

The criterion is hyperbolic excess speed, not boundedness. Layup is expected to
fit genuine interstellar objects, which are unbound and fast -- 3I/ATLAS arrives
at about 59 km/s -- so being unbound is never itself grounds for rejection.
"""

import os

import numpy as np
import pooch
import pytest

from layup.constants import (
    FLAG_CONVERGED,
    KM_S_IN_AU_PER_DAY,
    MAX_EXCESS_SPEED_KM_S,
    MU_SUN,
)
from layup.orbitfit import _implausible_excess_speed

FASTEST_KNOWN_INTERSTELLAR_KM_S = 59.0  # 3I/ATLAS


def _state_with_excess_speed(v_inf_km_s, r_au=1.0):
    """A state at ``r_au`` whose hyperbolic excess speed is ``v_inf_km_s``.

    From the vis-viva energy, v_inf^2 = v^2 - 2*GM/r, so a speed of
    sqrt(v_inf^2 + 2*GM/r) at radius r gives exactly the excess speed asked for.
    """
    v_inf = v_inf_km_s * KM_S_IN_AU_PER_DAY
    speed = np.sqrt(v_inf**2 + 2.0 * MU_SUN / r_au)
    return [r_au, 0.0, 0.0, 0.0, speed, 0.0]


# --------------------------------------------------------------------------
# The criterion itself. Constructed states, so nothing has to move the
# threshold to exercise both sides of it.
# --------------------------------------------------------------------------


def test_a_bound_orbit_is_never_implausible():
    """Circular at 1 au: negative energy, so there is no excess speed at all."""
    speed = np.sqrt(MU_SUN)
    assert _implausible_excess_speed([1.0, 0.0, 0.0, 0.0, speed, 0.0]) is False


def test_just_below_the_threshold_is_accepted():
    state = _state_with_excess_speed(MAX_EXCESS_SPEED_KM_S * 0.99)
    assert _implausible_excess_speed(state) is False


def test_just_above_the_threshold_is_rejected():
    state = _state_with_excess_speed(MAX_EXCESS_SPEED_KM_S * 1.01)
    assert _implausible_excess_speed(state) is True


def test_the_threshold_clears_the_fastest_known_interstellar_object():
    """Rejecting a real interstellar object would reject the bound-to-unbound
    case layup exists to handle, so the margin is checked rather than assumed."""
    state = _state_with_excess_speed(FASTEST_KNOWN_INTERSTELLAR_KM_S)
    assert _implausible_excess_speed(state) is False
    assert MAX_EXCESS_SPEED_KM_S > 3 * FASTEST_KNOWN_INTERSTELLAR_KM_S


def test_a_nan_state_is_left_to_the_convergence_flag():
    """A diverged fit is already reported as not converged; this check must not
    turn it into a different kind of failure."""
    assert _implausible_excess_speed([np.nan] * 6) is False


def test_a_state_at_the_origin_is_not_reported():
    """Radius zero makes the energy undefined rather than large."""
    assert _implausible_excess_speed([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) is False


# --------------------------------------------------------------------------
# End to end, through a real fit.
# --------------------------------------------------------------------------

CACHE = str(pooch.os_cache("layup"))
_EPHEM_AVAILABLE = os.path.exists(os.path.join(CACHE, "linux_p1550p2650.440")) and os.path.exists(
    os.path.join(CACHE, "sb441-n16.bsp")
)

requires_ephem = pytest.mark.skipif(
    not _EPHEM_AVAILABLE,
    reason=f"ASSIST ephemeris missing at {CACHE}; skipping the end-to-end test.",
)


@requires_ephem
def test_a_real_interstellar_orbit_is_accepted():
    """3I/ATLAS is unbound and fast, and layup ships its discovery arc. It must
    come through with the flag clear and the column clear."""
    import spiceypy as spice
    from numpy.lib import recfunctions as rfn

    from layup.orbitfit import _orbitfit
    from layup.utilities.data_processing_utilities import LayupObservatory
    from layup.utilities.data_utilities_for_tests import get_test_filepath
    from layup.utilities.file_io.CSVReader import CSVDataReader

    obs = CSVDataReader(
        get_test_filepath("3I_ATLAS_ades.csv"), "csv", primary_id_column_name="provID"
    ).read_rows()
    helper = LayupObservatory(cache_dir=CACHE)
    et = np.array([spice.str2et(t) for t in obs["obsTime"]], dtype="<f8")
    obs = rfn.append_fields(obs, "et", et, usemask=False, asrecarray=True)
    obs = rfn.merge_arrays(
        [obs, helper.obscodes_to_barycentric(obs)], flatten=True, asrecarray=True, usemask=False
    )
    row = _orbitfit(obs, cache_dir=CACHE, primary_id_column_name="provID", iod="gauss", engine="cartesian")[0]

    state = np.array([row[k] for k in ("x", "y", "z", "xdot", "ydot", "zdot")], dtype=float)
    energy = 0.5 * np.dot(state[3:], state[3:]) - MU_SUN / np.linalg.norm(state[:3])
    v_inf = np.sqrt(2 * energy) / KM_S_IN_AU_PER_DAY if energy > 0 else 0.0

    assert v_inf == pytest.approx(
        FASTEST_KNOWN_INTERSTELLAR_KM_S, abs=5.0
    ), f"expected an arrival speed near {FASTEST_KNOWN_INTERSTELLAR_KM_S} km/s, got {v_inf:.1f}"
    assert int(row["flag"]) == FLAG_CONVERGED, f"a real interstellar object was rejected (flag={row['flag']})"
    assert int(row["failed_physical"]) == 0
