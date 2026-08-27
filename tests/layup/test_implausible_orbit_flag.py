"""flag = 9: converged, but the orbit is not physically possible (issue #493).

A short arc can converge with an excellent reduced chi-square onto a state no
real object could occupy, because the arc does not constrain the velocity. The
chi-square gate cannot catch this -- it is anti-correlated with the failure,
since the less the object moves the better the fit (issue #485).

The criterion is hyperbolic excess speed, not boundedness. Layup is expected to
fit genuine interstellar objects, and those are unbound and fast: 3I/ATLAS
arrives at about 59 km/s. Only a speed far above any plausible arrival speed is
evidence of a bad fit rather than an unusual object.
"""

import numpy as np
import pytest

import layup.orbitfit as orbitfit_module
from layup.orbitfit import FLAG_IMPLAUSIBLE_ORBIT

GM_SUN = 2.9591220828559115e-4  # au^3/day^2
KM_S_IN_AU_DAY = 86400.0 / 149597870.7

DEFAULT_THRESHOLD_KM_S = 200.0
FASTEST_KNOWN_INTERSTELLAR_KM_S = 59.0  # 3I/ATLAS


@pytest.fixture(autouse=True)
def _restore_threshold():
    """The threshold is module-level state; put it back after each test."""
    original = orbitfit_module.MAX_EXCESS_SPEED_KM_S
    yield
    orbitfit_module.MAX_EXCESS_SPEED_KM_S = original


def set_max_v_inf(km_s):
    orbitfit_module.MAX_EXCESS_SPEED_KM_S = km_s


def get_max_v_inf():
    return orbitfit_module.MAX_EXCESS_SPEED_KM_S


def test_default_threshold_clears_the_fastest_known_interstellar_object():
    """The default must not reject a real interstellar object.

    Layup's stated advantage over OrbFit and OpenOrb is that it handles the
    bound-to-unbound transition, so a gate that rejects 3I/ATLAS would reject the
    capability the package advertises.
    """
    assert get_max_v_inf() == DEFAULT_THRESHOLD_KM_S
    assert get_max_v_inf() > 3 * FASTEST_KNOWN_INTERSTELLAR_KM_S


def test_threshold_is_configurable_and_round_trips_in_km_per_second():
    for km_s in (50.0, 100.0, 550.0):
        set_max_v_inf(km_s)
        assert get_max_v_inf() == pytest.approx(km_s)


def test_threshold_zero_disables_the_gate():
    set_max_v_inf(0.0)
    assert get_max_v_inf() == 0.0


# ---------------------------------------------------------------------------
# End-to-end, through a real fit. 3I/ATLAS is the natural fixture: it is a
# genuine interstellar object, so it is unbound and fast enough to sit on the
# interesting side of the gate, and layup already ships its discovery-arc
# astrometry. Moving the threshold across its excess speed must move the flag,
# which tests the gate itself rather than the accessor.
# ---------------------------------------------------------------------------

import os
import pooch

from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader

CACHE = str(pooch.os_cache("layup"))
_EPHEM_AVAILABLE = os.path.exists(os.path.join(CACHE, "linux_p1550p2650.440")) and os.path.exists(
    os.path.join(CACHE, "sb441-n16.bsp")
)

requires_ephem = pytest.mark.skipif(
    not _EPHEM_AVAILABLE,
    reason=f"ASSIST ephemeris missing at {CACHE}; skipping end-to-end gate test.",
)


def _fit_3i_atlas():
    """Fit 3I/ATLAS in-process and return (flag, excess speed in km/s)."""
    import spiceypy as spice
    from numpy.lib import recfunctions as rfn

    from layup.orbitfit import _orbitfit
    from layup.utilities.data_processing_utilities import LayupObservatory

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
    r = np.linalg.norm(state[:3])
    energy = 0.5 * np.dot(state[3:], state[3:]) - GM_SUN / r
    v_inf = np.sqrt(2 * energy) / KM_S_IN_AU_DAY if energy > 0 else 0.0
    return int(row["flag"]), v_inf


@requires_ephem
def test_real_interstellar_orbit_passes_at_the_default_threshold():
    flag, v_inf = _fit_3i_atlas()
    assert v_inf == pytest.approx(
        FASTEST_KNOWN_INTERSTELLAR_KM_S, abs=5.0
    ), f"3I/ATLAS should arrive near {FASTEST_KNOWN_INTERSTELLAR_KM_S} km/s, got {v_inf:.1f}"
    assert flag == 0, f"the default threshold rejected a real interstellar object (flag={flag})"


@requires_ephem
def test_gate_fires_when_the_threshold_drops_below_the_fitted_excess_speed():
    """Same fit, same data: only the threshold moves, so only the gate can
    explain a change in the flag."""
    _, v_inf = _fit_3i_atlas()
    set_max_v_inf(v_inf / 2.0)
    flag, _ = _fit_3i_atlas()
    assert flag == FLAG_IMPLAUSIBLE_ORBIT, f"expected flag=9 below the excess speed, got {flag}"


@requires_ephem
def test_disabled_gate_never_fires():
    _, v_inf = _fit_3i_atlas()
    set_max_v_inf(0.0)
    flag, _ = _fit_3i_atlas()
    assert flag == 0, f"gate fired while disabled (flag={flag})"
