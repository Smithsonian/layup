"""Inner-solar-system X05 arcs -- the regime a distant initial range misses.

These 55 arcs are Rubin-only astrometry (MPC obscode X05) for objects whose
cold-start Gauss fit ran past a 420 s deadline and returned nothing. They are
the cases an alternative initial-orbit method exists to cover. Published
semi-major axes run from 2.10 to 3.93 au: 54 are main belt and one, K25N52C at
3.93 au, sits in the Hilda region near the 3:2 resonance. The Herget prototype
in #502 starts its range iteration at 40 au, a trans-Neptunian value, so this
sample is the one that exercises that choice.

The reference elements are MPCORB's. They are fitted to each object's full
observation history rather than to these short arcs, so comparing against them
is an external check and not a restatement of layup's own answer.

Two things are deliberately *not* asserted:

* **How many arcs converge.** Choosing the initial range is open work, and a
  test that pinned today's count would fail on the improvement it exists to
  encourage. The count is reported instead.
* **The initial-orbit state itself.** It depends on the starting range, which
  is exactly what is expected to change, so pinning it now would freeze a
  number that is meant to move.

What is asserted is the part that must hold whatever the method does: an arc
that converges has to land on the published orbit.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from layup.utilities.data_utilities_for_tests import get_test_filepath

ARC_DIR = Path(get_test_filepath("herget_x05"))
MANIFEST = Path(get_test_filepath("herget_x05_truth.json"))

# Bounds on the published semi-major axes, wide enough to be a statement about
# the sample rather than a restatement of its extremes (measured 2.10 to 3.93).
A_MIN, A_MAX = 1.5, 4.5

# Agreement between a fit to a ~15 day arc and MPCORB's fit to the full history.
# The measured median over this sample is 2.6e-3.
MAX_DA_OVER_A = 0.05


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def test_every_arc_has_reference_elements(manifest):
    """A fixture arc with no published orbit could not be checked against
    anything, so the pairing is verified rather than assumed."""
    on_disk = {p.stem for p in ARC_DIR.glob("*.obs80")}
    assert on_disk, f"no arcs found in {ARC_DIR}"
    assert on_disk == set(manifest), (
        f"arcs and manifest disagree: "
        f"{sorted(on_disk - set(manifest))} unlisted, "
        f"{sorted(set(manifest) - on_disk)} missing"
    )
    for obj, entry in manifest.items():
        assert entry["a"] is not None, f"{obj} has no reference semi-major axis"
        assert entry["n_obs"] == sum(1 for _ in open(ARC_DIR / f"{obj}.obs80"))


def test_the_sample_stays_in_the_inner_solar_system(manifest):
    """The point of this fixture is the regime it covers, so the regime is
    checked. A distant object added here later would quietly turn it back into
    the case the 40 au starting range already handles."""
    a = np.array([e["a"] for e in manifest.values()])
    assert (
        (a > A_MIN) & (a < A_MAX)
    ).all(), f"outside the sampled regime: {sorted(a[(a <= A_MIN) | (a >= A_MAX)])}"
    assert len(a) >= 50, "the sample has been thinned; the regime claim weakens"


def test_the_arcs_are_short(manifest):
    """Short arcs are the whole difficulty. A long arc slipping into this
    fixture would make the sample easier without anyone noticing."""
    n = np.array([e["n_obs"] for e in manifest.values()])
    assert n.max() <= 30, f"an arc has {n.max()} observations; these should be short"


@pytest.mark.skipif(
    os.environ.get("LAYUP_RUN_X05_FITS") != "1",
    reason="fits 55 arcs; opt in with LAYUP_RUN_X05_FITS=1",
)
def test_converged_fits_agree_with_the_published_orbits(manifest, tmp_path):
    """Whatever initial range is used, an arc that converges must reproduce the
    published orbit. Arcs that do not converge are counted, not failed."""
    from layup.iod import iod_methods

    if "herget" not in iod_methods():
        pytest.skip("the herget IOD is not registered in this build")

    import spiceypy as spice
    from numpy.lib import recfunctions as rfn

    from layup.orbitfit import _orbitfit
    from layup.utilities.data_processing_utilities import LayupObservatory
    from layup.utilities.file_io.Obs80Reader import Obs80DataReader

    GM = 2.9591220828559115e-4
    obsv = LayupObservatory()
    checked, converged = 0, []

    for obj, entry in sorted(manifest.items()):
        rows = Obs80DataReader(str(ARC_DIR / f"{obj}.obs80")).read_rows()
        et = np.array([spice.str2et(t) for t in rows["obsTime"]], dtype="<f8")
        d = rfn.append_fields(rows, "et", et, usemask=False, asrecarray=True)
        d = rfn.merge_arrays(
            [d, obsv.obscodes_to_barycentric(d)], flatten=True, asrecarray=True, usemask=False
        )
        res = _orbitfit(d, primary_id_column_name="ObjID", iod="herget", engine="cartesian")[0]
        checked += 1
        if int(res["flag"]) != 0:
            continue
        state = np.array([res[k] for k in ("x", "y", "z", "xdot", "ydot", "zdot")], dtype=float)
        energy = 0.5 * state[3:] @ state[3:] - GM / np.linalg.norm(state[:3])
        if energy >= 0:
            continue  # an unbound fit has no semi-major axis to compare
        a_fit = -GM / (2 * energy)
        da = abs(a_fit - entry["a"]) / entry["a"]
        converged.append(da)
        assert da < MAX_DA_OVER_A, (
            f"{obj}: converged but a = {a_fit:.4f} against a published {entry['a']:.4f} "
            f"(|da/a| = {da:.2e})"
        )

    assert converged, f"no arc converged out of {checked}; the method is not running"
    print(
        f"\n{len(converged)}/{checked} converged; "
        f"median |da/a| = {np.median(converged):.2e}, worst = {max(converged):.2e}"
    )
