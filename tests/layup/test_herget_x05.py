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

This file carries the fixture and the checks that hold regardless of which
initial-orbit method is used. The test that actually fits these arcs and
compares against the published orbits belongs with the method itself, so it
travels with the Herget work in #502 rather than sitting here skipped.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from layup.utilities.data_utilities_for_tests import get_test_filepath

ARCS = Path(get_test_filepath("herget_x05.obs80"))
MANIFEST = Path(get_test_filepath("herget_x05_truth.json"))


def arcs_by_object():
    """The arcs live in one obs80 file, split here on the packed designation
    each line carries in columns 6-12. Keyed on the designation in the data
    rather than on any filename, so a mislabelled line cannot hide."""
    out = {}
    for line in ARCS.read_text().splitlines():
        if line.strip():
            out.setdefault(line[5:12], []).append(line)
    return out


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
    arcs = arcs_by_object()
    assert arcs, f"no arcs found in {ARCS}"
    assert set(arcs) == set(manifest), (
        f"arcs and manifest disagree: "
        f"{sorted(set(arcs) - set(manifest))} unlisted, "
        f"{sorted(set(manifest) - set(arcs))} missing"
    )
    for obj, entry in manifest.items():
        assert entry["a"] is not None, f"{obj} has no reference semi-major axis"
        assert entry["n_obs"] == len(arcs[obj]), (
            f"{obj}: {len(arcs[obj])} lines in {ARCS.name}, " f"manifest says {entry['n_obs']}"
        )


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
