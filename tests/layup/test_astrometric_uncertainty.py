"""The Veres (2017) sigma model must accept both catalog spellings (#548).

obs80 supplies the single-character MPC star-catalog CODE in column 72; ADES
supplies the NAME. The branches in `astrometric_uncertainty_Veres2017` are keyed
on names, so before #548 every name-keyed test was unreachable for obs80 input
and the observation fell through to the station's `else` value -- silently, and
by up to a factor of 15 in sigma, 225 in the weight.

`debias()` already normalised both spellings (#409). These pin the same
behaviour for the sigma path, and pin that the two spellings agree, which is the
property that actually matters: the same observation must be weighted the same
whether it arrived as obs80 or as ADES.
"""

import pytest

from layup.utilities.astrometric_uncertainty import (
    _CODE_TO_CATALOG_NAME,
    astrometric_uncertainty_Veres2017,
)
from layup.utilities.debiasing import MPC_CATALOGS

JD = 2458000.0

# (obsCode, program) for every station whose sigma depends on the catalog.
CATALOG_SENSITIVE_STATIONS = [
    ("G83", "2"),
    ("Y28", None),
    ("568", None),
    ("T09", None),
    ("T12", None),
    ("T14", None),
    ("309", "&"),
]


def sigma(stn, catalog, program=None):
    return astrometric_uncertainty_Veres2017(obsCode=stn, jd_tdb=JD, catalog=catalog, program=program)


@pytest.mark.parametrize("code,name", sorted(_CODE_TO_CATALOG_NAME.items()))
@pytest.mark.parametrize("stn,program", CATALOG_SENSITIVE_STATIONS)
def test_code_and_name_agree(stn, program, code, name):
    """The same observation must weigh the same from obs80 and from ADES."""
    assert sigma(stn, code, program) == sigma(stn, name, program)


def test_the_case_from_the_issue():
    """A Gaia-referenced observation from 568 was weighted 15x too loosely."""
    assert sigma("568", "Gaia1") == pytest.approx(0.1)
    assert sigma("568", "U") == pytest.approx(0.1)  # was 1.5


@pytest.mark.parametrize("code", ["V", "X"])
def test_the_modern_gaia_codes_decode(code):
    """Gaia DR2 and EDR3 are the two commonest catalogs in the archive and have
    no entry in the debiasing map, so a fix that borrowed it would miss them."""
    assert code not in MPC_CATALOGS.values(), "if this fails, borrow the debiasing map instead"
    assert sigma("568", code) == pytest.approx(0.1)


def test_usnob2_is_reachable():
    """Resolves the 'unsure of the abbreviation' note: MPC code s is USNO-B2.0."""
    assert sigma("568", "s") == pytest.approx(0.5)
    assert sigma("568", "USNOB2") == pytest.approx(0.5)


@pytest.mark.parametrize("code", ["L", "N", "R", "S"])
def test_an_undistinguished_code_is_unchanged(code):
    """Codes the model does not branch on must reach the same generic value they
    always did -- the decode must not perturb anything it was not aimed at."""
    assert sigma("568", code) == pytest.approx(1.5)
    assert sigma("703", code) == pytest.approx(0.8)


def test_no_catalog_still_takes_the_unknown_branch():
    assert sigma("568", None) == pytest.approx(1.5)
    assert sigma("999", None) == pytest.approx(1.5)


def test_every_name_the_model_branches_on_is_decodable():
    """Guards the map against the branches drifting away from it."""
    import re

    src = open("src/layup/utilities/astrometric_uncertainty.py").read()
    body = src.split("def astrometric_uncertainty_Veres2017", 1)[1]
    names = {m for m in re.findall(r'"([A-Za-z][A-Za-z0-9]{2,})"', body) if not m[0].isdigit()}
    branched = {n for n in names if n.startswith(("Gaia", "UCAC", "PPM", "USNO"))}
    missing = branched - set(_CODE_TO_CATALOG_NAME.values())
    assert not missing, f"catalog names with no code to decode from: {sorted(missing)}"
