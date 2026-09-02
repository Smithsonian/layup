"""Debiasing an MPC 80-column file has to actually debias it.

The 80-column format carries the star catalogue in column 72. Everything
downstream reads it under its ADES name, ``astCat`` -- the Farnocchia/Chesley
debiasing and the Veres (2017) weighting both key off it. The reader emitted it
as ``cat``, so ``astCat`` was absent, ``astcat_column_present`` was False, and
``catalog=None`` went to the bias lookup: the correction quietly did nothing and
the caller got back astrometry it believed had been corrected (#521).

Nothing raised, and nothing was logged, which is what made it worth a test
rather than a one-line rename. The tests below check the rename, that the value
read is the catalogue code and not some adjacent column, and that debiasing now
changes the astrometry it is given.
"""

import os

import numpy as np
import pytest

from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.Obs80Reader import Obs80DataReader

# 03666.txt carries a wide spread of catalogue codes; newy6_tiny.txt has none,
# which is the "historical astrometry" case that must stay working.
WITH_CATALOGS = "03666.txt"
WITHOUT_CATALOGS = "newy6_tiny.txt"


def _rows(filename, n=40):
    return Obs80DataReader(get_test_filepath(filename)).read_rows(block_start=0, block_size=n)


def test_the_reader_uses_the_ades_column_names():
    """Every other reader emits ``astCat``. This one emitting ``cat`` is what
    made the whole pipeline silently skip the correction."""
    names = _rows(WITH_CATALOGS).dtype.names
    assert "astCat" in names, "the star catalogue is not under its ADES name"
    assert "program" in names
    assert "cat" not in names, "the old name is still emitted; consumers will disagree"
    assert "prg" not in names


def test_the_catalogue_code_comes_from_column_72():
    """A rename that quietly picked up a neighbouring column would still pass
    the test above, so the values are checked against the column they must come
    from. The reader skips lines it cannot use, so the raw file and the returned
    rows do not line up positionally -- the alphabets are compared instead, and
    column 14 (the observing programme) is checked as the near miss it would be.
    """
    with open(get_test_filepath(WITH_CATALOGS)) as f:
        lines = [line.rstrip("\n") for line in f]
    col72 = {ln[71:72].strip() for ln in lines if len(ln) > 71} - {""}
    col14 = {ln[13:14].strip() for ln in lines if len(ln) > 13} - {""}

    got = {c.strip() for c in _rows(WITH_CATALOGS, 400)["astCat"]} - {""}
    assert got, "no catalogue codes were read at all"
    assert got <= col72, f"astCat holds codes absent from column 72: {sorted(got - col72)}"
    assert not got <= col14, "astCat is indistinguishable from column 14, the programme code"


def test_the_catalogue_codes_are_not_all_blank():
    """If they were, the end-to-end test below would pass without debiasing
    anything."""
    codes = {c for c in _rows(WITH_CATALOGS, 200)["astCat"] if c.strip()}
    assert len(codes) >= 2, f"expected a spread of catalogue codes, got {codes}"


def test_a_file_without_catalogue_codes_still_reads():
    """Historical astrometry has no catalogue. It must read cleanly and simply
    go undebiased, which is different from silently failing to debias data that
    does carry one."""
    rows = _rows(WITHOUT_CATALOGS)
    assert "astCat" in rows.dtype.names
    assert all(not c.strip() for c in rows["astCat"])


@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            os.environ.get("LAYUP_CACHE_DIR", os.path.expanduser("~/Library/Caches/layup")),
            "bias.dat",
        )
    ),
    reason="the debiasing table (bias.dat) is not in the cache",
)
def test_debiasing_changes_obs80_astrometry():
    """The bug itself: with the catalogue column present the correction is
    applied, and the astrometry that comes back differs from what went in."""
    from layup.utilities.debiasing import debias, generate_bias_dict
    from layup.utilities.datetime_conversions import convert_tdb_date_to_julian_date

    rows = _rows(WITH_CATALOGS, 200)
    catalogued = rows[[bool(c.strip()) for c in rows["astCat"]]]
    assert len(catalogued) > 0, "no catalogued observations to debias"

    bias_dict = generate_bias_dict()
    moved = 0
    for d in catalogued[:25]:
        ra, dec = debias(
            ra=float(d["ra"]),
            dec=float(d["dec"]),
            epoch_jd_tdb=convert_tdb_date_to_julian_date(d["obsTime"]),
            catalog=d["astCat"],
            bias_dict=bias_dict,
        )
        if ra != float(d["ra"]) or dec != float(d["dec"]):
            moved += 1
    assert moved > 0, "debiasing left every observation unchanged; it is still a no-op"


def test_an_uncatalogued_observation_comes_back_unchanged():
    """A missing catalogue is normal -- historical astrometry has none -- so the
    lookup must return the astrometry untouched rather than raise. This is the
    behaviour that made the bug invisible: correct on its own, wrong only
    because the column was absent for data that did have a catalogue.

    The warning `_orbitfit` now logs in this situation is not covered here; it
    sits inside the fitting entry point, which needs ephemeris-prepared input,
    and is exercised through the CLI rather than by this module.
    """
    from layup.utilities.debiasing import debias

    out = debias(ra=10.0, dec=20.0, epoch_jd_tdb=2460000.5, catalog=None, bias_dict={})
    assert np.allclose(out, (10.0, 20.0)), "an uncatalogued observation should be unchanged"
