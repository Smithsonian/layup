"""The incremental path must honour the id column it is given (#515).

``incremental_orbitfit`` takes ``primary_id_column_name``, but it was not
threaded through ``sequential_update`` to ``_observations_for_update``, which
read ``d["provID"]`` directly. So the sequential-update path worked only when
the id column happened to carry the default name, and raised otherwise -- on
data that ``orbitfit`` itself reads without complaint, since every other entry
point already threads the name.

The property pinned here is that the column NAME is not allowed to change the
ANSWER: the same observations under a different id column must produce the same
fit, not merely avoid raising.
"""

import numpy as np
import pytest

from layup.orbitfit import orbitfit, sequential_update
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader

ALT_ID = "objectName"  # anything that is not the default


@pytest.fixture(scope="module")
def split():
    """One object's observations, split old/new, with a prior fit over the old."""
    data = CSVDataReader(
        get_test_filepath("1_random_mpc_ADES_provIDs_no_sats_micro.csv"),
        "csv",
        primary_id_column_name="provID",
    ).read_rows()
    data = np.sort(data, order="obsTime", kind="mergesort")
    cut = (2 * len(data)) // 3
    seed = orbitfit(data, cache_dir=None)
    assert seed[0]["flag"] == 0, "seed fit did not converge"
    prior = orbitfit(data[:cut], cache_dir=None, initial_guess=seed)
    assert prior[0]["flag"] == 0, "prior fit did not converge"
    return {"prior": prior, "new": data[cut:], "all": data}


def _renamed(arr):
    """Same rows, one field renamed.

    Built field by field rather than with ``recfunctions.rename_fields``, which
    raises "Cannot change data-type for array of references" on the object-dtype
    columns these readers produce.
    """
    old = arr.dtype.names
    new = [ALT_ID if n == "provID" else n for n in old]
    out = np.empty(arr.shape, dtype=np.dtype([(nn, arr.dtype[n]) for nn, n in zip(new, old)]))
    for nn, n in zip(new, old):
        out[nn] = arr[n]
    return out


def test_update_runs_under_a_non_default_id_column(split):
    res = sequential_update(
        split["prior"],
        _renamed(split["new"]),
        cache_dir=None,
        all_data=_renamed(split["all"]),
        primary_id_column_name=ALT_ID,
    )
    assert res.flag == 0


def test_the_id_column_name_does_not_change_the_answer(split):
    """The fit is a function of the observations, not of what the column is called."""
    default = sequential_update(split["prior"], split["new"], cache_dir=None, all_data=split["all"])
    renamed = sequential_update(
        split["prior"],
        _renamed(split["new"]),
        cache_dir=None,
        all_data=_renamed(split["all"]),
        primary_id_column_name=ALT_ID,
    )
    assert renamed.flag == default.flag
    np.testing.assert_allclose(list(renamed.state), list(default.state), rtol=0, atol=0)
    np.testing.assert_allclose(list(renamed.cov), list(default.cov), rtol=0, atol=0)


def test_the_default_is_still_the_default(split):
    """Callers that never pass the argument must be unaffected."""
    res = sequential_update(split["prior"], split["new"], cache_dir=None, all_data=split["all"])
    assert res.flag == 0
