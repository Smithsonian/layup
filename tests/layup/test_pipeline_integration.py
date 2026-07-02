"""Full-workflow integration test: observations -> fit -> convert -> predict.

The per-feature suites test each stage in isolation; this drives the whole
pipeline end-to-end through real compute so a hand-off break between verbs (the
class of bug behind #384: `visualize` couldn't read the fitter's own output)
can't slip through. It runs on the bundled demo observations with the shared
default primary-id column (``provID``) and no per-stage flags.

Ephemeris-gated (needs `layup bootstrap`); everything runs single-worker
(in-process), so no spawned workers.
"""

from importlib.resources import files

import numpy as np

from layup.convert import convert
from layup.orbitfit import orbitfit
from layup.predict import predict
from layup.utilities.file_io.CSVReader import CSVDataReader

from _bk_guards import requires_ephem


class _PredictArgs:
    onsky_data = False


def _load_demo_observations():
    path = str(files("layup.data.demo.orbitfit").joinpath("holman_data_working.csv"))
    return np.atleast_1d(CSVDataReader(path, "csv", primary_id_column_name="provID").read_rows())


@requires_ephem
def test_observations_fit_convert_predict_pipeline():
    obs = _load_demo_observations()
    obj_ids = set(np.unique(obs["provID"]))

    # 1. Fit -------------------------------------------------------------------
    fit = np.atleast_1d(orbitfit(obs, cache_dir=None, primary_id_column_name="provID", num_workers=1))
    assert "provID" in fit.dtype.names
    assert set(np.unique(fit["provID"])) == obj_ids, "fit lost/renamed the object id"
    assert set(fit["FORMAT"]) == {"BCART_EQ"}, "orbitfit should emit BCART_EQ by default"
    state = np.array([fit["x"], fit["y"], fit["z"], fit["xdot"], fit["ydot"], fit["zdot"]])
    assert np.isfinite(state).all(), "fit produced a non-finite state (did not converge)"
    assert np.isfinite(fit["csq"]).all()

    # 2. Convert (fit output -> Keplerian) with the shared default id ----------
    kep = np.atleast_1d(convert(fit, "KEP", num_workers=1, primary_id_column_name="provID"))
    assert set(kep["FORMAT"]) == {"KEP"}
    assert set(np.unique(kep["provID"])) == obj_ids
    for col in ("a", "e", "inc"):
        assert np.isfinite(kep[col]).all()
    assert (kep["e"] >= 0).all()

    # 3. Predict on-sky positions from the same fit output ---------------------
    epoch_jd = float(fit["epochMJD_TDB"][0]) + 2400000.5
    times = epoch_jd + np.array([0.0, 1.0, 2.0])
    preds = np.atleast_1d(
        predict(
            fit,
            obscode="X05",
            times=times,
            num_workers=1,
            cache_dir=None,
            primary_id_column_name="provID",
            args=_PredictArgs(),
        )
    )
    assert len(preds) == len(obj_ids) * len(times), "one prediction per object per time expected"
    assert np.isfinite(preds["ra_deg"]).all() and np.isfinite(preds["dec_deg"]).all()
    assert ((preds["ra_deg"] >= 0) & (preds["ra_deg"] <= 360)).all()
    assert ((preds["dec_deg"] >= -90) & (preds["dec_deg"] <= 90)).all()
