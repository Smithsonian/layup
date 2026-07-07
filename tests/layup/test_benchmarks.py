"""Smoke tests for the benchmark harness (benchmarks/run_benchmarks.py).

These keep the benchmark code from bit-rotting as the APIs it drives evolve.
They are *not* performance assertions -- CI machine speed varies, so we only
check the harness runs and reports a positive rate, never a throughput floor.
"""

import importlib.util
import pathlib

import numpy as np

from _bk_guards import requires_ephem

_BENCH = pathlib.Path(__file__).parents[2] / "benchmarks" / "run_benchmarks.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_benchmarks", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_synth_orbits_shape():
    rb = _load()
    orbits = rb.synth_kep_orbits(10)
    assert len(orbits) == 10
    assert set(orbits["FORMAT"]) == {"KEP"}
    for col in ("a", "e", "inc", "node", "argPeri", "ma", "epochMJD_TDB"):
        assert col in orbits.dtype.names


@requires_ephem
def test_bench_convert_runs():
    """The convert benchmark runs on a tiny workload and reports positive throughput."""
    rb = _load()
    result = rb.bench_convert(n_orbits=64, warmup=8)
    assert result["seconds"] > 0
    assert result["n"] == 64
    assert result["unit"] == "rows/s"
    assert 64 / result["seconds"] > 0
    assert not np.isnan(result["seconds"])


@requires_ephem
def test_bench_residuals_runs():
    """The residuals benchmark runs and reports positive throughput. Since
    predict_sequence now uses the same sorted single-pass march as
    residuals_at_state, the two are comparable in cost (ratio ~1)."""
    rb = _load()
    result = rb.bench_residuals(reps=1)
    assert result["seconds"] > 0
    assert result["n"] > 0
    assert result["unit"] == "obs/s"
    assert not np.isnan(result["seconds"])
    # predict_sequence marches like residuals_at_state now, so it is no longer
    # structurally slower and the ratio sits near 1. Guard only against a
    # pathological regression (predict_sequence more than ~2x slower); a tight
    # bound would be flaky given CI machine-speed noise, per this file's intent.
    assert result["speedup"] > 0.5
