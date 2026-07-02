#!/usr/bin/env python
"""Throughput benchmarks for layup.

Produces the "how fast is it" numbers (rows/s for ``convert``, fits/s for orbit
fitting) on synthetic data, for the paper and for spotting regressions.

Usage::

    python benchmarks/run_benchmarks.py                 # default sizes, pretty table
    python benchmarks/run_benchmarks.py --orbits 20000  # bigger convert workload
    python benchmarks/run_benchmarks.py --json out.json # also write machine-readable results

Notes
-----
* JAX JIT-compiles on the first call, so every timed benchmark runs a small
  **warm-up** pass first and reports steady-state throughput.
* The fitting benchmark needs the ASSIST ephemeris (run ``layup bootstrap``);
  if it's missing (or any benchmark errors) that row is reported as skipped and
  the rest still run.
"""

import argparse
import json
import platform
import time

import numpy as np


def synth_kep_orbits(n, seed=42):
    """Generate ``n`` synthetic heliocentric Keplerian orbits (a structured array)."""
    rng = np.random.default_rng(seed)
    dtype = [
        ("ObjID", "U32"),
        ("FORMAT", "U8"),
        ("a", "f8"),
        ("e", "f8"),
        ("inc", "f8"),
        ("node", "f8"),
        ("argPeri", "f8"),
        ("ma", "f8"),
        ("epochMJD_TDB", "f8"),
    ]
    o = np.zeros(n, dtype=dtype)
    o["ObjID"] = [f"orb{i}" for i in range(n)]
    o["FORMAT"] = "KEP"
    o["a"] = rng.uniform(1.5, 40.0, n)
    o["e"] = rng.uniform(0.0, 0.6, n)
    o["inc"] = rng.uniform(0.0, 40.0, n)
    o["node"] = rng.uniform(0.0, 360.0, n)
    o["argPeri"] = rng.uniform(0.0, 360.0, n)
    o["ma"] = rng.uniform(0.0, 360.0, n)
    o["epochMJD_TDB"] = 60000.0
    return o


def bench_convert(n_orbits=5000, warmup=64):
    """Steady-state throughput of ``convert`` (KEP -> CART), single worker."""
    from layup.convert import convert

    # Warm up JAX's JIT so the timed run measures steady state, not compilation.
    convert(synth_kep_orbits(warmup), "CART", num_workers=1, primary_id_column_name="ObjID")

    data = synth_kep_orbits(n_orbits)
    t0 = time.perf_counter()
    convert(data, "CART", num_workers=1, primary_id_column_name="ObjID")
    dt = time.perf_counter() - t0
    return {"name": "convert (KEP->CART, 1 worker)", "n": n_orbits, "unit": "rows/s", "seconds": dt}


def bench_fit(obs_file="holman_data_working.csv"):
    """Wall-clock of a full ``orbitfit`` on a bundled demo observation set."""
    from importlib.resources import files

    from layup.orbitfit import orbitfit
    from layup.utilities.file_io.CSVReader import CSVDataReader

    # layup is installed unzipped, so the packaged demo file has a real path.
    path = str(files("layup.data.demo.orbitfit").joinpath(obs_file))
    data = np.atleast_1d(CSVDataReader(path, "csv", primary_id_column_name="provID").read_rows())
    n_obj = len(np.unique(data["provID"]))

    t0 = time.perf_counter()
    orbitfit(data, cache_dir=None, primary_id_column_name="provID")
    dt = time.perf_counter() - t0
    return {"name": f"orbitfit ({obs_file}, {n_obj} obj)", "n": n_obj, "unit": "fits/s", "seconds": dt}


BENCHMARKS = [bench_convert, bench_fit]


def _run_all(orbits):
    results = []
    for fn in BENCHMARKS:
        row = {"benchmark": fn.__name__}
        try:
            r = fn(n_orbits=orbits) if fn is bench_convert else fn()
            r["rate"] = r["n"] / r["seconds"] if r["seconds"] > 0 else float("inf")
            row.update({"status": "ok", **r})
        except Exception as exc:  # keep going; report what we could measure
            row.update({"status": "skipped", "error": f"{type(exc).__name__}: {exc}"})
        results.append(row)
    return results


def _print_table(results):
    print(f"\nlayup benchmarks  ({platform.platform()}, Python {platform.python_version()})")
    print("-" * 72)
    for r in results:
        if r["status"] == "ok":
            rate = r["rate"]
            rate_str = f"{rate:,.0f}" if rate >= 100 else f"{rate:.2f}"
            print(f"  {r['name']:<38} {rate_str:>12} {r['unit']:<8} ({r['seconds']:.3f}s, n={r['n']})")
        else:
            print(f"  {r['benchmark']:<38} SKIPPED  {r['error']}")
    print("-" * 72)


def main():
    ap = argparse.ArgumentParser(description="Run layup throughput benchmarks.")
    ap.add_argument("--orbits", type=int, default=5000, help="orbit count for the convert benchmark")
    ap.add_argument("--json", type=str, default=None, help="also write results to this JSON file")
    args = ap.parse_args()

    results = _run_all(args.orbits)
    _print_table(results)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
