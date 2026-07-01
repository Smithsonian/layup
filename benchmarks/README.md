# layup benchmarks

Throughput benchmarks for the numbers we cite (and to spot regressions).

```bash
python benchmarks/run_benchmarks.py                 # default sizes, pretty table
python benchmarks/run_benchmarks.py --orbits 20000  # larger convert workload
python benchmarks/run_benchmarks.py --json out.json # also emit machine-readable results
```

Current benchmarks:

- **convert** — `convert` throughput in rows/s (KEP → CART, single worker) on
  synthetic orbits.
- **orbitfit** — wall-clock of a full fit on the bundled demo observation set
  (needs the ASSIST ephemeris; run `layup bootstrap` first).

Notes:

- JAX JIT-compiles on the first call, so each timed benchmark runs a warm-up
  pass and reports steady-state throughput.
- Any benchmark that errors (e.g. the ephemeris is missing) is reported as
  *skipped*; the others still run.
- These are dev/paper tools, not part of the shipped package. `tests/layup/
  test_benchmarks.py` only smoke-tests that the harness runs — it makes no
  performance assertions (CI machine speed varies).

Planned additions: a short-tracklet fits/s throughput benchmark (matching the
"fits/sec" figure in the paper) and a `predict` rows/s benchmark.
