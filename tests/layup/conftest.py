"""Pytest configuration for the layup test suite.

Pin the native math threadpools to a single thread *before* numpy /
OpenBLAS / REBOUND get imported.  Under ``pytest -n auto`` each xdist
worker otherwise lets OpenBLAS (via numpy/scipy) and any OpenMP code
spin up a full threadpool, so N workers x N threads badly oversubscribes
a small CI runner (the GitHub Linux runners have ~4 cores).  On Linux
that manifested as the test step hanging for well over an hour even
though every individual orbit fit completes in under a second -- see the
per-case timing in the BK test-skip investigation.  One thread per
worker keeps the total thread count ~= core count and removes the hang.

``setdefault`` so an explicit environment setting (e.g. a developer
deliberately running multi-threaded) still wins.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
