"""Pytest configuration for the layup test suite.

Pin the native math threadpools to one thread *before* numpy / OpenBLAS /
REBOUND get imported. Under ``pytest -n auto`` each xdist worker would
otherwise spin up a full OpenBLAS/OpenMP threadpool, so N workers x N
threads oversubscribes a small CI runner (~4 cores) and hangs the Linux
test step. One thread per worker keeps the total thread count ~= core
count. ``setdefault`` lets an explicit environment setting win.
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
