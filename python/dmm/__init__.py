"""Distributed matrix multiplication benchmarks."""

import os

# Keep NumPy reference and analysis work independent of the C++ worker count.
for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
