"""Cannon multiplication with a configurable logical square grid."""

import numpy as np

from dmm import _native
from dmm.bsp import Trace
from dmm.execution import Executor

GRID = 4  # Default-grid compatibility constant; execution uses P.


def validate_size(size: int, *, P: int = 16, c: int = 1) -> None:
    """Validate the grid and common zero-padding size."""
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    _native.grid_size(P, c)
    if c != 1:
        raise ValueError("Cannon requires c = 1")
    _native.padded_size(size, P)


def matmul(A: np.ndarray, B: np.ndarray, executor: Executor, *, P: int = 16, c: int = 1) -> np.ndarray:
    """Multiply FP32 square matrices through the selected C++ executor."""
    return _native.cannon(A, B, executor, P=P, c=c)


def matmul_traced(A: np.ndarray, B: np.ndarray, executor: Executor, *, P: int = 16, c: int = 1, native_layout: bool = False) -> tuple[np.ndarray, Trace]:
    """Return the common route trace, or an explicit unpadded native-layout test run."""
    return _native.cannon_traced(A, B, executor, P=P, c=c, native_layout=native_layout)


def prepare(A: np.ndarray, B: np.ndarray, *, P: int = 16, c: int = 1):
    """Prepare reusable storage/schedule; run_into measures common rows to rows."""
    return _native.PreparedCannon(A, B, P=P, c=c)
