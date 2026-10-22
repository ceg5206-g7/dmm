"""Cannon multiplication with a configurable logical square grid."""

import numpy as np

from dmm import _native
from dmm.bsp import Trace
from dmm.execution import Executor

GRID = 4  # Default-grid compatibility constant; execution uses P.


def validate_size(size: int, *, P: int = 16, c: int = 1) -> None:
    """Validate the grid and reject sizes requiring padding."""
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    grid = _native.grid_size(P, c)
    if c != 1:
        raise ValueError("Cannon requires c = 1")
    if size % grid:
        raise ValueError(f"Cannon matrix size must be divisible by {grid}")


def matmul(A: np.ndarray, B: np.ndarray, executor: Executor, *, P: int = 16, c: int = 1) -> np.ndarray:
    """Multiply FP32 square matrices through the selected C++ executor."""
    return _native.cannon(A, B, executor, P=P, c=c)


def matmul_traced(A: np.ndarray, B: np.ndarray, executor: Executor, *, P: int = 16, c: int = 1) -> tuple[np.ndarray, Trace]:
    """Multiply like matmul and also return the per-superstep message trace."""
    return _native.cannon_traced(A, B, executor, P=P, c=c)
