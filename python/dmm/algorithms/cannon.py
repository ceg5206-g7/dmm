"""Cannon multiplication with a logical 4-by-4 block grid."""

import numpy as np

from dmm import _native
from dmm.bsp import Trace
from dmm.execution import Executor

GRID = 4


def validate_size(size: int) -> None:
    """Reject sizes that cannot form Cannon's fixed four-by-four block grid."""
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    if size % GRID:
        raise ValueError(f"Cannon matrix size must be divisible by {GRID}")


def matmul(A: np.ndarray, B: np.ndarray, executor: Executor) -> np.ndarray:
    """Multiply FP32 square matrices through the selected C++ executor."""
    return _native.cannon(A, B, executor)


def matmul_traced(A: np.ndarray, B: np.ndarray, executor: Executor) -> tuple[np.ndarray, Trace]:
    """Multiply like matmul and also return the per-superstep message trace."""
    return _native.cannon_traced(A, B, executor)
