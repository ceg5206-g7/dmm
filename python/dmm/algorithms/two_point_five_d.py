"""2.5D multiplication on c replicated q x q layers with cross-layer reduction."""

import numpy as np

from dmm import matmul as shared
from dmm.bsp import Trace
from dmm.execution import Executor


def validate_size(size: int, *, P: int = 16, c: int = 1) -> None:
    """Validate the 2.5D grid and common zero-padding size."""
    shared.validate_size(size, layout="2.5d", P=P, c=c)


def matmul(A: np.ndarray, B: np.ndarray, executor: Executor, *, P: int = 16, c: int = 1) -> np.ndarray:
    """Multiply FP32 square matrices through the selected C++ executor."""
    return shared.matmul(A, B, executor, layout="2.5d", P=P, c=c)


def matmul_traced(A: np.ndarray, B: np.ndarray, executor: Executor, *, P: int = 16, c: int = 1) -> tuple[np.ndarray, Trace]:
    """Return the common row-to-row route and its executed trace."""
    return shared.matmul_traced(A, B, executor, layout="2.5d", P=P, c=c)


def prepare(A: np.ndarray, B: np.ndarray, *, P: int = 16, c: int = 1):
    """Prepare reusable storage/schedule; run_into measures common rows to rows."""
    return shared.prepare(A, B, layout="2.5d", P=P, c=c)
