"""Direct 2.5D: row shards replicate straight to all layers; partial C reduce-scatters to rows."""

import numpy as np

from dmm import matmul as shared
from dmm.bsp import Trace
from dmm.execution import Executor


def validate_size(size: int, *, P: int = 16, c: int = 1) -> None:
    """Validate the direct 2.5D grid and common zero-padding size."""
    shared.validate_size(size, layout="2.5d-direct", P=P, c=c)


def matmul(A: np.ndarray, B: np.ndarray, executor: Executor, *, P: int = 16, c: int = 1) -> np.ndarray:
    """Multiply FP32 square matrices through the selected C++ executor."""
    return shared.matmul(A, B, executor, layout="2.5d-direct", P=P, c=c)


def matmul_traced(A: np.ndarray, B: np.ndarray, executor: Executor, *, P: int = 16, c: int = 1) -> tuple[np.ndarray, Trace]:
    """Return the direct-replication route (row shards straight to all layers) and its trace."""
    return shared.matmul_traced(A, B, executor, layout="2.5d-direct", P=P, c=c)


def prepare(A: np.ndarray, B: np.ndarray, *, P: int = 16, c: int = 1):
    """Prepare reusable storage/schedule; run_into measures common rows to rows."""
    return shared.prepare(A, B, layout="2.5d-direct", P=P, c=c)
