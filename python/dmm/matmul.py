"""Shared entry points for ordinary and retained matrix multiplication plans."""

import numpy as np

from dmm import _native
from dmm.bsp import Trace
from dmm.config import validate_case
from dmm.execution import Executor


def validate_size(size: int, *, layout: str, P: int, c: int) -> None:
    """Check layout and padding without allocating matrices or starting workers."""
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    validate_case(dict(layout=layout, P=P, c=c))
    _native.padded_size(size, P)


def prepare(A: np.ndarray, B: np.ndarray, *, layout: str, P: int, c: int):
    """Own input copies and build one reusable plan with common row-shard I/O."""
    return _native.prepare(A, B, layout=layout, P=P, c=c)


def matmul(A: np.ndarray, B: np.ndarray, executor: Executor, *, layout: str, P: int, c: int) -> np.ndarray:
    """Build and execute a temporary plan, returning an independent FP32 output."""
    return _native.matmul(A, B, executor, layout=layout, P=P, c=c)


def matmul_traced(A: np.ndarray, B: np.ndarray, executor: Executor, *, layout: str, P: int, c: int) -> tuple[np.ndarray, Trace]:
    """Return the product and the trace of the same common row-to-row plan."""
    return _native.matmul_traced(A, B, executor, layout=layout, P=P, c=c)
