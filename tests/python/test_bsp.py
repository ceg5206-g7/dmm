import numpy as np
import pytest

from dmm.algorithms import cannon
from dmm.bsp import bsp_cost, count
from dmm.execution import SerialExecutor, ThreadExecutor


@pytest.mark.parametrize("make_executor", [SerialExecutor, lambda: ThreadExecutor(2)])
def test_traced_cannon_matches_product_and_hand_counts(make_executor):
    rng = np.random.default_rng(5206)
    A = rng.uniform(-1, 1, (8, 8)).astype(np.float32)
    B = rng.uniform(-1, 1, (8, 8)).astype(np.float32)
    with make_executor() as executor:
        C, trace = cannon.matmul_traced(A, B, executor)
        np.testing.assert_array_equal(C, cannon.matmul(A, B, executor))
    np.testing.assert_allclose(C, A.astype(np.float64) @ B, rtol=1e-5, atol=1e-5)

    words, flops = 4, 16  # Block size 2.
    assert trace.nodes == 16
    assert [step.label for step in trace.supersteps] == ["skew"] + ["multiply"] * 4
    assert [len(step.messages) for step in trace.supersteps] == [24, 32, 32, 32, 0]
    assert all(message.words == words for step in trace.supersteps for message in step.messages)
    counts = count(trace)
    assert (counts.supersteps, counts.total_flops, counts.critical_flops) == (5, 2 * 8**3, 4 * flops)
    assert (counts.total_words, counts.critical_words, counts.messages) == (120 * words, 8 * words, 120)
    per_step = [count(step) for step in trace.supersteps]
    assert sum(step.critical_words for step in per_step) == counts.critical_words
    assert sum(step.critical_flops for step in per_step) == counts.critical_flops


def test_bsp_cost_recomputes_without_rerunning():
    A = np.ones((4, 4), dtype=np.float32)
    with SerialExecutor() as executor:
        _, trace = cannon.matmul_traced(A, A, executor)
    counts = count(trace)
    assert bsp_cost(counts) == 8
    assert bsp_cost(counts, g=10, l=100) == 8 + 10 * 8 + 100 * 5
    assert bsp_cost(counts, w=0.5) == 4
    with pytest.raises(ValueError, match="non-negative"):
        bsp_cost(counts, g=-1)
    with pytest.raises(TypeError):
        bsp_cost(counts, 1.0, 10.0)
