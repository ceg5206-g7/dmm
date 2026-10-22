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


@pytest.mark.parametrize("P", [1, 4, 16, 64, 256])
@pytest.mark.parametrize("workers", [0, 2])
def test_configured_cannon_grid_controls_product_and_trace(P, workers):
    rng = np.random.default_rng(5206)
    A = rng.uniform(-1, 1, (32, 32)).astype(np.float32)
    B = rng.uniform(-1, 1, (32, 32)).astype(np.float32)
    cannon.validate_size(32, P=P, c=1)
    executor = ThreadExecutor(workers) if workers else SerialExecutor()
    with executor:
        C, trace = cannon.matmul_traced(A, B, executor, P=P, c=1)
        np.testing.assert_array_equal(C, cannon.matmul(A, B, executor, P=P, c=1))
    np.testing.assert_allclose(C, A.astype(np.float64) @ B, atol=1e-4, rtol=1e-3)
    q = int(P**0.5)
    counts = count(trace)
    assert trace.nodes == P
    assert counts.supersteps == q + 1
    assert counts.total_flops == 2 * 32**3
    assert counts.messages == 2 * (q - 1) * (q + P)


@pytest.mark.parametrize("P,c", [(0, 1), (-1, 1), (True, 1), (4.0, 1), ("4", 1),
                                  (3, 1), (16, 0), (16, -1), (16, True), (16, 1.0),
                                  (16, 4), (64, 4)])
def test_invalid_cannon_configuration_rejected_by_python_and_native(P, c):
    from dmm import _native
    A = np.ones((16, 16), dtype=np.float32)
    with SerialExecutor() as executor:
        for call in (lambda: cannon.validate_size(16, P=P, c=c),
                     lambda: cannon.matmul(A, A, executor, P=P, c=c),
                     lambda: _native.cannon_traced(A, A, executor, P=P, c=c)):
            with pytest.raises(ValueError):
                call()


def test_shared_grid_validation_accepts_proposal_replication_grids():
    from dmm import _native
    assert _native.grid_size(64, 4) == 4
    assert _native.grid_size(256, 4) == 8


@pytest.mark.parametrize("workers", [0, 2])
def test_trace_exposes_per_node_payload_peaks_and_count_takes_maximum(workers):
    A = np.ones((8, 8), dtype=np.float32)
    executor = ThreadExecutor(workers) if workers else SerialExecutor()
    with executor:
        _, trace = cannon.matmul_traced(A, A, executor)
    assert len(trace.peak_words) == 16
    assert all(len(step.peak_words) == 16 for step in trace.supersteps)
    # Three local tiles plus at most two incoming and two outgoing tile copies.
    assert max(trace.peak_words) == 7 * 2**2
    assert count(trace).peak_words == max(trace.peak_words)
    assert max(count(step).peak_words for step in trace.supersteps) == count(trace).peak_words
