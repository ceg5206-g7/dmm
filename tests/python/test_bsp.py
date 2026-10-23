import numpy as np
import pytest

from dmm.algorithms import cannon
from dmm.bsp import bsp_cost, count, count_native
from dmm.execution import SerialExecutor, ThreadExecutor


@pytest.mark.parametrize("make_executor", [SerialExecutor, lambda: ThreadExecutor(2)])
def test_traced_cannon_matches_product_and_hand_counts(make_executor):
    rng = np.random.default_rng(5206)
    A = rng.uniform(-1, 1, (16, 16)).astype(np.float32)
    B = rng.uniform(-1, 1, (16, 16)).astype(np.float32)
    with make_executor() as executor:
        C, trace = cannon.matmul_traced(A, B, executor)
        np.testing.assert_allclose(C, cannon.matmul(A, B, executor), atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(C, A.astype(np.float64) @ B, rtol=1e-5, atol=1e-5)

    words, flops = 16, 128  # Padded block size 4.
    steps = [s for s in trace.supersteps if s.label not in ("input_conversion", "output_conversion")]
    assert trace.nodes == 16
    assert [step.label for step in steps] == ["skew"] + ["multiply"] * 4
    assert [len(step.messages) for step in steps] == [24, 32, 32, 32, 0]
    assert all(message.words == words for step in steps for message in step.messages)
    counts = count_native(trace)
    assert (counts.supersteps, counts.total_flops, counts.critical_flops) == (5, 2 * 16**3, 4 * flops)
    assert (counts.total_words, counts.critical_words, counts.messages) == (120 * words, 8 * words, 120)
    per_step = [count(step) for step in steps]
    assert sum(step.critical_words for step in per_step) == counts.critical_words
    assert sum(step.critical_flops for step in per_step) == counts.critical_flops


def test_bsp_cost_recomputes_without_rerunning():
    A = np.ones((4, 4), dtype=np.float32)
    with SerialExecutor() as executor:
        _, trace = cannon.matmul_traced(A, A, executor)
    counts = count_native(trace)
    assert bsp_cost(counts) == 512
    assert bsp_cost(counts, g=10, l=100) == 512 + 10 * 128 + 100 * 5
    assert bsp_cost(counts, w=0.5) == 256
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
    assert counts.supersteps == q + (7 if P > 1 else 1)
    padded = ((32 + P - 1) // P) * P
    assert counts.total_flops == 2 * padded**3
    assert counts.messages == 2 * (q - 1) * (q + P) + 3 * P * (q - 1)


@pytest.mark.parametrize("P,c", [(0, 1), (-1, 1), (True, 1), (4.0, 1), ("4", 1),
                                  (3, 1), (16, 0), (16, -1), (16, True), (16, 1.0),
                                  (16, 4), (64, 4)])
def test_invalid_cannon_configuration_rejected_by_python_and_native(P, c):
    from dmm import _native
    A = np.ones((16, 16), dtype=np.float32)
    with SerialExecutor() as executor:
        for call in (lambda: cannon.validate_size(16, P=P, c=c),
                     lambda: cannon.matmul(A, A, executor, P=P, c=c),
                     lambda: _native.matmul_traced(A, A, executor, P=P, c=c)):
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
    # The common plan retains conversion scratch/cache capacity.
    assert max(trace.peak_words) == 8 * 4**2
    assert count(trace).peak_words == max(trace.peak_words)
    assert max(count(step).peak_words for step in trace.supersteps) == count(trace).peak_words


@pytest.mark.parametrize("n", [17, 65, 257])
def test_common_padding_crops_and_reports_executed_dimensions(n):
    rng = np.random.default_rng(5206)
    A = rng.uniform(-1, 1, (n, n)).astype(np.float32)
    B = rng.uniform(-1, 1, (n, n)).astype(np.float32)
    cannon.validate_size(n)
    with ThreadExecutor(2) as executor:
        C, trace = cannon.matmul_traced(A, B, executor)
    np.testing.assert_allclose(C, A.astype(np.float64) @ B.astype(np.float64), atol=1e-4, rtol=1e-3)
    assert trace.original_size == n
    assert trace.padded_size == ((n + 15) // 16) * 16
    assert count(trace).total_flops == 2 * trace.padded_size**3
    for step in trace.supersteps:
        for message in step.messages:
            assert message.regions
            assert sum(r.rows * r.cols for r in message.regions) == message.words
            for region in message.regions:
                assert region.row + region.rows <= trace.padded_size
                assert region.col + region.cols <= trace.padded_size


def test_common_and_native_costs_have_explicit_conversion_boundary_and_hand_counts():
    from dmm.bsp import count_native
    A = np.ones((16, 16), dtype=np.float32)
    with SerialExecutor() as executor:
        C, trace = cannon.matmul_traced(A, A, executor)
    np.testing.assert_array_equal(C, np.full_like(A, 16))
    end = count(trace)
    body = count_native(trace)
    assert (end.supersteps, end.total_flops, end.critical_flops) == (11, 8192, 512)
    assert (end.total_words, end.critical_words, end.messages) == (2496, 164, 264)
    assert (body.supersteps, body.total_words, body.critical_words) == (5, 1920, 128)
    assert (body.total_flops, body.messages) == (8192, 120)
    # The common plan retains conversion scratch/caches during its native body.
    assert body.peak_words == 8 * 4**2


@pytest.mark.parametrize("P", [1, 4, 16, 64, 256])
def test_executed_kernel_records_cover_every_block_triple_once(P):
    from dmm.bsp import validate_coverage
    A = np.ones((17, 17), dtype=np.float32)
    with ThreadExecutor(2) as executor:
        C, trace = cannon.matmul_traced(A, A, executor, P=P)
    np.testing.assert_array_equal(C, np.full((17, 17), 17, dtype=np.float32))
    validate_coverage(trace)
    q = int(P**0.5)
    records = [r for step in trace.supersteps for r in step.products]
    assert len(records) == q**3
    assert {(r.product.i, r.product.k, r.product.j) for r in records} == {
        (i, k, j) for i in range(q) for k in range(q) for j in range(q)}
    assert (trace.coverage.row_parts, trace.coverage.inner_parts, trace.coverage.col_parts) == (q, q, q)
    assert all(r.product.m == r.product.n == r.product.inner == trace.padded_size // q for r in records)
