from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from dmm import _native
from dmm.algorithms.cannon import prepare
from dmm.execution import SerialExecutor, ThreadExecutor


@pytest.mark.parametrize('P', [1, 4, 16])
def test_prepared_owns_inputs_reuses_output_and_records_actual_work(P):
    rng = np.random.default_rng(32)
    a = rng.random((17, 17), dtype=np.float32) / 32
    b = rng.random((17, 17), dtype=np.float32) / 32
    expected = a.astype(np.float64) @ b.astype(np.float64)
    plan = prepare(a, b, P=P)
    a.fill(0)
    b.fill(0)
    output = np.empty((17, 17), dtype=np.float32)
    with ThreadExecutor(2) as executor:
        for _ in range(3):
            output.fill(np.nan)
            timing = plan.run_into(output, executor)
            np.testing.assert_allclose(output, expected, atol=1e-4, rtol=1e-3)
            assert timing.end_to_end.total > 0
            assert timing.end_to_end.total == pytest.approx(sum(getattr(timing.end_to_end, key)
                for key in ('compute', 'copy', 'synchronization')))
            _native.validate_coverage(plan.trace())
            counts = _native.count(plan.trace())
            padded = ((17 + P - 1) // P) * P
            assert counts.total_flops == 2 * padded ** 3
    with SerialExecutor() as executor:
        for bad in [output.astype(np.float64), output[:, ::2], output.ravel(), np.empty((18, 18), dtype=np.float32)]:
            with pytest.raises((ValueError, TypeError)):
                plan.run_into(bad, executor)
        output.flags.writeable = False
        with pytest.raises(ValueError):
            plan.run_into(output, executor)


def test_shared_prepared_workspace_serializes_runs_with_shared_executor():
    a = np.arange(25, dtype=np.float32).reshape(5, 5) / 32
    plan = prepare(a, a, P=4)
    with ThreadExecutor(2) as executor:
        def run(_):
            out = np.empty_like(a)
            plan.run_into(out, executor)
            return out
        with ThreadPoolExecutor(3) as callers:
            outputs = list(callers.map(run, range(6)))
    for output in outputs:
        np.testing.assert_allclose(output, a @ a, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize('field,value', [('P', True), ('P', 3), ('c', 2)])
def test_preparation_validates_inputs_and_grid(field, value):
    a = np.eye(4, dtype=np.float32)
    with pytest.raises((ValueError, TypeError)):
        prepare(a, a, **{field: value})


def test_preparation_rejects_nonfinite_inputs():
    a = np.eye(4, dtype=np.float32)
    a[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        prepare(a, a)


def _trace_signature(trace):
    coverage = trace.coverage
    return (
        trace.nodes, trace.original_size, trace.padded_size,
        (coverage.row_parts, coverage.inner_parts, coverage.col_parts), tuple(trace.peak_words),
        [(step.label, tuple(step.flops), tuple(step.peak_words),
          [(m.source, m.destination, m.tag, m.words,
            None if m.region is None else (m.region.row, m.region.col, m.region.rows, m.region.cols))
           for m in step.messages],
          [(p.node, p.product.i, p.product.k, p.product.j, p.product.m, p.product.n, p.product.inner)
           for p in step.products]) for step in trace.supersteps],
    )


@pytest.mark.parametrize('P', [1, 4, 16])
@pytest.mark.parametrize('workers', [0, 2])
def test_ordinary_and_reused_plan_have_identical_executed_trace(P, workers):
    from dmm.algorithms import cannon
    rng = np.random.default_rng(5206)
    a = rng.uniform(-1, 1, (17, 17)).astype(np.float32)
    b = rng.uniform(-1, 1, (17, 17)).astype(np.float32)
    reference = a.astype(np.float64) @ b.astype(np.float64)
    prepared = cannon.prepare(a, b, P=P, c=1)
    executor = ThreadExecutor(workers) if workers else SerialExecutor()
    with executor:
        expected, trace = cannon.matmul_traced(a, b, executor, P=P, c=1)
        np.testing.assert_allclose(expected, reference, atol=1e-4, rtol=1e-3)
        signature = _trace_signature(trace)
        output = np.empty_like(a)
        for _ in range(2):
            prepared.run_into(output, executor)
            np.testing.assert_array_equal(output, expected)
            assert _trace_signature(prepared.trace()) == signature
        repeated, repeated_trace = cannon.matmul_traced(a, b, executor, P=P, c=1)
        np.testing.assert_array_equal(repeated, expected)
        assert _trace_signature(repeated_trace) == signature


def test_public_shared_entry_points_use_the_same_plan_and_binding():
    from dmm import matmul as shared
    from dmm.algorithms import cannon
    a = np.eye(5, dtype=np.float32)
    shared.validate_size(5, layout='cannon', P=4, c=1)
    job = shared.prepare(a, a, layout='cannon', P=4, c=1)
    assert type(job) is _native.PreparedMatmul
    with SerialExecutor() as executor:
        output = np.empty_like(a)
        job.run_into(output, executor)
        expected, trace = cannon.matmul_traced(a, a, executor, P=4)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(shared.matmul(a, a, executor, layout='cannon', P=4, c=1), expected)
        assert _trace_signature(job.trace()) == _trace_signature(trace)
    for layout in ['ring', '2.5d', 'unknown']:
        with pytest.raises(ValueError, match='implemented|unknown'):
            shared.prepare(a, a, layout=layout, P=64, c=1)
