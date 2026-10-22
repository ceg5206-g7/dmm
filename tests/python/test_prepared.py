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
