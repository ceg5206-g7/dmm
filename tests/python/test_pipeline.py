import csv
from dataclasses import replace
import json

import numpy as np
import pytest

from dmm.analysis import analyze_results, summarize_results
from dmm import _native
import dmm.benchmark as benchmark
from dmm.benchmark import run_benchmark
from dmm.config import BenchmarkConfig, DatasetConfig
from dmm.datasets import generate_dataset, load_pair
from dmm.algorithms import cannon
from dmm.execution import SerialExecutor, ThreadExecutor


def make_executor(workers=0):
    return SerialExecutor() if workers == 0 else ThreadExecutor(workers)


@pytest.fixture
def dataset(tmp_path):
    config = DatasetConfig(tmp_path / "data", 5206, [0, 1], [4, 12], [0, 255])
    generate_dataset(config)
    return config


def benchmark_config(dataset, tmp_path):
    return BenchmarkConfig(dataset.data_dir, tmp_path / "run.csv",
                           ["cannon"], {"sequential": {}, "threaded": {"workers": [1, 2]}},
                           [0, 1], [4, 12], repeats=2, warmups=1)


def test_native_pipeline_outputs_samples_metadata_and_plots(dataset, tmp_path):
    config = benchmark_config(dataset, tmp_path)
    assert run_benchmark(config) == 24
    metadata = json.loads(config.output.with_suffix(".json").read_text())
    assert metadata["benchmark"]["repeats"] == 2
    assert metadata["benchmark"]["warmups"] == 1
    assert metadata["native"]["backend"] == "pybind11"
    assert metadata["dataset"]["dtype"] == "float32"
    assert metadata["dataset"]["reference_dtype"] == "float64"
    assert metadata["native"]["reference_dtype"] == "float64"
    assert metadata["native"]["input_dtype"] == "float32"
    assert metadata["native"]["timer"]["name"] == "time.perf_counter"
    assert "BSP supersteps" in metadata["native"]["execution"]
    with config.output.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 24
    summary = summarize_results(config.output)
    assert len(summary) == 6
    assert all(row["group_count"] == 2 and row["repeats"] == 2 for row in summary)
    assert all(row["speedup"] == 1 for row in summary if row["workers"] is None)
    output = analyze_results(config.output)
    for name in ("summary.csv", "time.png", "speedup.png"):
        assert (output / name).stat().st_size > 0
    before = config.output.read_bytes()
    with pytest.raises(FileExistsError):
        run_benchmark(config)
    assert config.output.read_bytes() == before
    with pytest.raises(FileExistsError):
        analyze_results(config.output)


def test_benchmark_accepts_fortran_order_dataset(dataset, tmp_path):
    config = replace(benchmark_config(dataset, tmp_path), groups=[0], sizes=[4],
                     modes={"sequential": {}}, warmups=0, repeats=1)
    a, b = load_pair(dataset.data_dir, 0, 4)
    for name, matrix in (("A", a), ("B", b)):
        np.save(dataset.data_dir / f"G0/{name}_4.npy", np.asfortranarray(matrix), allow_pickle=False)
    assert run_benchmark(config) == 1


@pytest.mark.parametrize("array", [
    np.ones((4, 4), dtype=np.float64), np.ones((4, 4), dtype=np.int64),
    np.ones((4, 4), dtype=">f4"), np.ones((3, 3), dtype=np.float32),
    np.ones((4, 3), dtype=np.float32), np.ones(4, dtype=np.float32),
    np.empty((0, 0), dtype=np.float32), np.full((4, 4), np.nan, dtype=np.float32),
    np.full((4, 4), np.inf, dtype=np.float32), [[1, 2], [3, 4]],
])
def test_binding_rejects_invalid_arrays(array):
    with pytest.raises((ValueError, TypeError)):
        _native.cannon(array, np.eye(4, dtype=np.float32), make_executor())


@pytest.mark.parametrize("workers", [-1, 10**6])
def test_binding_rejects_invalid_workers(workers):
    with pytest.raises(ValueError):
        make_executor(workers)


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_binding_reuses_executor_with_independent_outputs(workers):
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    b = a.T.copy()
    executor = make_executor(workers)
    expected = a.astype(np.float64) @ b.astype(np.float64)
    first = _native.cannon(a, b, executor)
    first[0, 0] = -999
    np.testing.assert_allclose(_native.cannon(a, b, executor), expected)
    a.fill(0)
    np.testing.assert_array_equal(_native.cannon(a, b, executor), np.zeros((4, 4)))
    del executor
    assert first[0, 0] == -999


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_binding_preserves_matmul_tile_order(workers):
    a = np.tile(np.array([1e8, 1, -1e8, 1], dtype=np.float32), (4, 1))
    b = np.ones((4, 4), dtype=np.float32)
    expected = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    assert _native.GRID == 4
    np.testing.assert_array_equal(_native.cannon(a, b, make_executor(workers)), expected)


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_binding_supports_readonly_noncontiguous_inputs(workers):
    a = np.arange(16, dtype=np.float32).reshape(4, 4).T
    b = np.arange(16, dtype=np.float32).reshape(4, 4)[:, ::-1]
    a.flags.writeable = False
    b.flags.writeable = False
    a_before, b_before = a.copy(), b.copy()
    np.testing.assert_allclose(_native.cannon(a, b, make_executor(workers)),
                               a.astype(np.float64) @ b.astype(np.float64))
    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)


def test_warmups_and_validation_are_outside_algorithm_call_timing(dataset, tmp_path, monkeypatch):
    config = replace(benchmark_config(dataset, tmp_path), groups=[0], sizes=[4],
                     modes={"sequential": {}}, warmups=3, repeats=2)
    events = []
    multiply = cannon.matmul
    validate = benchmark._validate_product

    def tracked(a, b, executor):
        events.append("multiply")
        return multiply(a, b, executor)

    def checked(actual, reference):
        events.append("validate")
        return validate(actual, reference)

    timestamps = iter([0, 0.25, 1, 1.5])

    def clock():
        events.append("clock")
        return next(timestamps)

    monkeypatch.setattr(cannon, "matmul", tracked)
    monkeypatch.setattr(benchmark, "perf_counter", clock)
    monkeypatch.setattr(benchmark, "_validate_product", checked)
    assert run_benchmark(config) == 2
    assert events == ["multiply"] * 3 + ["clock", "multiply", "clock", "validate"] * 2
    with config.output.open(newline="") as handle:
        assert [float(row["time_s"]) for row in csv.DictReader(handle)] == [0.25, 0.5]


def test_configured_mode_worker_and_group_order_is_preserved(dataset, tmp_path):
    config = replace(benchmark_config(dataset, tmp_path), groups=[1, 0], repeats=1, warmups=0,
                     modes={"threaded": {"workers": [2, 1]}, "sequential": {}})
    assert run_benchmark(config) == 12
    with config.output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["mode"], row["workers"], row["group"], row["size"]) for row in rows] == [
        (mode, workers, group, size)
        for mode, workers in (("threaded", "2"), ("threaded", "1"), ("sequential", ""))
        for group in ("1", "0") for size in ("4", "12")
    ]


@pytest.mark.parametrize("elapsed", [0, -1, float("nan"), float("inf")])
def test_invalid_measured_time_is_not_recorded(dataset, tmp_path, monkeypatch, elapsed):
    config = replace(benchmark_config(dataset, tmp_path), groups=[0], sizes=[4],
                     modes={"sequential": {}}, warmups=0, repeats=1)
    timestamps = iter([0, elapsed])
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(timestamps))
    with pytest.raises(ValueError, match="positive and finite"):
        run_benchmark(config)
    with config.output.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == []


@pytest.mark.parametrize("invalid", [-1, 0, True, 1.5])
def test_thread_executor_rejects_invalid_workers(invalid):
    with pytest.raises(ValueError, match="workers|worker count"):
        ThreadExecutor(invalid)


def test_executor_context_closes_after_failure():
    a = np.eye(4, dtype=np.float32)
    executor = ThreadExecutor(2)
    assert ThreadExecutor is _native.ThreadExecutor
    assert SerialExecutor is _native.SerialExecutor
    np.testing.assert_array_equal(cannon.matmul(a, a, executor), a)
    with pytest.raises(ValueError, match="deliberate failure"):
        with executor as selected:
            assert selected is executor
            with pytest.raises(RuntimeError, match="already active"):
                executor.__enter__()
            np.testing.assert_array_equal(cannon.matmul(a, a, executor), a)
            raise ValueError("deliberate failure")
    executor.close()  # Closing twice is harmless.
    with pytest.raises(RuntimeError, match="closed"):
        cannon.matmul(a, a, executor)
    with pytest.raises(RuntimeError, match="closed"):
        executor.__enter__()
    with ThreadExecutor(2) as fresh:
        np.testing.assert_array_equal(cannon.matmul(a, a, fresh), a)
    np.testing.assert_array_equal(cannon.matmul(a, a, SerialExecutor()), a)


def test_concurrent_calls_share_one_native_executor_safely():
    from concurrent.futures import ThreadPoolExecutor

    a = np.arange(144, dtype=np.float32).reshape(12, 12)
    a.flags.writeable = False
    executor = make_executor(2)
    with ThreadPoolExecutor(2) as callers:
        outputs = list(callers.map(lambda _: _native.cannon(a, a, executor), range(8)))
    for output in outputs:
        np.testing.assert_allclose(output, a.astype(np.float64) @ a.astype(np.float64))
