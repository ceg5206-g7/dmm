import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from dmm.algorithms import cannon
from dmm.config import BenchmarkConfig, DatasetConfig
import dmm.config as config_module
import dmm.benchmark as benchmark
from dmm.datasets import generate_dataset
from dmm.execution import ThreadExecutor, SerialExecutor
import numpy as np
import pytest


def make_config(data_dir, output, **options):
    values = dict(data_dir=data_dir, output=output, cases=[{"layout": "cannon", "P": 16, "c": 1}],
                  modes={"sequential": {}}, groups=[0], sizes=[16], warmups=0)
    values.update(options)
    return BenchmarkConfig(**values)


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "data"
    generate_dataset(DatasetConfig(root, 5206, [0, 1], [16, 32], [0, 255]))
    return root


def test_csv_writer_flushes_header_and_rows_and_closes_on_failure(tmp_path):
    output = tmp_path / "results" / "test.csv"
    result = dict(layout="cannon", P=16, c=1, mode="sequential", workers=None,
                  group=0, pair=0, size=16, repeat=0, time_s=0.25, max_abs_error=0, frobenius_relative_error=0)
    with pytest.raises(RuntimeError, match="calculation failed"):
        with benchmark.open_results_csv(output) as write_result:
            with output.open(newline="") as handle:
                reader = csv.DictReader(handle)
                assert reader.fieldnames == list(benchmark.CSV_FIELDS)
                assert list(reader) == []
            write_result(result)
            with output.open(newline="") as handle:
                row, = csv.DictReader(handle)
            assert row == {key: "" if value is None else str(value) for key, value in result.items()}
            raise RuntimeError("calculation failed")
    with pytest.raises(ValueError, match="closed file"):
        write_result(result)
    with output.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [row]


def test_benchmark_pair_times_only_the_algorithm(monkeypatch):
    events = []
    clock_values = iter([10.0, 10.25])
    a = np.eye(4, dtype=np.float32)
    b = a.copy()
    expected = a.copy()
    executor = SerialExecutor()

    def clock():
        events.append("clock")
        return next(clock_values)

    def multiply(left, right, selected):
        events.append("multiply")
        assert left is a and right is b and selected is executor
        return expected

    monkeypatch.setattr(benchmark, "perf_counter", clock)
    result, elapsed = benchmark.benchmark_pair(multiply, a, b, executor)
    assert result is expected
    assert elapsed == 0.25
    assert events == ["clock", "multiply", "clock"]


def test_validation_uses_saved_reference_and_reports_bad_cell():
    reference = np.array([[19, 22], [43, 50]], dtype=np.float64)
    c = reference.astype(np.float32)
    benchmark._validate_product(c, reference)
    c[1, 1] = 51
    with pytest.raises(ValueError, match=r"Incorrect result at \(1, 1\): expected 50.0, got 51.0"):
        benchmark._validate_product(c, reference)


@pytest.mark.parametrize("result", [None, [[1]], np.zeros((3, 3), dtype=np.float32), np.zeros((2, 2)),
                                  np.full((2, 2), np.nan, dtype=np.float32),
                                  np.full((2, 2), np.inf, dtype=np.float32)])
def test_validation_rejects_invalid_result_format(result):
    a = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="Result must be"):
        benchmark._validate_product(result, a)


@pytest.mark.parametrize("repeats", [1, 3])
def test_runner_reuses_executor_and_excludes_setup_and_validation(data_dir, tmp_path, monkeypatch, repeats):
    events = []
    output = tmp_path / "results" / "test.csv"
    original_load = benchmark.load_pair
    original_load_reference = benchmark.load_reference
    original_validate = benchmark._validate_product
    original_save_metadata = benchmark.save_run_metadata
    clock_values = iter(value for start, elapsed in ((10.0, 0.25), (20.0, 0.5))
                        for _ in range(repeats) for value in (start, start + elapsed))

    class TrackedExecutor(SerialExecutor):
        def __enter__(self):
            events.append("enter")
            assert output.with_suffix(".json").exists()
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")
            return False

    def load(root, group, size):
        events.append(f"load {size}")
        return original_load(root, group, size)

    def save_metadata(config):
        events.append("metadata")
        return original_save_metadata(config)

    def load_reference(root, group, size):
        events.append(f"reference {size}")
        return original_load_reference(root, group, size)

    def clock():
        events.append("clock")
        return next(clock_values)

    def multiply(a, b, executor, **options):
        events.append("multiply")
        assert not a.flags.writeable and not b.flags.writeable
        return cannon.matmul(a, b, executor)

    def validate(c, reference):
        events.append("validate")
        if c.shape[0] == 32:
            with output.open(newline="") as handle:
                assert len(list(csv.DictReader(handle))) == events.count("validate") - 1
        return original_validate(c, reference)

    monkeypatch.setattr(benchmark, "SerialExecutor", TrackedExecutor)
    monkeypatch.setattr(benchmark, "save_run_metadata", save_metadata)
    monkeypatch.setattr(benchmark, "load_pair", load)
    monkeypatch.setattr(benchmark, "load_reference", load_reference)
    monkeypatch.setattr(benchmark, "perf_counter", clock)
    monkeypatch.setattr(benchmark, "import_module", lambda name: SimpleNamespace(matmul=multiply, validate_size=cannon.validate_size))
    monkeypatch.setattr(benchmark, "_validate_product", validate)

    assert benchmark.run_benchmark(make_config(data_dir, output, sizes=[16, 32], repeats=repeats)) == 2 * repeats
    expected_events = ["metadata", "enter"]
    for size in (16, 32):
        expected_events += [f"load {size}", f"reference {size}"]
        expected_events += ["clock", "multiply", "clock", "validate"] * repeats
    assert events == expected_events + ["exit"]
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(benchmark.CSV_FIELDS)
        rows = list(reader)
    assert [row["time_s"] for row in rows] == ["0.25"] * repeats + ["0.5"] * repeats
    assert [row["size"] for row in rows] == ["16"] * repeats + ["32"] * repeats
    assert [row["pair"] for row in rows] == ["0"] * repeats + ["1"] * repeats
    assert [row["repeat"] for row in rows] == [str(repeat) for repeat in range(repeats)] * 2
    assert all(row["mode"] == "sequential" and row["workers"] == "" for row in rows)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("repeats", [1, 3])
def test_threaded_uses_real_pool_and_records_worker_count(data_dir, tmp_path, monkeypatch, workers, repeats):
    created = []

    def create_pool(workers):
        created.append(workers)
        return ThreadExecutor(workers)

    monkeypatch.setattr(benchmark, "ThreadExecutor", create_pool)
    monkeypatch.setattr(config_module.os, "cpu_count", lambda: 6)
    output = tmp_path / "mp.csv"
    assert benchmark.run_benchmark(make_config(data_dir, output, repeats=repeats,
                                   modes={"threaded": {"workers": [workers]}})) == repeats
    assert created == [workers]
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["repeat"] for row in rows] == [str(repeat) for repeat in range(repeats)]
    assert all(row["mode"] == "threaded" and row["workers"] == str(workers)
               and float(row["time_s"]) >= 0 for row in rows)


def test_subset_uses_the_assignment_pair_index(data_dir, tmp_path):
    output = tmp_path / "subset.csv"
    benchmark.run_benchmark(make_config(data_dir, output, sizes=[32]))
    with output.open(newline="") as handle:
        row, = csv.DictReader(handle)
    assert row["pair"] == "1"


def test_existing_csv_is_not_overwritten(data_dir, tmp_path):
    output = tmp_path / "existing.csv"
    output.write_text("keep existing results")
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(make_config(data_dir, output))
    assert output.read_text() == "keep existing results"
    assert not output.with_suffix(".json").exists()


def test_later_failure_preserves_completed_rows(data_dir, tmp_path):
    np.save(data_dir / "G0" / "B_32.npy", np.zeros((32, 32), dtype=np.uint8))
    output = tmp_path / "partial.csv"
    with pytest.raises(ValueError, match="dtype"):
        benchmark.run_benchmark(make_config(data_dir, output, sizes=[16, 32]))
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["size"] == "16"
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["benchmark"]["sizes"] == [16, 32]
    assert metadata["dataset"]["seed"] == 5206


def test_incorrect_result_is_not_recorded(data_dir, tmp_path, monkeypatch):
    def incorrect(a, b, executor, **options):
        return np.zeros_like(a)

    monkeypatch.setattr(benchmark, "import_module", lambda name: SimpleNamespace(matmul=incorrect, validate_size=cannon.validate_size))
    output = tmp_path / "incorrect.csv"
    with pytest.raises(ValueError, match="Incorrect result"):
        benchmark.run_benchmark(make_config(data_dir, output))
    with output.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == []


def test_each_repeat_is_validated_and_failure_preserves_only_completed_measurements(data_dir, tmp_path, monkeypatch):
    calls = []

    def multiply(a, b, executor, **options):
        calls.append((a, b, executor))
        return cannon.matmul(a, b, executor) if len(calls) == 1 else np.zeros_like(a)

    monkeypatch.setattr(benchmark, "import_module", lambda name: SimpleNamespace(
        matmul=multiply, validate_size=cannon.validate_size))
    output = tmp_path / "partial_repeats.csv"
    with pytest.raises(ValueError, match="Incorrect result"):
        benchmark.run_benchmark(make_config(data_dir, output, repeats=3))
    assert len(calls) == 2
    assert all(first is second for first, second in zip(calls[0], calls[1]))
    with output.open(newline="") as handle:
        row, = csv.DictReader(handle)
    assert row["repeat"] == "0"
    assert json.loads(output.with_suffix(".json").read_text())["benchmark"]["repeats"] == 3


@pytest.mark.parametrize("options,message", [
    ({"cases": [dict(layout="unknown", P=16, c=1)]}, "layout"),
    ({"cases": [dict(layout="cannon", P=16, c=1)] * 2}, "unique"),
    ({"cases": "cannon"}, "list"),
    ({"modes": {"unknown": {}}}, "mode"),
    ({"modes": {}}, "non-empty"),
    ({"modes": []}, "mapping"),
    ({"modes": {"sequential": None}}, "mapping"),
    ({"modes": {"sequential": {"workers": [1]}}}, "fields"),
    ({"modes": {"threaded": {}}}, "fields"),
    ({"modes": {"threaded": {"workers": [1], "typo": 2}}}, "fields"),
    ({"modes": {"threaded": {"workers": 2}}}, "list"),
    ({"modes": {"threaded": {"workers": []}}}, "non-empty"),
    ({"modes": {"threaded": {"workers": [0]}}}, "workers"),
    ({"modes": {"threaded": {"workers": [True]}}}, "workers"),
    ({"modes": {"threaded": {"workers": [7]}}}, "workers"),
    ({"modes": {"threaded": {"workers": [1, 1]}}}, "unique"),
    ({"groups": [-1]}, "groups"),
    ({"groups": [True]}, "groups"),
    ({"groups": [0, 0]}, "unique"),
    ({"groups": []}, "non-empty"),
    ({"sizes": []}, "sizes"),
    ({"sizes": [0]}, "sizes"),
    ({"sizes": [16.0]}, "sizes"),
    ({"sizes": [32, 16]}, "ascending"),
    ({"sizes": [16, 16]}, "unique"),
    ({"repeats": 0}, "repeats"),
    ({"repeats": True}, "repeats"),
])
def test_invalid_configuration_does_not_create_output(tmp_path, monkeypatch, options, message):
    monkeypatch.setattr(config_module.os, "cpu_count", lambda: 8)
    output = tmp_path / "results" / "invalid.csv"
    with pytest.raises(ValueError, match=message):
        benchmark.run_benchmark(make_config(tmp_path / "data", output, **options))
    assert not output.parent.exists()


@pytest.mark.parametrize("cpu_count,workers,limit", [(2, 3, 2), (None, 2, 1)])
def test_process_count_is_limited_by_available_cpus(tmp_path, monkeypatch, cpu_count, workers, limit):
    monkeypatch.setattr(config_module.os, "cpu_count", lambda: cpu_count)
    config = make_config(tmp_path, tmp_path / "test.csv", modes={"threaded": {"workers": [workers]}})
    with pytest.raises(ValueError, match="workers") as error:
        benchmark.run_benchmark(config)
    assert str(tuple(range(1, limit + 1))) in str(error.value)


def test_all_combinations_reuse_one_executor_per_mode_and_process_count(data_dir, tmp_path, monkeypatch):
    from itertools import product

    created = []
    calls = []
    checks = []
    expected_checks = [("cannon", 16), ("cannon", 32), ("ring", 16), ("ring", 32)]

    def create_serial():
        created.append(None)
        return SerialExecutor()

    def create_pool(workers):
        created.append(workers)
        return ThreadExecutor(workers)

    def load_algorithm(module_name):
        def validate_size(size, **options):
            checks.append((module_name.rsplit(".", 1)[1], size))
            cannon.validate_size(size)

        def multiply(a, b, executor, **options):
            assert checks == expected_checks
            calls.append((module_name, a.shape[0]))
            return cannon.matmul(a, b, executor)

        return SimpleNamespace(matmul=multiply, validate_size=validate_size)

    monkeypatch.setattr(config_module.os, "cpu_count", lambda: 6)
    monkeypatch.setattr(benchmark, "SerialExecutor", create_serial)
    monkeypatch.setattr(benchmark, "ThreadExecutor", create_pool)
    # The second name exercises dispatch; only Cannon is currently implemented.
    monkeypatch.setattr(benchmark, "import_module", load_algorithm)
    output = tmp_path / "combinations.csv"
    config = make_config(data_dir, output, cases=[dict(layout="cannon", P=16, c=1), dict(layout="ring", P=16, c=1)], groups=[0, 1], sizes=[16, 32],
                         modes={"sequential": {}, "threaded": {"workers": [1, 2]}})
    assert benchmark.run_benchmark(config) == 24
    assert created == [None, 1, 2]
    assert checks == expected_checks
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = [
        (algorithm, mode, count, str(group), str(size))
        for (mode, count), algorithm, group, size in product(
            [("sequential", ""), ("threaded", "1"), ("threaded", "2")],
            ["cannon", "ring"], [0, 1], [16, 32],
        )
    ]
    assert [(r["layout"], r["mode"], r["workers"], r["group"], r["size"]) for r in rows] == expected
    assert calls == [(f"dmm.algorithms.{algorithm}", int(size)) for algorithm, _, _, _, size in expected]


def test_unimplemented_algorithm_fails_before_creating_output(data_dir, tmp_path):
    output = tmp_path / "results" / "missing.csv"
    with pytest.raises(ValueError, match="ring.*not implemented"):
        benchmark.run_benchmark(make_config(data_dir, output, cases=[dict(layout="cannon", P=16, c=1), dict(layout="ring", P=16, c=1)]))
    assert not output.parent.exists()


def test_custom_dataset_dimensions_and_range_are_used(tmp_path):
    root = tmp_path / "custom"
    generate_dataset(DatasetConfig(root, 123, [5], [20, 12], [-4, 8]))
    output = tmp_path / "custom.csv"
    config = make_config(root, output, groups=[5], sizes=[12, 20])
    assert benchmark.run_benchmark(config) == 2
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["group"], r["size"], r["pair"]) for r in rows] == [("5", "12", "1"), ("5", "20", "0")]


@pytest.mark.parametrize("options", [{"groups": [5]}, {"sizes": [8]}])
def test_missing_dataset_selection_fails_before_creating_output(data_dir, tmp_path, options):
    output = tmp_path / "results" / "missing.csv"
    with pytest.raises(ValueError, match="dataset manifest"):
        benchmark.run_benchmark(make_config(data_dir, output, **options))
    assert not output.parent.exists()


def test_benchmark_never_recomputes_the_reference(data_dir, tmp_path, monkeypatch):
    def unexpected_multiply(*args, **kwargs):
        pytest.fail("References must be loaded, not recomputed during the benchmark")

    monkeypatch.setattr(np, "matmul", unexpected_multiply)
    config = make_config(data_dir, tmp_path / "reused.csv", groups=[0, 1], sizes=[16, 32])
    assert benchmark.run_benchmark(config) == 4


def test_missing_reference_does_not_run_the_algorithm(data_dir, tmp_path, monkeypatch):
    (data_dir / "G0" / "C_ref_16.npy").unlink()

    def unexpected_multiply(*args):
        pytest.fail("Missing references must fail before the timed algorithm call")

    monkeypatch.setattr(benchmark, "benchmark_pair", unexpected_multiply)
    output = tmp_path / "missing_reference.csv"
    with pytest.raises(FileNotFoundError, match="C_ref_16"):
        benchmark.run_benchmark(make_config(data_dir, output))
    with output.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == []


def test_saved_reference_is_the_validation_source(data_dir, tmp_path):
    np.save(data_dir / "G0" / "C_ref_16.npy", np.zeros((16, 16), dtype=np.float64))
    output = tmp_path / "bad_reference.csv"
    with pytest.raises(ValueError, match="expected 0"):
        benchmark.run_benchmark(make_config(data_dir, output))
    with output.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == []


def test_metadata_captures_configuration_manifest_and_environment(data_dir, tmp_path, monkeypatch):
    output = tmp_path / "results" / "run.csv"
    config = make_config(data_dir, output, groups=[1, 0], sizes=[16, 32], repeats=3,
                         modes={"sequential": {}, "threaded": {"workers": [1, 2]}})
    monkeypatch.setattr(benchmark.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(benchmark.platform, "python_version", lambda: "3.12.test")
    monkeypatch.setattr(benchmark.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "test-cpu")
    monkeypatch.setattr(benchmark.platform, "machine", lambda: "test-architecture")
    before = datetime.now(timezone.utc)
    path = benchmark.save_run_metadata(config)
    after = datetime.now(timezone.utc)
    metadata = json.loads(path.read_text())
    assert path == output.with_suffix(".json")
    assert before <= datetime.fromisoformat(metadata["started_at"]) <= after
    assert metadata["benchmark"] == {
        "data_dir": str(data_dir), "output": str(output), "cases": [{"layout": "cannon", "P": 16, "c": 1}],
        "modes": {"sequential": {}, "threaded": {"workers": [1, 2]}},
        "groups": [1, 0], "sizes": [16, 32], "repeats": 3, "warmups": 0,
    }
    assert metadata["dataset"] == json.loads((data_dir / "manifest.json").read_text())
    assert metadata["environment"] == {
        "python_version": "3.12.test", "numpy_version": np.__version__, "platform": "test-platform",
        "cpu": {"processor": "test-cpu", "architecture": "test-architecture", "logical_count": 8},
    }
    config.groups.append(5)
    assert json.loads(path.read_text())["benchmark"]["groups"] == [1, 0]
    assert not output.exists()


def test_metadata_records_resolved_paths_without_mutating_configuration(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = make_config(Path("data"), Path("results/run.csv"))
    path = benchmark.save_run_metadata(config)
    settings = json.loads(path.read_text())["benchmark"]
    assert settings["data_dir"] == str(data_dir)
    assert settings["output"] == str(tmp_path / "results" / "run.csv")
    assert config.data_dir == Path("data") and config.output == Path("results/run.csv")


@pytest.mark.parametrize("error", [
    FileNotFoundError("git unavailable"), subprocess.CalledProcessError(128, "git"),
    subprocess.TimeoutExpired("git", 5),
])
def test_git_metadata_can_be_unavailable(data_dir, tmp_path, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(benchmark.subprocess, "run", fail)
    path = benchmark.save_run_metadata(make_config(data_dir, tmp_path / "run.csv"))
    assert json.loads(path.read_text())["git"] == {"commit": None, "dirty": None}


@pytest.mark.parametrize("change", ["tracked", "untracked"])
def test_git_metadata_uses_the_code_checkout_not_the_working_directory(data_dir, tmp_path, monkeypatch, change):
    repo = tmp_path / "checkout"
    repo.mkdir()
    hooks = tmp_path / "empty-hooks"
    hooks.mkdir()

    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
             "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={hooks}", *arguments],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init")
    source = repo / "python" / "dmm" / "benchmark.py"
    source.parent.mkdir(parents=True)
    source.write_text("# tracked source\n")
    git("add", ".")
    git("commit", "-m", "Initial source")
    commit = git("rev-parse", "HEAD")
    monkeypatch.setattr(benchmark, "__file__", str(source))
    monkeypatch.chdir(tmp_path)
    path = benchmark.save_run_metadata(make_config(data_dir, tmp_path / "clean.csv"))
    assert json.loads(path.read_text())["git"] == {"commit": commit, "dirty": False}
    changed = source if change == "tracked" else repo / "new-file.txt"
    changed.write_text("changed\n")
    path = benchmark.save_run_metadata(make_config(data_dir, tmp_path / "dirty.csv"))
    assert json.loads(path.read_text())["git"] == {"commit": commit, "dirty": True}


def test_missing_cpu_details_are_recorded_as_unknown(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark.os, "cpu_count", lambda: None)
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "")
    path = benchmark.save_run_metadata(make_config(data_dir, tmp_path / "run.csv"))
    cpu = json.loads(path.read_text())["environment"]["cpu"]
    assert cpu["processor"] is None and cpu["logical_count"] is None


def test_existing_metadata_prevents_a_new_csv_and_preserves_both_files(data_dir, tmp_path):
    output = tmp_path / "run.csv"
    path = output.with_suffix(".json")
    path.write_text("keep existing metadata")
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(make_config(data_dir, output))
    assert not output.exists()
    output.write_text("keep existing CSV")
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(make_config(data_dir, output))
    assert output.read_text() == "keep existing CSV"
    assert path.read_text() == "keep existing metadata"


@pytest.mark.parametrize("name", ["run.json", "run.JSON"])
def test_metadata_cannot_use_the_csv_filename(data_dir, tmp_path, name):
    output = tmp_path / name
    with pytest.raises(ValueError, match="different filename"):
        benchmark.run_benchmark(make_config(data_dir, output))
    assert not output.exists()


def test_failed_metadata_write_is_cleaned_up_before_starting_an_experiment(data_dir, tmp_path, monkeypatch):
    output = tmp_path / "run.csv"
    metadata_path = output.with_suffix(".json")
    original_open = Path.open

    class FailedWrite:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def write(self, text):
            self.handle.write("{")
            raise OSError("metadata disk full")

        def __exit__(self, *args):
            self.handle.close()

    def open_file(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return FailedWrite(handle) if path == metadata_path else handle

    def unexpected_executor():
        pytest.fail("An experiment must not start without its metadata")

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(benchmark, "SerialExecutor", unexpected_executor)
    with pytest.raises(OSError, match="metadata disk full"):
        benchmark.run_benchmark(make_config(data_dir, output))
    assert not output.exists() and not metadata_path.exists()


@pytest.mark.parametrize("existing_results", [False, True])
def test_invalid_algorithm_size_fails_before_any_experiment_setup(tmp_path, monkeypatch, existing_results):
    data_dir = tmp_path / "data"
    generate_dataset(DatasetConfig(data_dir, 5206, [0], [4, 7], [0, 255]))
    output = tmp_path / "results" / "invalid.csv"
    if existing_results:
        output.parent.mkdir()
        output.write_text("keep existing results")
        output.with_suffix(".json").write_text("keep existing metadata")

    def unexpected_setup(*args, **kwargs):
        pytest.fail("Invalid algorithm sizes must fail before experiment setup or calculation")

    for name in ("save_run_metadata", "open_results_csv", "SerialExecutor", "ThreadExecutor",
                 "load_pair", "load_reference", "benchmark_pair", "perf_counter"):
        monkeypatch.setattr(benchmark, name, unexpected_setup)
    def reject_seven(size, **options):
        if size == 7:
            raise ValueError("test size limit is 4")

    monkeypatch.setattr(cannon, "validate_size", reject_seven)
    config = make_config(data_dir, output, sizes=[4, 7],
                         modes={"sequential": {}, "threaded": {"workers": [1]}})
    with pytest.raises(ValueError, match="cannon.*size 7.*test size limit"):
        benchmark.run_benchmark(config)
    if existing_results:
        assert output.read_text() == "keep existing results"
        assert output.with_suffix(".json").read_text() == "keep existing metadata"
    else:
        assert not output.parent.exists()


def test_later_algorithms_are_validated_before_any_experiment_runs(data_dir, tmp_path, monkeypatch):
    checks = []

    def unexpected_setup(*args, **kwargs):
        pytest.fail("All algorithm/size combinations must be checked before starting")

    def load_algorithm(module_name):
        name = module_name.rsplit(".", 1)[1]

        def validate_size(size, **options):
            checks.append((name, size))
            if name == "ring" and size == 32:
                raise ValueError("test size limit is 16")

        return SimpleNamespace(matmul=unexpected_setup, validate_size=validate_size)

    monkeypatch.setattr(benchmark, "import_module", load_algorithm)
    for name in ("save_run_metadata", "open_results_csv", "SerialExecutor", "ThreadExecutor", "load_pair"):
        monkeypatch.setattr(benchmark, name, unexpected_setup)
    output = tmp_path / "results" / "invalid.csv"
    config = make_config(data_dir, output, cases=[dict(layout="cannon", P=16, c=1), dict(layout="ring", P=16, c=1)], sizes=[16, 32])
    with pytest.raises(ValueError, match="ring.*size 32.*test size limit is 16"):
        benchmark.run_benchmark(config)
    assert checks == [("cannon", 16), ("cannon", 32), ("ring", 16), ("ring", 32)]
    assert not output.parent.exists()


@pytest.mark.parametrize("P", [1, 4, 16, 64, 256])
def test_benchmark_passes_grid_to_algorithm_and_records_metadata(data_dir, tmp_path, monkeypatch, P):
    calls = []
    original = cannon.matmul

    def multiply(a, b, executor, *, P, c):
        calls.append((P, c))
        return original(a, b, executor, P=P, c=c)

    monkeypatch.setattr(cannon, "matmul", multiply)
    output = tmp_path / f"P{P}.csv"
    config = make_config(data_dir, output, cases=[dict(layout="cannon", P=P, c=1)], warmups=1)
    assert benchmark.run_benchmark(config) == 1
    assert calls == [(P, 1), (P, 1)]
    settings = json.loads(output.with_suffix(".json").read_text())["benchmark"]
    assert (settings["cases"][0]["P"], settings["cases"][0]["c"]) == (P, 1)


@pytest.mark.parametrize("P,c", [(3, 1), (64, 4), (16, 4)])
def test_invalid_algorithm_grid_fails_before_output(data_dir, tmp_path, P, c):
    output = tmp_path / "results" / "invalid.csv"
    with pytest.raises(ValueError, match="c|grid"):
        benchmark.run_benchmark(make_config(data_dir, output, cases=[dict(layout="cannon", P=P, c=c)]))
    assert not output.parent.exists()


def test_mixed_grid_yaml_runs_real_cannon_and_keeps_results_separate(data_dir, tmp_path):
    import yaml
    from dmm.config import load_benchmark_config
    from dmm.analysis import analyze_results, summarize_results

    cases = [dict(layout="cannon", P=P, c=1) for P in (16, 64, 256)]
    values = dict(data_dir=str(data_dir), output="mixed.csv", cases=cases,
                  modes={"sequential": {}, "threaded": {"workers": [2]}},
                  groups=[0, 1], sizes=[16, 32], repeats=2, warmups=1)
    path = tmp_path / "mixed.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    config = load_benchmark_config(path)
    assert benchmark.run_benchmark(config) == 48
    with config.output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {int(row["P"]) for row in rows} == {16, 64, 256}
    assert [(row["layout"], int(row["P"]), int(row["c"])) for row in rows[:24]] == [
        ("cannon", P, 1) for P in (16, 64, 256) for _ in range(8)]
    saved = json.loads(config.output.with_suffix(".json").read_text())
    assert saved["benchmark"]["cases"] == cases
    summary = summarize_results(config.output)
    assert len(summary) == 12
    assert all(row["group_count"] == 2 and row["repeats"] == 2 for row in summary)
    assert len({(r["layout"], r["P"], r["c"], r["mode"], r["size"]) for r in summary}) == 12
    assert (analyze_results(config.output) / "time.png").is_file()


def test_dispatch_preserves_layout_and_replication_per_case(data_dir, tmp_path, monkeypatch):
    cases = [dict(layout="cannon", P=16, c=1), dict(layout="2.5d", P=64, c=4)]
    checks, calls = [], []

    def load(module):
        def validate_size(size, *, P, c):
            checks.append((module, size, P, c))

        def multiply(a, b, executor, *, P, c):
            calls.append((module, P, c))
            return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)

        return SimpleNamespace(validate_size=validate_size, matmul=multiply)

    monkeypatch.setattr(benchmark, "import_module", load)
    output = tmp_path / "dispatch.csv"
    assert benchmark.run_benchmark(make_config(data_dir, output, cases=cases, warmups=1)) == 2
    assert checks == [("dmm.algorithms.cannon", 16, 16, 1),
                      ("dmm.algorithms.two_point_five_d", 16, 64, 4)]
    assert calls == [("dmm.algorithms.cannon", 16, 1)] * 2 + [
                    ("dmm.algorithms.two_point_five_d", 64, 4)] * 2
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["layout"], r["P"], r["c"]) for r in rows] == [
        ("cannon", "16", "1"), ("2.5d", "64", "4")]


@pytest.mark.parametrize("case", [dict(layout="ring", P=16, c=1), dict(layout="2.5d", P=64, c=4)])
def test_unimplemented_layout_after_valid_case_fails_before_any_execution(data_dir, tmp_path, monkeypatch, case):
    def unexpected(*args, **kwargs):
        pytest.fail("preflight must check every case before execution")

    monkeypatch.setattr(cannon, "matmul", unexpected)
    output = tmp_path / "missing" / "run.csv"
    with pytest.raises(ValueError, match="not implemented"):
        benchmark.run_benchmark(make_config(data_dir, output, cases=[dict(layout="cannon", P=16, c=1), case]))
    assert not output.parent.exists()
