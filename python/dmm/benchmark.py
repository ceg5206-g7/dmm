"""Time matrix multiplication and write independently verified CSV results."""

import csv
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import import_module
from functools import partial
from itertools import product
import json
from math import isfinite
import os
from pathlib import Path
import platform
import subprocess
from time import get_clock_info, perf_counter

import numpy as np

from dmm import _native
from dmm.config import BenchmarkConfig, LAYOUT_MODULES
from dmm.datasets import load_manifest, load_pair, load_reference
from dmm.execution import Executor, SerialExecutor, ThreadExecutor

CSV_FIELDS = ("layout", "P", "c", "mode", "workers", "group", "pair", "size", "repeat", "time_s",
              "max_abs_error", "frobenius_relative_error")


def save_run_metadata(config: BenchmarkConfig) -> Path:
    """Save an experiment snapshot beside its CSV without replacing any file."""
    config.validate()
    output = Path(config.output).resolve()
    if output.suffix.lower() == ".json":
        raise ValueError("Benchmark output must have a different filename from its .json metadata")
    metadata_path = output.with_suffix(".json")
    settings = asdict(config)
    settings["data_dir"] = str(Path(config.data_dir).resolve())
    settings["output"] = str(output)
    manifest = json.loads((Path(config.data_dir) / "manifest.json").read_text(encoding="utf-8"))

    git = {"commit": None, "dirty": None}
    source_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=normal"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        git = {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.SubprocessError):
        pass

    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": settings,
        "dataset": manifest,
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "cpu": {
                "processor": platform.processor() or None,
                "architecture": platform.machine(),
                "logical_count": os.cpu_count(),
            },
        },
        "git": git,
        "native": {
            "backend": "pybind11", "compiler": _native.compiler, "build_type": _native.build_type,
            "execution": "Per-case logical grids; BSP supersteps with copied messages",
            "input_dtype": "float32", "reference_dtype": "float64",
            "timer": {"name": "time.perf_counter", **vars(get_clock_info("perf_counter"))},
            "timing": "algorithm call including input/output copies; excludes worker startup and validation",
        },
    }
    contents = json.dumps(metadata, indent=2) + "\n"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    handle = metadata_path.open("x", encoding="utf-8")
    try:
        with handle:
            handle.write(contents)
    except BaseException:
        metadata_path.unlink(missing_ok=True)
        raise
    return metadata_path


@contextmanager
def open_results_csv(output: str | Path) -> Generator[Callable[[dict], None]]:
    """Create a new CSV and yield a function that writes and flushes each row."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        handle.flush()

        def write_result(result: dict) -> None:
            writer.writerow(result)
            handle.flush()

        yield write_result


def benchmark_pair(matmul, A: np.ndarray, B: np.ndarray, executor: Executor) -> tuple[np.ndarray, float]:
    """Time only the algorithm call; the caller owns setup and validation."""
    start = perf_counter()
    C = matmul(A, B, executor)
    elapsed = perf_counter() - start
    return C, elapsed


def _validate_product(C: np.ndarray, reference: np.ndarray) -> dict:
    """Compare with the independent FP64 reference outside timing."""
    if (not isinstance(C, np.ndarray) or C.shape != reference.shape
            or C.dtype != np.float32 or not np.isfinite(C).all()):
        raise ValueError("Result must be a finite float32 NumPy array matching the input shape")
    difference = np.abs(C.astype(np.float64) - reference)
    incorrect = difference > 1e-4 + 1e-3 * np.abs(reference)
    if np.any(incorrect):
        i, j = np.unravel_index(np.argmax(incorrect), C.shape)
        raise ValueError(f"Incorrect result at ({i}, {j}): expected {reference[i, j]}, got {C[i, j]}")
    reference_norm = np.linalg.norm(reference)
    error_norm = np.linalg.norm(difference)
    return dict(max_abs_error=float(difference.max()),
                frobenius_relative_error=float(error_norm / reference_norm if reference_norm else error_norm))


def run_benchmark(config: BenchmarkConfig) -> int:
    """Run all configured combinations, reusing a pool for each worker count."""
    config.validate()
    algorithms = []
    for case in config.cases:
        name = case["layout"]
        module_name = LAYOUT_MODULES[name]
        try:
            algorithm = import_module(f"dmm.algorithms.{module_name}")
        except ModuleNotFoundError as error:
            if error.name != f"dmm.algorithms.{module_name}":
                raise
            raise ValueError(f"Algorithm {name!r} is not implemented") from error
        for size in config.sizes:
            try:
                algorithm.validate_size(size, P=case["P"], c=case["c"])
            except ValueError as error:
                raise ValueError(f"Algorithm {name!r} does not support size {size}: {error}") from error
        algorithms.append((case, partial(algorithm.matmul, P=case["P"], c=case["c"])))

    dataset = load_manifest(config.data_dir)
    for name in ("groups", "sizes"):
        if any(value not in getattr(dataset, name) for value in getattr(config, name)):
            raise ValueError(f"Selected {name} must be present in the dataset manifest")

    completed = 0
    metadata_path = save_run_metadata(config)
    with ExitStack() as resources:
        try:
            write_result = resources.enter_context(open_results_csv(config.output))
        except BaseException:
            metadata_path.unlink(missing_ok=True)
            raise

        def run_with_executor(executor: Executor, mode: str, workers: int | None = None) -> int:
            completed = 0
            with executor:
                for (case, matmul), group_id, size in product(
                    algorithms, config.groups, config.sizes,
                ):
                    A, B = load_pair(config.data_dir, group_id, size)
                    reference = load_reference(config.data_dir, group_id, size)
                    A.flags.writeable = False
                    B.flags.writeable = False
                    for _ in range(config.warmups):
                        matmul(A, B, executor)
                    for repeat in range(config.repeats):
                        C, elapsed = benchmark_pair(matmul, A, B, executor)
                        if not isfinite(elapsed) or elapsed <= 0:
                            raise ValueError("measured time must be positive and finite")
                        errors = _validate_product(C, reference)
                        write_result({
                            **case,
                            "mode": mode,
                            "workers": workers,
                            "group": group_id,
                            "pair": dataset.sizes.index(size),
                            "size": size,
                            "repeat": repeat,
                            "time_s": elapsed,
                            **errors,
                        })
                        completed += 1
            return completed

        for mode, options in config.modes.items():
            if mode == "sequential":
                completed += run_with_executor(SerialExecutor(), mode)
            else:
                for workers in options["workers"]:
                    completed += run_with_executor(ThreadExecutor(workers=workers), mode, workers)
    return completed
