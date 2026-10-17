# Distributed Matrix Multiplication

CEG5206 experiments on matrix layouts and communication costs. C++20 implements
Cannon and the worker executor; pybind11 exposes them to Python. Python handles
shared FP32 NumPy datasets, FP64 references, warmup, repeated timing, CSV output
and plots.

Cannon follows `matmul`'s fixed 4 × 4 logical grid: prepare 16 tasks, compute
one complete output block per task, then assemble the output. It represents
the logical shifts through block views. Node-local storage, message
copies, communication accounting, BSP costs, 1D ring and 2.5D are future work.
These timings describe the current single-machine implementation.

## Install and quick start

Requires a C++20 compiler, CMake 3.21+ and uv. Python is pinned to 3.12.
Installing the Python package builds the C++ extension in Release mode through
scikit-build-core and CMake. Google Benchmark is no longer a dependency.

```sh
uv sync --locked
uv run --locked pytest -q

uv run dmm generate --config configs/dataset_smoke.yaml
uv run dmm bench --config configs/benchmark_smoke.yaml
uv run dmm analyze --input results/benchmark_smoke.csv
```

Smoke covers sizes 4 and 12, two input groups, sequential and one/two-worker
execution, one untimed warmup and two measurements per case: 24 samples.
Its tiny timings verify the pipeline, not performance. Remove worker count 2
from the YAML on a host with only one available logical CPU.

For the regular experiment:

```sh
uv run dmm generate --config configs/dataset.yaml
uv run dmm bench --config configs/benchmark_lite.yaml
uv run dmm analyze --input results/benchmark_lite.csv
```

The dataset contains sizes 16 through 2048. Lite measures sizes through 256,
three groups, sequential and one/two-worker execution, with two warmups and
one measurement per case: 45 samples, matching `matmul` lite. The full
configuration uses all sizes, five warmups and three measurements per case:
216 samples, matching `matmul` full. Increase `repeats` in YAML for more samples.

Reuse a complete dataset instead of regenerating it. YAML paths are relative
to the configuration file. Inputs and result files are never overwritten;
choose a fresh output path when rerunning, including when replacing older
Google Benchmark results. The measurement methods differ, so retain each run's
metadata with its CSV.

After changing C++ sources, rebuild the installed extension:

```sh
uv sync --reinstall-package dmm
```

For standalone CMake development and C++ tests:

```sh
uv run cmake --preset release
uv run cmake --build --preset release --parallel
ctest --preset release
```

Replace `release` with `debug` for a debug build/test. These builds do not
replace the installed Python extension; use the reinstall command above for
Python runs.

Native tests use GoogleTest 1.17.0, fetched by CMake with a pinned commit and
SHA-256 checksum when `BUILD_TESTING=ON`. The first test configuration needs
network access. Python package builds set `BUILD_TESTING=OFF` and do not fetch
GoogleTest. CTest discovers each GoogleTest case separately, including worker
counts 0, 1, 2 and 5. To inspect or run a subset:

```sh
ctest --preset release -N
ctest --preset release -R MatrixBlockViewTest
```

## Python API

```python
from dmm.algorithms.cannon import matmul
from dmm.execution import SerialExecutor, ThreadExecutor
import numpy as np

a = np.eye(4, dtype=np.float32)
b = np.ones((4, 4), dtype=np.float32)

with SerialExecutor() as executor:
    c = matmul(a, b, executor)

with ThreadExecutor(workers=2) as executor:
    c = matmul(a, b, executor)
    c_again = matmul(a, b, executor)  # Reuse workers across calls and sizes.
```

The `matmul(A, B, executor)` interface matches the original Python framework.
`dmm.execution` directly re-exports the bound C++ classes; it defines no Python
executor classes. The Python executor object is the bound C++ executor itself.
The thin algorithm module calls `_native.cannon(A, B, executor)`;
C++ owns input copies, task preparation, scalar multiplication and assembly.
Python retains configuration, datasets, dispatch, measurement, validation and
analysis. There is no Python process pool: positive worker counts use
persistent C++ threads, while the serial executor runs on the caller.

On the C++ side, `Executor` is an abstract synchronous task interface with
`SerialExecutor` and `ThreadExecutor` implementations. `make_executor(0)` selects
serial execution; positive counts select a thread pool. `Cannon` receives an
`Executor&` and does not depend on either implementation. Virtual dispatch occurs
once per task batch, outside the scalar multiplication loops.

`Algorithm<T>` is a C++20 concept checking that a const algorithm object can
call `matmul(const Matrix<float>& A, const Matrix<float>& B, Executor& executor)`
and return exactly `Matrix<float>` by value. `Cannon` is an ordinary stateless struct, checked with
`static_assert(Algorithm<Cannon>)`; algorithms use no inheritance or virtual
methods. Each call borrows A/B and the executor until it returns, and returns
an independent matrix owned by the caller. The same algorithm object can be
reused with different inputs and executors; earlier results stay unchanged.

```cpp
dmm::Cannon algorithm;
dmm::Matrix<float> C = algorithm.matmul(A, B, executor);
```

pybind11 exposes `Executor`, `SerialExecutor` and `ThreadExecutor` directly.
The `_native.cannon` function handles NumPy conversion, GIL release and the
algorithm call. Thread pools start at construction and close on context exit
or explicit `close()`. A closed pool cannot be reopened; create another executor
for another context. Destruction also closes any remaining workers.

Inputs must be finite, non-empty, square, matching native FP32 arrays. Other
dtypes are rejected; convert explicitly in Python. Read-only, transposed and
negative-stride inputs are supported. Sizes must be divisible by 4; padding
is not implemented. The 16 output-block tasks are independent of the worker
count. The grid is fixed and has no YAML/API option.

Each call copies A/B into owned C++ matrices and returns an independent NumPy
output. Copies are included in algorithm-call timing. C++ computation releases
the GIL; calls sharing a thread executor serialize their task batches.
The executor also waits for an active batch before closing its workers.

`Matrix<T>` owns a square matrix of `T` values. Cannon uses `Matrix<float>` for
FP32 computation; native correctness tests accumulate an independent FP64
reference in `Matrix<double>`. `validate` compares the two matrices and checks
both their dimensions and storage lengths.

`Matrix<T>::block(row, col, size)` returns a read-only `MatrixBlockView<T>` containing
the block's start pointer, square size and original row stride. Creating or
copying a view does not allocate or copy matrix elements. Access uses
`view(i, j)`; block creation checks the storage shape and bounds, and debug
builds assert element indices. Keep the source matrix alive and do not replace
or reallocate its storage while a view is in use.

The C++ implementation follows `matmul.algorithms.cannon`'s algorithm:

1. Prepare 16 tasks in output-block row-major order. Each contains all four
   operand pairs selected by `k = (i + j + stage) % 4`, using `MatrixBlockView<float>`
   for each input block.
2. Map tasks to the sequential executor or worker threads. Each task creates
   a zeroed local tile and accumulates all four pairs with scalar `i-k-j` loops.
3. Wait for the tasks and assemble their tiles into a newly allocated output.

Inputs are not shifted in memory, and there is no synchronization after each
stage or separate per-round API. Communication simulation is deferred.

## Timing, correctness and results

YAML `warmups` is the number of discarded calls before each case's measured
calls; `repeats` is the number of independent measured calls. Defaults are
five warmups and one measurement; the example configs set both explicitly.
Each case is one algorithm/mode/workers/group/size setting. There is no automatic iteration calibration.

Python surrounds `matmul(A, B, executor)` with `time.perf_counter()`, matching
`matmul`'s measurement boundary. Timing includes the Python wrapper, binding,
input validation/copies, task preparation, local tile allocation/zeroing, all
block products, worker dispatch/completion, assembly and output copying.
Input/reference loading, worker creation, warmup, correctness checks and file
writes are outside timing. One executor is reused for every algorithm/group/size
at a worker count. The configured mode and worker order is preserved; within
each mode, algorithms, groups and sizes follow the original Cartesian product.
Inputs are marked read-only before calling the algorithm.

Execution uses C++ threads, with no Python process-pool serialization cost.
For tiny matrices, binding and dispatch costs may dominate. Timings from the
previous persistent-runner boundary are not directly comparable to these runs.

Every measured result is checked against the saved NumPy FP64 product using
`abs(actual - reference) <= 1e-4 + 1e-3 * abs(reference)`. Non-finite results
and invalid times are rejected; differences and error norms are computed in
FP64. As in `matmul`, each validated row is immediately written and flushed.
A later failure preserves earlier rows and run metadata; the failed result is
not recorded. Partial results do not imply the configured experiment completed.

Each run writes:

- `benchmark_lite.csv`: one row per measured call, with times in seconds,
  algorithm/mode/workers/group/pair/size/repeat identifiers, maximum absolute
  error and Frobenius relative error. Serial `workers` is empty; `pair` is the
  size index in the dataset manifest, as in `matmul`. The fixed grid is not a column.
- `benchmark_lite.json`: the original `started_at`, `benchmark`, `dataset`,
  `environment` and `git` structure, plus a `native` section for compiler/build
  type, FP types, timing boundary and timer details. Settings include warmups. A dirty flag does not preserve local changes.
- `benchmark_lite_analysis/`: `summary.csv`, `time.png` and `speedup.png`.

Analysis averages repetitions within each group, then averages group means.
Speedup is the matching sequential mean divided by the threaded mean, using
the same algorithm, size and groups. Summary fields, aggregation and plots
match `matmul`; CSV files without `repeat` are treated as one measurement. Duplicate samples, incomplete repeats, missing baselines,
mismatched groups and invalid times are rejected. There is no assumed speedup.

## Dataset compatibility with matmul

The dataset layout, configuration and loading interfaces are carried over from
`matmul`. The defaults use seed 5206, groups 0/1/2, sizes 16 through 2048,
and value bounds [0, 255]. Values are uniformly sampled and rounded to FP32.
Generation uses `np.random.default_rng([seed, group_id, size])`, so the same
settings reproduce the same inputs independently of generation order.

```text
data/
  manifest.json
  G0/
    A_16.npy
    B_16.npy
    C_ref_16.npy
    ...
  G1/
  G2/
```

A/B are saved as `float32`; `C_ref` is computed from those stored FP32 inputs
using NumPy FP64 multiplication/accumulation, and saved as `float64`, matching
the proposal. The manifest keeps seed, groups, sizes, value range and NumPy
version, with `dtype: float32`, `reference_dtype: float64` and
`reference: numpy.matmul(float64)`. Files are read with `allow_pickle=False`,
with shape, dtype, finiteness, range and selection validation.

The original Python interfaces are preserved: `generate_matrix`,
`reference_matmul`, `generate_dataset`, `load_manifest` (returning
`DatasetConfig`), `load_pair`, `load_reference`, `load_dataset_config` and
`load_benchmark_config`.

Python loads the FP32 arrays directly and passes them to the C++ binding,
which copies them into owned row-major storage inside the algorithm call. C++ output uses FP32;
validation uses the FP64 reference and numerical tolerance.

Earlier int64 `.npy` datasets and raw `.f32`/`.f64` datasets are incompatible.
Generate a new floating-point `.npy` dataset in a fresh directory; existing data
is never overwritten. The
package sets OMP, MKL, OpenBLAS, Accelerate and NumExpr thread limits to one
at import, as in `matmul`. Import `dmm` before NumPy for these limits to apply.

## Documentation

The [dmm_doc repository](https://github.com/ceg5206-g7/dmm_doc) is checked out as
the `doc/` submodule and follows the MkDocs structure used by `matmul/doc`.
It covers usage, Python/C++ architecture, Cannon, measurement rules and APIs.

```sh
git submodule update --init --recursive
uv sync --locked
uv run --locked pytest doc/tests -q
cd doc
uvx --with-requirements requirements.txt mkdocs build --strict
```

Use `mkdocs serve --strict` with the same uvx options for a local preview.
See [doc/README.md](doc/README.md) for the build and update workflow.

## Layout and checks

- `include/dmm/`, `src/cannon.cpp`, `src/execution.cpp`: shared C++ core.
- `src/bindings.cpp`: pybind11 array validation, ownership and executor/Cannon API.
- `python/dmm/`: Python configuration, datasets, timing and plotting.
- `configs/`: smoke, lite and full settings.
- `doc/`: separate MkDocs documentation repository, tracked as a Git submodule.
- `tests/cpp/`: native GoogleTest cases for Matrix, MatrixBlockView, Cannon and Executor,
  each in its own file.
- `tests/python/`: Python configuration, dataset, analysis and binding/pipeline tests.
- `data*/`, `results/`: generated inputs, records and plots, ignored by Git.

CI installs the extension, builds GoogleTest cases, runs CTest/pytest, then executes
the smoke CLI workflow. Tests cover Cannon's task accumulation order, FP64
comparisons, different block sizes, zero/identity/ones/cancellation, repeated runs,
worker errors, invalid arrays, output ownership and concurrent calls, dataset
validation, executor lifecycle, warmup/timing boundaries and result comparability.
The original configuration, CLI, environment, benchmark and analysis tests are
ported with FP and native-executor adjustments. Algorithm names remain
`cannon`, `fox`, `strassen`, `cw`; unimplemented algorithms fail before setup,
as in `matmul`. Only Cannon is currently implemented.
