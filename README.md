# Distributed Matrix Multiplication

CEG5206 experiments on matrix layouts and communication costs. C++20 implements
Cannon and the worker executor; pybind11 exposes them to Python. Python handles
shared FP32 NumPy datasets, FP64 references, warmup, repeated timing, CSV output
and plots.

Cannon uses a configurable q × q logical grid (P=q², default P=16) on a
BSP runtime: each node owns its A/B/C blocks, and the initial skew and the
round shifts are real copied messages. Communication is counted per superstep,
not delayed; BSP costs are computed from those counts afterwards (see
[Communication model](#communication-model)). Common row-shard input/output conversion and zero padding are implemented;
1D ring and 2.5D remain future work. Per-node payload memory
peaks are tracked for local tiles and messages. Wall-clock
timings describe the single-machine implementation, including message copies.

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
configuration uses all sizes, five warmups and fifteen measurements per case:
1080 samples. Increase `repeats` in YAML for more samples.

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
    c = matmul(a, b, executor)  # Default P=16, c=1.
    c = matmul(a, b, executor, P=4, c=1)

with ThreadExecutor(workers=2) as executor:
    c = matmul(a, b, executor)
    c_again = matmul(a, b, executor)  # Reuse workers across calls and sizes.

from dmm.algorithms.cannon import matmul_traced
from dmm.bsp import bsp_cost, count

with SerialExecutor() as executor:
    c, trace = matmul_traced(a, b, executor)
counts = count(trace)  # Totals; count(trace.supersteps[s]) gives one superstep.
cost = bsp_cost(counts, w=1, g=10, l=100)  # Recompute for any g/l without rerunning.
```

The `matmul(A, B, executor, *, P=16, c=1)` interface preserves existing calls.
`validate_size` and `matmul_traced` accept the same keyword options.
`dmm.execution` directly re-exports the bound C++ classes; it defines no Python
executor classes. The Python executor object is the bound C++ executor itself.
The thin algorithm module calls the common `dmm.matmul` entry points;
C++ owns input copies, node-local blocks, messages, scalar multiplication and assembly.
Python retains configuration, datasets, dispatch, measurement, validation and
analysis. There is no Python process pool: positive worker counts use
persistent C++ threads, while the serial executor runs on the caller.

On the C++ side, `Executor` is an abstract synchronous task interface with
`SerialExecutor` and `ThreadExecutor` implementations. `make_executor(0)` selects
serial execution; positive counts select a thread pool. The common runtime receives an
`Executor&` and does not depend on either implementation. Virtual dispatch occurs
once per superstep batch, outside the scalar multiplication loops.

Cannon's schedule is the plain
`MatmulDefinition build_cannon(BspPlan&, const MatmulConfig&)` function in
`src/algorithms/cannon.cpp`. Configuration is immutable; the returned definition
binds `input_a_rows`, `input_b_rows`, `output_c_rows` and `product_grid`. The shared
`PreparedMatmul` owns input copies, padding, reset/execute/gather, locking and
timing. Ordinary calls create it temporarily; `cannon.prepare` retains it.
New layouts supply one builder and thin Python forwarders. See the
[Cannon contribution walkthrough](doc/docs/contribute.md).

```cpp
#include "dmm/algorithms.hpp"
auto C = dmm::matmul(A, B, executor, "cannon", 64, 1);
dmm::PreparedMatmul job(A, B, 64, 1, dmm::build_cannon);
std::vector<float> output(A.values.size());
job.run_into(output, executor);
```

pybind11 exposes `Executor`, `SerialExecutor` and `ThreadExecutor` directly.
The `_native.matmul` function handles NumPy conversion, GIL release and the
algorithm call. Thread pools start at construction and close on context exit
or explicit `close()`. A closed pool cannot be reopened; create another executor
for another context. Destruction also closes any remaining workers.

Inputs must be finite, non-empty, square, matching native FP32 arrays. Other
dtypes are rejected; convert explicitly in Python. Read-only, transposed and
negative-stride inputs are supported. Any positive size is accepted on the common route: inputs are zero-padded
to a multiple of P and outputs cropped. Traces report original_size and padded_size. P is independent of the worker count H. Cannon requires
a positive square P and c=1. The shared `dmm::grid_size(P, c)` C++ helper
(and `_native.grid_size` binding) validates 2D/2.5D grids with P=c*q*q and c dividing q;
for example (64,4) and (256,4) are valid 2.5D grids, but Cannon rejects c>1.

Benchmark YAML lists each layout/P/c configuration separately. All cases run
across the shared `sizes`, `groups`, and `modes`, in list order within each mode:

```yaml
data_dir: ../data
output: ../results/mixed.csv
cases:
  - {layout: cannon, P: 16, c: 1}
  - {layout: cannon, P: 64, c: 1}
  - {layout: cannon, P: 256, c: 1}
modes:
  sequential: {}
  threaded:
    workers: [2]
groups: [0]
sizes: [16, 32]
repeats: 15
warmups: 5
```

Run the ready-to-use example with `uv run dmm bench --config configs/benchmark_mixed.yaml`,
then `uv run dmm analyze --input results/benchmark_mixed.csv`.
Each case requires exactly `layout`, `P`, and `c`; duplicate cases and old global
`algorithms`/`P`/`c` keys are rejected. Convert an old `algorithms: [cannon], P: 64, c: 1`
configuration to `cases: [{layout: cannon, P: 64, c: 1}]`.
Valid layouts are `ring`, `cannon`, and `2.5d`. Ring requires c=1; Cannon requires
c=1 and square P; 2.5D requires P=c*q*q and c dividing q. For example a future
`{layout: 2.5d, P: 64, c: 4}` case is valid, but its algorithm is not yet implemented.
Only Cannon currently runs. Missing layouts and invalid configurations fail during
preflight, before dataset loading or output creation; no case is silently skipped.

The dispatcher loads `dmm.algorithms.ring`, `dmm.algorithms.cannon`, or
`dmm.algorithms.two_point_five_d`. Each module provides
`validate_size(size, *, P, c)` and `matmul(A, B, executor, *, P, c)`.
Both warmups and measured calls receive that case's P/c. Worker pools are reused
across cases. CSV rows, summaries, plot series and serial baselines keep
`layout/P/c` separate; JSON metadata saves the complete ordered case list.

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

The native Cannon body runs on P=q² logical nodes, node `(i, j)`
having id `qi + j`, in q+1 supersteps. The default common route first converts
A and B from row shards (two supersteps each) and converts C back afterwards
(two more); P=1 conversions are local with no extra steps. The native body is:

1. `skew`: each node starts with its owned A_ij and B_ij; C is initialized to zero. A_ij is sent
   to `(i, j - i)` and B_ij to `(i - j, j)`; row 0 of A and column 0 of B stay put.
2. `multiply` × q: each node receives the blocks sent to it, accumulates
   `A_{i,k} B_{k,j}` with `k = (i + j + round) % q` using the shared blocked `gemm_add` kernel (tile 32),
   and, in the first q−1 rounds, sends A to its left and B to its upper neighbour.
3. The common route converts C tiles back to row shards, then the host adapter
   gathers/crops the result. All routes use this same output adapter.

Accumulation order is fixed for a given padded layout; padding can change the
block schedule. Results are verified against the independent FP64 reference.

## Communication model

`BspPlan` in `include/dmm/plan.hpp` is the only execution engine. Layout
builders declare node-owned buffers and `add_zero/add_copy/add_receive/add_gemm/add_send`
operations per superstep. `finalize` validates the schedule and prepares message
slots. `execute` interprets the operation lists through the borrowed executor;
it does not generate machine code. Each step has copy, kernel and send batches,
but the model counts one logical superstep. Messages must be consumed exactly
once in the immediately following step; all cross-node data is explicitly copied.

`Superstep` handles accept only current-step operations. Each node declares
copies/receives, then GEMMs, then sends. A message matches by sending step and
source/destination pair, with one next-step receive. Multiple components between
the same pair form one ordered bundle; no public tag or message table is needed.
Shapes and arity are checked at finalization. Equal-shaped components must keep
the agreed positional order.

`include/dmm/trace.hpp` holds trace/count types and model cost functions. Critical
counts sum per-step maxima, with h=max(sent,received). The cost is
T=w*W_crit+g*C_crit+l*S. No sleep emulates a network; saved traces support
parameter sweeps without repeating matrix arithmetic. See
[system design](doc/docs/design.md) for the lifecycle and measurement boundaries.

## Timing, correctness and results

YAML `warmups` is the number of discarded calls before each case's measured
calls; `repeats` is the number of independent measured calls. Defaults are
five warmups and fifteen measurements; the example configs set both explicitly.
Each case is one algorithm/mode/workers/group/size setting. There is no automatic iteration calibration.

Python surrounds `matmul(A, B, executor)` with `time.perf_counter()`, matching
`matmul`'s measurement boundary. Timing includes the Python wrapper, binding,
input validation/copies, node-local block copies, message copies and delivery,
all block products, worker dispatch/completion per superstep, assembly and output copying.
Input/reference loading, worker creation, warmup, correctness checks and file
writes are outside timing. One executor is reused for every algorithm/group/size
at a worker count. The configured mode and worker order is preserved; within
each mode, cases, groups and sizes follow their Cartesian product.
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
  layout/P/c/mode/workers/group/pair/size/repeat identifiers, maximum absolute
  error and Frobenius relative error. Serial `workers` is empty; `pair` is the
  size index in the dataset manifest, as in `matmul`. Each row records its own layout, P and c.
- `benchmark_lite.json`: the original `started_at`, `benchmark`, `dataset`,
  `environment` and `git` structure, plus a `native` section for compiler/build
  type, FP types, timing boundary and timer details. Settings include warmups. A dirty flag does not preserve local changes.
- `benchmark_lite_analysis/`: `summary.csv`, `time.png` and `speedup.png`.

Analysis reports median and IQR (Q75−Q25, linear interpolation) within each
input group, then takes the median of group medians and the median of group IQRs.
The IQR is repeat variability, not variability between input groups.
Speedup is the matching sequential median divided by the threaded median, using
the same layout, P, c, size and groups. CSV files without `repeat` are treated as one measurement. Historical CSV files
with an `algorithm` column are still readable; their unknown P/c are left blank
in summaries rather than inferred. Duplicate samples, incomplete repeats, missing baselines,
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

- `include/dmm/`, `src/plan_build.cpp`, `src/plan_execute.cpp`, `src/trace.cpp`: common C++ runtime.
- `src/algorithms/`: builder implementations and layout selection.
- `src/bindings.cpp`: pybind11 array validation, ownership and common executor/matmul API.
- `python/dmm/`: Python configuration, datasets, saved traces, timing and plotting.
- `configs/`: smoke, lite and full settings.
- `doc/`: separate MkDocs documentation repository, tracked as a Git submodule.
- `tests/cpp/`: native GoogleTest cases for Matrix, MatrixBlockView, BSP, Cannon and Executor,
  each in its own file.
- `tests/python/`: Python configuration, dataset, analysis and binding/pipeline tests.
- `data*/`, `results/`: generated inputs, records and plots, ignored by Git.

CI installs the extension, builds GoogleTest cases, runs CTest/pytest, then executes
the smoke CLI workflow. Tests cover BSP delivery and protocol checks, Cannon's hand-computed
communication counts and accumulation order, FP64
comparisons, different block sizes, zero/identity/ones/cancellation, repeated runs,
worker errors, invalid arrays, output ownership and concurrent calls, dataset
validation, executor lifecycle, warmup/timing boundaries and result comparability.
The original configuration, CLI, environment, benchmark and analysis tests are
ported with FP and native-executor adjustments. Layout names are `ring`, `cannon`, `2.5d`; unimplemented layouts fail before setup.
Only Cannon is currently implemented.

The shared rectangular FP32 kernel is `include/dmm/detail/gemm.hpp`; all layouts
must use its fixed tile 32 and common compilation flags. It accepts row strides,
accumulates into C, supports tile tails, and performs no allocation.

`include/dmm/collectives.hpp` provides fixed binary-tree broadcast and sum
reduction over disjoint ordered node groups. All groups share supersteps;
messages and reduction FLOPs are recorded by BSP, including the final receive
step. For each group, rank r has parent (r-1)/2. Singleton groups are no-ops.

Node payload ownership uses move-only `Buffer` from `include/dmm/detail/memory.hpp`.
Allocate through `BspPlan::allocate_buffer` so local tiles,
packing buffers, replicas and messages share one per-node memory ledger.
`Trace.peak_words` and each superstep contain per-node peaks; `Counts.peak_words`
is their maximum, and count aggregation takes max rather than sum for memory.
Host Matrix/oracle storage is excluded from logical-node payload; whole-process
memory requires a separate measurement.

Common row shards and base-layer block conversion are provided by
`include/dmm/redistribution.hpp`. Both directions copy real messages with global
`ElementRegion` rectangles. Packing and output buffers use the node memory ledger.
`count_native(trace)` excludes exactly input/output conversion from the full
trace, keeping the same actual padded dimensions. At n=16, P=16, the full path
has S=11, F_total=8192, W_crit=512, C_total=2496 and C_crit=164; the native body
has S=5, C_total=1920 and C_crit=128. Both metrics come from the same execution;
there is no standalone native-input mode.

The versioned [project proposal](doc/docs/proposal.pdf) now specifies C++20
in both its platform contract and build deliverables, matching CMake and the
public span interfaces. Its remaining acceptance conditions still apply.

`Superstep::add_gemm` couples the shared kernel call with actual (i,k,j), GEMM
shape and FLOP records. The builder returns `product_grid` to its block partition
and callers run `dmm.bsp.validate_coverage(trace)` to reject missing/duplicate blocks,
wrong shapes and inconsistent node work. The independent FP64 oracle remains
necessary; trace metadata is not included in payload-memory budgets.


For real trace capture and offline BSP scans:

```sh
uv run dmm model --config configs/model_smoke.yaml
# Use another YAML/output_dir to change gamma/lambda/budgets without rerunning:
uv run dmm model --config configs/my_model.yaml --traces results/model_smoke/traces
```

`model.csv` stores normalized T_BSP/w0 as T_model; t_measured stays empty.
Raw trace JSONs, input/file hashes and a correctness CSV support audit/reuse.
Phase/native costs, budget feasibility, ties and infeasible cells are explicit.
See [model commands and limits](doc/docs/model.md). The full proposal profile
is 24 executions/192 scenarios/144 selection cells, and still requires the
unimplemented ring/2.5D algorithms; its preflight refuses to fabricate results.
New trace captures use schema 2 with ordered message `regions`; saved schema-1
traces remain readable. `python/dmm/trace.py` provides shared serialization,
validation and offline loading for model and measurement.


Prepared measurements and independent local calibration:

```sh
uv run dmm measure --config configs/measure_serial.yaml
uv run dmm measure --config configs/measure_threaded.yaml
```

These measure common rows to rows after preallocation, with 5 warmups/15 repeats,
frozen microkernel w/g/l and held-out residuals. The four-worker profile requires
four physical cores. See [the measurement protocol](doc/docs/measurement.md).
Ring and 2.5D still belong to teammates; the full 24-run comparison awaits them.
