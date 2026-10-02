# Distributed Matrix Multiplication

CEG5206 project: compare the data layouts and communication costs of 1D ring,
2D Cannon and replicated 2.5D matrix multiplication.

This initial scaffold contains no experiment implementation. The C++ executable
has an empty entry point; Python dependencies and test cases are not added yet.

## Setup and build

Requires a C++17 compiler, CMake 3.16+ and uv. Python is pinned to 3.12;
`uv sync` installs it if needed.

```sh
uv sync --locked
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/dmm
```

Run future analysis scripts with `uv run scripts/<script>.py`.
Add Python dependencies with `uv add <dependency>` and commit `uv.lock`.

## Layout

- `src/`: C++ kernel, executor and layouts when implemented.
- `tests/`: future correctness and counting checks.
- `scripts/`: future experiment and analysis scripts.
- `results/`: generated records and plots, ignored by Git.

GitHub Actions checks the locked Python environment, builds the C++ scaffold
and runs its empty entry point on pushes and pull requests. It does not run
algorithm tests yet. Slide build files and generated presentations are ignored.
