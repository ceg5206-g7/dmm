"""Load and validate YAML dataset and benchmark configurations."""

from dataclasses import MISSING, dataclass, field, fields
import os
from math import isfinite
from pathlib import Path

import yaml
import numpy as np

LAYOUT_MODULES = {"ring": "ring", "cannon": "cannon", "2.5d": "two_point_five_d"}
MODES = ("sequential", "threaded")


def _selection(name, values, allowed) -> None:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    if any(type(value) is not type(allowed[0]) or value not in allowed for value in values):
        raise ValueError(f"{name} must contain only values from {allowed}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


def _dimensions(groups, sizes) -> None:
    for name, values, minimum in (("groups", groups, 0), ("sizes", sizes, 1)):
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{name} must be a non-empty list")
        if any(type(value) is not int or value < minimum for value in values):
            raise ValueError(f"{name} must contain integers >= {minimum}")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must be unique")


def validate_value_range(value_range) -> None:
    """Check finite bounds representable in FP32, including constant inputs."""
    limit = float(np.finfo(np.float32).max)
    if (
        not isinstance(value_range, (list, tuple)) or len(value_range) != 2
        or any(type(value) not in (int, float) for value in value_range)
        or not -limit <= value_range[0] <= value_range[1] <= limit
    ):
        raise ValueError("value_range must contain two finite FP32 bounds with low <= high")


def validate_case(case: dict) -> None:
    """Validate the layout-specific logical grid without requiring its algorithm."""
    from dmm import _native

    if not isinstance(case, dict) or set(case) != {"layout", "P", "c"}:
        raise ValueError("each case requires exactly these fields: layout, P, c")
    if not isinstance(case["layout"], str) or case["layout"] not in LAYOUT_MODULES:
        raise ValueError(f"case layout must be one of {tuple(LAYOUT_MODULES)}")
    for name in ("P", "c"):
        if type(case[name]) is not int or case[name] <= 0:
            raise ValueError(f"case {name} must be a positive integer")
    if case["layout"] != "2.5d" and case["c"] != 1:
        raise ValueError(f"{case['layout']} requires c = 1")
    if case["layout"] != "ring":
        _native.grid_size(case["P"], case["c"])


def validate_cases(cases) -> None:
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    seen = set()
    for case in cases:
        validate_case(case)
        key = (case["layout"], case["P"], case["c"])
        if key in seen:
            raise ValueError("cases must be unique")
        seen.add(key)


@dataclass
class DatasetConfig:
    """Specify reproducible matrix generation in a new directory."""

    data_dir: Path
    seed: int
    groups: list[int]
    sizes: list[int]
    value_range: list[float]

    def validate(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        _dimensions(self.groups, self.sizes)
        validate_value_range(self.value_range)


@dataclass
class BenchmarkConfig:
    """Run each layout/P/c case across shared modes, groups and sizes."""

    data_dir: Path
    output: Path
    cases: list[dict]
    modes: dict[str, dict]
    groups: list[int]
    sizes: list[int]
    repeats: int = 15
    warmups: int = 5

    def validate(self) -> None:
        if type(self.repeats) is not int or self.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if type(self.warmups) is not int or self.warmups < 0:
            raise ValueError("warmups must be a non-negative integer")
        validate_cases(self.cases)
        _dimensions(self.groups, self.sizes)
        if any(left >= right for left, right in zip(self.sizes, self.sizes[1:])):
            raise ValueError("sizes must be in ascending order")
        if not isinstance(self.modes, dict) or not self.modes:
            raise ValueError("modes must be a non-empty mapping")
        for mode, options in self.modes.items():
            if mode not in MODES:
                raise ValueError(f"mode must be one of {MODES}")
            if not isinstance(options, dict):
                raise ValueError(f"{mode} options must be a mapping; use {{}} for no options")
            expected = set() if mode == "sequential" else {"workers"}
            if set(options) != expected:
                raise ValueError(f"{mode} requires exactly these fields: {sorted(expected)}")
            if mode == "threaded":
                limit = min(6, os.cpu_count() or 1)
                _selection("threaded.workers", options["workers"], tuple(range(1, limit + 1)))


@dataclass
class ModelConfig:
    """Execute each configuration once; sweep saved counts without rerunning matrices."""
    data_dir: Path
    output_dir: Path
    cases: list[dict]
    sizes: list[int]
    group: int = 0
    workers: int = 0
    gammas: list[float] = field(default_factory=lambda: [1, 10, 100, 1000])
    lambdas: list[float] = field(default_factory=lambda: [100, 10000])
    budgets: list[float] = field(default_factory=lambda: [8, 32])

    def validate(self) -> None:
        validate_cases(self.cases)
        _dimensions([self.group], self.sizes)
        if self.sizes != sorted(self.sizes):
            raise ValueError("sizes must be in ascending order")
        if type(self.workers) is not int or not 0 <= self.workers <= (os.cpu_count() or 1):
            raise ValueError("workers must be between zero and available logical CPU count")
        for name, minimum in (("gammas", 0), ("lambdas", 0), ("budgets", 0)):
            values = getattr(self, name)
            if (not isinstance(values, list) or not values
                    or any(type(v) not in (int, float) or not isfinite(v) or v < minimum
                           or (name == "budgets" and v == 0) for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{name} must contain unique finite non-negative values (budgets positive)")


def _load_config(path, config_type, path_fields):
    path = Path(path).resolve()
    try:
        with path.open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error
    config_fields = fields(config_type)
    expected = {field.name for field in config_fields}
    required = {field.name for field in config_fields if field.default is MISSING and field.default_factory is MISSING}
    if not isinstance(values, dict) or not required <= set(values) <= expected:
        raise ValueError(f"Configuration requires fields: {sorted(required)}; optional fields: {sorted(expected - required)}")
    for key in path_fields:
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} must be a non-empty path string")
        values[key] = path.parent / values[key]
    config = config_type(**values)
    config.validate()
    return config


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load benchmark settings, resolving paths relative to the YAML file."""
    return _load_config(path, BenchmarkConfig, ("data_dir", "output"))


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load generation settings, resolving paths relative to the YAML file."""
    return _load_config(path, DatasetConfig, ("data_dir",))


def load_model_config(path: str | Path) -> ModelConfig:
    """Load a trace/model experiment with paths relative to its YAML file."""
    return _load_config(path, ModelConfig, ("data_dir", "output_dir"))
