"""Load and validate YAML dataset and benchmark configurations."""

from dataclasses import MISSING, dataclass, fields
import os
from pathlib import Path

import yaml
import numpy as np

ALGORITHMS = ("cannon", "fox", "strassen", "cw")
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
    """Describe the Cartesian product of algorithms, modes, groups and sizes."""

    data_dir: Path
    output: Path
    algorithms: list[str]
    modes: dict[str, dict]
    groups: list[int]
    sizes: list[int]
    repeats: int = 1
    warmups: int = 5
    P: int = 16
    c: int = 1

    def validate(self) -> None:
        if type(self.repeats) is not int or self.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if type(self.warmups) is not int or self.warmups < 0:
            raise ValueError("warmups must be a non-negative integer")
        for name in ("P", "c"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _selection("algorithms", self.algorithms, ALGORITHMS)
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
