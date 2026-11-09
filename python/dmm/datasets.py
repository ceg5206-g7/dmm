"""Generate and load NumPy datasets with FP32 inputs and FP64 references."""

import json
from pathlib import Path

import numpy as np

from dmm.config import DatasetConfig, validate_value_range


def generate_matrix(size: int, value_range: tuple[float, float] | list[float], rng: np.random.Generator) -> np.ndarray:
    """Return uniformly sampled values rounded to a square FP32 matrix."""
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    validate_value_range(value_range)
    low, high = value_range
    return rng.uniform(low, high, size=(size, size)).astype(np.float32)


def reference_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply the stored FP32 inputs using independent FP64 accumulation."""
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError("A and B must be NumPy arrays")
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1] or A.shape != B.shape:
        raise ValueError("A and B must be non-empty square matrices with matching shapes")
    if A.dtype != np.float32 or B.dtype != np.float32:
        raise TypeError("A and B must have dtype float32")
    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ValueError("A and B must contain finite values")
    return np.matmul(A.astype(np.float64), B.astype(np.float64))


def generate_dataset(config: DatasetConfig) -> None:
    """Write inputs, NumPy reference products and a completion manifest."""
    config.validate()
    root = Path(config.data_dir)
    # shortcut: interrupted generation leaves a partial directory; remove it before retrying.
    root.mkdir(parents=True)
    for group_id in config.groups:
        group_dir = root / f"G{group_id}"
        group_dir.mkdir()
        for size in config.sizes:
            rng = np.random.default_rng([config.seed, group_id, size])
            A = generate_matrix(size, config.value_range, rng)
            B = generate_matrix(size, config.value_range, rng)
            reference = reference_matmul(A, B)
            for name, matrix in (("A", A), ("B", B), ("C_ref", reference)):
                np.save(group_dir / f"{name}_{size}.npy", matrix, allow_pickle=False)

    manifest = {
        "seed": config.seed,
        "numpy_version": np.__version__,
        "groups": config.groups,
        "sizes": config.sizes,
        "value_range": config.value_range,
        "dtype": "float32",
        "reference_dtype": "float64",
        "reference": "numpy.matmul(float64)",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def load_manifest(data_dir: str | Path) -> DatasetConfig:
    """Read the generation settings recorded with a completed dataset."""
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    required = ("seed", "groups", "sizes", "value_range")
    if not isinstance(manifest, dict) or any(key not in manifest for key in (*required, "dtype", "reference_dtype", "reference")):
        raise ValueError("Dataset manifest is incomplete; regenerate the dataset")
    if manifest["dtype"] != "float32":
        raise ValueError("Dataset manifest dtype must be float32; regenerate the dataset")
    if manifest["reference_dtype"] != "float64" or manifest["reference"] != "numpy.matmul(float64)":
        raise ValueError("Dataset reference must use float64 accumulation; regenerate the dataset")
    config = DatasetConfig(data_dir=root, **{key: manifest[key] for key in required})
    config.validate()
    return config


def _pair_config(data_dir: str | Path, group_id: int, size: int) -> DatasetConfig:
    if type(group_id) is not int or group_id < 0:
        raise ValueError("group_id must be a non-negative integer")
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    config = load_manifest(data_dir)
    if group_id not in config.groups or size not in config.sizes:
        raise ValueError("Requested group_id and size must be present in the dataset manifest")
    return config


def _load_matrix(path: Path, size: int, dtype) -> np.ndarray:
    matrix = np.load(path, allow_pickle=False)
    if not isinstance(matrix, np.ndarray):
        matrix.close()
        raise ValueError(f"{path} must contain a NumPy array")
    if matrix.shape != (size, size):
        raise ValueError(f"{path} must have shape {(size, size)}")
    if matrix.dtype != dtype:
        raise ValueError(f"{path} must have dtype {np.dtype(dtype).name}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path} must contain finite values")
    return matrix


def load_pair(data_dir: str | Path, group_id: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load one pair, rejecting unsupported IDs and invalid matrix contents."""
    config = _pair_config(data_dir, group_id, size)

    low, high = np.asarray(config.value_range, dtype=np.float32)
    matrices = []
    for name in ("A", "B"):
        path = Path(data_dir) / f"G{group_id}" / f"{name}_{size}.npy"
        matrix = _load_matrix(path, size, np.float32)
        if matrix.min() < low or matrix.max() > high:
            raise ValueError(f"{path} must contain values between {low} and {high}")
        matrices.append(matrix)
    return matrices[0], matrices[1]


def load_reference(data_dir: str | Path, group_id: int, size: int) -> np.ndarray:
    """Load a saved reference product; its values need not match the input range."""
    _pair_config(data_dir, group_id, size)
    return _load_matrix(Path(data_dir) / f"G{group_id}" / f"C_ref_{size}.npy", size, np.float64)
