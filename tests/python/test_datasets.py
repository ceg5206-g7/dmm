from dataclasses import replace
import json
from pathlib import Path

from dmm.config import DatasetConfig, load_dataset_config

import dmm.datasets as datasets
import numpy as np
import pytest


DEFAULT = load_dataset_config(Path(__file__).parents[2] / "configs" / "dataset.yaml")
SAMPLE_SIZES = [16, 32, 64]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("datasets") / "data"
    datasets.generate_dataset(replace(DEFAULT, data_dir=root, sizes=SAMPLE_SIZES))
    return root


def test_generate_matrix_has_reproducible_fp32_fractional_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng([5206, 0, 16])
    a = datasets.generate_matrix(16, DEFAULT.value_range, rng)
    b = datasets.generate_matrix(16, DEFAULT.value_range, rng)
    for matrix in (a, b):
        assert matrix.shape == (16, 16)
        assert matrix.dtype == np.float32
        assert 0 <= matrix.min() <= matrix.max() <= 255
    assert a[0, :4].tolist() == [244.48135375976562, 164.74769592285156, 137.889892578125, 174.1058807373047]
    assert b[0, :4].tolist() == [87.34440612792969, 188.90977478027344, 182.7567901611328, 83.64093780517578]
    assert np.any(a != np.floor(a))
    assert list(tmp_path.iterdir()) == []


def test_dataset_layout_and_manifest(dataset):
    expected = {"manifest.json"} | {
        f"G{group_id}/{name}_{size}.npy"
        for group_id in DEFAULT.groups
        for size in SAMPLE_SIZES
        for name in ("A", "B", "C_ref")
    }
    assert {str(path.relative_to(dataset)) for path in dataset.rglob("*") if path.is_file()} == expected
    assert json.loads((dataset / "manifest.json").read_text()) == {
        "seed": 5206,
        "numpy_version": np.__version__,
        "groups": [0, 1, 2],
        "sizes": SAMPLE_SIZES,
        "dtype": "float32",
        "reference_dtype": "float64",
        "value_range": [0, 255],
        "reference": "numpy.matmul(float64)",
    }


@pytest.mark.parametrize("group_id", DEFAULT.groups)
@pytest.mark.parametrize("size", SAMPLE_SIZES)
def test_load_every_pair(dataset, group_id, size):
    a, b = datasets.load_pair(dataset, group_id, size)
    for matrix in (a, b):
        assert matrix.shape == (size, size)
        assert matrix.dtype == np.float32
        assert 0 <= matrix.min() <= matrix.max() <= 255
    assert not np.array_equal(a, b)
    reference = datasets.load_reference(dataset, group_id, size)
    assert reference.dtype == np.float64
    np.testing.assert_array_equal(reference, a.astype(np.float64) @ b.astype(np.float64))


def test_reproducibility_is_independent_of_order_and_subset(dataset, tmp_path):
    config = replace(DEFAULT, data_dir=tmp_path / "regenerated", groups=[2, 0], sizes=[32, 16])
    datasets.generate_dataset(config)
    for group_id in config.groups:
        for size in config.sizes:
            for original, repeated in zip(
                datasets.load_pair(dataset, group_id, size),
                datasets.load_pair(config.data_dir, group_id, size),
            ):
                np.testing.assert_array_equal(original, repeated)
            np.testing.assert_array_equal(
                datasets.load_reference(dataset, group_id, size),
                datasets.load_reference(config.data_dir, group_id, size),
            )


def test_seed_and_group_produce_different_inputs(dataset, tmp_path):
    config = replace(DEFAULT, data_dir=tmp_path / "other_seed", seed=5202, groups=[0, 1], sizes=[16])
    datasets.generate_dataset(config)
    for original, changed_seed, changed_group in zip(
        datasets.load_pair(dataset, 0, 16),
        datasets.load_pair(config.data_dir, 0, 16),
        datasets.load_pair(dataset, 1, 16),
    ):
        assert not np.array_equal(original, changed_seed)
        assert not np.array_equal(original, changed_group)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "5206"])
def test_invalid_seed_does_not_create_directory(tmp_path, seed):
    root = tmp_path / "data"
    with pytest.raises(ValueError, match="seed"):
        datasets.generate_dataset(replace(DEFAULT, data_dir=root, seed=seed))
    assert not root.exists()


@pytest.mark.parametrize("size", [0, -1, True, 16.0, "16"])
def test_generate_matrix_rejects_invalid_size(size):
    with pytest.raises(ValueError, match="size"):
        datasets.generate_matrix(size, DEFAULT.value_range, np.random.default_rng(5206))


@pytest.mark.parametrize("value_range", [(2, 1), (np.nan, 3), (0, np.inf), (0, True), None, [0], [0, 1, 2]])
def test_generate_matrix_rejects_invalid_range(value_range):
    with pytest.raises(ValueError, match="value_range"):
        datasets.generate_matrix(7, value_range, np.random.default_rng(5206))


@pytest.mark.parametrize("value_range", [(-3, 3), (9, 9), (0.1, 0.9)])
def test_generate_matrix_supports_custom_sizes_and_ranges(value_range):
    matrix = datasets.generate_matrix(7, value_range, np.random.default_rng(5206))
    assert matrix.shape == (7, 7)
    assert matrix.dtype == np.float32
    assert value_range[0] <= matrix.min() <= matrix.max() <= value_range[1]


@pytest.mark.parametrize("value", [-float(np.finfo(np.float32).max), float(np.finfo(np.float32).max)])
def test_generate_matrix_supports_fp32_endpoints(value):
    matrix = datasets.generate_matrix(2, (value, value), np.random.default_rng(5206))
    assert matrix.dtype == np.float32
    assert np.all(matrix == value)


def test_existing_directory_is_untouched(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep this file")
    with pytest.raises(FileExistsError):
        datasets.generate_dataset(replace(DEFAULT, data_dir=tmp_path))
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep this file"


def test_write_failure_is_reported_without_a_completion_manifest(tmp_path, monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(datasets.np, "save", fail_save)
    root = tmp_path / "data"
    with pytest.raises(OSError, match="disk full"):
        datasets.generate_dataset(replace(DEFAULT, data_dir=root))
    assert not (root / "manifest.json").exists()


@pytest.fixture
def small_dataset(tmp_path):
    config = DatasetConfig(tmp_path / "data", 5206, [0], [16], [0, 255])
    datasets.generate_dataset(config)
    return config.data_dir


@pytest.mark.parametrize("group_id,size", [(-1, 16), (3, 16), ("0", 16), (True, 16), (0, 8), (0, 16.0)])
def test_invalid_pair_request(small_dataset, group_id, size):
    with pytest.raises(ValueError):
        datasets.load_pair(small_dataset, group_id, size)


@pytest.mark.parametrize("missing", ["A", "B"])
def test_missing_matrix(small_dataset, missing):
    (small_dataset / "G0" / f"{missing}_16.npy").unlink()
    with pytest.raises(FileNotFoundError, match=f"{missing}_16"):
        datasets.load_pair(small_dataset, 0, 16)


@pytest.mark.parametrize("name", ["A", "B"])
@pytest.mark.parametrize("invalid,match", [
    (np.zeros((16, 15), dtype=np.float32), "shape"),
    (np.zeros((16, 16), dtype=np.int64), "dtype"),
    (np.full((16, 16), np.nan, dtype=np.float32), "finite"),
    (np.full((16, 16), np.inf, dtype=np.float32), "finite"),
    (np.full((16, 16), -1, dtype=np.float32), "between"),
    (np.full((16, 16), 256, dtype=np.float32), "between"),
    (np.full((16, 16), "unsafe", dtype=object), "Object arrays"),
])
def test_invalid_matrix(small_dataset, name, invalid, match):
    np.save(small_dataset / "G0" / f"{name}_16.npy", invalid)
    with pytest.raises(ValueError, match=match):
        datasets.load_pair(small_dataset, 0, 16)


def test_corrupt_file(small_dataset):
    (small_dataset / "G0" / "A_16.npy").write_bytes(b"not a NumPy file")
    with pytest.raises(ValueError):
        datasets.load_pair(small_dataset, 0, 16)


def test_archive_is_rejected(small_dataset):
    with (small_dataset / "G0" / "A_16.npy").open("wb") as handle:
        np.savez(handle, data=np.zeros((16, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="NumPy array"):
        datasets.load_pair(small_dataset, 0, 16)


@pytest.mark.parametrize("value_range", [[-9, -2], [300, 400], [7, 7], [0.1, 0.1]])
def test_custom_dataset_settings_round_trip(tmp_path, value_range):
    config = DatasetConfig(tmp_path / "custom", 0, [5, 7], [7, 12], value_range)
    datasets.generate_dataset(config)
    assert datasets.load_manifest(config.data_dir) == config
    for group in config.groups:
        for size in config.sizes:
            for matrix in datasets.load_pair(config.data_dir, group, size):
                assert matrix.shape == (size, size)
                assert value_range[0] <= matrix.min() <= matrix.max() <= value_range[1]
    invalid = np.full((7, 7), value_range[1] + 1, dtype=np.float32)
    np.save(config.data_dir / "G5" / "A_7.npy", invalid)
    with pytest.raises(ValueError, match="between"):
        datasets.load_pair(config.data_dir, 5, 7)


@pytest.mark.parametrize("changes", [
    {"groups": []}, {"groups": [0, 0]}, {"groups": [-1]}, {"groups": [True]},
    {"sizes": []}, {"sizes": [16, 16]}, {"sizes": [0]}, {"sizes": [4.0]},
    {"value_range": [3, 2]}, {"value_range": [0]}, {"value_range": [0, True]},
    {"value_range": [-1e39, 0]}, {"value_range": [0, 1e39]},
    {"value_range": [np.nan, 1]}, {"value_range": [0, np.inf]},
])
def test_invalid_generation_settings_do_not_create_directory(tmp_path, changes):
    config = replace(DEFAULT, data_dir=tmp_path / "invalid", **changes)
    with pytest.raises(ValueError):
        datasets.generate_dataset(config)
    assert not config.data_dir.exists()


@pytest.mark.parametrize("contents", ["[]", "{}", "not JSON", '{"dtype": "int64"}'])
def test_invalid_manifest_is_rejected(small_dataset, contents):
    (small_dataset / "manifest.json").write_text(contents)
    with pytest.raises(ValueError):
        datasets.load_pair(small_dataset, 0, 16)


@pytest.mark.parametrize("dtype", ["int64", "uint8"])
def test_incorrect_manifest_dtype_is_rejected(small_dataset, dtype):
    path = small_dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["dtype"] = dtype
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dtype"):
        datasets.load_pair(small_dataset, 0, 16)


def test_generation_computes_each_reference_once(tmp_path, monkeypatch):
    calls = []
    multiply = datasets.reference_matmul

    def tracked(a, b):
        calls.append(a.shape[0])
        return multiply(a, b)

    monkeypatch.setattr(datasets, "reference_matmul", tracked)
    config = DatasetConfig(tmp_path / "data", 5206, [0, 1], [4, 8], [0, 255])
    datasets.generate_dataset(config)
    assert calls == [4, 8, 4, 8]
    assert len(list(config.data_dir.rglob("*.npy"))) == 12


def test_reference_write_failure_leaves_no_completion_manifest(tmp_path, monkeypatch):
    save = np.save

    def fail_reference(path, matrix, **kwargs):
        if path.name.startswith("C_ref_"):
            raise OSError("reference write failed")
        save(path, matrix, **kwargs)

    monkeypatch.setattr(datasets.np, "save", fail_reference)
    root = tmp_path / "data"
    with pytest.raises(OSError, match="reference write failed"):
        datasets.generate_dataset(DatasetConfig(root, 5206, [0], [4], [0, 255]))
    assert (root / "G0" / "A_4.npy").exists()
    assert not (root / "manifest.json").exists()


@pytest.mark.parametrize("field,value", [("reference_dtype", "float32"), ("reference", "numpy.matmul")])
def test_incorrect_reference_metadata_is_rejected(small_dataset, field, value):
    path = small_dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reference"):
        datasets.load_reference(small_dataset, 0, 16)


def test_missing_reference_is_reported(small_dataset):
    (small_dataset / "G0" / "C_ref_16.npy").unlink()
    with pytest.raises(FileNotFoundError, match="C_ref_16"):
        datasets.load_reference(small_dataset, 0, 16)


@pytest.mark.parametrize("invalid,message", [
    (np.zeros((16, 15), dtype=np.float32), "shape"),
    (np.zeros((16, 16), dtype=np.float32), "dtype"),
    (np.zeros((16, 16), dtype=np.int64), "dtype"),
    (np.full((16, 16), np.nan, dtype=np.float64), "finite"),
    (np.full((16, 16), np.inf, dtype=np.float64), "finite"),
    (np.full((16, 16), 0, dtype=object), "Object arrays"),
])
def test_invalid_reference_is_rejected(small_dataset, invalid, message):
    np.save(small_dataset / "G0" / "C_ref_16.npy", invalid)
    with pytest.raises(ValueError, match=message):
        datasets.load_reference(small_dataset, 0, 16)


@pytest.mark.parametrize("group,size", [(True, 16), (0, 16.0), (5, 16), (0, 8)])
def test_invalid_reference_selection(small_dataset, group, size):
    with pytest.raises(ValueError):
        datasets.load_reference(small_dataset, group, size)
