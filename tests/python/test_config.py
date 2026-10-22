from pathlib import Path

import pytest
import yaml

from dmm.config import load_benchmark_config, load_dataset_config


@pytest.fixture
def values():
    return {
        "data_dir": "data", "output": "results/run.csv",
        "cases": [{"layout": "cannon", "P": 16, "c": 1}, {"layout": "ring", "P": 16, "c": 1}],
        "modes": {"sequential": {}, "threaded": {"workers": [1]}},
        "groups": [0, 2], "sizes": [16, 64],
    }


def test_paths_are_relative_to_config_not_working_directory(tmp_path, monkeypatch, values):
    directory = tmp_path / "experiment"
    directory.mkdir()
    path = directory / "run.yaml"
    path.write_text(yaml.safe_dump(values))
    monkeypatch.chdir(tmp_path)
    config = load_benchmark_config(path)
    assert config.data_dir == directory / "data"
    assert config.output == directory / "results/run.csv"
    assert config.cases == values["cases"]
    assert config.modes == values["modes"]
    assert config.groups == [0, 2] and config.sizes == [16, 64]


def test_absolute_paths_are_preserved(tmp_path, values):
    values["data_dir"] = str(tmp_path / "absolute-data")
    values["output"] = str(tmp_path / "absolute.csv")
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(values))
    config = load_benchmark_config(path)
    assert config.data_dir == Path(values["data_dir"])
    assert config.output == Path(values["output"])


@pytest.mark.parametrize("content", ["", "[]", "42", "{modes: [", "!!python/object:builtins.object {}"])
def test_invalid_or_unsafe_yaml_is_rejected(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_benchmark_config(path)


@pytest.mark.parametrize("field,value", [
    ("data_dir", None), ("data_dir", 42), ("output", " "), ("output", []),
    ("cases", []), ("groups", [True]), ("sizes", [32, 16]),
    ("modes", {"sequential": {"workers": [1]}}),
    ("modes", {"threaded": {"workers": 1}}),
])
def test_bad_fields_are_rejected(tmp_path, values, field, value):
    values[field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match=field if field != "modes" else "fields|workers"):
        load_benchmark_config(path)


@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_missing_and_unknown_keys_are_rejected(tmp_path, values, change):
    if change == "missing":
        del values["groups"]
    else:
        values["group"] = [0]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="fields"):
        load_benchmark_config(path)


def test_repository_example_loads():
    config = load_benchmark_config(Path(__file__).parents[2] / "configs" / "benchmark.yaml")
    assert config.cases == [{"layout": "cannon", "P": 16, "c": 1}]
    assert config.data_dir.resolve() == Path(__file__).parents[2] / "data"
    assert config.output.resolve() == Path(__file__).parents[2] / "results" / "benchmark.csv"
    assert config.repeats == 3
    assert config.sizes == [16, 32, 64, 128, 256, 512, 1024, 2048]


@pytest.mark.parametrize("repeats", [None, 1, 3])
def test_benchmark_repeats_defaults_to_one_and_accepts_positive_integers(tmp_path, values, repeats):
    if repeats is not None:
        values["repeats"] = repeats
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(values))
    assert load_benchmark_config(path).repeats == (repeats or 1)


@pytest.mark.parametrize("repeats", [0, -1, True, 1.5, "3", None])
def test_invalid_repeats_are_rejected(tmp_path, values, repeats):
    values["repeats"] = repeats
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="repeats must be a positive integer"):
        load_benchmark_config(path)


def test_dataset_config_paths_and_custom_fields(tmp_path, monkeypatch):
    directory = tmp_path / "experiment"
    directory.mkdir()
    path = directory / "dataset.yaml"
    path.write_text(yaml.safe_dump({
        "data_dir": "custom", "seed": 0, "groups": [5, 2],
        "sizes": [12, 7], "value_range": [-8, 400],
    }))
    monkeypatch.chdir(tmp_path)
    config = load_dataset_config(path)
    assert config.data_dir == directory / "custom"
    assert config.seed == 0
    assert config.groups == [5, 2]
    assert config.sizes == [12, 7]
    assert config.value_range == [-8, 400]


@pytest.mark.parametrize("changes,message", [
    ({"seed": True}, "seed"), ({"data_dir": ""}, "data_dir"),
    ({"groups": [0, 0]}, "unique"), ({"sizes": [0]}, "sizes"),
    ({"value_range": [1, 0]}, "value_range"), ({"dtype": "int32"}, "fields"),
    ({"repeats": 3}, "fields"),
])
def test_dataset_yaml_validation(tmp_path, changes, message):
    values = dict(data_dir="data", seed=5206, groups=[0], sizes=[16], value_range=[0, 255])
    values.update(changes)
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match=message):
        load_dataset_config(path)


def test_repository_dataset_example_loads():
    config = load_dataset_config(Path(__file__).parents[2] / "configs" / "dataset.yaml")
    assert config.groups == [0, 1, 2]
    assert config.value_range == [0, 255]
    assert config.data_dir.resolve() == Path(__file__).parents[2] / "data"


@pytest.mark.parametrize("P,c", [(16, 1), (64, 1), (256, 1), (64, 4), (256, 4)])
def test_yaml_loads_logical_nodes_and_replication(tmp_path, values, P, c):
    values["cases"] = [dict(layout="2.5d" if c > 1 else "cannon", P=P, c=c)]
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(values))
    config = load_benchmark_config(path)
    assert (config.cases[0]["P"], config.cases[0]["c"]) == (P, c)


@pytest.mark.parametrize("field,value", [("P", 0), ("P", -1), ("P", True), ("P", 16.0),
                                         ("c", 0), ("c", -1), ("c", True), ("c", "4")])
def test_yaml_rejects_invalid_logical_node_and_replication_types(tmp_path, values, field, value):
    values["cases"][0][field] = value
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match=field):
        load_benchmark_config(path)


@pytest.mark.parametrize("cases", [None, {}, [], "cannon", [None], ["cannon"],
    [{"layout": "cannon", "P": 16}],
    [{"layout": "cannon", "P": 16, "c": 1, "extra": 0}],
    [{"layout": ["cannon"], "P": 16, "c": 1}],
    [{"layout": "fox", "P": 16, "c": 1}],
    [{"layout": "cannon", "P": 3, "c": 1}],
    [{"layout": "ring", "P": 16, "c": 4}],
    [{"layout": "2.5d", "P": 16, "c": 4}],
    [{"layout": "cannon", "P": 16, "c": 1}] * 2,
])
def test_invalid_case_lists_are_rejected(tmp_path, values, cases):
    values["cases"] = cases
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError):
        load_benchmark_config(path)


@pytest.mark.parametrize("field,value", [("algorithms", ["cannon"]), ("P", 64), ("c", 4)])
def test_old_global_layout_options_are_rejected(tmp_path, values, field, value):
    values[field] = value
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="fields"):
        load_benchmark_config(path)


def test_ring_case_does_not_require_square_node_count(tmp_path, values):
    values["cases"] = [dict(layout="ring", P=3, c=1)]
    path = tmp_path / "ring.yaml"
    path.write_text(yaml.safe_dump(values))
    assert load_benchmark_config(path).cases == values["cases"]
