import csv
import json
from pathlib import Path
import shutil
from statistics import fmean
import subprocess
import sys

import dmm.cli as cli
from dmm.config import DatasetConfig
from dmm.datasets import generate_dataset, generate_matrix, load_pair, load_reference
import numpy as np
import pytest
import yaml


@pytest.mark.parametrize("arguments,filename", [
    (["generate"], "configs/dataset.yaml"),
    (["generate", "--config", "custom.yaml"], "custom.yaml"),
])
def test_generate_arguments(tmp_path, monkeypatch, capsys, arguments, filename):
    (tmp_path / filename).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / filename).write_text(yaml.safe_dump({
        "data_dir": "custom", "seed": 0, "groups": [4], "sizes": [12], "value_range": [-2, 7],
    }))
    calls = []
    monkeypatch.setattr(cli, "generate_dataset", lambda config: calls.append(config))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["dmm", *arguments])
    cli.main()
    root = (tmp_path / filename).parent
    assert calls == [DatasetConfig(root / "custom", 0, [4], [12], [-2, 7])]
    assert str(root / "custom") in capsys.readouterr().out


@pytest.mark.parametrize("arguments,changes,exit_code,message", [
    ([], {}, 2, "required"),
    (["unknown"], {}, 2, "invalid choice"),
    (["generate", "--config", "missing.yaml"], {}, 1, "missing.yaml"),
    (["generate"], {"seed": "invalid"}, 1, "non-negative"),
    (["generate"], {"seed": -1}, 1, "non-negative"),
    (["generate"], {"groups": []}, 1, "non-empty"),
    (["generate"], {}, 1, "File exists"),
])
def test_errors_leave_existing_data_untouched(tmp_path, monkeypatch, capsys, arguments, changes, exit_code, message):
    monkeypatch.chdir(tmp_path)
    values = dict(data_dir="data", seed=5206, groups=[0], sizes=[16], value_range=[0, 255])
    values.update(changes)
    (tmp_path / "configs").mkdir()
    values["data_dir"] = "../data"
    (tmp_path / "configs" / "dataset.yaml").write_text(yaml.safe_dump(values))
    (tmp_path / "data").mkdir()
    sentinel = tmp_path / "data" / "keep.txt"
    sentinel.write_text("keep this file")
    monkeypatch.setattr(sys, "argv", ["dmm", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == exit_code
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err
    assert "Generated" not in captured.out
    assert list((tmp_path / "data").iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep this file"


def test_installed_command_generates_dataset_outside_project(tmp_path):
    directory = tmp_path / "experiment"
    directory.mkdir()
    config_path = directory / "dataset.yaml"
    config_path.write_text(yaml.safe_dump({
        "data_dir": "custom", "seed": 123, "groups": [5, 2],
        "sizes": [12, 16], "value_range": [-4, 8],
    }))
    executable = shutil.which("dmm", path=str(Path(sys.executable).parent))
    assert executable is not None
    result = subprocess.run(
        [executable, "generate", "--config", str(config_path)],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    root = directory / "custom"
    assert f"Generated dataset in {root}" in result.stdout
    assert len(list(root.rglob("*.npy"))) == 12
    assert json.loads((root / "manifest.json").read_text())["seed"] == 123
    for group in (5, 2):
        for size in (12, 16):
            rng = np.random.default_rng([123, group, size])
            for saved in load_pair(root, group, size):
                np.testing.assert_array_equal(saved, generate_matrix(size, (-4, 8), rng))
            a, b = load_pair(root, group, size)
            np.testing.assert_array_equal(load_reference(root, group, size), a.astype(np.float64) @ b.astype(np.float64))
    bench_path = directory / "benchmark.yaml"
    bench_path.write_text(yaml.safe_dump({
        "data_dir": "custom", "output": "results.csv", "cases": [{"layout": "cannon", "P": 16, "c": 1}],
        "modes": {"sequential": {}}, "groups": [5, 2], "sizes": [12, 16],
    }))
    result = subprocess.run(
        [executable, "bench", "--config", str(bench_path)],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "Saved 4 benchmark result(s)" in result.stdout
    with (directory / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["group"], r["size"]) for r in rows] == [("5", "12"), ("5", "16"), ("2", "12"), ("2", "16")]


@pytest.mark.parametrize("arguments,filename", [
    (["bench"], "configs/benchmark.yaml"),
    (["bench", "--config", "custom.yaml"], "custom.yaml"),
])
def test_bench_arguments(tmp_path, monkeypatch, capsys, arguments, filename):
    config_path = tmp_path / filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump({
        "data_dir": "data", "output": "out.csv", "cases": [{"layout": "cannon", "P": 16, "c": 1}],
        "modes": {"sequential": {}}, "groups": [0, 1], "sizes": [16, 32],
    }))
    calls = []
    monkeypatch.setattr(cli, "run_benchmark", lambda config: calls.append(config) or 4)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["dmm", *arguments])
    cli.main()
    config, = calls
    assert config.data_dir == config_path.parent / "data"
    assert config.output == config_path.parent / "out.csv"
    assert config.groups == [0, 1] and config.sizes == [16, 32]
    assert f"Saved 4 benchmark result(s) to {config.output}" in capsys.readouterr().out


@pytest.mark.parametrize("content,message", [
    (None, "benchmark.yaml"),
    ("modes: [", "Invalid YAML"),
    ("{}", "fields"),
    (yaml.safe_dump({
        "data_dir": "missing", "output": "out.csv", "cases": [{"layout": "cannon", "P": 16, "c": 1}],
        "modes": {"sequential": {}}, "groups": [0], "sizes": [16],
    }), "manifest.json"),
])
def test_bench_errors_are_reported_without_tracebacks(tmp_path, monkeypatch, capsys, content, message):
    monkeypatch.chdir(tmp_path)
    if content is not None:
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "benchmark.yaml").write_text(content)
    monkeypatch.setattr(sys, "argv", ["dmm", "bench"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err
    assert "Saved" not in captured.out


@pytest.mark.parametrize("repeats", [None, 3])
@pytest.mark.parametrize("node_counts", [(16,), (16, 64, 256)])
def test_installed_bench_command_outside_project(tmp_path, repeats, node_counts):
    root = tmp_path / "experiment"
    generate_dataset(DatasetConfig(root / "data", 5206, [0], [16], [0, 255]))
    config_path = root / "run.yaml"
    settings = {
        "data_dir": "data", "output": "out.csv", "cases": [dict(layout="cannon", P=P, c=1) for P in node_counts],
        "modes": {"sequential": {}, "threaded": {"workers": [1, 2]}},
        "groups": [0], "sizes": [16],
    }
    if repeats is not None:
        settings["repeats"] = repeats
    config_path.write_text(yaml.safe_dump(settings, sort_keys=False))
    repeat_count = repeats or 1
    executable = shutil.which("dmm", path=str(Path(sys.executable).parent))
    assert executable is not None
    result = subprocess.run([executable, "bench", "--config", str(config_path)],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert f"Saved {3 * len(node_counts) * repeat_count} benchmark result(s)" in result.stdout
    metadata = json.loads((root / "out.json").read_text())
    assert metadata["benchmark"]["output"] == str(root / "out.csv")
    assert metadata["benchmark"]["repeats"] == repeat_count
    assert metadata["dataset"]["seed"] == 5206
    assert metadata["git"]["commit"]
    with (root / "out.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["mode"], r["workers"], int(r["P"]), int(r["repeat"])) for r in rows] == [
        (mode, workers, P, repeat)
        for mode, workers in [("sequential", ""), ("threaded", "1"), ("threaded", "2")]
        for P in node_counts for repeat in range(repeat_count)]
    for row in rows:
        assert row["layout"] == "cannon"
        assert (row["group"], row["pair"], row["size"]) == ("0", "0", "16")
        assert float(row["time_s"]) >= 0
    result = subprocess.run([executable, "analyze", "--input", str(root / "out.csv")],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    analysis_dir = root / "out_analysis"
    assert f"Saved analysis to {analysis_dir}" in result.stdout
    with (analysis_dir / "summary.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 3 * len(node_counts)
    for aggregated in summary:
        baseline = fmean(float(row["time_s"]) for row in rows
                         if row["mode"] == "sequential" and row["P"] == aggregated["P"])
        selected = [float(row["time_s"]) for row in rows
                    if row["mode"] == aggregated["mode"] and row["workers"] == aggregated["workers"]
                    and row["P"] == aggregated["P"] and row["c"] == aggregated["c"]]
        assert aggregated["group_count"] == "1"
        assert aggregated["repeats"] == str(repeat_count)
        assert float(aggregated["mean_time_s"]) == pytest.approx(fmean(selected))
        assert float(aggregated["speedup"]) == pytest.approx(baseline / fmean(selected))
    for filename in ("time.png", "speedup.png"):
        assert (analysis_dir / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_rejects_unsupported_algorithm_sizes_before_creating_results(tmp_path, monkeypatch, capsys):
    generate_dataset(DatasetConfig(tmp_path / "data", 5206, [0], [4, 7], [0, 255]))
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({
        "data_dir": "data", "output": "results/invalid.csv", "cases": [{"layout": "cannon", "P": 16, "c": 1}],
        "modes": {"sequential": {}}, "groups": [0], "sizes": [4, 7],
    }))
    from dmm.algorithms import cannon

    def reject_seven(size, **options):
        if size == 7:
            raise ValueError("test size limit is 4")

    monkeypatch.setattr(cannon, "validate_size", reject_seven)
    monkeypatch.setattr(sys, "argv", ["dmm", "bench", "--config", str(path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "cannon" in captured.err and "size 7" in captured.err and "test size limit" in captured.err
    assert "Traceback" not in captured.err and "Saved" not in captured.out
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("extra,expected_output", [([], None), (["--output-dir", "custom/plots"], Path("custom/plots"))])
def test_analyze_arguments(monkeypatch, capsys, extra, expected_output):
    calls = []

    def analyze(input_path, output_dir):
        calls.append((input_path, output_dir))
        return output_dir or Path("results/run_analysis")

    monkeypatch.setattr(cli, "analyze_results", analyze)
    monkeypatch.setattr(sys, "argv", ["dmm", "analyze", "--input", "results/run.csv", *extra])
    cli.main()
    assert calls == [(Path("results/run.csv"), expected_output)]
    assert f"Saved analysis to {expected_output or Path('results/run_analysis')}" in capsys.readouterr().out


@pytest.mark.parametrize("arguments,content,exit_code,message", [
    (["analyze"], None, 2, "--input"),
    (["analyze", "--input", "missing.csv"], None, 1, "missing.csv"),
    (["analyze", "--input", "run.csv"], "wrong,header\n", 1, "CSV columns must be"),
    (["analyze", "--input", "run.csv"],
     "algorithm,mode,workers,group,pair,size,time_s,max_abs_error,frobenius_relative_error\ncannon,threaded,1,0,0,16,1,0,0\n", 1,
     "Missing sequential baseline"),
])
def test_analyze_errors_are_reported_without_tracebacks(tmp_path, monkeypatch, capsys, arguments, content, exit_code, message):
    monkeypatch.chdir(tmp_path)
    if content is not None:
        (tmp_path / "run.csv").write_text(content)
    monkeypatch.setattr(sys, "argv", ["dmm", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == exit_code
    captured = capsys.readouterr()
    assert message in captured.err and "Traceback" not in captured.err
    assert "Saved analysis" not in captured.out
    assert not (tmp_path / "run_analysis").exists()
