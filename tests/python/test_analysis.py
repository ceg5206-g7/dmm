import csv
from pathlib import Path

import dmm.analysis as analysis
from dmm.benchmark import CSV_FIELDS
import pytest


def write_results(path, rows, fields=CSV_FIELDS):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def rows():
    results = []
    for algorithm, algorithm_factor in (("cannon", 1), ("fox", 2)):
        for pair, size, size_factor in ((0, 16, 1), (1, 32, 3)):
            for mode, workers, times in (("sequential", None, (2, 6)),
                                           ("threaded", 1, (4, 8)),
                                           ("threaded", 2, (1, 3))):
                for group, elapsed in zip((2, 5), times):
                    results.append(dict(algorithm=algorithm, mode=mode, workers=workers,
                                        group=group, pair=pair, size=size, repeat=0,
                                        time_s=elapsed * algorithm_factor * size_factor,
                                        max_abs_error=0, frobenius_relative_error=0))
    return results


def test_summary_averages_groups_and_uses_each_algorithm_and_size_baseline(tmp_path, rows):
    path = write_results(tmp_path / "run.csv", reversed(rows))
    summary = analysis.summarize_results(str(path))
    expected = []
    for algorithm, factor in (("cannon", 1), ("fox", 2)):
        for mode, workers, average, speedup in (("sequential", None, 4, 1),
                                                 ("threaded", 1, 6, 2 / 3),
                                                 ("threaded", 2, 2, 2)):
            for size, size_factor in ((16, 1), (32, 3)):
                expected.append(dict(algorithm=algorithm, mode=mode, workers=workers,
                                     size=size, group_count=2, repeats=1, mean_time_s=average * factor * size_factor,
                                     speedup=speedup))
    assert summary == expected


@pytest.mark.parametrize("changes", [{"algorithm": "fox"}, {"size": 32}])
def test_baseline_must_match_algorithm_and_size(tmp_path, changes):
    serial = dict(algorithm="cannon", mode="sequential", workers=None, group=0, pair=0, size=16, repeat=0, time_s=4)
    parallel = dict(serial, mode="threaded", workers=1, **changes)
    path = write_results(tmp_path / "run.csv", [serial, parallel])
    with pytest.raises(ValueError, match="Missing sequential baseline"):
        analysis.analyze_results(path)
    assert not (tmp_path / "run_analysis").exists()


@pytest.mark.parametrize("group", [None, 8])
def test_partial_or_different_groups_are_rejected(tmp_path, rows, group):
    selected = next(row for row in rows if row["mode"] == "threaded")
    if group is None:
        rows.remove(selected)
    else:
        selected["group"] = group
    with pytest.raises(ValueError, match="Group mismatch.*cannon.*16"):
        analysis.summarize_results(write_results(tmp_path / "run.csv", rows))


def test_duplicate_results_are_rejected(tmp_path, rows):
    rows.append(rows[0])
    with pytest.raises(ValueError, match="row 26: duplicate result"):
        analysis.summarize_results(write_results(tmp_path / "run.csv", rows))


@pytest.mark.parametrize("changes,message", [
    ({"algorithm": "unknown"}, "unknown algorithm"),
    ({"mode": "unknown"}, "unknown algorithm or mode"),
    ({"group": -1}, "non-negative"),
    ({"pair": -1}, "non-negative"),
    ({"size": 0}, "positive"),
    ({"size": "16.5"}, "invalid literal"),
    ({"group": ""}, "invalid literal"),
    ({"repeat": -1}, "repeat must be a non-negative integer"),
    ({"repeat": ""}, "invalid literal"),
    ({"repeat": "1.5"}, "invalid literal"),
    ({"repeat": "True"}, "invalid literal"),
    ({"workers": 1}, "sequential workers must be empty"),
    ({"mode": "threaded", "workers": 0}, "workers must be positive"),
    ({"mode": "threaded", "workers": -1}, "workers must be positive"),
    ({"mode": "threaded", "workers": ""}, "invalid literal"),
    ({"mode": "threaded", "workers": "1.5"}, "invalid literal"),
    ({"time_s": "invalid"}, "could not convert"),
    ({"time_s": 0}, "positive and finite"),
    ({"time_s": -1}, "positive and finite"),
    ({"time_s": "nan"}, "positive and finite"),
    ({"time_s": "inf"}, "positive and finite"),
    ({"time_s": "-inf"}, "positive and finite"),
])
def test_invalid_values_report_the_csv_row_before_creating_output(tmp_path, rows, changes, message):
    rows[0].update(changes)
    path = write_results(tmp_path / "run.csv", rows)
    with pytest.raises(ValueError, match=f"row 2: .*{message}"):
        analysis.analyze_results(path, tmp_path / "plots")
    assert not (tmp_path / "plots").exists()


@pytest.mark.parametrize("fields", [(), CSV_FIELDS[:-1], (*CSV_FIELDS, "extra"), (*CSV_FIELDS, "size")])
def test_invalid_headers_are_rejected(tmp_path, fields):
    path = write_results(tmp_path / "run.csv", [], fields)
    with pytest.raises(ValueError, match="CSV columns must be"):
        analysis.summarize_results(path)


def test_empty_results_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="contains no results"):
        analysis.summarize_results(write_results(tmp_path / "run.csv", []))


@pytest.mark.parametrize("line,message", [
    ("cannon,sequential,,0,0,16", "row 2: row must contain"),
    ("cannon,sequential,,0,0,16,0,1,extra", "row 2: row must contain"),
    ('cannon,sequential,,0,0,16,"1', "Invalid benchmark CSV"),
])
def test_malformed_rows_are_rejected(tmp_path, line, message):
    path = tmp_path / "run.csv"
    path.write_text(",".join(CSV_FIELDS) + "\n" + line + "\n")
    with pytest.raises(ValueError, match=message):
        analysis.summarize_results(path)


def test_unrepresentable_speedup_is_rejected(tmp_path):
    serial = dict(algorithm="cannon", mode="sequential", workers=None, group=0, pair=0, size=16, repeat=0, time_s=1e308)
    parallel = dict(serial, mode="threaded", workers=1, time_s=1e-308)
    with pytest.raises(ValueError, match="Speedup must be positive and finite"):
        analysis.summarize_results(write_results(tmp_path / "run.csv", [serial, parallel]))


def test_repeated_measurements_are_averaged_within_each_group(tmp_path, rows):
    repeated = [dict(row, repeat=repeat, time_s=row["time_s"] * factor)
                for row in rows for repeat, factor in enumerate((0.5, 1, 1.5))]
    summary = analysis.summarize_results(write_results(tmp_path / "run.csv", reversed(repeated)))
    assert len(summary) == 12
    assert all(row["group_count"] == 2 and row["repeats"] == 3 for row in summary)
    selected = [row for row in summary if row["algorithm"] == "cannon" and row["size"] == 16]
    assert [row["mean_time_s"] for row in selected] == [4, 6, 2]
    assert [row["speedup"] for row in selected] == pytest.approx([1, 2 / 3, 2])


@pytest.mark.parametrize("repeat_ids", [[1], [0, 2], [0, 1, 2**60]])
def test_non_contiguous_repeat_ids_are_rejected(tmp_path, rows, repeat_ids):
    repeated = [dict(row, repeat=repeat) for row in rows for repeat in repeat_ids]
    with pytest.raises(ValueError, match="Incomplete or inconsistent repeats"):
        analysis.summarize_results(write_results(tmp_path / "run.csv", repeated))


@pytest.mark.parametrize("whole_setting", [False, True])
def test_missing_repeats_in_a_group_or_execution_setting_are_rejected(tmp_path, rows, whole_setting):
    repeated = [dict(row, repeat=repeat) for row in rows for repeat in range(3)
                if not (row["mode"] == "threaded" and row["workers"] == 2 and repeat == 2
                        and (whole_setting or row["group"] == 2))]
    path = write_results(tmp_path / "run.csv", repeated)
    with pytest.raises(ValueError, match="Incomplete or inconsistent repeats.*threaded"):
        analysis.analyze_results(path)
    assert not (tmp_path / "run_analysis").exists()


def test_legacy_csv_without_repeat_column_is_treated_as_one_measurement(tmp_path, rows):
    current = analysis.summarize_results(write_results(tmp_path / "current.csv", rows))
    fields = tuple(field for field in CSV_FIELDS if field != "repeat")
    legacy_rows = [{field: row[field] for field in fields} for row in rows]
    legacy = analysis.summarize_results(write_results(tmp_path / "legacy.csv", legacy_rows, fields))
    assert legacy == current
    assert all(row["repeats"] == 1 for row in legacy)


@pytest.mark.parametrize("custom_output", [False, True])
def test_analysis_writes_summary_and_real_pngs_without_changing_input(tmp_path, rows, monkeypatch, custom_output):
    from matplotlib.figure import Figure
    from matplotlib.image import imread

    figures = {}
    original_save = Figure.savefig

    def save(figure, path, **kwargs):
        figures[Path(path).name] = figure
        return original_save(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    path = write_results(tmp_path / "run.csv", rows)
    original_input = path.read_bytes()
    destination = tmp_path / "custom" / "plots" if custom_output else None
    output = analysis.analyze_results(str(path), str(destination) if destination else None)
    assert output == (destination or tmp_path / "run_analysis")
    assert path.read_bytes() == original_input
    assert {item.name for item in output.iterdir()} == {"summary.csv", "time.png", "speedup.png"}
    with (output / "summary.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(analysis.SUMMARY_FIELDS)
        saved = list(reader)
    expected = analysis.summarize_results(path)
    assert saved == [{name: "" if row[name] is None else str(row[name]) for name in analysis.SUMMARY_FIELDS}
                     for row in expected]
    for filename in ("time.png", "speedup.png"):
        pixels = imread(output / filename)
        assert pixels.shape[:2] == (900, 1350)
        assert pixels.min() < 0.5 and pixels.max() > 0.9
    time_curve = next(line for line in figures["time.png"].axes[0].lines if line.get_label() == "cannon: sequential")
    assert list(time_curve.get_xdata()) == [16, 32]
    assert list(time_curve.get_ydata()) == [4, 12]
    speedup_curve = next(line for line in figures["speedup.png"].axes[0].lines if line.get_label() == "cannon: N=16")
    assert list(speedup_curve.get_xdata()) == [1, 2]
    assert list(speedup_curve.get_ydata()) == pytest.approx([2 / 3, 2])


def test_sequential_only_results_have_unit_speedup_and_no_parallel_curve(tmp_path, rows, monkeypatch):
    from matplotlib.figure import Figure

    figures = {}
    original_save = Figure.savefig

    def save(figure, path, **kwargs):
        figures[Path(path).name] = figure
        return original_save(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    path = write_results(tmp_path / "serial.csv", [row for row in rows if row["mode"] == "sequential"])
    assert all(row["speedup"] == 1 for row in analysis.summarize_results(path))
    analysis.analyze_results(path)
    axis = figures["speedup.png"].axes[0]
    assert [line.get_label() for line in axis.lines] == ["Sequential baseline"]
    assert any(text.get_text() == "No threaded results" for text in axis.texts)


@pytest.mark.parametrize("existing_file", [False, True])
def test_existing_output_is_preserved(tmp_path, rows, existing_file):
    path = write_results(tmp_path / "run.csv", rows)
    output = tmp_path / "run_analysis"
    if existing_file:
        output.write_text("keep this file")
        sentinel = output
    else:
        output.mkdir()
        sentinel = output / "summary.csv"
        sentinel.write_text("keep this file")
    with pytest.raises(FileExistsError):
        analysis.analyze_results(path)
    assert sentinel.read_text() == "keep this file"
    if not existing_file:
        assert list(output.iterdir()) == [sentinel]
