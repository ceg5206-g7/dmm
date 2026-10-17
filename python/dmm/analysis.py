"""Summarize benchmark timings and plot runtime and parallel speedup."""

import csv
from math import isfinite
from pathlib import Path
from statistics import mean

from dmm.benchmark import CSV_FIELDS
from dmm.config import ALGORITHMS, MODES

SUMMARY_FIELDS = ("algorithm", "mode", "workers", "size", "group_count", "repeats", "mean_time_s", "speedup")


def _load_timings(input_path: str | Path) -> dict[tuple[str, str, int | None, int], dict[int, dict[int, float]]]:
    """Load and validate CSV rows, grouping timings by execution setting and size."""
    timings = {}
    try:
        with Path(input_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            legacy_fields = tuple(field for field in CSV_FIELDS if field != "repeat")
            if reader.fieldnames is None or sorted(reader.fieldnames) not in (sorted(CSV_FIELDS), sorted(legacy_fields)):
                raise ValueError(f"Benchmark CSV columns must be: {', '.join(CSV_FIELDS)}")
            for row in reader:
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("row must contain all benchmark columns")
                    algorithm, mode = row["algorithm"].strip(), row["mode"].strip()
                    if algorithm not in ALGORITHMS or mode not in MODES:
                        raise ValueError("unknown algorithm or mode")
                    group, pair, size = (int(row[name]) for name in ("group", "pair", "size"))
                    if group < 0 or pair < 0 or size <= 0:
                        raise ValueError("group and pair must be non-negative; size must be positive")
                    repeat = int(row.get("repeat", "0"))
                    if repeat < 0:
                        raise ValueError("repeat must be a non-negative integer")
                    if mode == "sequential":
                        if row["workers"].strip():
                            raise ValueError("sequential workers must be empty")
                        workers = None
                    else:
                        workers = int(row["workers"])
                        if workers <= 0:
                            raise ValueError("threaded workers must be positive")
                    elapsed = float(row["time_s"])
                    if not isfinite(elapsed) or elapsed <= 0:
                        raise ValueError("time_s must be positive and finite")
                    samples = timings.setdefault((algorithm, mode, workers, size), {}).setdefault(group, {})
                    if repeat in samples:
                        raise ValueError("duplicate result for the same algorithm, mode, workers, group, size and repeat")
                    samples[repeat] = elapsed
                except ValueError as error:
                    raise ValueError(f"Invalid benchmark CSV row {reader.line_num}: {error}") from error
    except csv.Error as error:
        raise ValueError(f"Invalid benchmark CSV: {error}") from error
    if not timings:
        raise ValueError("Benchmark CSV contains no results")
    # shortcut: infer repeat count from CSV; use run metadata when complete-run validation is required.
    repeat_count = max(len(samples) for groups in timings.values() for samples in groups.values())
    repeat_ids = set(range(repeat_count))
    for (algorithm, mode, workers, size), groups in timings.items():
        for group, samples in groups.items():
            if samples.keys() != repeat_ids:
                raise ValueError(f"Incomplete or inconsistent repeats for algorithm {algorithm!r}, mode {mode!r}, "
                                 f"workers {workers}, size {size}, group {group}; expected IDs {sorted(repeat_ids)}")
    return timings


def summarize_results(input_path: str | Path) -> list[dict]:
    """Average groups and compare each setting with its matching serial baseline."""
    timings = _load_timings(input_path)
    averages = {
        key: mean(mean(samples.values()) for samples in groups.values())
        for key, groups in timings.items()
    }
    summary = []
    for key in sorted(timings, key=lambda key: (key[0], key[1] != "sequential", key[2] or 0, key[3])):
        algorithm, mode, workers, size = key
        baseline = (algorithm, "sequential", None, size)
        if baseline not in timings:
            raise ValueError(f"Missing sequential baseline for algorithm {algorithm!r}, size {size}")
        if timings[key].keys() != timings[baseline].keys():
            raise ValueError(f"Group mismatch for algorithm {algorithm!r}, size {size}, workers {workers}")
        speedup = averages[baseline] / averages[key]
        if not isfinite(speedup) or speedup <= 0:
            raise ValueError(f"Speedup must be positive and finite for algorithm {algorithm!r}, size {size}")
        summary.append(dict(algorithm=algorithm, mode=mode, workers=workers, size=size,
                            group_count=len(timings[key]), repeats=len(next(iter(timings[key].values()))),
                            mean_time_s=averages[key], speedup=speedup))
    return summary


def _plot_runtime(summary: list[dict], output_path: Path) -> None:
    """Plot mean runtime against matrix size for each execution setting."""
    # Use Agg directly so analysis also works on machines without a GUI.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    runtime_series = {}
    for row in summary:
        runtime_series.setdefault((row["algorithm"], row["mode"], row["workers"]), []).append(row)

    figure = Figure(figsize=(9, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for (algorithm, mode, workers), rows in runtime_series.items():
        rows = sorted(rows, key=lambda row: row["size"])
        setting = "sequential" if mode == "sequential" else f"{workers} thread(s)"
        axis.plot([row["size"] for row in rows], [row["mean_time_s"] for row in rows],
                  marker="o", label=f"{algorithm}: {setting}")
    sizes = sorted({row["size"] for row in summary})
    axis.set_xscale("log", base=2)
    axis.set_xticks(sizes, labels=[str(size) for size in sizes])
    axis.set_yscale("log")
    axis.set(xlabel="Matrix size (N)", ylabel="Mean time (s)", title="Matrix multiplication runtime")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")
    figure.savefig(output_path, dpi=150)


def _plot_speedup(summary: list[dict], output_path: Path) -> None:
    """Plot speedup against worker count with a sequential baseline."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    speedup_series = {}
    for row in summary:
        if row["mode"] == "threaded":
            speedup_series.setdefault((row["algorithm"], row["size"]), []).append(row)

    figure = Figure(figsize=(9, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.axhline(1, color="gray", linestyle="--", label="Sequential baseline")
    for (algorithm, size), rows in speedup_series.items():
        rows = sorted(rows, key=lambda row: row["workers"])
        axis.plot([row["workers"] for row in rows], [row["speedup"] for row in rows],
                  marker="o", label=f"{algorithm}: N={size}")
    if speedup_series:
        axis.set_xticks(sorted({row["workers"] for row in summary if row["workers"] is not None}))
    else:
        axis.text(0.5, 0.5, "No threaded results", transform=axis.transAxes, ha="center")
        axis.set_xticks([])
    axis.set(xlabel="Worker threads", ylabel="Speedup (sequential / threaded)",
             title="Matrix multiplication speedup")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")
    figure.savefig(output_path, dpi=150)


def analyze_results(input_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Write summary.csv and two PNG plots into a new analysis directory."""
    input_path = Path(input_path)
    summary = summarize_results(input_path)
    output_dir = Path(output_dir) if output_dir is not None else input_path.with_name(f"{input_path.stem}_analysis")
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    _plot_runtime(summary, output_dir / "time.png")
    _plot_speedup(summary, output_dir / "speedup.png")
    return output_dir
