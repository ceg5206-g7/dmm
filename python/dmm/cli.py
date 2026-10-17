"""Command-line entry point for the matrix multiplication benchmarks."""

import argparse
from pathlib import Path

from dmm.analysis import analyze_results
from dmm.benchmark import run_benchmark
from dmm.config import load_benchmark_config, load_dataset_config
from dmm.datasets import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="dmm")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate the shared matrix dataset")
    generate.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"), help="YAML configuration (default: configs/dataset.yaml)")
    bench = commands.add_parser("bench", help="Benchmark and verify matrix multiplication")
    bench.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"), help="YAML configuration (default: configs/benchmark.yaml)")
    analyze = commands.add_parser("analyze", help="Summarize benchmark results and plot timings and speedup")
    analyze.add_argument("--input", type=Path, required=True, help="Benchmark CSV to analyze")
    analyze.add_argument("--output-dir", type=Path, help="New output directory (default: <input_stem>_analysis beside the CSV)")
    args = parser.parse_args()

    try:
        if args.command == "generate":
            config = load_dataset_config(args.config)
            generate_dataset(config)
            print(f"Generated dataset in {config.data_dir}")
        elif args.command == "bench":
            config = load_benchmark_config(args.config)
            completed = run_benchmark(config)
            print(f"Saved {completed} benchmark result(s) to {config.output}")
        else:
            output_dir = analyze_results(args.input, args.output_dir)
            print(f"Saved analysis to {output_dir}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"dmm: error: {error}\n")
