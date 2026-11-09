"""Capture real BSP traces and recompute normalized model costs and memory feasibility."""

from hashlib import sha256
from importlib import import_module
from itertools import product
import json
from math import isclose, isfinite
from pathlib import Path

from dmm import _native
from dmm.benchmark import _validate_product, source_revision
from dmm.config import LAYOUT_MODULES, ModelConfig
from dmm.datasets import load_manifest, load_pair, load_reference
from dmm.execution import SerialExecutor, ThreadExecutor
from dmm.trace import (MODEL_FIELDS, PERFORMANCE_FIELDS, write_csv, write_json, trace_name,
                       trace_record, save_trace_manifest, load_traces)

SELECTION_FIELDS = ("n", "P", "gamma", "lambda", "mu", "M_cap", "feasible", "infeasible",
                    "best", "T_model", "native_best", "native_T_model")


def capture_traces(config: ModelConfig, directory: Path):
    """Preflight all cases, execute real data once and save validated immutable trace files."""
    config.validate()
    algorithms = []
    for case in config.cases:
        name = f"dmm.algorithms.{LAYOUT_MODULES[case['layout']]}"
        try:
            module = import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
            raise ValueError(f"Layout {case['layout']!r} is not implemented; no traces fabricated") from error
        if any(not callable(getattr(module, name, None)) for name in ("validate_size", "matmul_traced")):
            raise ValueError(f"Layout {case['layout']!r} must implement validate_size and matmul_traced")
        for n in config.sizes:
            module.validate_size(n, P=case["P"], c=case["c"])
        algorithms.append((case, module))
    dataset = load_manifest(config.data_dir)
    if config.group not in dataset.groups or any(n not in dataset.sizes for n in config.sizes):
        raise ValueError("model group/sizes must be present in dataset manifest")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    files, correctness = [], []
    executor = ThreadExecutor(config.workers) if config.workers else SerialExecutor()
    with executor:
        for case, module in algorithms:
            for n in config.sizes:
                a, b = load_pair(config.data_dir, config.group, n)
                reference = load_reference(config.data_dir, config.group, n)
                a.flags.writeable = False
                b.flags.writeable = False
                output, trace = module.matmul_traced(a, b, executor, P=case["P"], c=case["c"])
                errors = _validate_product(output, reference)
                hashes = {"A": sha256(a.tobytes()).hexdigest(), "B": sha256(b.tobytes()).hexdigest()}
                saved = trace_record(trace, case, config.workers or 1, errors, hashes)
                path = directory / trace_name(n, case)
                write_json(path, saved)
                files.append({"name": path.name, "sha256": sha256(path.read_bytes()).hexdigest()})
                correctness.append(dict(n=n, **case, **errors))
    save_trace_manifest(config, directory, files, correctness,
                        "correctness/counting only; no statistical wall-clock measurements")
    return directory


def _model_row(record, phase, counts, gamma, latency):
    cost = counts["W_crit"] + gamma * counts["C_crit"] + latency * counts["S"]
    if not isfinite(cost):
        raise ValueError("model cost is not finite")
    return {"n": record["n"], "n_padded": record["n_padded"], "P": record["P"],
            "H": record["H"], "layout": record["layout"], "c": record["c"], "phase": phase,
            **counts, "gamma": gamma, "lambda": latency, "T_model": cost, "t_measured": "",
            **dict.fromkeys(PERFORMANCE_FIELDS, "")}


def _winner(candidates, scope):
    if not candidates:
        return "infeasible", ""
    best = min(value[scope] for value in candidates)
    names = "|".join(value["name"] for value in candidates if isclose(value[scope], best, rel_tol=1e-12, abs_tol=1e-9))
    return names, best


def build_model_tables(records, config):
    """Each gamma/lambda scenario is deterministic; budgets filter the same results."""
    rows, phases, selections = [], [], []
    for record, metrics in records:
        for gamma, latency in product(config.gammas, config.lambdas):
            rows.append(_model_row(record, "end_to_end", metrics["end_to_end"], gamma, latency))
            phases.extend(_model_row(record, phase, counts, gamma, latency)
                          for phase, counts in metrics.items() if phase != "end_to_end")
    for n, P, gamma, latency, mu in product(config.sizes, sorted({c["P"] for c in config.cases}),
                                           config.gammas, config.lambdas, config.budgets):
        # peak_words counts zero-padded payload, so the budget uses the padded
        # dimension of this problem/cell rather than the unpadded original size.
        cap = mu * _native.padded_size(n, P) ** 2 / P
        candidates, native_candidates, impossible = [], [], []
        for record, metrics in records:
            if (record["n"], record["P"]) != (n, P):
                continue
            name = f"{record['layout']}(c={record['c']})"
            value = {"name": name,
                     "end": _model_row(record, "end_to_end", metrics["end_to_end"], gamma, latency)["T_model"],
                     "native": _model_row(record, "native", metrics["native"], gamma, latency)["T_model"]}
            if metrics["end_to_end"]["peak_words"] <= cap:
                candidates.append(value)
            else:
                impossible.append(name)
            if metrics["native"]["peak_words"] <= cap:
                native_candidates.append(value)
        best, cost = _winner(candidates, "end")
        native, native_cost = _winner(native_candidates, "native")
        selections.append(dict(n=n, P=P, gamma=gamma, **{"lambda": latency}, mu=mu, M_cap=cap,
                               feasible="|".join(v["name"] for v in candidates), infeasible="|".join(impossible),
                               best=best, T_model=cost, native_best=native, native_T_model=native_cost))
    return rows, phases, selections


def run_model(config: ModelConfig, traces=None):
    """Create a new model report; --traces recomputes without loading matrices or algorithms."""
    config.validate()
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"model output already exists: {output}")
    if traces is None:
        traces = capture_traces(config, output / "traces")
    records = load_traces(traces, config)
    rows, phases, choices = build_model_tables(records, config)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "model.csv", MODEL_FIELDS, rows)
    write_csv(output / "phase_model.csv", MODEL_FIELDS, phases)
    write_csv(output / "layout_selection.csv", SELECTION_FIELDS, choices)
    write_json(output / "model.json", {"schema": 1, "units": "T_model stores T_BSP/w0 in FLOP-equivalents; w=w0",
                "assumptions": "equal-speed logical nodes; ideal fully connected symmetric full-duplex network; no congestion or overlap",
                "word_bytes": 4,
                "formula": "W_crit + gamma*C_crit + lambda*S", "w_over_w0": 1,
                "scenarios": len(rows), "selection_cells": len(choices), "executed_configurations": len(records),
                "gammas": config.gammas, "lambdas": config.lambdas, "budgets": config.budgets,
                "analysis_git": source_revision(),
                "trace_directory": str(Path(traces).resolve()),
                "trace_manifest_sha256": sha256((Path(traces) / "manifest.json").read_bytes()).hexdigest(),
                "measurement": "t_measured intentionally blank; capture is not a timing experiment",
                "native_budget": "native peak tested separately against the same per-node cap"})
    from dmm.model_plots import plot_model
    plot_model(rows, phases, choices, config, output)
    return output
