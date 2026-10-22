"""Capture real BSP traces and recompute normalized model costs and memory feasibility."""

import csv
from dataclasses import asdict
from hashlib import sha256
from importlib import import_module
from itertools import product
import json
from math import isclose, isfinite, isqrt
from pathlib import Path

from dmm import _native
from dmm.benchmark import _validate_product, source_revision
from dmm.config import LAYOUT_MODULES, ModelConfig, validate_case
from dmm.datasets import load_manifest, load_pair, load_reference
from dmm.execution import SerialExecutor, ThreadExecutor

COUNT_FIELDS = ("F_total", "W_crit", "C_total", "C_crit", "S", "peak_words", "messages")
MODEL_FIELDS = ("n", "n_padded", "P", "H", "layout", "c", "phase", *COUNT_FIELDS,
                "gamma", "lambda", "T_model", "t_measured")
SELECTION_FIELDS = ("n", "P", "gamma", "lambda", "mu", "M_cap", "feasible", "infeasible",
                    "best", "T_model", "native_best", "native_T_model")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def trace_record(trace, case, H, errors, input_hashes):
    """Serialize the executed trace, not an analytic reconstruction of its schedule."""
    _native.validate_coverage(trace)
    record = {"schema": 1, "n": trace.original_size, "n_padded": trace.padded_size,
              **case, "H": H, "coverage": [trace.coverage.row_parts, trace.coverage.inner_parts,
                                         trace.coverage.col_parts],
              "peak_words": list(trace.peak_words), "correctness": errors,
              "inputs_sha256": input_hashes, "supersteps": []}
    for step in trace.supersteps:
        messages = []
        for msg in step.messages:
            region = msg.region
            messages.append({"source": msg.source, "destination": msg.destination, "tag": msg.tag,
                             "words": msg.words, "region": None if region is None else
                             [region.row, region.col, region.rows, region.cols]})
        products = [{"node": r.node, "i": r.product.i, "k": r.product.k, "j": r.product.j,
                     "m": r.product.m, "n": r.product.n, "inner": r.product.inner}
                    for r in step.products]
        record["supersteps"].append(dict(label=step.label, flops=list(step.flops),
                                         peak_words=list(step.peak_words), messages=messages, products=products))
    metrics = validate_trace_record(record)
    actual = _native.count(trace)
    expected = (actual.total_flops, actual.critical_flops, actual.total_words, actual.critical_words,
                actual.supersteps, actual.peak_words, actual.messages)
    if tuple(metrics["end_to_end"][f] for f in COUNT_FIELDS) != expected:
        raise RuntimeError("independent saved-trace counts disagree with native counts")
    native = _native.count_native(trace)
    expected_native = (native.total_flops, native.critical_flops, native.total_words, native.critical_words,
                       native.supersteps, native.peak_words, native.messages)
    if tuple(metrics["native"][f] for f in COUNT_FIELDS) != expected_native:
        raise RuntimeError("independent native-body counts disagree with native runtime")
    return record


def _merge(counts):
    rows = list(counts)
    return {f: (max((r[f] for r in rows), default=0) if f == "peak_words"
                else sum(r[f] for r in rows)) for f in COUNT_FIELDS}


def validate_trace_record(record):
    """Independently check saved messages, block coverage and critical-phase counts."""
    if not isinstance(record, dict) or type(record.get("schema")) is not int or record.get("schema") != 1:
        raise ValueError("unsupported trace schema")
    case = {name: record.get(name) for name in ("layout", "P", "c")}
    validate_case(case)
    P = case["P"]
    n, padded = _integer(record.get("n"), "n", 1), _integer(record.get("n_padded"), "n_padded", 1)
    if padded != ((n + P - 1) // P) * P:
        raise ValueError("saved trace must use common padding to a multiple of P")
    _integer(record.get("H"), "H", 1)
    peaks = record.get("peak_words")
    if not isinstance(peaks, list) or len(peaks) != P:
        raise ValueError("trace peaks must cover every node")
    for words in peaks:
        _integer(words, "peak_words")
    errors, hashes = record.get("correctness"), record.get("inputs_sha256")
    if (not isinstance(errors, dict) or set(errors) != {"max_abs_error", "frobenius_relative_error"}
            or any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in errors.values())):
        raise ValueError("trace needs finite non-negative correctness metrics")
    if (not isinstance(hashes, dict) or set(hashes) != {"A", "B"}
            or any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                   for v in hashes.values())):
        raise ValueError("trace needs input content hashes")
    parts = record.get("coverage")
    expected_parts = [P, P, 1] if case["layout"] == "ring" else [isqrt(P // case["c"])] * 3
    if parts != expected_parts or any(type(x) is not int for x in parts):
        raise ValueError("trace coverage partition does not match layout")
    seen, per_phase, all_steps, native_steps = set(), {}, [], []
    steps = record.get("supersteps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("trace must contain executed supersteps")
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("label"), str) or not step["label"]:
            raise ValueError("superstep needs a phase label")
        for field in ("flops", "peak_words"):
            values = step.get(field)
            if not isinstance(values, list) or len(values) != P:
                raise ValueError(f"superstep {field} must cover every node")
            for v in values:
                _integer(v, field)
        if any(v > peaks[i] for i, v in enumerate(step["peak_words"])):
            raise ValueError("step memory peak exceeds trace peak")
        sent, received, work = [0] * P, [0] * P, [0] * P
        messages, products = step.get("messages"), step.get("products")
        if not isinstance(messages, list) or not isinstance(products, list):
            raise ValueError("superstep messages/products must be lists")
        unique = set()
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("message must be a record")
            source = _integer(msg.get("source"), "source")
            target = _integer(msg.get("destination"), "destination")
            tag = _integer(msg.get("tag"), "tag")
            words = _integer(msg.get("words"), "words")
            if source >= P or target >= P or source == target or (source, target, tag) in unique:
                raise ValueError("invalid or duplicate message endpoints/tag")
            unique.add((source, target, tag))
            region = msg.get("region")
            if words and region is None:
                raise ValueError("matrix payload is missing its element region")
            if region is not None:
                if not isinstance(region, list) or len(region) != 4:
                    raise ValueError("invalid message region")
                for v in region:
                    _integer(v, "region")
                row, col, height, width = region
                if height == 0 or width == 0 or height * width != words or row + height > padded or col + width > padded:
                    raise ValueError("message region does not match matrix/payload")
            sent[source] += words
            received[target] += words
        if any(step["peak_words"][i] < max(sent[i], received[i]) for i in range(P)):
            raise ValueError("payload peak is smaller than simultaneously queued messages")
        for r in products:
            if not isinstance(r, dict):
                raise ValueError("product must be a record")
            for f in ("node", "i", "k", "j", "m", "n", "inner"):
                _integer(r.get(f), f)
            triple = (r["i"], r["k"], r["j"])
            if (r["node"] >= P or any(t >= b for t, b in zip(triple, parts)) or triple in seen
                    or (r["m"], r["inner"], r["n"]) != tuple(padded // b for b in parts)):
                raise ValueError("invalid or duplicate executed product coverage")
            if step["peak_words"][r["node"]] < r["m"] * r["n"] + max(r["m"] * r["inner"], r["n"] * r["inner"]):
                raise ValueError("payload peak cannot hold executed GEMM operands and output")
            seen.add(triple)
            work[r["node"]] += 2 * r["m"] * r["n"] * r["inner"]
        if any(w > f for w, f in zip(work, step["flops"])):
            raise ValueError("executed products exceed recorded node FLOPs")
        counts = dict(F_total=sum(step["flops"]), W_crit=max(step["flops"]), C_total=sum(sent),
                      C_crit=max(max(sent), max(received)), S=1,
                      peak_words=max(step["peak_words"]), messages=len(messages))
        all_steps.append(counts)
        per_phase.setdefault(step["label"], []).append(counts)
        if step["label"] not in ("input_conversion", "output_conversion"):
            native_steps.append(counts)
    if len(seen) != parts[0] * parts[1] * parts[2]:
        raise ValueError("missing executed product coverage")
    if case["layout"] != "ring" and P > 1 and not {"input_conversion", "output_conversion"} <= per_phase.keys():
        raise ValueError("matrix trace is missing common input/output conversion")
    if case["layout"] == "2.5d" and case["c"] > 1 and not {"replication", "reduction"} <= per_phase.keys():
        raise ValueError("replicated matrix trace is missing replication/reduction")
    result = {name: _merge(values) for name, values in per_phase.items()}
    result["end_to_end"], result["native"] = _merge(all_steps), _merge(native_steps)
    result["end_to_end"]["peak_words"] = max(peaks)
    return result


def _trace_name(n, case):
    return f"n{n}_{case['layout']}_P{case['P']}_c{case['c']}.json"


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
                path = directory / _trace_name(n, case)
                _write_json(path, saved)
                files.append({"name": path.name, "sha256": sha256(path.read_bytes()).hexdigest()})
                correctness.append(dict(n=n, **case, **errors))
    save_trace_manifest(config, directory, files, correctness,
                        "correctness/counting only; no statistical wall-clock measurements")
    return directory


def save_trace_manifest(config, directory, files, correctness, purpose):
    """Finish a checksum-verified capture usable by the shared offline model pipeline."""
    _csv(directory / "correctness.csv", ("n", "layout", "P", "c", "max_abs_error", "frobenius_relative_error"), correctness)
    settings = asdict(config)
    settings["data_dir"], settings["output_dir"] = str(config.data_dir), str(config.output_dir)
    _write_json(directory / "manifest.json", {"schema": 1, "settings": settings, "files": files,
                "compiler": _native.compiler, "build_type": _native.build_type, "git": source_revision(),
                "correctness_sha256": sha256((directory / "correctness.csv").read_bytes()).hexdigest(),
                "validation_rule": "abs(error) <= 1e-4 + 1e-3 * abs(FP64 reference)",
                "input_manifest": json.loads((Path(config.data_dir) / "manifest.json").read_text()),
                "purpose": purpose})


def load_traces(directory, config):
    """Verify capture completeness and hashes before using saved counts."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not isinstance(manifest, dict) or type(manifest.get("schema")) is not int or manifest.get("schema") != 1:
        raise ValueError("unsupported trace manifest")
    settings, files = manifest.get("settings"), manifest.get("files")
    if not isinstance(settings, dict) or not isinstance(files, list):
        raise ValueError("trace manifest needs settings and files")
    for name in ("cases", "sizes", "group", "workers"):
        if settings.get(name) != getattr(config, name):
            raise ValueError(f"saved trace {name} does not match requested execution settings")
    if any(not isinstance(f, dict) or set(f) != {"name", "sha256"}
           or not isinstance(f["name"], str) or not isinstance(f["sha256"], str) for f in files):
        raise ValueError("invalid trace file inventory")
    expected = {_trace_name(n, c) for n, c in product(config.sizes, config.cases)}
    if len(files) != len(expected) or {f["name"] for f in files} != expected:
        raise ValueError("trace capture is incomplete or duplicated")
    if sha256((directory / "correctness.csv").read_bytes()).hexdigest() != manifest.get("correctness_sha256"):
        raise ValueError("correctness summary checksum mismatch")
    records = []
    for item in files:
        path = directory / item["name"]
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"trace checksum mismatch: {path.name}")
        record = json.loads(raw)
        metrics = validate_trace_record(record)
        case = {k: record[k] for k in ("layout", "P", "c")}
        if path.name != _trace_name(record["n"], case) or record["H"] != (config.workers or 1):
            raise ValueError("trace identity does not match manifest")
        records.append((record, metrics))
    return records


def _model_row(record, phase, counts, gamma, latency):
    cost = counts["W_crit"] + gamma * counts["C_crit"] + latency * counts["S"]
    if not isfinite(cost):
        raise ValueError("model cost is not finite")
    return {"n": record["n"], "n_padded": record["n_padded"], "P": record["P"],
            "H": record["H"], "layout": record["layout"], "c": record["c"], "phase": phase,
            **counts, "gamma": gamma, "lambda": latency, "T_model": cost, "t_measured": ""}


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
        cap = mu * n * n / P
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


def _csv(path, fields, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    _csv(output / "model.csv", MODEL_FIELDS, rows)
    _csv(output / "phase_model.csv", MODEL_FIELDS, phases)
    _csv(output / "layout_selection.csv", SELECTION_FIELDS, choices)
    _write_json(output / "model.json", {"schema": 1, "units": "T_model stores T_BSP/w0 in FLOP-equivalents; w=w0",
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
