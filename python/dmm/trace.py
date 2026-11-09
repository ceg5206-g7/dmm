"""Executed trace serialization, validation and checksum-verified offline replay."""

import csv
from dataclasses import asdict
from hashlib import sha256
from itertools import product
import json
from math import isfinite, isqrt
from pathlib import Path

from dmm import _native
from dmm.benchmark import source_revision
from dmm.config import REPLICATED_LAYOUTS, validate_case
from dmm._native import (Trace, SuperstepRecord, MessageRecord, ElementRegion,
                         ProductRecord, ProductRegion, CoverageGrid, Counts,
                         count, count_native, bsp_cost, validate_coverage)

COUNT_FIELDS = ("F_total", "W_crit", "C_total", "C_crit", "S", "peak_words", "messages")
PERFORMANCE_FIELDS = ("gflops", "speedup", "parallel_efficiency")
MODEL_FIELDS = ("n", "n_padded", "P", "H", "layout", "c", "phase", *COUNT_FIELDS,
                "gamma", "lambda", "T_model", "t_measured", *PERFORMANCE_FIELDS)


def write_json(path, value):
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
    record = {"schema": 2, "n": trace.original_size, "n_padded": trace.padded_size,
              **case, "H": H, "coverage": [trace.coverage.row_parts, trace.coverage.inner_parts,
                                         trace.coverage.col_parts],
              "peak_words": list(trace.peak_words), "correctness": errors,
              "inputs_sha256": input_hashes, "supersteps": []}
    for step in trace.supersteps:
        messages = []
        for msg in step.messages:
            messages.append({"source": msg.source, "destination": msg.destination,
                             "words": msg.words, "regions": [
                                 [r.row, r.col, r.rows, r.cols] for r in msg.regions]})
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
    if not isinstance(record, dict) or type(record.get("schema")) is not int or record.get("schema") not in (1, 2):
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
            words = _integer(msg.get("words"), "words")
            if record["schema"] == 1:
                key = (source, target, _integer(msg.get("tag"), "tag"))
                regions = [] if msg.get("region") is None else [msg["region"]]
            else:
                key = (source, target)
                regions = msg.get("regions")
                if not isinstance(regions, list) or not regions:
                    raise ValueError("message needs an ordered nonempty regions list")
            if source >= P or target >= P or source == target or key in unique:
                raise ValueError("invalid or duplicate message endpoints")
            unique.add(key)
            area = 0
            for region in regions:
                if not isinstance(region, list) or len(region) != 4:
                    raise ValueError("invalid message region")
                for v in region:
                    _integer(v, "region")
                row, col, height, width = region
                if height == 0 or width == 0 or row + height > padded or col + width > padded:
                    raise ValueError("message region is outside matrix")
                area += height * width
            if area != words:
                raise ValueError("message regions do not match payload words")
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
    # Ring works on row shards; direct 2.5D fuses conversion into replication/reduction.
    fused = case["layout"] == "ring" or (case["layout"] == "2.5d-direct" and case["c"] > 1)
    if not fused and P > 1 and not {"input_conversion", "output_conversion"} <= per_phase.keys():
        raise ValueError("matrix trace is missing common input/output conversion")
    if case["layout"] in REPLICATED_LAYOUTS and case["c"] > 1 and not {"replication", "reduction"} <= per_phase.keys():
        raise ValueError("replicated matrix trace is missing replication/reduction")
    result = {name: _merge(values) for name, values in per_phase.items()}
    result["end_to_end"], result["native"] = _merge(all_steps), _merge(native_steps)
    result["end_to_end"]["peak_words"] = max(peaks)
    return result


def trace_name(n, case):
    return f"n{n}_{case['layout']}_P{case['P']}_c{case['c']}.json"


def save_trace_manifest(config, directory, files, correctness, purpose):
    """Finish a checksum-verified capture usable by the shared offline model pipeline."""
    write_csv(directory / "correctness.csv", ("n", "layout", "P", "c", "max_abs_error", "frobenius_relative_error"), correctness)
    settings = asdict(config)
    settings["data_dir"], settings["output_dir"] = str(config.data_dir), str(config.output_dir)
    write_json(directory / "manifest.json", {"schema": 1, "settings": settings, "files": files,
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
    expected = {trace_name(n, c) for n, c in product(config.sizes, config.cases)}
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
        if path.name != trace_name(record["n"], case) or record["H"] != (config.workers or 1):
            raise ValueError("trace identity does not match manifest")
        records.append((record, metrics))
    return records


def write_csv(path, fields, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
