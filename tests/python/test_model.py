import copy
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dmm.algorithms import cannon
from dmm.config import DatasetConfig, ModelConfig, load_model_config
from dmm.datasets import generate_dataset
from dmm.model import build_model_tables, capture_traces, run_model, _winner
from dmm.trace import load_traces, trace_record, validate_trace_record
from dmm.execution import SerialExecutor


@pytest.fixture
def config(tmp_path):
    data = tmp_path / "data"
    generate_dataset(DatasetConfig(data, 5206, [0], [4, 12], [-1, 1]))
    return ModelConfig(data, tmp_path / "report", [dict(layout="cannon", P=P, c=1) for P in (4, 16)],
                       [4, 12], gammas=[1, 10], lambdas=[100], budgets=[8, 32])


def test_main_profile_has_24_executions_192_scenarios_and_144_budget_cells():
    cfg = load_model_config(Path(__file__).parents[2] / "configs/model.yaml")
    assert len(cfg.sizes) * len(cfg.cases) == 24
    assert len(cfg.sizes) * len(cfg.cases) * len(cfg.gammas) * len(cfg.lambdas) == 192
    assert len(cfg.sizes) * 3 * len(cfg.gammas) * len(cfg.lambdas) * len(cfg.budgets) == 144
    assert cfg.budgets == [8, 32]


def test_model_yaml_paths_and_numeric_validation(config, tmp_path):
    values = dict(data_dir="data", output_dir="results", cases=config.cases, sizes=config.sizes)
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(values))
    loaded = load_model_config(path)
    assert loaded.data_dir == tmp_path / "data"
    assert loaded.output_dir == tmp_path / "results"
    for field, value in (("gammas", [True]), ("lambdas", [float("inf")]), ("budgets", [0]),
                         ("gammas", [1, 1]), ("workers", -1), ("sizes", [12, 4])):
        changed = dict(values, **{field: value})
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError):
            load_model_config(path)


def test_real_capture_and_recompute_never_reruns_algorithms_or_loads_inputs(config, monkeypatch):
    output = run_model(config)
    with (output / "model.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert all(row["t_measured"] == "" and row["H"] == "1" for row in rows)
    assert all(row[k] == "" for row in rows for k in ("gflops", "speedup", "parallel_efficiency"))
    for row in rows:
        expected = int(row["W_crit"]) + float(row["gamma"]) * int(row["C_crit"]) + float(row["lambda"]) * int(row["S"])
        assert float(row["T_model"]) == expected
    with (output / "layout_selection.csv").open(newline="") as handle:
        choices = list(csv.DictReader(handle))
    assert len(choices) == 16
    # The budget is measured in padded payload words: M_cap = mu * n_padded^2 / P.
    for row in choices:
        padded = ((int(row["n"]) + int(row["P"]) - 1) // int(row["P"])) * int(row["P"])
        assert float(row["M_cap"]) == float(row["mu"]) * padded**2 / int(row["P"])
    assert any(r["best"] != "infeasible" for r in choices)
    metadata = json.loads((output / "model.json").read_text())
    assert metadata["executed_configurations"] == 4
    assert metadata["peak_semantics"] == "live-payload"
    assert metadata["superstep_semantics"] == "delivery-merged"
    for name in ("layout_selection", "cost_breakdown", "conversion_breakdown", "native_vs_common", "peak_memory"):
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG")

    def forbidden(*args, **kwargs):
        pytest.fail("offline sweep must not execute algorithms or load matrices")
    monkeypatch.setattr(cannon, "matmul_traced", forbidden)
    import dmm.model as model
    monkeypatch.setattr(model, "load_pair", forbidden)
    changed = replace(config, data_dir=Path("does-not-exist"), output_dir=output.with_name("recomputed"), gammas=[100])
    replay = run_model(changed, output / "traces")
    with (replay / "model.csv").open(newline="") as handle:
        replayed = list(csv.DictReader(handle))
    assert len(replayed) == 4
    assert all(float(r["gamma"]) == 100 for r in replayed)


def test_trace_checksum_and_capture_completeness_are_checked(config):
    root = capture_traces(config, config.output_dir / "traces")
    first = next(p for p in root.glob("n*.json"))
    first.write_text(first.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        load_traces(root, config)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"].pop()
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        load_traces(root, config)


def _record():
    A = np.ones((16, 16), dtype=np.float32)
    with SerialExecutor() as executor:
        _, trace = cannon.matmul_traced(A, A, executor)
    return trace_record(trace, dict(layout="cannon", P=16, c=1), 1,
                        dict(max_abs_error=0, frobenius_relative_error=0), {"A": "0" * 64, "B": "1" * 64})


@pytest.mark.parametrize("change", ["missing", "duplicate", "shape", "work", "memory", "message", "schema"])
def test_offline_validator_rejects_corrupted_audit_records(change):
    record = _record()
    step = next(s for s in record["supersteps"] if s["products"])
    if change == "missing":
        step["products"].pop()
    elif change == "duplicate":
        step["products"].append(copy.deepcopy(step["products"][0]))
    elif change == "shape":
        step["products"][0]["m"] = 1
    elif change == "work":
        step["flops"][0] = 0
    elif change == "memory":
        step["peak_words"] = [0] * 16
    elif change == "message":
        sent = next(s for s in record["supersteps"] if s["messages"])
        sent["messages"].append(copy.deepcopy(sent["messages"][0]))
    else:
        record["schema"] = True
    with pytest.raises(ValueError):
        validate_trace_record(record)


def test_budget_filters_reuse_costs_and_allow_exact_ties(config):
    record = _record()
    metrics = validate_trace_record(record)
    cfg = replace(config, cases=[dict(layout="cannon", P=16, c=1)],
                  sizes=[16], gammas=[1], lambdas=[100], budgets=[1, 32])
    rows, phases, choices = build_model_tables([(record, metrics)], cfg)
    assert len(rows) == 1 and len(choices) == 2
    assert choices[0]["best"] == "infeasible"
    assert choices[1]["best"] == "cannon(c=1)"
    assert choices[0]["M_cap"] == 16
    assert any(row["phase"] == "native" for row in phases)
    assert _winner([dict(name="a", cost=5), dict(name="b", cost=5)], "cost") == ("a|b", 5)


def test_budget_uses_padded_payload_words_for_zero_padded_sizes():
    """n=12 with P=16 pads to 16, so M_cap = mu * 16^2 / 16 and the 116-word peak fits mu=8."""
    A = np.ones((12, 12), dtype=np.float32)
    with SerialExecutor() as executor:
        _, trace = cannon.matmul_traced(A, A, executor, P=16, c=1)
    record = trace_record(trace, dict(layout="cannon", P=16, c=1), 1,
                          dict(max_abs_error=0, frobenius_relative_error=0), {"A": "0" * 64, "B": "1" * 64})
    assert record["n"] == 12 and record["n_padded"] == 16
    cfg = ModelConfig(Path("unused"), Path("unused"), [dict(layout="cannon", P=16, c=1)], [12],
                      gammas=[1], lambdas=[100], budgets=[8])
    _, _, choices = build_model_tables([(record, validate_trace_record(record))], cfg)
    assert choices[0]["M_cap"] == 128.0
    assert choices[0]["best"] == "cannon(c=1)", "peak 116 words must fit the padded 128-word budget"


def test_missing_algorithm_preflight_creates_no_output_and_never_fabricates_counts(config, monkeypatch):
    from importlib import import_module

    def load(name):
        if name.endswith(".ring"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return import_module(name)

    monkeypatch.setattr("dmm.model.import_module", load)
    cfg = replace(config, cases=[dict(layout="cannon", P=4, c=1), dict(layout="ring", P=4, c=1)])
    with pytest.raises(ValueError, match="not implemented"):
        run_model(cfg)
    assert not cfg.output_dir.exists()


def test_existing_report_is_preserved(config):
    config.output_dir.mkdir()
    sentinel = config.output_dir / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        run_model(config)
    assert sentinel.read_text() == "keep"


def test_cli_captures_then_recomputes_after_dataset_is_removed(config, tmp_path):
    import subprocess
    import sys
    cli = str(Path(sys.executable).with_name("dmm"))
    values = dict(data_dir=str(config.data_dir), output_dir=str(config.output_dir), cases=config.cases,
                  sizes=[4], gammas=[1], lambdas=[100], budgets=[8, 32])
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(values))
    first = subprocess.run([cli, "model", "--config", str(path)], cwd=tmp_path,
                           capture_output=True, text=True, check=True)
    assert "Saved model analysis" in first.stdout
    config.data_dir.rename(config.data_dir.with_name("unused-data"))
    values.update(output_dir=str(tmp_path / "offline"), gammas=[10, 100])
    path.write_text(yaml.safe_dump(values))
    second = subprocess.run([cli, "model", "--config", str(path), "--traces", str(config.output_dir / "traces")],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "Saved model analysis" in second.stdout
    with (tmp_path / "offline/model.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 and {r["gamma"] for r in rows} == {"10", "100"}


def test_manifest_structure_and_requested_execution_identity_are_checked(config):
    root = capture_traces(config, config.output_dir / "traces")
    original = json.loads((root / "manifest.json").read_text())
    with pytest.raises(ValueError, match="workers"):
        load_traces(root, replace(config, workers=1))
    for manifest in ([], dict(original, schema=True), dict(original, settings=None),
                     dict(original, files=[None]), dict(original, schema=1),
                     dict(original, peak_semantics=None),
                     dict(original, superstep_semantics="legacy")):
        (root / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            load_traces(root, config)


def test_schema_two_serializes_ordered_regions_and_one_message_per_pair():
    record = _record()
    assert record['schema'] == 2
    for step in record['supersteps']:
        assert len({(m['source'], m['destination']) for m in step['messages']}) == len(step['messages'])
        for msg in step['messages']:
            assert 'tag' not in msg and 'region' not in msg
            assert sum(r[2] * r[3] for r in msg['regions']) == msg['words']
    original = validate_trace_record(record)
    msg = next(m for s in record['supersteps'] for m in s['messages'] if m['regions'][0][3] >= 2)
    row, col, height, width = msg['regions'][0]
    msg['regions'][0:1] = [[row, col, height, 1], [row, col + 1, height, width - 1]]
    assert validate_trace_record(record) == original


@pytest.mark.parametrize('failure', ['empty', 'missing', 'zero', 'shape', 'outside', 'duplicate_with_tag'])
def test_schema_two_rejects_invalid_component_regions_and_duplicate_pairs(failure):
    record = _record()
    step = next(s for s in record['supersteps'] if s['messages'])
    msg = step['messages'][0]
    if failure == 'empty':
        msg['regions'] = []
    elif failure == 'missing':
        del msg['regions']
    elif failure == 'zero':
        msg['regions'][0][2] = 0
    elif failure == 'shape':
        msg['regions'][0][3] += 1
    elif failure == 'outside':
        msg['regions'][0][0] = record['n_padded']
    else:
        duplicate = copy.deepcopy(msg)
        duplicate['tag'] = 999
        step['messages'].append(duplicate)
    with pytest.raises(ValueError):
        validate_trace_record(record)


def test_real_legacy_schema_one_capture_stays_readable_but_is_not_mixed(tmp_path):
    """Historical schema-1 records still validate; old captures are not replayed as current counts."""
    from hashlib import sha256

    fixture = Path(__file__).parents[1] / 'fixtures/trace_v1_cannon.json'
    raw = fixture.read_bytes()
    legacy = json.loads(raw)
    assert legacy['schema'] == 1
    metrics = validate_trace_record(legacy)
    assert metrics['native']['F_total'] == 2 * 4 ** 3
    assert metrics['native']['messages'] == 12
    assert metrics['end_to_end']['S'] == 9
    root = tmp_path / 'legacy'
    root.mkdir()
    name = 'n4_cannon_P4_c1.json'
    (root / name).write_bytes(raw)
    (root / 'correctness.csv').write_text('saved historical summary\n')
    cfg = ModelConfig(tmp_path / 'missing-inputs', tmp_path / 'report',
                      [dict(layout='cannon', P=4, c=1)], [4], gammas=[10], lambdas=[100], budgets=[32])
    (root / 'manifest.json').write_text(json.dumps(dict(schema=1,
        settings=dict(cases=cfg.cases, sizes=cfg.sizes, workers=cfg.workers, group=cfg.group),
        files=[dict(name=name, sha256=sha256(raw).hexdigest())],
        correctness_sha256=sha256((root / 'correctness.csv').read_bytes()).hexdigest())))
    # The old capture used allocated-capacity peaks and unmerged supersteps, so it
    # must not be mixed into a current sweep; capture it again instead.
    with pytest.raises(ValueError, match='recapture'):
        load_traces(root, cfg)
    assert not cfg.output_dir.exists()
    assert fixture.read_bytes() == raw


def _replicated_record(layout):
    from dmm.algorithms import two_point_five_d, two_point_five_d_direct
    module = {"2.5d": two_point_five_d, "2.5d-direct": two_point_five_d_direct}[layout]
    A = np.arange(64, dtype=np.float32).reshape(8, 8) / 64
    with SerialExecutor() as executor:
        C, trace = module.matmul_traced(A, A, executor, P=8, c=2)
    np.testing.assert_allclose(C, A.astype(np.float64) @ A, atol=1e-4, rtol=1e-3)
    return trace_record(trace, dict(layout=layout, P=8, c=2), 1,
                        dict(max_abs_error=0, frobenius_relative_error=0), {"A": "0" * 64, "B": "1" * 64})


def test_direct_replication_trace_fuses_conversion_and_moves_fewer_words():
    """The ablation has no conversion phases, so native and end-to-end coincide."""
    direct = validate_trace_record(_replicated_record("2.5d-direct"))
    base = validate_trace_record(_replicated_record("2.5d"))
    assert "input_conversion" not in direct and "output_conversion" not in direct
    assert {"replication", "reduction"} <= direct.keys()
    assert direct["native"] == direct["end_to_end"]
    assert (direct["end_to_end"]["S"], direct["end_to_end"]["C_crit"]) == (5, 80)
    assert (base["end_to_end"]["S"], base["end_to_end"]["C_crit"]) == (8, 128)
    # The base-layer route still requires its conversion phases.
    record = _replicated_record("2.5d")
    record["supersteps"] = [s for s in record["supersteps"] if s["label"] != "output_conversion"]
    with pytest.raises(ValueError, match="conversion"):
        validate_trace_record(record)


def test_direct_ablation_profile_extends_the_frozen_main_profile():
    root = Path(__file__).parents[2] / "configs"
    main, ablation = load_model_config(root / "model.yaml"), load_model_config(root / "model_direct.yaml")
    extra = [case for case in ablation.cases if case not in main.cases]
    assert extra == [dict(layout="2.5d-direct", P=64, c=4), dict(layout="2.5d-direct", P=256, c=4)]
    assert [case for case in ablation.cases if case in main.cases] == main.cases
    assert (ablation.sizes, ablation.gammas, ablation.lambdas, ablation.budgets) == (
        main.sizes, main.gammas, main.lambdas, main.budgets)
