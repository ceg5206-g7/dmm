import csv
from dataclasses import replace
from hashlib import sha256
import json

import pytest

from dmm import _native
from dmm.analysis import timing_statistics
from dmm.config import DatasetConfig, MeasurementConfig, load_measurement_config
from dmm.datasets import generate_dataset
from dmm.measurement import run_measurement
from dmm.trace import validate_trace_record


@pytest.fixture
def config(tmp_path):
    data = tmp_path / 'data'
    generate_dataset(DatasetConfig(data, 51, [0], [5, 12], [-1, 1]))
    return MeasurementConfig(data, tmp_path / 'measured', [dict(layout='cannon', P=1, c=1)], [5, 12], warmups=1, repeats=3)


def test_real_calibration_frozen_across_heldout_sizes_and_reproducible_csv(config):
    output = run_measurement(config)
    manifest = json.loads((output / 'manifest.json').read_text())
    calibration = json.loads((output / 'calibration.json').read_text())
    assert manifest['executed_configurations'] == 2
    for name, checksum in manifest['artifacts_sha256'].items():
        assert sha256((output / name).read_bytes()).hexdigest() == checksum
    parameters = calibration['parameters_seconds']
    assert all(v > 0 for v in parameters.values())
    micros = list(csv.DictReader((output / 'microbenchmarks.csv').open()))
    assert len(micros) == 7 * config.repeats
    for key, kind in zip(('w', 'g', 'l'), ('compute', 'copy', 'synchronization')):
        median, iqr = timing_statistics([float(r['seconds_per_unit']) for r in micros if r['kind'] == kind])
        assert median == parameters[key]
        assert iqr == calibration['iqr_seconds_per_unit'][key]
    samples = list(csv.DictReader((output / 'samples.csv').open()))
    models = list(csv.DictReader((output / 'measured_model.csv').open()))
    assert len(samples) == 2 * 7 * config.repeats
    assert set(int(r['n']) for r in models) == {5, 12}
    assert not set(config.sizes) & set(calibration['micro_sizes'])
    # P = 1 has nothing to skew: zeroing C joins its single multiply superstep.
    assert {r['phase'] for r in models} == {'end_to_end', 'native', 'multiply'}
    for row in models:
        assert all(float(row[k]) == parameters[k] for k in ('w', 'g', 'l'))
        prediction = sum(parameters[k] * float(row[c]) for k, c in [('w', 'W_crit'), ('g', 'C_crit'), ('l', 'S')])
        assert float(row['T_model_seconds']) == pytest.approx(prediction)
        assert float(row['T_model']) * parameters['w'] == pytest.approx(prediction)
        assert float(row['residual_seconds']) == pytest.approx(float(row['t_measured']) - prediction)
        assert float(row['relative_residual']) == pytest.approx((float(row['t_measured']) - prediction) / float(row['t_measured']))
        if row['phase'] == 'end_to_end':
            durations = [float(s['total']) for s in samples if s['n'] == row['n'] and s['phase'] == row['phase']]
            assert timing_statistics(durations) == (float(row['total_median_s']), float(row['total_iqr_s']))
            assert float(row['gflops']) == pytest.approx(2 * int(row['n']) ** 3 / (1e9 * float(row['t_measured'])))
            assert float(row['speedup']) == float(row['parallel_efficiency']) == 1
            assert float(row['baseline_time_s']) == float(row['t_measured'])
            assert row['baseline_P'] == row['baseline_H'] == '1'
            assert row['baseline_source'] == 'self'
        else:
            assert all(row[k] == '' for k in ('gflops', 'speedup', 'parallel_efficiency', 'baseline_time_s'))
    for item in manifest['traces']:
        trace = json.loads((output / 'traces' / item['name']).read_text())
        validate_trace_record(trace)
    from dmm.config import ModelConfig
    from dmm.model import build_model_tables
    from dmm.trace import load_traces
    model_config = ModelConfig(config.data_dir, output / 'offline', config.cases, config.sizes)
    records = load_traces(output / 'traces', model_config)
    assert len(records) == 2
    assert len(build_model_tables(records, model_config)[0]) == 16
    assert (output / 'heldout_residuals.png').stat().st_size > 1000
    with pytest.raises(FileExistsError):
        run_measurement(config)


def test_serial_single_node_has_zero_inactive_components_and_valid_native_summary(config):
    output = run_measurement(replace(config, cases=[dict(layout='cannon', P=1, c=1)]))
    rows = list(csv.DictReader((output / 'samples.csv').open()))
    assert all(float(r['total']) == 0 for r in rows if r['phase'] in ('replication', 'reduction', 'input_conversion', 'output_conversion'))
    assert timing_statistics([0, 0, 1], allow_zero=True) == (0, .5)
    with pytest.raises(ValueError):
        timing_statistics([0, 1])


def test_threaded_preflight_requires_physical_cores_without_creating_output(config, monkeypatch):
    monkeypatch.setattr('dmm.measurement.physical_cores', lambda: 1)
    with pytest.raises(ValueError, match='physical core'):
        run_measurement(replace(config, workers=2))
    assert not config.output_dir.exists()


def test_missing_teammate_algorithms_fail_preflight(config, monkeypatch):
    def load(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr('dmm.measurement.import_module', load)
    with pytest.raises(ValueError, match='not implemented'):
        run_measurement(replace(config, cases=[dict(layout='ring', P=1, c=1)]))
    assert not config.output_dir.exists()


def test_corrupt_timing_fails_before_completion_manifest(config, monkeypatch):
    from dmm.algorithms import cannon
    real_prepare = cannon.prepare
    class CorruptPrepared:
        def __init__(self, *args, **kwargs):
            self.real = real_prepare(*args, **kwargs)
        def run_into(self, output, executor):
            timing = self.real.run_into(output, executor)
            return type('Broken', (), {'phases': timing.phases, 'end_to_end': type('Timing', (), dict(total=-1, compute=0, copy=0, synchronization=0))()})()
    monkeypatch.setattr(cannon, 'prepare', CorruptPrepared)
    with pytest.raises(ValueError, match='invalid timing'):
        run_measurement(config)
    assert not (config.output_dir / 'manifest.json').exists()


def test_measurement_profile_paths_and_holdout_rules(tmp_path):
    import yaml
    path = tmp_path / 'measure.yaml'
    values = dict(data_dir='data', output_dir='run', cases=[dict(layout='cannon', P=1, c=1)], sizes=[256, 512])
    path.write_text(yaml.safe_dump(values))
    config = load_measurement_config(path)
    assert config.data_dir == tmp_path / 'data'
    assert (config.warmups, config.repeats) == (5, 15)
    for change in [dict(sizes=[64]), dict(workers=True), dict(warmups=-1), dict(repeats=0), dict(sizes=[512, 256])]:
        with pytest.raises(ValueError):
            replace(config, **change).validate()


def test_real_measurement_cli_works_outside_checkout(config, tmp_path):
    import subprocess
    import sys
    import yaml
    path = tmp_path / 'measurement.yaml'
    path.write_text(yaml.safe_dump(dict(data_dir=str(config.data_dir), output_dir=str(config.output_dir),
        cases=config.cases, sizes=[5], warmups=0, repeats=1)))
    cli = str(__import__('pathlib').Path(sys.executable).parent / 'dmm')
    result = subprocess.run([cli, 'measure', '--config', str(path)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Saved calibrated measurements' in result.stdout
    manifest = json.loads((config.output_dir / 'manifest.json').read_text())
    assert manifest['executed_configurations'] == 1


def _ends(directory):
    with (directory / 'measured_model.csv').open() as handle:
        return {int(row['n']): row for row in csv.DictReader(handle) if row['phase'] == 'end_to_end'}


@pytest.mark.parametrize('workers,P', [(0, 4), (0, 16), (2, 4)])
def test_oversubscribed_measurement_rejected_before_inputs_or_output(config, workers, P):
    bad = replace(config, data_dir=config.data_dir / 'does-not-exist',
                  cases=[dict(layout='cannon', P=P, c=1)], workers=workers)
    with pytest.raises(ValueError, match='P <= H.*dmm model'):
        run_measurement(bad)
    assert not config.output_dir.exists()


def test_thread_efficiency_uses_explicit_serial_baseline_and_real_H(config, monkeypatch):
    monkeypatch.setattr('dmm.measurement.physical_cores', lambda: 2)
    baseline = run_measurement(config)
    current = run_measurement(replace(config, output_dir=config.output_dir.with_name('threaded'), workers=2),
                              baseline=baseline)
    serial_rows = _ends(baseline)
    for n, row in _ends(current).items():
        assert row['P'] == '1' and row['H'] == '2'
        ratio = float(serial_rows[n]['t_measured']) / float(row['t_measured'])
        assert float(row['speedup']) == pytest.approx(ratio)
        assert float(row['parallel_efficiency']) == pytest.approx(ratio / 2)
        assert row['baseline_source'] == str(baseline.resolve())
        assert float(row['baseline_time_s']) == float(serial_rows[n]['t_measured'])
    manifest = json.loads((current / 'manifest.json').read_text())
    assert manifest['serial_baseline'] == dict(directory=str(baseline.resolve()),
        manifest_sha256=sha256((baseline / 'manifest.json').read_bytes()).hexdigest())


def test_missing_or_partial_baseline_leaves_efficiency_blank(config, monkeypatch):
    monkeypatch.setattr('dmm.measurement.physical_cores', lambda: 2)
    baseline = run_measurement(replace(config, sizes=[5]))
    threaded = replace(config, output_dir=config.output_dir.with_name('partial'), workers=2)
    rows = _ends(run_measurement(threaded, baseline=baseline))
    assert rows[5]['speedup'] != ''
    assert rows[12]['speedup'] == rows[12]['parallel_efficiency'] == rows[12]['baseline_time_s'] == ''
    no_baseline = run_measurement(replace(threaded, output_dir=config.output_dir.with_name('no-baseline')))
    assert all(row['speedup'] == row['parallel_efficiency'] == '' and float(row['gflops']) > 0
               for row in _ends(no_baseline).values())


@pytest.mark.parametrize('change', ['schema', 'boundary', 'implementation', 'host', 'protocol', 'worker', 'checksum'])
def test_incompatible_baseline_fails_preflight(config, monkeypatch, change):
    monkeypatch.setattr('dmm.measurement.physical_cores', lambda: 2)
    baseline = run_measurement(config)
    path = baseline / 'manifest.json'
    manifest = json.loads(path.read_text())
    if change == 'schema':
        manifest['schema'] = True
    elif change == 'boundary':
        manifest['boundary'] = 'includes allocation'
    elif change == 'implementation':
        manifest['implementation_sha256']['native'] = '0' * 64
    elif change == 'host':
        manifest['environment_before']['host'] = 'another-machine'
    elif change == 'protocol':
        manifest['settings']['repeats'] += 1
    elif change == 'worker':
        manifest['settings']['workers'] = 1
    else:
        with (baseline / 'measured_model.csv').open('a') as handle:
            handle.write('corrupted\n')
    path.write_text(json.dumps(manifest))
    current = replace(config, output_dir=config.output_dir.with_name('rejected'), workers=2)
    with pytest.raises(ValueError, match='baseline'):
        run_measurement(current, baseline=baseline)
    assert not current.output_dir.exists()


def test_changed_input_cannot_reuse_baseline_even_with_same_n(config, monkeypatch):
    import numpy as np
    monkeypatch.setattr('dmm.measurement.physical_cores', lambda: 2)
    baseline = run_measurement(config)
    path = config.data_dir / 'G0/A_5.npy'
    matrix = np.load(path)
    matrix[0, 0] = 0
    np.save(path, matrix)
    current = replace(config, output_dir=config.output_dir.with_name('changed-input'), workers=2)
    with pytest.raises(ValueError, match='baseline input hashes'):
        run_measurement(current, baseline=baseline)
    assert not (current.output_dir / 'manifest.json').exists()


def test_gflops_uses_original_n_when_padding_adds_work(config):
    from dmm.measurement import physical_cores
    import os
    if (physical_cores() or 0) < 4 or (os.cpu_count() or 1) < 4:
        pytest.skip('P=4 formal measurement requires four physical cores')
    output = run_measurement(replace(config, cases=[dict(layout='cannon', P=4, c=1)], workers=4))
    for n, row in _ends(output).items():
        assert float(row['gflops']) == pytest.approx(2 * n ** 3 / (1e9 * float(row['t_measured'])))
        if n == 5:
            assert int(row['n_padded']) == 8
            assert int(row['F_total']) == 2 * 8 ** 3


def test_measurement_cli_baseline_argument_is_forwarded(tmp_path, monkeypatch, capsys):
    import sys
    import yaml
    from dmm.cli import main
    path = tmp_path / 'measure.yaml'
    path.write_text(yaml.safe_dump(dict(data_dir='data', output_dir='output',
                                      cases=[dict(layout='cannon', P=1, c=1)], sizes=[256])))
    received = []
    def run(config, baseline=None):
        received.append(baseline)
        return config.output_dir
    monkeypatch.setattr('dmm.cli.run_measurement', run)
    monkeypatch.setattr(sys, 'argv', ['dmm', 'measure', '--config', str(path), '--baseline', 'saved-serial'])
    main()
    assert received == [__import__('pathlib').Path('saved-serial')]
    assert 'Saved calibrated measurements' in capsys.readouterr().out


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'nonfinite', 'truncated'])
def test_invalid_baseline_rows_are_rejected(config, monkeypatch, change):
    monkeypatch.setattr('dmm.measurement.physical_cores', lambda: 2)
    baseline = run_measurement(config)
    path = baseline / 'measured_model.csv'
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    first = next(r for r in rows if r['phase'] == 'end_to_end')
    if change == 'missing':
        rows.remove(first)
    elif change == 'duplicate':
        rows.append(first.copy())
    elif change == 'nonfinite':
        first['t_measured'] = 'inf'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        if change == 'truncated':
            handle.write('4,1\n')
    manifest_path = baseline / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['artifacts_sha256'][path.name] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    current = replace(config, output_dir=config.output_dir.with_name('invalid-rows'), workers=2)
    with pytest.raises(ValueError):
        run_measurement(current, baseline=baseline)
    assert not current.output_dir.exists()
