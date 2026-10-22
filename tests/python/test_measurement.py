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
from dmm.model import validate_trace_record


@pytest.fixture
def config(tmp_path):
    data = tmp_path / 'data'
    generate_dataset(DatasetConfig(data, 51, [0], [5, 12], [-1, 1]))
    return MeasurementConfig(data, tmp_path / 'measured', [dict(layout='cannon', P=4, c=1)], [5, 12], warmups=1, repeats=3)


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
    assert {r['phase'] for r in models} == {'end_to_end', 'native', 'input_conversion', 'skew', 'multiply', 'output_conversion'}
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
    for item in manifest['traces']:
        trace = json.loads((output / 'traces' / item['name']).read_text())
        validate_trace_record(trace)
    from dmm.config import ModelConfig
    from dmm.model import load_traces, build_model_tables
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


def test_missing_teammate_algorithms_fail_preflight(config):
    with pytest.raises(ValueError, match='not implemented'):
        run_measurement(replace(config, cases=[dict(layout='ring', P=4, c=1)]))
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
