"""Independent local calibration and held-out, allocation-free matrix measurements."""

from dataclasses import asdict
from hashlib import sha256
from importlib import import_module
import json
from math import isfinite
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

from dmm import _native
from dmm.analysis import timing_statistics
from dmm.benchmark import _validate_product, source_revision
from dmm.config import LAYOUT_MODULES, MeasurementConfig
from dmm.datasets import load_manifest, load_pair, load_reference
from dmm.execution import SerialExecutor, ThreadExecutor
from dmm.model import MODEL_FIELDS, _csv, _write_json, _trace_name, save_trace_manifest, trace_record, validate_trace_record

PARTS = ('total', 'compute', 'copy', 'synchronization')
MICROS = [('compute', n, 16) for n in (32, 64, 96)] + [('copy', n, 128) for n in (1024, 4096, 16384)] + [
    ('synchronization', 1, 100)]


def physical_cores():
    """Return a detected physical core count; never treat logical CPUs as physical."""
    if platform.system() == 'Darwin':
        result = subprocess.run(['/usr/sbin/sysctl', '-n', 'hw.physicalcpu'], capture_output=True, text=True, check=True)
        return int(result.stdout)
    if platform.system() == 'Linux':
        pairs = set()
        allowed = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
        for record in Path('/proc/cpuinfo').read_text().split('\n\n'):
            fields = dict(line.split(':', 1) for line in record.splitlines() if ':' in line)
            fields = {k.strip(): v.strip() for k, v in fields.items()}
            if allowed is not None and int(fields.get('processor', -1)) not in allowed:
                continue
            if 'physical id' in fields and 'core id' in fields:
                pairs.add((fields['physical id'], fields['core id']))
        return len(pairs) or None
    return None


def _positive_time(value):
    if not isfinite(value) or value <= 0:
        raise ValueError('measurement must return finite positive seconds')
    return value


def calibrate(executor, H, warmups, repeats):
    """Freeze one w/g/l per executor context using only independent microkernels."""
    rows, rates = [], {kind: [] for kind in ('compute', 'copy', 'synchronization')}
    for kind, size, iterations in MICROS:
        units = iterations * (2 * size ** 3 if kind == 'compute' else size if kind == 'copy' else 1)
        for _ in range(warmups):
            _positive_time(_native.microbenchmark(kind, size, iterations, H, executor))
        for repeat in range(repeats):
            seconds = _positive_time(_native.microbenchmark(kind, size, iterations, H, executor))
            rates[kind].append(seconds / units)
            rows.append(dict(kind=kind, size=size, iterations=iterations, H=H, repeat=repeat,
                             seconds=seconds, units=units, seconds_per_unit=seconds / units))
    parameters, variability = {}, {}
    for key, kind in zip(('w', 'g', 'l'), ('compute', 'copy', 'synchronization')):
        median, iqr = timing_statistics(rates[kind])
        parameters[key], variability[key] = median, iqr
    return parameters, variability, rows


def _environment(cores):
    return dict(system=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                physical_cores=cores, logical_cpus=os.cpu_count(), compiler=_native.compiler,
                build_type=_native.build_type, load_average=list(os.getloadavg()),
                affinity_policy='unbound persistent std::threads; static round-robin logical tasks; same policy for micro/full runs',
                affinity_mask=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)


def _timing_rows(timing):
    try:
        phases = timing.phases
        if len(phases) != len(_native.phase_names):
            raise ValueError('invalid timing phase count')
        values = [('end_to_end', timing.end_to_end), *zip(_native.phase_names, phases)]
    except AttributeError as error:
        raise ValueError('invalid timing protocol') from error
    for phase, value in values:
        parts = {name: getattr(value, name) for name in PARTS}
        if any(not isfinite(v) or v < 0 for v in parts.values()):
            raise ValueError('invalid timing decomposition')
        if phase == 'end_to_end':
            _positive_time(parts['total'])
        if not np.isclose(parts['total'], sum(parts[p] for p in PARTS[1:]), rtol=1e-9, atol=1e-9):
            raise ValueError('timing components do not add up to wall time')
        yield phase, parts


def run_measurement(config: MeasurementConfig):
    """No per-matrix fit: save raw samples, frozen parameters, traces and residuals."""
    config.validate()
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f'measurement output already exists: {output}')
    cores = physical_cores()
    if config.workers and (cores is None or config.workers > cores):
        raise ValueError('threaded validation requires a detected physical core count >= workers')
    modules = []
    for case in config.cases:
        name = f"dmm.algorithms.{LAYOUT_MODULES[case['layout']]}"
        try:
            module = import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
            raise ValueError(f"Layout {case['layout']!r} is not implemented") from error
        if not callable(getattr(module, 'prepare', None)) or not callable(getattr(module, 'validate_size', None)):
            raise ValueError(f"Layout {case['layout']!r} must expose prepare and validate_size")
        for n in config.sizes:
            module.validate_size(n, P=case['P'], c=case['c'])
        modules.append((case, module))
    manifest = load_manifest(config.data_dir)
    if config.group not in manifest.groups or not set(config.sizes) <= set(manifest.sizes):
        raise ValueError('measurement group/sizes must be present in dataset manifest')
    output.mkdir(parents=True, exist_ok=False)
    before = _environment(cores)
    samples, summaries, traces, correctness, trace_correctness = [], [], [], [], []
    trace_dir = output / 'traces'
    trace_dir.mkdir()
    H = config.workers or 1
    executor = ThreadExecutor(config.workers) if config.workers else SerialExecutor()
    with executor:
        parameters, variability, micro_rows = calibrate(executor, H, config.warmups, config.repeats)
        _csv(output / 'microbenchmarks.csv', ('kind', 'size', 'iterations', 'H', 'repeat', 'seconds', 'units', 'seconds_per_unit'), micro_rows)
        _write_json(output / 'calibration.json', dict(parameters_seconds=parameters, iqr_seconds_per_unit=variability,
            H=H, warmups=config.warmups, repeats=config.repeats, micro_sizes=[32, 64, 96],
            method='median/IQR of equally weighted normalized micro samples across sizes; frozen before any matrix run',
            compute='critical local shared GEMM seconds/FLOP', copy='critical local send+receive copy seconds/word',
            synchronization='wall seconds/logical superstep with three empty blocking executor batches',
            caveat='effective local emulator coefficients, not a measured network; no per-case refit'))
        for case, module in modules:
            for n in config.sizes:
                a, b = load_pair(config.data_dir, config.group, n)
                reference = load_reference(config.data_dir, config.group, n)
                hashes = {'A': sha256(a.tobytes()).hexdigest(), 'B': sha256(b.tobytes()).hexdigest()}
                a.flags.writeable = b.flags.writeable = False
                prepared = module.prepare(a, b, P=case['P'], c=case['c'])
                result = np.empty_like(a, order='C')
                for _ in range(config.warmups):
                    prepared.run_into(result, executor)
                    _validate_product(result, reference)
                case_samples = []
                for repeat in range(config.repeats):
                    timing = prepared.run_into(result, executor)
                    errors = _validate_product(result, reference)
                    for phase, parts in _timing_rows(timing):
                        row = dict(n=n, **case, H=H, phase=phase, repeat=repeat, **parts)
                        samples.append(row); case_samples.append(row)
                    correctness.append(dict(n=n, **case, H=H, repeat=repeat, **errors))
                record = trace_record(prepared.trace(), case, H, errors, hashes)
                path = trace_dir / _trace_name(n, case)
                _write_json(path, record)
                traces.append(dict(name=path.name, sha256=sha256(path.read_bytes()).hexdigest()))
                trace_correctness.append(dict(n=n, **case, **errors))
                metrics = validate_trace_record(record)
                for phase, counts in metrics.items():
                    if phase == 'native':
                        relevant = [row for row in case_samples if row['phase'] not in ('end_to_end', 'input_conversion', 'output_conversion')]
                        repeats = [{part: sum(r[part] for r in relevant if r['repeat'] == i) for part in PARTS}
                                   for i in range(config.repeats)]
                    else:
                        repeats = [row for row in case_samples if row['phase'] == phase]
                    parts = {}
                    for part in PARTS:
                        median, iqr = timing_statistics([r[part] for r in repeats], allow_zero=True)
                        parts[f'{part}_median_s'], parts[f'{part}_iqr_s'] = median, iqr
                    w, g, l = (parameters[k] for k in ('w', 'g', 'l'))
                    prediction = w * counts['W_crit'] + g * counts['C_crit'] + l * counts['S']
                    measured = parts['total_median_s']
                    summaries.append(dict(n=n, n_padded=record['n_padded'], **case, H=H, phase=phase, **counts,
                        gamma=g / w, **{'lambda': l / w}, T_model=prediction / w, t_measured=measured,
                        w=w, g=g, l=l, T_model_seconds=prediction, residual_seconds=measured - prediction,
                        relative_residual=(measured - prediction) / measured, **parts))
    save_trace_manifest(config, trace_dir, traces, trace_correctness,
                        'prepared measurement trace; retained cache memory counted; use measured_model.csv for statistical times')
    sample_fields = ('n', 'layout', 'P', 'c', 'H', 'phase', 'repeat', *PARTS)
    extra_fields = ('w', 'g', 'l', 'T_model_seconds', 'residual_seconds', 'relative_residual',
                    *(f'{part}_{stat}_s' for part in PARTS for stat in ('median', 'iqr')))
    _csv(output / 'samples.csv', sample_fields, samples)
    _csv(output / 'measured_model.csv', (*MODEL_FIELDS, *extra_fields), summaries)
    _csv(output / 'correctness.csv', ('n', 'layout', 'P', 'c', 'H', 'repeat', 'max_abs_error', 'frobenius_relative_error'), correctness)
    _plot_residuals(summaries, output)
    settings = asdict(config)
    settings['data_dir'], settings['output_dir'] = str(config.data_dir), str(output)
    artifacts = {p.name: sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    _write_json(output / 'manifest.json', dict(schema=1, settings=settings, git=source_revision(),
        environment_before=before, environment_after=_environment(cores), artifacts_sha256=artifacts,
        trace_manifest_sha256=sha256((trace_dir / 'manifest.json').read_bytes()).hexdigest(),
        traces=traces, executed_configurations=len(modules) * len(config.sizes),
        input_manifest=json.loads((Path(config.data_dir) / 'manifest.json').read_text()),
        boundary='common row-shard input to common row-shard output; includes packing/conversion/copies/zeroing/compute/scheduling; excludes allocation/host adapters/validation',
        model_units='T_model is FLOP-equivalents; T_model_seconds and t_measured are seconds',
        holdout='full matrix sizes never used to fit w/g/l; frozen micro calibration only; residuals are reported without refitting'))
    return output


def _plot_residuals(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    end = [row for row in rows if row['phase'] == 'end_to_end']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    labels = [f"{r['layout']} P={r['P']} n={r['n']}" for r in end]
    x = np.arange(len(end))
    axes[0].bar(x - .2, [r['t_measured'] for r in end], .4, label='Measured median')
    axes[0].bar(x + .2, [r['T_model_seconds'] for r in end], .4, label='Frozen micro model')
    axes[0].set_ylabel('Seconds'); axes[0].legend()
    axes[1].bar(x, [100 * r['relative_residual'] for r in end])
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set_ylabel('(Measured - predicted) / measured (%)')
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha='right')
    fig.suptitle('Held-out full matrices: local emulator calibration, no refitting')
    fig.savefig(output / 'heldout_residuals.png', dpi=160)
    plt.close(fig)
