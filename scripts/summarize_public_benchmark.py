"""Summarize the locked five-seed public pilot without selecting test checkpoints."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.public_benchmark import atomic_json, file_hash


def score(pred, true):
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    return float(np.mean(np.linalg.norm(pred-true, axis=1) / np.maximum(np.linalg.norm(true, axis=1), 1e-12)))


def stats(values):
    return dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)), values=values)


def summarize(folder, data_dir):
    protocol = json.loads((folder/'protocol.json').read_text())
    if file_hash(data_dir/'pairs.npz') != protocol['cache_sha256']:
        raise ValueError('cache mismatch')
    groups = {}
    shared = None
    for architecture in ('deeponet_mlp', 'deeponet_fourier', 'fno'):
        runs = []
        for seed in range(5):
            path = folder/f'{architecture}_s{seed}'/'result.json'
            run = json.loads(path.read_text())
            identity = run['identity']
            cfg = identity['configuration']
            if cfg['architecture'] != architecture or cfg['seed'] != seed:
                raise ValueError(f'incorrect identity: {path}')
            if identity['data_protocol'] != protocol or cfg['steps'] != 1500:
                raise ValueError(f'protocol mismatch: {path}')
            common = {k: v for k, v in identity.items() if k != 'configuration'}
            common['configuration'] = {k: v for k, v in cfg.items() if k not in ('architecture', 'seed')}
            if shared is not None and shared != common:
                raise ValueError(f'incomparable runs: {path}')
            shared = common
            runs.append(run)
        group = dict(seeds=list(range(5)), real_parameter_count=runs[0]['real_parameter_count'],
                     train_seconds=stats([r['train_seconds'] for r in runs]),
                     result_files=[f'{architecture}_s{s}/result.json' for s in range(5)])
        for split in ('train', 'val', 'test'):
            group[split] = {metric: stats([r['metrics'][split][metric] for r in runs])
                            for metric in ('mean_relative_l2', 'relative_fro', 'rmse')}
            if architecture != 'fno':
                group[split]['projection'] = {}
                for key in ('projection', 'projection_rcond_1e6'):
                    audits = [r['metrics'][split][key] for r in runs]
                    group[split]['projection'][key] = {
                        metric: stats([a[metric] for a in audits]) for metric in
                        ('rank', 'projection_mean_relative_l2', 'projection_relative_fro',
                         'closure_error', 'prediction_outside_span')}
                    group[split]['projection'][key]['representation_share_squared_error'] = stats(
                        [a['representation_sq']/max(a['total_sq'], 1e-30) for a in audits])
        groups[architecture] = group
    with np.load(data_dir/'pairs.npz') as arrays:
        baselines = {}
        for split in ('val', 'test'):
            x, y = arrays[split+'_x'], arrays[split+'_y']
            baselines[split] = {
                'persistence': score(x, y),
                'initial_spatial_mean': score(np.broadcast_to(x.mean(axis=1, keepdims=True), y.shape), y),
                'training_target_mean_field': score(np.broadcast_to(arrays['train_y'].mean(axis=0), y.shape), y)}
    result = dict(protocol='public_fixed_t1_v1', completed_runs=15, groups=groups, baselines=baselines,
                  common_identity=shared, baseline_note='No fitted hyperparameters; mean-field baseline uses training targets only.',
                  limitations=['One PDE, one target time, one subset split, five training seeds.',
                               'Equal optimizer updates are not equal parameter count or compute.',
                               'Projection uses held-out labels offline and relaxed per-sample bias.',
                               'No significance, physics-informed, generalization-to-resolution or SOTA claim.'])
    atomic_json(folder/'summary.json', result)
    for name, group in groups.items():
        print(name, 'test L2', group['test']['mean_relative_l2'], 'parameters', group['real_parameter_count'])
        if name != 'fno':
            print('  oracle', group['test']['projection']['projection']['projection_mean_relative_l2'])
    print('Baselines:', baselines)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/public_fixed_t1_v1')
    parser.add_argument('--data-dir', default='external_data/pdebench/fixed_t1_v1')
    args = parser.parse_args()
    summarize(ROOT/args.out, ROOT/args.data_dir)
