"""Audit complete same-POD groups before drawing coefficient-map conclusions."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.public_benchmark import atomic_json, file_hash


def summarize():
    root = ROOT/'results/public_pod_readout_v1'
    summary = json.loads((root/'summary.json').read_text())
    if summary['completed_runs'] != 30 or len(summary['groups']) != 6:
        raise ValueError('incomplete experiment')
    baseline = json.loads((ROOT/'results/public_fixed_t1_v1/summary.json').read_text())['baselines']['val']['initial_spatial_mean']
    common = None
    comparisons = []
    coefficient_shares = []
    max_closure = 0.
    for rank in (16, 32, 64):
        arms = {}
        shared_context = None
        ridge = next(r for r in summary['ridge'] if r['rank'] == rank)
        for kind in ('linear', 'mlp'):
            runs = []
            for seed in range(5):
                folder = root/f'{kind}_r{rank}_s{seed}'
                record = json.loads((folder/'result.json').read_text())
                identity = record['identity']
                expected = dict(rank=rank, kind=kind, seed=seed, hidden=128, layers=3, steps=3000,
                                batch_size=32, lr=.001, lr_min=.00001, eval_every=250, checkpoint_every=50)
                if identity['config'] != expected:
                    raise ValueError('config mismatch')
                context = identity['context']
                if context['source'] != summary['source'] or context['protocol'] != summary['protocol']:
                    raise ValueError('summary identity mismatch')
                if shared_context is not None and context != shared_context:
                    raise ValueError('basis/preprocessing source mismatch')
                shared_context = context
                if file_hash(ROOT/context['pod_checkpoint']) != context['pod_checkpoint_sha256']:
                    raise ValueError('POD checkpoint changed')
                if common is not None and identity['environment'] != common:
                    raise ValueError('environment mismatch')
                common = identity['environment']
                manifest = json.loads((folder/'manifest.json').read_text())
                if manifest['status'] != 'completed' or manifest['identity'] != identity:
                    raise ValueError('manifest mismatch')
                if [h['step'] for h in record['history']] != list(range(0, 3001, 250)):
                    raise ValueError('history incomplete')
                if record['metrics'] != record['history'][-1]['metrics']:
                    raise ValueError('final metric mismatch')
                # Historical baseline used float32 spatial means; this experiment uses float64.
                if abs(record['history'][0]['metrics']['val']['prediction']['mean_relative_l2']-baseline)>1e-10:
                    raise ValueError('zero baseline mismatch')
                for h in record['history']:
                    for split in ('train', 'val'):
                        m = h['metrics'][split]
                        if m['oracle'] != ridge['metrics'][split]['oracle']:
                            raise ValueError('oracle mismatch')
                        max_closure = max(max_closure, abs(m['closure']))
                runs.append(record)
            arms[kind] = runs
            group = next(g for g in summary['groups'] if g['rank'] == rank and g['kind'] == kind)
            coefficient_shares.append(dict(rank=rank, kind=kind,
                validation_coefficient_energy_fraction=[r['metrics']['val']['coefficient_relative_fro']**2/
                    r['metrics']['val']['prediction']['relative_fro']**2 for r in runs],
                final_tail_train_coefficient=[r['history'][-1]['metrics']['train']['coefficient_relative_fro']-
                    r['history'][-4]['metrics']['train']['coefficient_relative_fro'] for r in runs],
                final_tail_validation_l2=[r['history'][-1]['metrics']['val']['prediction']['mean_relative_l2']-
                    r['history'][-4]['metrics']['val']['prediction']['mean_relative_l2'] for r in runs]))
            for split in ('train', 'val'):
                for metric in ('mean_relative_l2', 'relative_fro', 'rmse', 'coefficient_relative_fro'):
                    values = [r['metrics'][split][metric] if metric == 'coefficient_relative_fro'
                              else r['metrics'][split]['prediction'][metric] for r in runs]
                    expected_stats = dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)), values=values)
                    if group['metrics'][split][metric] != expected_stats:
                        raise ValueError('aggregate mismatch')
        pairs = {}
        for split in ('train', 'val'):
            for metric in ('mean_relative_l2', 'rmse'):
                delta = [m['metrics'][split]['prediction'][metric]-l['metrics'][split]['prediction'][metric]
                         for m, l in zip(arms['mlp'], arms['linear'])]
                pairs[split+'_'+metric] = dict(mlp_minus_linear=delta, improved_seeds=sum(d<0 for d in delta))
        comparisons.append(dict(rank=rank, paired=pairs,
            mlp_beats_ridge_val=sum(r['metrics']['val']['prediction']['mean_relative_l2']<ridge['metrics']['val']['prediction']['mean_relative_l2'] for r in arms['mlp']),
            mlp_beats_baseline_val=sum(r['metrics']['val']['prediction']['mean_relative_l2']<baseline for r in arms['mlp']),
            mlp_train_projection_fit=sum(r['metrics']['train']['coefficient_relative_fro']<=.01 for r in arms['mlp'])))
    audit = dict(completed_runs=30, initial_mean_validation_l2=baseline, comparisons=comparisons,
                 max_squared_frobenius_closure_error=max_closure, coefficient_shares=coefficient_shares,
                 source=summary['source'], environment=common,
                 interpretation='Development validation only. Nonlinearity, capacity, regularization and finite optimization remain distinct explanations. No significance or held-out claim.')
    atomic_json(root/'audit.json', audit)
    for g in summary['groups']:
        train, val = g['metrics']['train'], g['metrics']['val']
        print(g['rank'], g['kind'], 'train', train['mean_relative_l2'], 'val', val['mean_relative_l2'],
              'train_coefficient', train['coefficient_relative_fro']['mean'])
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == '__main__':
    summarize()
