"""Independently replay selection and confirm paired statistics from saved records."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.public_benchmark import atomic_json, file_hash
from scripts.run_public_ridge_residual import OUT, ANCHOR, digest, config, validate_lock, paired_interval


def audit():
    plan = json.loads((OUT/'plan.json').read_text())
    lock = json.loads((OUT/'selection_lock.json').read_text())
    summary = json.loads((OUT/'development_summary.json').read_text())
    validate_lock(plan, lock)
    if len(summary['groups']) != 8 or len(summary['ridge']) != 8:
        raise ValueError('incomplete grid')
    if [r['alpha'] for r in summary['ridge']] != plan['ridge_alphas']:
        raise ValueError('ridge grid mismatch')
    expected_keys = [(w, p) for w in plan['widths'] for p in plan['penalties']]
    if [(g['width'], g['penalty']) for g in summary['groups']] != expected_keys:
        raise ValueError('grid order mismatch')
    excluded = set(range(128))
    for ids in plan['protocol']['sample_ids'].values():
        excluded.update(ids)
    if len(set(plan['confirmation_ids'])) != 500 or set(plan['confirmation_ids']) & excluded:
        raise ValueError('confirmation overlap')
    context = dict(plan_sha256=digest(plan), anchor_sha256=plan['anchor_sha256'])
    anchor = torch.load(ANCHOR, map_location='cpu', weights_only=False)['model']
    count, zero_selected, steps, max_closure = 0, 0, 0, 0.
    common_environment = None
    group_details = []
    for group in summary['groups']:
        values, selected_steps, stopped_steps = [], [], []
        for seed, entry in zip(plan['seeds'], group['runs']):
            expected_name = f"w{group['width']}_p{group['penalty']:g}_s{seed}"
            if entry['name'] != expected_name:
                raise ValueError('seed order mismatch')
            folder = OUT/entry['name']
            r = json.loads((folder/'result.json').read_text())
            cfg = config(plan, group['width'], group['penalty'], seed)
            if r['identity']['config'] != cfg or r['identity']['context'] != context:
                raise ValueError('configuration mismatch')
            if common_environment is not None and r['identity']['environment'] != common_environment:
                raise ValueError('environment mismatch')
            common_environment = r['identity']['environment']
            if file_hash(folder/'result.json') != entry['result_sha256']:
                raise ValueError('result hash mismatch')
            if file_hash(folder/'checkpoints/selected.pt') != entry['selected_sha256']:
                raise ValueError('checkpoint hash mismatch')
            cp = torch.load(folder/'checkpoints/selected.pt', map_location='cpu', weights_only=False)
            if cp['identity'] != r['identity'] or cp['step'] != r['best_step']:
                raise ValueError('selected state identity mismatch')
            for k in anchor:
                if not np.array_equal(anchor[k], cp['anchor'][k]):
                    raise ValueError('frozen anchor mutated')
            manifest = json.loads((folder/'manifest.json').read_text())
            if manifest['status'] != 'completed' or manifest['identity'] != r['identity']:
                raise ValueError('completion mismatch')
            if r['history'][0]['metrics']['val']['prediction'] != summary['baseline']['val']:
                raise ValueError('initial ridge prediction mismatch')
            best = r['history'][0]
            bad = 0
            for index, h in enumerate(r['history'][1:], 1):
                if h['step'] != index*plan['eval_every']:
                    raise ValueError('history step mismatch')
                value = h['metrics']['val']['prediction']['mean_relative_l2']
                if value < best['metrics']['val']['prediction']['mean_relative_l2']-plan['min_delta']:
                    best, bad = h, 0
                else:
                    bad += 1
                if h['step'] >= plan['min_steps'] and bad >= plan['patience'] and h != r['history'][-1]:
                    raise ValueError('continued beyond declared early stop')
                for split in ('train', 'val'):
                    max_closure = max(max_closure, abs(h['metrics'][split]['closure']))
            if r['best_step'] != best['step'] or r['metrics'] != best['metrics'] or r['metrics'] != entry['metrics']:
                raise ValueError('best selection mismatch')
            if r['stopped_step'] != r['history'][-1]['step']:
                raise ValueError('stop mismatch')
            if r['stopped_step'] < plan['steps'] and not (r['stopped_step'] >= plan['min_steps'] and bad >= plan['patience']):
                raise ValueError('unjustified early stop')
            values.append(r['metrics']['val']['prediction']['mean_relative_l2'])
            selected_steps.append(r['best_step'])
            stopped_steps.append(r['stopped_step'])
            count += 1
            steps += r['stopped_step']
            zero_selected += r['best_step'] == 0
        if len(group['runs']) != 5 or float(np.mean(values)) != group['mean_validation_l2'] or float(np.std(values, ddof=1)) != group['std_validation_l2']:
            raise ValueError('group statistics mismatch')
        group_details.append(dict(width=group['width'], penalty=group['penalty'],
            validation_mean=group['mean_validation_l2'], validation_std=group['std_validation_l2'],
            selected_steps=selected_steps, stopped_steps=stopped_steps))
    expected = min(summary['groups'], key=lambda g: g['mean_validation_l2'])
    if lock['selected_group'] != expected:
        raise ValueError('group selection mismatch')
    if lock['expanded_ridge'] != min(summary['ridge'], key=lambda r: r['metrics']['val']['mean_relative_l2']):
        raise ValueError('ridge selection mismatch')
    if lock['use_residual'] != (expected['mean_validation_l2'] < summary['baseline']['val']['mean_relative_l2']-plan['min_delta']):
        raise ValueError('fallback selection mismatch')
    result = dict(plan_sha256=digest(plan), completed_runs=count, total_updates=steps,
                  selected_initial_checkpoint_count=zero_selected, group_details=group_details,
                  frozen_anchor_exactly_preserved=True, max_squared_closure_error=max_closure)
    confirmation_path = OUT/'confirmation.json'
    if confirmation_path.exists():
        c = json.loads(confirmation_path.read_text())
        if c['ids'] != plan['confirmation_ids'] or c['identity'] != dict(plan_sha256=digest(plan), selection_lock_sha256=file_hash(OUT/'selection_lock.json')):
            raise ValueError('confirmation identity mismatch')
        if [r['name'] for r in c['per_seed']] != [r['name'] for r in lock['selected_group']['runs']]:
            raise ValueError('confirmation seed selection mismatch')
        residual = np.mean([r['per_trajectory_l2'] for r in c['per_seed']], axis=0)
        if residual.shape != (plan['confirmation_count'],) or c['use_residual'] != lock['use_residual']:
            raise ValueError('confirmation size/procedure mismatch')
        for name, values in c['baseline_per_trajectory_l2'].items():
            pair = paired_interval(residual-np.array(values), plan['bootstrap_seed'], plan['bootstrap_replicates'])
            if pair != c['paired_comparisons'][name]:
                raise ValueError('paired confidence interval mismatch')
        if float(np.mean(residual)) != c['residual_mean_l2']:
            raise ValueError('confirmation aggregate mismatch')
        base_mean = float(np.mean(c['baseline_per_trajectory_l2']['frozen_ridge']))
        gain = (base_mean-float(residual.mean()))/base_mean
        success = c['paired_comparisons']['frozen_ridge']['ci95'][1] < 0 and gain >= .01
        if gain != c['relative_gain_vs_frozen_ridge'] or success != c['primary_success']:
            raise ValueError('primary decision mismatch')
        result['confirmation_statistics_reproduced'] = True
        result['primary_success'] = c['primary_success']
    atomic_json(OUT/'audit.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    audit()
