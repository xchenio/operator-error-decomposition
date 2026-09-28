"""Frozen ridge + small residual, development selection, sealed fresh confirmation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.public_benchmark import atomic_json, file_hash, extract_pair
from src.resume import atomic_save, random_state, restore_random
from src.utils import env_fingerprint
from scripts.run_public_benchmark import metrics, source_hash
from scripts.run_public_pod_readout import prepare, make_branch, learning_rate
from scripts.run_public_pod_ridge import fit_ridge, predict

OUT = ROOT/'results/public_ridge_residual_v1'
DATA = ROOT/'external_data/pdebench/fixed_t1_v1'
ANCHOR = ROOT/'results/public_pod_ridge_v1/checkpoints/r16_a10.pt'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_digest():
    files = [Path(__file__), ROOT/'scripts/run_public_pod_readout.py', ROOT/'scripts/run_public_pod_ridge.py']
    return hashlib.sha256((''.join(file_hash(p) for p in files)+source_hash()).encode()).hexdigest()


def git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def event(folder, name, **values):
    with (Path(folder)/'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(event=name, utc=datetime.now(timezone.utc).isoformat(), **values))+'\n')
        f.flush()
        os.fsync(f.fileno())


def choose_ids(protocol, total=10000, count=500):
    # Exclude all historical splits, plus conservatively reserve the generic smoke prefix.
    excluded = set(range(128))
    for values in protocol['sample_ids'].values():
        excluded.update(values)
    available = np.array(sorted(set(range(total))-excluded), dtype=np.int64)
    if len(available) < count:
        raise ValueError('insufficient unused trajectories')
    return np.random.default_rng(2026092801).permutation(available)[:count].tolist()


def locked_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('immutable record mismatch: '+str(path))
    else:
        atomic_json(path, value)


def create_plan():
    protocol = json.loads((DATA/'protocol.json').read_text())
    plan = dict(version=1, source=source_digest(), protocol=protocol,
        anchor_sha256=file_hash(ANCHOR), rank=16, widths=[16, 32], penalties=[0., .1, 1., 10.],
        seeds=list(range(5)), ridge_alphas=[.0001, .001, .01, .1, 1., 10., 100., 1000.],
        steps=3000, batch_size=32, lr=.001, lr_min=.00001, weight_decay=.0001,
        layers=1, eval_every=100, checkpoint_every=50, patience=5, min_steps=500, min_delta=.0001,
        confirmation_ids=choose_ids(protocol), confirmation_seed=2026092801,
        confirmation_count=500, bootstrap_seed=20260928, bootstrap_replicates=5000,
        selection='minimum mean development L2 across five seeds, each with baseline-inclusive early stopping; grid-order ties; no seed selection; fallback if gain <=1e-4',
        primary='mean of five per-trajectory relative L2 errors minus frozen alpha10 ridge; paired trajectory bootstrap 95% CI upper<0 AND relative mean gain>=1%',
        secondary='expanded-grid ridge and initial-mean comparisons; descriptive only; no retuning after confirmation')
    OUT.mkdir(parents=True, exist_ok=True)
    locked_json(OUT/'plan.json', plan)
    return plan


def config(plan, width, penalty, seed):
    keys = ('rank', 'layers', 'steps', 'batch_size', 'lr', 'lr_min', 'weight_decay',
            'eval_every', 'checkpoint_every', 'patience', 'min_steps', 'min_delta')
    return dict({k: plan[k] for k in keys}, kind='mlp', hidden=width, penalty=penalty, seed=seed)


def score_coefficients(coef, prepared, q):
    pred = coef@q.T+prepared['offset']
    scores = metrics(pred, prepared['y'])
    energy = np.sum(prepared['y']**2)
    coefficient_sq = float(np.sum((coef-prepared['coef'])**2)/energy)
    closure = scores['relative_fro']**2-prepared['oracle']['relative_fro']**2-coefficient_sq
    if abs(closure) > 1e-10:
        raise ValueError('projection closure failure')
    return dict(prediction=scores, oracle=prepared['oracle'],
                coefficient_relative_fro=float(np.sqrt(coefficient_sq)), closure=closure)


def train_one(cfg, arrays, anchor, context, folder, device, stop_after=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(config=cfg, context=context, environment=env_fingerprint(device))
    manifest = folder/'manifest.json'
    result_path = folder/'result.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity'] != identity:
        raise ValueError('identity mismatch')
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['identity'] != identity:
            raise ValueError('result identity mismatch')
        print('[skip]', folder.name, flush=True)
        return result
    commit = git_commit()
    atomic_json(manifest, dict(identity=identity, status='running', git_commit=commit))
    prepared, scale = prepare(arrays, anchor)
    bases = {s: p['x']@anchor['weights']+anchor['intercept'] for s, p in prepared.items()}
    branch = make_branch(cfg, prepared['train']['x'].shape[1], device)
    opt = torch.optim.AdamW(branch.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    rng = np.random.default_rng(cfg['seed']+3000)
    xs = {s: torch.tensor(p['x'], dtype=torch.float32, device=device) for s, p in prepared.items()}
    target = torch.tensor((prepared['train']['coef']-bases['train'])/scale, dtype=torch.float32, device=device)
    step, bad, best_step = 0, 0, 0
    history = []
    best_value = float('inf')
    best_state = None
    cp_path = folder/'checkpoints/latest.pt'
    if cp_path.exists():
        cp = torch.load(cp_path, map_location='cpu', weights_only=False)
        if cp['identity'] != identity:
            raise ValueError('checkpoint identity mismatch')
        branch.load_state_dict(cp['model'])
        opt.load_state_dict(cp['optimizer'])
        restore_random(cp['rng'], rng)
        step, bad, best_step, best_value = cp['step'], cp['bad'], cp['best_step'], cp['best_value']
        best_state, history = cp['best_state'], cp['history']
    event(folder, 'resume' if history else 'start', step=step, git_commit=commit)

    def evaluate():
        branch.eval()
        scores = {}
        with torch.no_grad():
            for split, p in prepared.items():
                delta = branch(xs[split]).cpu().numpy().astype(np.float64)*scale
                scores[split] = score_coefficients(bases[split]+delta, p, anchor['q'])
                scores[split]['correction_relative_fro'] = float(np.linalg.norm(delta)/np.linalg.norm(p['y']))
        branch.train()
        return dict(step=step, metrics=scores)

    def checkpoint():
        atomic_save(dict(identity=identity, step=step, bad=bad, best_step=best_step, best_value=best_value,
            model=branch.state_dict(), optimizer=opt.state_dict(), rng=random_state(rng),
            best_state=best_state, history=history, anchor=anchor, scale=scale), cp_path)
        atomic_json(folder/'progress.json', dict(step=step, best_step=best_step, bad=bad,
                                               last_evaluation=history[-1], status='running'))

    if not history:
        history.append(evaluate())
        best_value = history[0]['metrics']['val']['prediction']['mean_relative_l2']
        best_state = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
        checkpoint()
    end = min(stop_after, cfg['steps']) if stop_after is not None else cfg['steps']
    while step < end and not (step >= cfg['min_steps'] and bad >= cfg['patience']):
        idx = rng.choice(len(target), size=min(cfg['batch_size'], len(target)), replace=False)
        opt.param_groups[0]['lr'] = learning_rate(cfg, step)
        opt.zero_grad(set_to_none=True)
        delta = branch(xs['train'][idx])
        loss = ((delta-target[idx]).square().sum(1).mean()+cfg['penalty']*delta.square().sum(1).mean())/target.new_tensor(arrays['train_y'].shape[1])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward()
        opt.step()
        step += 1
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            history.append(evaluate())
            value = history[-1]['metrics']['val']['prediction']['mean_relative_l2']
            if value < best_value-cfg['min_delta']:
                best_value, best_step, bad = value, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
            else:
                bad += 1
            event(folder, 'evaluation', step=step, best_step=best_step, validation_l2=value)
            print(folder.name, step, 'val', round(value, 6), 'best_step', best_step, flush=True)
        if step % cfg['checkpoint_every'] == 0 or step == end:
            checkpoint()
    terminal = step >= cfg['steps'] or (step >= cfg['min_steps'] and bad >= cfg['patience'])
    if not terminal:
        atomic_json(manifest, dict(identity=identity, status='interrupted', git_commit=commit))
        event(folder, 'paused_for_test', step=step)
        return None
    checkpoint()
    selected = next(h for h in history if h['step'] == best_step)
    selected_path = folder/'checkpoints/selected.pt'
    atomic_save(dict(identity=identity, model=best_state, anchor=anchor, scale=scale, step=best_step), selected_path)
    result = dict(identity=identity, git_commit=commit, history=history, stopped_step=step,
        best_step=best_step, metrics=selected['metrics'], selected_sha256=file_hash(selected_path),
        stop_reason='patience' if step < cfg['steps'] else 'max_steps',
        trainable_parameters=sum(p.numel() for p in branch.parameters()))
    atomic_json(result_path, result)
    atomic_json(manifest, dict(identity=identity, status='completed', git_commit=commit))
    atomic_json(folder/'progress.json', dict(step=step, best_step=best_step, status='completed'))
    event(folder, 'completed', step=step, best_step=best_step)
    return result


def fit_fixed_basis_ridge(arrays, anchor, alpha):
    x = np.asarray(arrays['train_x'], dtype=np.float64)
    y = np.asarray(arrays['train_y'], dtype=np.float64)
    u, s, vt = np.linalg.svd((x-anchor['mean'])/anchor['std'], full_matrices=False)
    pre = dict(mean=anchor['mean'], std=anchor['std'], q=anchor['q'], u=u, s=s, vt=vt,
               residual=y-x.mean(axis=1, keepdims=True), n=len(x))
    return fit_ridge(pre, anchor['q'].shape[1], alpha)


def train_and_select(plan, device):
    if (OUT/'selection_lock.json').exists():
        lock = json.loads((OUT/'selection_lock.json').read_text())
        validate_lock(plan, lock)
        print('[skip] selection already sealed')
        return lock
    if file_hash(DATA/'pairs.npz') != plan['protocol']['cache_sha256']:
        raise ValueError('cache hash mismatch')
    with np.load(DATA/'pairs.npz') as z:
        arrays = {k: z[k].copy() for k in ('train_x', 'train_y', 'val_x', 'val_y')}
    saved = torch.load(ANCHOR, map_location='cpu', weights_only=False)
    if saved['identity']['protocol'] != plan['protocol']:
        raise ValueError('anchor protocol mismatch')
    anchor = saved['model']
    baseline = {s: metrics(predict(anchor, arrays[s+'_x']), arrays[s+'_y']) for s in ('train', 'val')}
    context = dict(plan_sha256=digest(plan), anchor_sha256=plan['anchor_sha256'])
    atomic_json(OUT/'manifest.json', dict(status='running_development', context=context, git_commit=git_commit()))
    ridge = []
    for alpha in plan['ridge_alphas']:
        path = OUT/'ridge'/f'a{alpha:g}.json'
        cp_path = OUT/'ridge/checkpoints'/f'a{alpha:g}.pt'
        if path.exists():
            record = json.loads(path.read_text())
            if record['context'] != context or file_hash(cp_path) != record['sha256']:
                raise ValueError('ridge record mismatch')
        else:
            model = fit_fixed_basis_ridge(arrays, anchor, alpha)
            scores = {s: metrics(predict(model, arrays[s+'_x']), arrays[s+'_y']) for s in ('train', 'val')}
            atomic_save(dict(context=context, model=model, alpha=alpha), cp_path)
            record = dict(context=context, alpha=alpha, metrics=scores, sha256=file_hash(cp_path))
            atomic_json(path, record)
        ridge.append(record)
    groups = []
    for width in plan['widths']:
        for penalty in plan['penalties']:
            runs = []
            for seed in plan['seeds']:
                name = f'w{width}_p{penalty:g}_s{seed}'
                r = train_one(config(plan, width, penalty, seed), arrays, anchor, context, OUT/name, device)
                runs.append(dict(name=name, best_step=r['best_step'], stopped_step=r['stopped_step'],
                    selected_sha256=r['selected_sha256'], result_sha256=file_hash(OUT/name/'result.json'), metrics=r['metrics']))
            values = [r['metrics']['val']['prediction']['mean_relative_l2'] for r in runs]
            groups.append(dict(width=width, penalty=penalty, mean_validation_l2=float(np.mean(values)),
                               std_validation_l2=float(np.std(values, ddof=1)), runs=runs))
            atomic_json(OUT/'development_summary.json', dict(context=context, groups=groups, ridge=ridge, baseline=baseline))
    selected = min(groups, key=lambda g: g['mean_validation_l2'])
    best_ridge = min(ridge, key=lambda r: r['metrics']['val']['mean_relative_l2'])
    lock = dict(plan_sha256=digest(plan), source=plan['source'], git_commit=git_commit(),
        selected_group=selected, expanded_ridge=best_ridge,
        use_residual=selected['mean_validation_l2'] < baseline['val']['mean_relative_l2']-plan['min_delta'],
        baseline=baseline, development_summary_sha256=file_hash(OUT/'development_summary.json'))
    locked_json(OUT/'selection_lock.json', lock)
    atomic_json(OUT/'manifest.json', dict(status='selection_locked', context=context, git_commit=git_commit()))
    print('SEALED', selected['width'], selected['penalty'], selected['mean_validation_l2'], 'use_residual', lock['use_residual'], flush=True)
    return lock


def validate_lock(plan, lock):
    if lock['plan_sha256'] != digest(plan) or lock['source'] != source_digest():
        raise ValueError('selection identity mismatch')
    if file_hash(OUT/'development_summary.json') != lock['development_summary_sha256']:
        raise ValueError('development summary changed')
    for r in lock['selected_group']['runs']:
        if file_hash(OUT/r['name']/'checkpoints/selected.pt') != r['selected_sha256'] or file_hash(OUT/r['name']/'result.json') != r['result_sha256']:
            raise ValueError('selected model or result changed')
    alpha = lock['expanded_ridge']['alpha']
    if file_hash(OUT/'ridge/checkpoints'/f'a{alpha:g}.pt') != lock['expanded_ridge']['sha256']:
        raise ValueError('expanded ridge changed')


def paired_interval(delta, seed=20260928, replicates=5000):
    delta = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(len(delta), size=(replicates, len(delta)))].mean(axis=1)
    return dict(mean=float(delta.mean()), ci95=np.quantile(draws, [.025, .975]).tolist())


def confirmation(plan, device):
    lock_path = OUT/'selection_lock.json'
    lock = json.loads(lock_path.read_text())
    validate_lock(plan, lock)
    # Model selection must be versioned before ANY new confirmation target is loaded.
    committed = json.loads(subprocess.check_output(['git', 'show', 'HEAD:'+lock_path.relative_to(ROOT).as_posix()], cwd=ROOT, text=True))
    if committed != lock:
        raise ValueError('selection lock must be committed before confirmation')
    result_path = OUT/'confirmation.json'
    identity = dict(plan_sha256=digest(plan), selection_lock_sha256=file_hash(lock_path))
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['identity'] != identity:
            raise ValueError('confirmation identity mismatch')
        print('[skip] completed confirmation; no labels reloaded')
        return result
    source = Path(plan['protocol']['source'])
    print('Verifying full source SHA-256 before confirmation', flush=True)
    if file_hash(source) != plan['protocol']['source_sha256']:
        raise ValueError('source data hash mismatch')
    cache = ROOT/'external_data/pdebench/confirmation_residual_v1'
    cache.mkdir(parents=True, exist_ok=True)
    chunks = []
    with h5py.File(source, 'r') as f:
        if list(f['tensor'].shape) != plan['protocol']['source_shape']:
            raise ValueError('source shape changed')
        for start in range(0, len(plan['confirmation_ids']), 50):
            ids = plan['confirmation_ids'][start:start+50]
            path = cache/f'chunk_{start:04d}.pt'
            chunk_identity = dict(identity=identity, ids=ids)
            if path.exists():
                chunk = torch.load(path, map_location='cpu', weights_only=False)
                if chunk['identity'] != chunk_identity:
                    raise ValueError('confirmation chunk mismatch')
            else:
                x, y = extract_pair(f['tensor'], ids, plan['protocol']['time_index'], plan['protocol']['spatial_stride'])
                chunk = dict(identity=chunk_identity, x=x, y=y)
                atomic_save(chunk, path)
            chunks.append(chunk)
            atomic_json(OUT/'confirmation_progress.json', dict(identity=identity, extracted=start+len(ids), total=len(plan['confirmation_ids'])))
    x = np.concatenate([c['x'] for c in chunks]).astype(np.float64)
    y = np.concatenate([c['y'] for c in chunks]).astype(np.float64)
    anchor = torch.load(ANCHOR, map_location='cpu', weights_only=False)['model']
    alpha = lock['expanded_ridge']['alpha']
    stronger = torch.load(OUT/'ridge/checkpoints'/f'a{alpha:g}.pt', map_location='cpu', weights_only=False)['model']
    denom = np.maximum(np.linalg.norm(y, axis=1), 1e-12)
    predictions = dict(frozen_ridge=predict(anchor, x), expanded_ridge=predict(stronger, x),
                       initial_mean=np.broadcast_to(x.mean(axis=1, keepdims=True), y.shape))
    per_seed = []
    for r in lock['selected_group']['runs']:
        cp = torch.load(OUT/r['name']/'checkpoints/selected.pt', map_location='cpu', weights_only=False)
        branch = make_branch(cp['identity']['config'], x.shape[1], device)
        branch.load_state_dict(cp['model'])
        branch.eval()
        with torch.no_grad():
            normalized = torch.tensor((x-anchor['mean'])/anchor['std'], dtype=torch.float32, device=device)
            correction = branch(normalized).cpu().numpy().astype(np.float64)*cp['scale']
        pred = predictions['frozen_ridge']+correction@anchor['q'].T if lock['use_residual'] else predictions['frozen_ridge'].copy()
        error = np.linalg.norm(pred-y, axis=1)/denom
        per_seed.append(dict(name=r['name'], best_step=r['best_step'], metrics=metrics(pred, y), per_trajectory_l2=error.tolist()))
    errors = {k: np.linalg.norm(v-y, axis=1)/denom for k, v in predictions.items()}
    residual_error = np.mean([r['per_trajectory_l2'] for r in per_seed], axis=0)
    comparisons = {k: paired_interval(residual_error-v, plan['bootstrap_seed'], plan['bootstrap_replicates']) for k, v in errors.items()}
    gain = float((errors['frozen_ridge'].mean()-residual_error.mean())/errors['frozen_ridge'].mean())
    result = dict(identity=identity, confirmation_git_commit=git_commit(), ids=plan['confirmation_ids'],
        use_residual=lock['use_residual'], baseline_metrics={k: metrics(v, y) for k, v in predictions.items()},
        baseline_per_trajectory_l2={k: v.tolist() for k, v in errors.items()}, per_seed=per_seed,
        residual_mean_l2=float(residual_error.mean()), residual_seed_std=float(np.std([r['metrics']['mean_relative_l2'] for r in per_seed], ddof=1)),
        paired_comparisons=comparisons, relative_gain_vs_frozen_ridge=gain,
        primary_success=bool(comparisons['frozen_ridge']['ci95'][1]<0 and gain>=.01),
        limitations='One fresh same-dataset confirmation split; bootstrap trajectories conditional on fixed five models, not 2500 independent units or cross-PDE proof. No retuning on confirmation.')
    atomic_json(result_path, result)
    atomic_json(OUT/'confirmation_progress.json', dict(identity=identity, status='completed', total=len(x)))
    atomic_json(OUT/'manifest.json', dict(status='completed', plan_sha256=digest(plan), git_commit=git_commit()))
    print('CONFIRMATION', result['residual_mean_l2'], result['paired_comparisons'], 'primary_success', result['primary_success'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['plan', 'train', 'confirm'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    plan = create_plan()
    if args.stage == 'plan':
        print('Locked', len(plan['confirmation_ids']), 'unused IDs; no confirmation labels loaded')
    elif args.stage == 'train':
        train_and_select(plan, torch.device(args.device))
    else:
        confirmation(plan, torch.device(args.device))


if __name__ == '__main__':
    main()
