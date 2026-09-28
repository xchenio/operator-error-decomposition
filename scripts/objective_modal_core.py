"""Weighted residual core; strict-CV fixed-step mode and masked corrections."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import numpy as np
import torch
from scripts.run_public_ridge_residual import (git_commit, event, prepare, make_branch,
    learning_rate, score_coefficients, env_fingerprint, atomic_json, atomic_save,
    random_state, restore_random, file_hash)


def sample_weights(y, objective):
    energy = np.mean(np.asarray(y, dtype=np.float64)**2, axis=1)
    floor = max(float(np.quantile(energy, .05)), 1e-12)
    if objective == 'mse':
        return np.ones(len(y)), floor
    if objective != 'normalized':
        raise ValueError('unknown objective')
    weights = 1/np.maximum(energy, floor)
    return weights/weights.mean(), floor


def stable_mask(fold_gains):
    """Rows=folds, columns=modes; folds already average two seed risks."""
    gains = np.asarray(fold_gains, dtype=np.float64)
    if gains.ndim != 2 or not np.isfinite(gains).all():
        raise ValueError('invalid gains')
    return ((gains > 0).all(axis=0) & (gains.mean(axis=0) >= .01)).astype(float)


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
    mask = torch.tensor(cfg['mask'], dtype=torch.float32, device=device)
    weights_np, floor = sample_weights(arrays['train_y'], cfg['objective'])
    weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
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
                delta = (branch(xs[split])*mask).cpu().numpy().astype(np.float64)*scale
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
    while step < end and not (not cfg['fixed_final'] and step >= cfg['min_steps'] and bad >= cfg['patience']):
        idx = rng.choice(len(target), size=min(cfg['batch_size'], len(target)), replace=False)
        opt.param_groups[0]['lr'] = learning_rate(cfg, step)
        opt.zero_grad(set_to_none=True)
        delta = branch(xs['train'][idx])*mask
        loss = (weights[idx]*((delta-target[idx]).square().sum(1)+cfg['penalty']*delta.square().sum(1))).mean()/arrays['train_y'].shape[1]
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward()
        opt.step()
        step += 1
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            history.append(evaluate())
            value = history[-1]['metrics']['val']['prediction']['mean_relative_l2']
            if (cfg['fixed_final'] and step == cfg['steps']) or (not cfg['fixed_final'] and value < best_value-cfg['min_delta']):
                best_value, best_step, bad = value, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
            else:
                bad += 1
            event(folder, 'evaluation', step=step, best_step=best_step, validation_l2=value)
            print(folder.name, step, 'val', round(value, 6), 'best_step', best_step, flush=True)
        if step % cfg['checkpoint_every'] == 0 or step == end:
            checkpoint()
    terminal = step >= cfg['steps'] or (not cfg['fixed_final'] and step >= cfg['min_steps'] and bad >= cfg['patience'])
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
        trainable_parameters=sum(p.numel() for p in branch.parameters()), denominator_floor=floor,
        weights_mean=float(np.mean(weights_np)), weight_max=float(np.max(weights_np)))
    atomic_json(result_path, result)
    atomic_json(manifest, dict(identity=identity, status='completed', git_commit=commit))
    atomic_json(folder/'progress.json', dict(step=step, best_step=best_step, status='completed'))
    event(folder, 'completed', step=step, best_step=best_step)
    return result
