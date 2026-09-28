"""Matched fixed-POD readouts; train/validation only, interruption-safe."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import MLP
from src.public_benchmark import atomic_json, file_hash
from src.resume import atomic_save, random_state, restore_random
from src.utils import set_seed, env_fingerprint
from scripts.run_public_benchmark import metrics, source_hash
from scripts.run_public_pod_ridge import predict


def prepare(arrays, model):
    """Use the exact stored float64 basis; one train-only scalar, no mode weighting."""
    q = model['q']
    if not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-12, rtol=0):
        raise ValueError('POD basis is not orthonormal')
    scale = float(np.asarray(arrays['train_y'], dtype=np.float64).std())
    if scale <= 0:
        raise ValueError('zero target scale')
    out = {}
    for split in ('train', 'val'):
        x = np.asarray(arrays[split + '_x'], dtype=np.float64)
        y = np.asarray(arrays[split + '_y'], dtype=np.float64)
        offset = x.mean(axis=1, keepdims=True)
        coef = (y - offset) @ q
        oracle = coef @ q.T + offset
        out[split] = dict(x=(x-model['mean'])/model['std'], y=y,
                          offset=offset, coef=coef, oracle=metrics(oracle, y))
    return out, scale


def make_branch(cfg, in_dim, device):
    set_seed(cfg['seed'] + 2000)
    if cfg['kind'] == 'mlp':
        branch = MLP(in_dim, cfg['rank'], cfg['hidden'], cfg['layers'])
        last = branch.net[-1]
    elif cfg['kind'] == 'linear':
        branch = torch.nn.Linear(in_dim, cfg['rank'])
        last = branch
    else:
        raise ValueError('unknown readout')
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.zeros_(last.bias)
    return branch.to(device)


def learning_rate(cfg, step):
    phase = step / max(cfg['steps'] - 1, 1)
    return cfg['lr_min'] + .5*(cfg['lr']-cfg['lr_min'])*(1+math.cos(math.pi*phase))


def train_one(cfg, arrays, model, identity_context, folder, device, stop_after=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(config=cfg, context=identity_context, environment=env_fingerprint(device))
    result_path = folder/'result.json'
    manifest_path = folder/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['identity'] != identity:
        raise ValueError('identity mismatch')
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['identity'] != identity:
            raise ValueError('result identity mismatch')
        print('[skip]', folder.name, flush=True)
        return result
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    atomic_json(manifest_path, dict(identity=identity, status='running', git_commit=commit))
    prepared, scale = prepare(arrays, model)
    branch = make_branch(cfg, prepared['train']['x'].shape[1], device)
    opt = torch.optim.Adam(branch.parameters(), lr=cfg['lr'])
    rng = np.random.default_rng(cfg['seed']+3000)
    xs = {s: torch.tensor(prepared[s]['x'], dtype=torch.float32, device=device) for s in prepared}
    target = torch.tensor(prepared['train']['coef']/scale, dtype=torch.float32, device=device)
    cp_path = folder/'checkpoints/latest.pt'
    history = []
    step = 0
    if cp_path.exists():
        cp = torch.load(cp_path, map_location='cpu', weights_only=False)
        if cp['identity'] != identity:
            raise ValueError('checkpoint identity mismatch')
        branch.load_state_dict(cp['model'])
        opt.load_state_dict(cp['optimizer'])
        restore_random(cp['rng'], rng)
        step, history = cp['step'], cp['history']

    def evaluate():
        branch.eval()
        scores = {}
        with torch.no_grad():
            for split, data in prepared.items():
                coef = branch(xs[split]).cpu().numpy().astype(np.float64)*scale
                pred = coef @ model['q'].T + data['offset']
                score = metrics(pred, data['y'])
                coefficient_sq = float(np.sum((coef-data['coef'])**2)/np.sum(data['y']**2))
                closure = score['relative_fro']**2 - data['oracle']['relative_fro']**2 - coefficient_sq
                if abs(closure) > 1e-10:
                    raise ValueError('projection closure failure')
                scores[split] = dict(prediction=score, oracle=data['oracle'],
                                     coefficient_relative_fro=math.sqrt(coefficient_sq), closure=closure)
        branch.train()
        return dict(step=step, lr=None if step == 0 else learning_rate(cfg, step-1), metrics=scores)

    def checkpoint():
        atomic_save(dict(identity=identity, step=step, model=branch.state_dict(),
                         optimizer=opt.state_dict(), rng=random_state(rng), history=history,
                         pod_model=model, coefficient_scale=scale), cp_path)
        atomic_json(folder/'progress.json', dict(step=step, total_steps=cfg['steps'],
                    last_evaluation=history[-1], status='running'))

    if not history:
        history.append(evaluate())
        checkpoint()
    end = min(cfg['steps'], stop_after) if stop_after is not None else cfg['steps']
    while step < end:
        index = rng.choice(len(target), size=min(cfg['batch_size'], len(target)), replace=False)
        opt.param_groups[0]['lr'] = learning_rate(cfg, step)
        opt.zero_grad(set_to_none=True)
        # Coefficient error / spatial dimension equals the nonconstant part of field MSE.
        loss = (branch(xs['train'][index])-target[index]).square().sum(dim=1).mean()/xs['train'].shape[1]
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        opt.step()
        step += 1
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            history.append(evaluate())
            print(folder.name, step, 'val', round(history[-1]['metrics']['val']['prediction']['mean_relative_l2'], 6), flush=True)
        if step % cfg['checkpoint_every'] == 0 or step == end:
            checkpoint()
    if step < cfg['steps']:
        return None
    result = dict(identity=identity, git_commit=commit, history=history, metrics=history[-1]['metrics'],
                  trainable_parameters=sum(p.numel() for p in branch.parameters()),
                  coefficient_scale=scale, selection='fixed final step; no seed/checkpoint selection')
    atomic_json(result_path, result)
    atomic_json(manifest_path, dict(identity=identity, status='completed', git_commit=commit))
    atomic_json(folder/'progress.json', dict(step=step, total_steps=cfg['steps'], status='completed',
                                           last_evaluation=history[-1]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data = ROOT/'external_data/pdebench/fixed_t1_v1'
    out = ROOT/'results/public_pod_readout_v1'
    ridge_dir = ROOT/'results/public_pod_ridge_v1'
    protocol = json.loads((data/'protocol.json').read_text())
    if file_hash(data/'pairs.npz') != protocol['cache_sha256']:
        raise ValueError('cache mismatch')
    with np.load(data/'pairs.npz') as z:
        arrays = {k: z[k].copy() for k in ('train_x', 'train_y', 'val_x', 'val_y')}
    source = hashlib.sha256((file_hash(Path(__file__))+file_hash(ROOT/'scripts/run_public_pod_ridge.py')+source_hash()).encode()).hexdigest()
    summary = dict(source=source, protocol=protocol, groups=[], ridge=[], completed_runs=0)
    for rank in (16, 32, 64):
        candidates = [json.loads(p.read_text()) for p in ridge_dir.glob(f'r{rank}_a*.json')]
        best = min(candidates, key=lambda r: r['metrics']['val']['prediction']['mean_relative_l2'])
        path = ridge_dir/'checkpoints'/f"r{rank}_a{best['alpha']:g}.pt"
        saved = torch.load(path, weights_only=False, map_location='cpu')
        if saved['identity'] != best['identity'] or saved['identity']['protocol'] != protocol:
            raise ValueError('ridge source identity mismatch')
        model = saved['model']
        for split in ('train', 'val'):
            check = metrics(predict(model, arrays[split+'_x']), arrays[split+'_y'])
            if check != best['metrics'][split]['prediction']:
                raise ValueError('ridge metric reproduction failed')
        summary['ridge'].append(dict(rank=rank, alpha=best['alpha'], metrics=best['metrics']))
        context = dict(protocol=protocol, source=source, pod_checkpoint_sha256=file_hash(path),
                       pod_checkpoint=str(path.relative_to(ROOT)), ridge_identity=best['identity'])
        for kind in ('linear', 'mlp'):
            records = []
            for seed in range(5):
                cfg = dict(rank=rank, kind=kind, seed=seed, hidden=128, layers=3, steps=3000,
                           batch_size=32, lr=.001, lr_min=.00001, eval_every=250, checkpoint_every=50)
                record = train_one(cfg, arrays, model, context, out/f'{kind}_r{rank}_s{seed}', torch.device(args.device))
                records.append(record)
            group = dict(rank=rank, kind=kind, seeds=list(range(5)), metrics={})
            for split in ('train', 'val'):
                group['metrics'][split] = {}
                for metric in ('mean_relative_l2', 'relative_fro', 'rmse'):
                    values = [r['metrics'][split]['prediction'][metric] for r in records]
                    group['metrics'][split][metric] = dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)), values=values)
                values = [r['metrics'][split]['coefficient_relative_fro'] for r in records]
                group['metrics'][split]['coefficient_relative_fro'] = dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)), values=values)
                group['metrics'][split]['oracle'] = records[0]['metrics'][split]['oracle']
            summary['groups'].append(group)
            summary['completed_runs'] += 5
            atomic_json(out/'summary.json', summary)
    print('Completed', summary['completed_runs'], flush=True)


if __name__ == '__main__':
    main()
