"""Transfer the locked schedule control to 1000 functions, batch32 only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_public_benchmark import metrics, source_hash
from src.models import DeepONet, MLP
from src.public_benchmark import atomic_json, file_hash
from src.resume import atomic_save, random_state, restore_random
from src.utils import set_seed, env_fingerprint
from scripts.run_public_coefficients import paired_bases


def learning_rate(cfg, update_index):
    """Zero-based update index; recomputable from checkpoint step, no hidden scheduler state."""
    if cfg['schedule'] == 'constant': return cfg['lr']
    if cfg['schedule'] != 'cosine': raise ValueError('unknown schedule')
    phase = update_index/max(cfg['steps']-1,1)
    return cfg['min_lr'] + .5*(cfg['lr']-cfg['min_lr'])*(1+math.cos(math.pi*phase))


def train_one(cfg, arrays, protocol, source, folder, device, stop_after=None):
    if not cfg['zero_init'] or cfg['n_train'] != 1000: raise ValueError('locked zero-init n=1000 protocol')
    if cfg['batch_size'] != 32: raise ValueError('locked batch choices')
    learning_rate(cfg,0)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    identity = dict(configuration=cfg, protocol=protocol, parent_checkpoint_sha256=file_hash(source),
                    source_sha256=hashlib.sha256((source_hash()+file_hash(ROOT/'scripts/run_public_coefficients.py')+file_hash(Path(__file__))).encode()).hexdigest(),
                    environment=env_fingerprint(device))
    final = folder/'result.json'
    if final.exists():
        result = json.loads(final.read_text())
        if result['identity'] != identity: raise ValueError('completed identity mismatch')
        print('[skip]', folder.name, flush=True)
        return result
    manifest = folder/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity'] != identity:
        raise ValueError('manifest identity mismatch')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    atomic_json(manifest, dict(status='running', identity=identity, git_commit=commit))
    parent = torch.load(source, map_location='cpu', weights_only=False)
    if parent['identity']['data_protocol'] != protocol: raise ValueError('parent data mismatch')
    parent_cfg = parent['identity']['configuration']
    if parent_cfg['seed'] != cfg['seed'] or parent_cfg['architecture'] != cfg['architecture']:
        raise ValueError('parent model mismatch')
    trunk = DeepONet(branch_in=len(arrays['x']), trunk_in=1, width=64,
                    trunk_type='fourier' if cfg['architecture']=='deeponet_fourier' else 'mlp')
    trunk.load_state_dict(parent['model'])
    with torch.no_grad(): phi = trunk.trunk_features(torch.tensor(arrays['x'][:,None])).numpy()
    raw, white, basis_info = paired_bases(phi)
    basis = torch.tensor(white, device=device)
    subset = np.random.default_rng(20260923).permutation(len(arrays['train_x']))[:cfg['n_train']]
    if len(subset) != cfg['n_train']: raise ValueError('insufficient training samples')
    arrays = {k: v[subset] if k in ('train_x','train_y') else v for k,v in arrays.items()}
    basis_info['training_row_indices'] = subset.tolist()
    set_seed(cfg['seed']+2000)
    branch = MLP(len(arrays['x']), basis.shape[1], 128, 3).to(device)
    if cfg['zero_init']:
        torch.nn.init.zeros_(branch.net[-1].weight)
        torch.nn.init.zeros_(branch.net[-1].bias)
    optimizer = torch.optim.Adam(branch.parameters(), lr=cfg['lr'])
    rng = np.random.default_rng(cfg['seed']+3000)
    n = protocol['normalization']
    scale = max(n['target_std'], 1e-6)
    inputs = {s: torch.tensor((arrays[s+'_x']-n['input_mean'])/max(n['input_std'],1e-6), device=device)
              for s in ('train','val')}
    offsets = {s: arrays[s+'_x'].mean(axis=1, keepdims=True) for s in inputs}
    q = np.linalg.svd(white.astype('float64'), full_matrices=False)[0][:,:basis_info['rank']]
    oracles = {}
    for split in inputs:
        truth = arrays[split+'_y'].astype('float64')
        residual = truth-offsets[split]
        projected = (residual@q)@q.T+offsets[split]
        oracles[split] = metrics(projected, truth)
    targets = torch.tensor((arrays['train_y']-offsets['train'])/scale, device=device)
    step, elapsed, history, milestones = 0, 0., [], {}
    cp = folder/'checkpoints/latest.pt'
    if cp.exists():
        saved = torch.load(cp,map_location=device,weights_only=False)
        if saved['identity'] != identity: raise ValueError('checkpoint identity mismatch')
        branch.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        restore_random(saved['random'],rng)
        step, elapsed, history, milestones = saved['step'], saved['train_seconds'], saved['history'], saved['milestones']
    def event(kind, **kw):
        with (folder/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(event=kind,time_unix=time.time(),**kw))+'\n'); f.flush()
    event('resumed' if cp.exists() else 'started',step=step)
    @torch.no_grad()
    def evaluate(split):
        pred = torch.cat([branch(x)@basis.T for x in inputs[split].split(64)]).cpu().numpy()*scale+offsets[split]
        score = metrics(pred, arrays[split+'_y'])
        score['projection_excess_relative_fro'] = float(np.sqrt(max(0.,score['relative_fro']**2-oracles[split]['relative_fro']**2)))
        return score
    def save():
        atomic_save(dict(identity=identity,model=branch.state_dict(),optimizer=optimizer.state_dict(),
                         random=random_state(rng),step=step,train_seconds=elapsed,history=history,
                         milestones=milestones,basis=basis.detach().cpu(),basis_info=basis_info),cp)
        atomic_json(folder/'progress.json',dict(status='running',step=step,total_steps=cfg['steps'],next_lr=learning_rate(cfg,min(step,cfg['steps']-1))))
    if not cp.exists():
        initial = {s:evaluate(s) for s in ('train','val')}
        milestones['0'] = initial
        event('initial',metrics=initial)
        save()
    while step < cfg['steps']:
        if device.type=='cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        end = min(step+cfg['checkpoint_every'],cfg['steps'])
        if stop_after is not None: end=min(end,stop_after)
        while step < end:
            for group in optimizer.param_groups: group['lr']=learning_rate(cfg,step)
            idx = rng.choice(len(targets),min(cfg['batch_size'],len(targets)),replace=False)
            pred = branch(inputs['train'][idx])@basis.T
            loss = ((pred-targets[idx])**2).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); step+=1
        if device.type=='cuda': torch.cuda.synchronize()
        elapsed += time.perf_counter()-start
        if step%cfg['eval_every']==0 or step==cfg['steps']:
            row = dict(step=step,learning_rate=optimizer.param_groups[0]['lr'],examples_seen=step*cfg['batch_size'],training=evaluate('train'),validation=evaluate('val'),train_seconds=elapsed,loss=float(loss.detach()))
            history.append(row); event('evaluation',**row)
        if step in (750,1500,cfg['steps']):
            milestones[str(step)] = {s:evaluate(s) for s in ('train','val')}
        save()
        if stop_after is not None and step>=stop_after: return None
    result = dict(identity=identity,git_commit=commit,basis_info=basis_info,history=history,
                  milestones=milestones,metrics=milestones[str(cfg['steps'])],train_seconds=elapsed,oracles=oracles,
                  trainable_parameters=sum(p.numel() for p in branch.parameters()),
                  selection='predeclared 750/1500/final updates; no test selection',
                  examples_seen=step*cfg['batch_size'],schedule_state=dict(completed_updates=step,next_lr=learning_rate(cfg,min(step,cfg['steps']-1))),
                  timing_note='update time excluding validation/save; equal steps is not equal compute or data exposure',
                  limitations='trunk and normalization trained on original 1000 samples; small-subset fit diagnostic, NOT few-shot evaluation; validation only')
    atomic_json(final,result)
    atomic_json(manifest,dict(status='completed',identity=identity,git_commit=commit))
    atomic_json(folder/'progress.json',dict(status='completed',step=step))
    event('completed',step=step)
    print(folder.name,'train_excess',round(result['metrics']['train']['projection_excess_relative_fro'],5),'val',round(result['metrics']['val']['mean_relative_l2'],5),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results/public_scale1000_v1')
    ap.add_argument('--steps',type=int,default=3000)
    ap.add_argument('--seeds',default='0,1,2,3,4')
    args=ap.parse_args()
    data=ROOT/'external_data/pdebench/fixed_t1_v1'
    protocol=json.loads((data/'protocol.json').read_text())
    if file_hash(data/'pairs.npz')!=protocol['cache_sha256']: raise ValueError('data changed')
    with np.load(data/'pairs.npz') as z: arrays={k:z[k].copy() for k in z.files if not k.startswith('test_')}
    for architecture in ('deeponet_mlp','deeponet_fourier'):
        for seed in map(int,args.seeds.split(',')):
            source=ROOT/f'results/public_fixed_t1_v1/{architecture}_s{seed}/checkpoints/latest.pt'
            for schedule in ('constant','cosine'):
                for batch_size in (32,):
                    cfg=dict(architecture=architecture,seed=seed,n_train=1000,zero_init=True,
                             steps=args.steps,lr=1e-3,min_lr=1e-5,schedule=schedule,batch_size=batch_size,
                             checkpoint_every=50,eval_every=250)
                    name=f'{architecture}_s{seed}_{schedule}_b{batch_size}'
                    train_one(cfg,arrays,protocol,source,ROOT/args.out/name,torch.device('cuda'))


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
