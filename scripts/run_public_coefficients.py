"""Locked fixed-trunk conditioning x mean-lifting factorial intervention."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
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


def paired_bases(features, rcond=1e-6):
    """Equal-size, equal-RMS, same retained span; only singular scaling differs."""
    a = np.column_stack([np.asarray(features, dtype=np.float64), np.ones(len(features))])
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    keep = s > s[0]*rcond
    u, retained, vt = u[:, keep], s[keep], vt[keep]
    raw = (u*retained)@vt
    white = u@vt
    raw /= np.sqrt(np.mean(raw**2))
    white /= np.sqrt(np.mean(white**2))
    return raw.astype('float32'), white.astype('float32'), dict(
        rcond=rcond, rank=int(keep.sum()), singular_values=s.tolist(),
        retained_condition=float(retained[0]/retained[-1]), columns=a.shape[1])


def train_one(cfg, arrays, protocol, source, folder, device, stop_after=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    identity = dict(configuration=cfg, protocol=protocol, parent_checkpoint_sha256=file_hash(source),
                    source_sha256=hashlib.sha256((source_hash()+file_hash(Path(__file__))).encode()).hexdigest(),
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
    basis = torch.tensor(white if cfg['whiten'] else raw, device=device)
    set_seed(cfg['seed']+2000)
    branch = MLP(len(arrays['x']), basis.shape[1], 128, 3).to(device)
    optimizer = torch.optim.Adam(branch.parameters(), lr=cfg['lr'])
    rng = np.random.default_rng(cfg['seed']+3000)
    n = protocol['normalization']
    scale = max(n['target_std'], 1e-6)
    inputs = {s: torch.tensor((arrays[s+'_x']-n['input_mean'])/max(n['input_std'],1e-6), device=device)
              for s in ('train','val','test')}
    offsets = {s: arrays[s+'_x'].mean(axis=1, keepdims=True) if cfg['mean_lift'] else
               np.full((len(arrays[s+'_x']),1),n['target_mean'],dtype='float32') for s in inputs}
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
        return metrics(pred, arrays[split+'_y'])
    def save():
        atomic_save(dict(identity=identity,model=branch.state_dict(),optimizer=optimizer.state_dict(),
                         random=random_state(rng),step=step,train_seconds=elapsed,history=history,
                         milestones=milestones,basis=basis.detach().cpu(),basis_info=basis_info),cp)
        atomic_json(folder/'progress.json',dict(status='running',step=step,total_steps=cfg['steps']))
    while step < cfg['steps']:
        if device.type=='cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        end = min(step+cfg['checkpoint_every'],cfg['steps'])
        if stop_after is not None: end=min(end,stop_after)
        while step < end:
            idx = rng.choice(len(targets),min(32,len(targets)),replace=False)
            pred = branch(inputs['train'][idx])@basis.T
            loss = ((pred-targets[idx])**2).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); step+=1
        if device.type=='cuda': torch.cuda.synchronize()
        elapsed += time.perf_counter()-start
        if step%cfg['eval_every']==0 or step==cfg['steps']:
            row = dict(step=step,validation=evaluate('val'),train_seconds=elapsed,loss=float(loss.detach()))
            history.append(row); event('evaluation',**row)
        if step in (1500,cfg['steps']):
            milestones[str(step)] = {s:evaluate(s) for s in ('train','val','test')}
        save()
        if stop_after is not None and step>=stop_after: return None
    result = dict(identity=identity,git_commit=commit,basis_info=basis_info,history=history,
                  milestones=milestones,metrics=milestones[str(cfg['steps'])],train_seconds=elapsed,
                  trainable_parameters=sum(p.numel() for p in branch.parameters()),
                  selection='predeclared 1500 and final updates; no test-based selection',
                  limitations='frozen previously supervised trunk; extra training; samplewise constant allowed in every arm')
    atomic_json(final,result)
    atomic_json(manifest,dict(status='completed',identity=identity,git_commit=commit))
    atomic_json(folder/'progress.json',dict(status='completed',step=step))
    event('completed',step=step)
    print(folder.name,'val',round(result['metrics']['val']['mean_relative_l2'],5),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results/public_coefficients_v1')
    ap.add_argument('--steps',type=int,default=3000)
    ap.add_argument('--seeds',default='0,1,2,3,4')
    args=ap.parse_args()
    data=ROOT/'external_data/pdebench/fixed_t1_v1'
    protocol=json.loads((data/'protocol.json').read_text())
    if file_hash(data/'pairs.npz')!=protocol['cache_sha256']: raise ValueError('data changed')
    with np.load(data/'pairs.npz') as z: arrays={k:z[k].copy() for k in z.files}
    for architecture in ('deeponet_mlp','deeponet_fourier'):
        for seed in map(int,args.seeds.split(',')):
            source=ROOT/f'results/public_fixed_t1_v1/{architecture}_s{seed}/checkpoints/latest.pt'
            for whiten,mean_lift in ((False,False),(True,False),(False,True),(True,True)):
                cfg=dict(architecture=architecture,seed=seed,whiten=whiten,mean_lift=mean_lift,
                         steps=args.steps,lr=1e-3,checkpoint_every=50,eval_every=250)
                name=f'{architecture}_s{seed}_w{int(whiten)}_m{int(mean_lift)}'
                train_one(cfg,arrays,protocol,source,ROOT/args.out/name,torch.device('cuda'))


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
