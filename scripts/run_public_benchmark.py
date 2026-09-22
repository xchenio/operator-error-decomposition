"""Matched public supervised controls, step-resumable with immutable run identities."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.fno import FNO1d
from src.models import DeepONet
from src.projection_audit import projection_audit
from src.public_benchmark import atomic_json, file_hash
from src.resume import atomic_save, random_state, restore_random
from src.utils import set_seed, env_fingerprint


def source_hash():
    files = [Path(__file__), ROOT/'src/fno.py', ROOT/'src/models.py', ROOT/'src/resume.py',
             ROOT/'src/public_benchmark.py', ROOT/'src/projection_audit.py', ROOT/'src/utils.py',
             ROOT/'src/derivatives.py']
    h=hashlib.sha256()
    for path in files: h.update(path.name.encode()); h.update(path.read_bytes())
    return h.hexdigest()


def metrics(pred,true):
    pred,true=np.asarray(pred,dtype=np.float64),np.asarray(true,dtype=np.float64)
    residual=pred-true
    return dict(mean_relative_l2=float(np.mean(np.linalg.norm(residual,axis=1)/np.maximum(np.linalg.norm(true,axis=1),1e-12))),
                relative_fro=float(np.linalg.norm(residual)/max(np.linalg.norm(true),1e-12)),
                rmse=float(np.sqrt(np.mean(residual**2))))


def train_one(cfg, arrays, protocol, folder, device, stop_after=None):
    """stop_after is an integration-test interruption hook, excluded from run identity."""
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    set_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    identity=dict(configuration=cfg,source_sha256=source_hash(),data_protocol=protocol,
                  environment=env_fingerprint(device))
    result_path=folder/'result.json'
    if result_path.exists():
        previous=json.loads(result_path.read_text(encoding='utf-8'))
        if previous['identity']!=identity: raise ValueError(f'completed run identity mismatch: {folder}')
        print('[skip completed]',folder.name,flush=True)
        return previous
    manifest=folder/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:
        raise ValueError(f'run identity mismatch: {folder}')
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    atomic_json(manifest,dict(status='running',identity=identity,git_commit=git_commit))
    def event(kind,**kw):
        with (folder/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(time_unix=time.time(),event=kind,**kw))+'\n')
            f.flush()
    n=protocol['normalization']
    xmean,xstd=n['input_mean'],max(n['input_std'],1e-6)
    ymean,ystd=n['target_mean'],max(n['target_std'],1e-6)
    xs={k:torch.tensor((arrays[k+'_x']-xmean)/xstd,device=device) for k in ('train','val','test')}
    ys=torch.tensor((arrays['train_y']-ymean)/ystd,device=device)
    coords=torch.tensor(arrays['x'][:,None],device=device,dtype=torch.float32)
    if cfg['architecture']=='fno':
        model=FNO1d(width=32,modes=16,layers=4).to(device)
        predict=lambda inputs: model(inputs,coords)
    else:
        model=DeepONet(branch_in=xs['train'].shape[1],trunk_in=1,width=64,
                      branch_hidden=128,trunk_hidden=128,branch_layers=3,trunk_layers=3,
                      trunk_type='fourier' if cfg['architecture']=='deeponet_fourier' else 'mlp').to(device)
        predict=lambda inputs: model(inputs,coords)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    rng=np.random.default_rng(cfg['seed']+1000)
    cp=folder/'checkpoints/latest.pt'
    step,train_seconds,history=0,0.,[]
    if cp.exists():
        saved=torch.load(cp,map_location=device,weights_only=False)
        if saved['identity']!=identity: raise ValueError('checkpoint identity mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        restore_random(saved['random'],rng)
        step,train_seconds,history=saved['step'],saved['train_seconds'],saved['history']
        event('resumed',step=step)
    else: event('started',step=0)
    @torch.no_grad()
    def evaluate(split):
        model.eval()
        pred=torch.cat([predict(v) for v in xs[split].split(64)]).cpu().numpy()*ystd+ymean
        model.train()
        return pred,metrics(pred,arrays[split+'_y'])
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
    def save():
        atomic_save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
                         random=random_state(rng),step=step,history=history,train_seconds=train_seconds),cp)
        atomic_json(folder/'progress.json',dict(status='running',step=step,total_steps=cfg['steps'],train_seconds=train_seconds))
    while step<cfg['steps']:
        sync(); started=time.perf_counter()
        end=min(step+cfg['checkpoint_every'],cfg['steps'])
        if stop_after is not None: end=min(end,stop_after)
        while step<end:
            idx=rng.choice(len(ys),size=min(cfg['batch_size'],len(ys)),replace=False)
            pred=predict(xs['train'][idx])
            loss=((pred-ys[idx])**2).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            step+=1
        sync(); train_seconds+=time.perf_counter()-started
        if step%cfg['eval_every']==0 or step==cfg['steps']:
            _,val=evaluate('val')
            row=dict(step=step,loss=float(loss.detach()),validation=val,train_seconds=train_seconds)
            history.append(row); event('evaluation',**row)
            print(folder.name,step,'val=',round(val['mean_relative_l2'],5),'train_s=',round(train_seconds,2),flush=True)
        save()
        if stop_after is not None and step>=stop_after:
            event('interrupted_for_test',step=step)
            return None
    scores={}
    for split in ('train','val','test'):
        pred,score=evaluate(split)
        scores[split]=score
        if cfg['architecture']!='fno':
            with torch.no_grad(): phi=model.trunk_features(coords).cpu().numpy().astype('float64')
            basis=np.column_stack([phi,np.ones(len(phi))])
            score['projection']=projection_audit(basis,arrays[split+'_y'].T,pred.T)
            score['projection_rcond_1e6']=projection_audit(basis,arrays[split+'_y'].T,pred.T,rcond=1e-6)
    parameters=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
    result=dict(identity=identity,git_commit=git_commit,metrics=scores,history=history,
                train_seconds=train_seconds,real_parameter_count=parameters,selection='last fixed-budget checkpoint',
                timing_note='optimizer update time includes batch sampling; excludes validation/checkpoint and final diagnostics',
                projection_note='DeepONet only; offline relaxed per-sample bias; not pure optimization attribution')
    atomic_json(result_path,result)
    atomic_json(manifest,dict(status='completed',identity=identity,git_commit=git_commit))
    atomic_json(folder/'progress.json',dict(status='completed',step=step,train_seconds=train_seconds))
    event('completed',step=step,test=scores['test'])
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='external_data/pdebench/fixed_t1_v1')
    ap.add_argument('--out',default='results/public_fixed_t1_v1')
    ap.add_argument('--steps',type=int,default=1500)
    ap.add_argument('--seeds',default='0,1,2,3,4')
    ap.add_argument('--architectures',default='deeponet_mlp,deeponet_fourier,fno')
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args()
    data_dir=ROOT/args.data_dir
    protocol=json.loads((data_dir/'protocol.json').read_text())
    if file_hash(data_dir/'pairs.npz')!=protocol['cache_sha256']: raise ValueError('pair cache changed')
    with np.load(data_dir/'pairs.npz') as z: arrays={k:z[k].copy() for k in z.files}
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'protocol.json',protocol)
    device=torch.device(args.device)
    for architecture in args.architectures.split(','):
        if architecture not in ('deeponet_mlp','deeponet_fourier','fno'): raise ValueError(architecture)
        for seed in map(int,args.seeds.split(',')):
            cfg=dict(architecture=architecture,seed=seed,steps=args.steps,lr=1e-3,batch_size=32,
                     checkpoint_every=50,eval_every=250)
            train_one(cfg,arrays,protocol,out/f'{architecture}_s{seed}',device)


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
