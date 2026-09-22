"""Offline supervised coefficient intervention with frozen learned trunk (pilot)."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models import DeepONet, MLP
from src.pdes import make_problem
from src.data import dataset_cache_path
from src.resume import atomic_save, random_state, restore_random
from src.projection_audit import projection_audit
from src.utils import set_seed


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args()
    paths=[ROOT/'results'/d/n for d,n in [
        ('expV','burgers2d-w32_pi_baseline_nd0_anone1.0_s0.pt'),
        ('expV','burgers2d-w32_pi_align_nd0_aprior1.0_s0.pt'),
        ('expW','burgers2d-w32_pi_align_nd0_arandom1.0_s0.pt'),
        ('expW','burgers2d-fourier-w32_pi_baseline_nd0_anone1.0_s0.pt')]]
    out=ROOT/'results/coefficient_pilot'
    out.mkdir(parents=True,exist_ok=True)
    for path in paths:
        target=out/(path.stem+f'_branchseed{args.seed}_steps{args.steps}.json')
        if target.exists():
            continue
        set_seed(args.seed)
        ck=torch.load(path,map_location='cpu',weights_only=False)
        c=ck['cfg']
        p=make_problem(ck['pde'],**ck['pde_kwargs'])
        cache=dataset_cache_path(ROOT,ck['pde'],c['pde_kwargs'],c['n_train'],c['n_val'],c['n_test'])
        if not cache.exists():
            raise FileNotFoundError(cache)
        with np.load(cache) as z:
            splits={k:(z[k+'_u'].copy(),z[k+'_s'].T.astype(np.float64)) for k in ('train','val','test')}
        net=DeepONet(p.n_sensors,p.trunk_dim,**ck['model_kwargs'])
        net.load_state_dict(ck['state_dict'])
        with torch.no_grad():
            phi=net.trunk_features(torch.as_tensor(p.grid,dtype=torch.float32)).numpy().astype(np.float64)
        a=np.column_stack([phi,np.ones(len(phi))])
        q,sv,_=np.linalg.svd(a,full_matrices=False)
        q=q[:,sv>sv[0]*1e-12]
        mu=splits['train'][0].mean(axis=0)
        sd=splits['train'][0].std(axis=0).clip(1e-6)
        coef=q.T@splits['train'][1]
        scale=float(np.sqrt(np.mean(coef**2)))
        x=torch.tensor((splits['train'][0]-mu)/sd,dtype=torch.float32)
        y=torch.tensor(coef.T/scale,dtype=torch.float32)
        branch=MLP(x.shape[1],q.shape[1],128,3)
        opt=torch.optim.Adam(branch.parameters(),lr=1e-3)
        rng=np.random.default_rng(args.seed)
        checkpoint=out/'checkpoints'/(target.stem+'.pt')
        identity=dict(steps=args.steps,seed=args.seed,source=hashlib.sha256(path.read_bytes()).hexdigest(),
                      data=hashlib.sha256(cache.read_bytes()).hexdigest(),script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        start=0
        if checkpoint.exists():
            state=torch.load(checkpoint,weights_only=False)
            if state['identity']!=identity:
                raise ValueError('pilot resume mismatch')
            branch.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            restore_random(state['random'],rng); start=state['step']
        history=[]
        t0=time.monotonic()
        for step in range(start,args.steps):
            idx=rng.choice(len(x),min(32,len(x)),replace=False)
            loss=((branch(x[idx])-y[idx])**2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if (step+1)%50==0 or step+1==args.steps:
                atomic_save(dict(identity=identity,model=branch.state_dict(),optimizer=opt.state_dict(),
                                 random=random_state(rng),step=step+1),checkpoint)
                event=dict(step=step+1,loss=float(loss.detach()))
                history.append(event)
                with target.with_suffix('.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(event)+'\n')
        metrics={}
        for key,(inputs,s) in splits.items():
            with torch.no_grad():
                predicted=branch(torch.tensor((inputs-mu)/sd,dtype=torch.float32)).numpy().T*scale
                original=phi@net.branch_coeffs(torch.tensor(inputs)).numpy().astype(np.float64).T+float(net.bias.detach())
            metrics[key]=dict(original=projection_audit(a,s,original),
                              supervised_coefficients=projection_audit(a,s,q@predicted))
        report=dict(identity=identity,checkpoint=str(path.relative_to(ROOT)),metrics=metrics,
                    protocol='offline supervised frozen-basis pilot; 200 labeled training functions; relaxed samplewise constant; not a zero-data method',
                    latest_segment_seconds=time.monotonic()-t0)
        tmp=target.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2),encoding='utf-8'); tmp.replace(target)
        m=metrics['test']
        print(path.stem,'original=',m['original']['actual_relative_fro'],
              'after=',m['supervised_coefficients']['actual_relative_fro'],
              'oracle=',m['original']['projection_relative_fro'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
