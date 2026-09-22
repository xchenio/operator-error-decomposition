"""Re-evaluate saved checkpoints without overwriting historical results or retraining."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data import dataset_cache_path
from src.models import DeepONet
from src.pdes import make_problem
from src.projection_audit import projection_audit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dirs',default='expQ,expV,expW')
    ap.add_argument('--out',default='results/audit_v2')
    args=ap.parse_args()
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for directory in args.dirs.split(','):
        for path in sorted((ROOT/'results'/directory).glob('*.pt')):
            target=out/(directory+'__'+path.stem+'.json')
            if target.exists():
                continue
            ck=torch.load(path,map_location='cpu',weights_only=False)
            cfg=ck['cfg']
            pde=make_problem(ck['pde'],**ck['pde_kwargs'])
            cache=dataset_cache_path(ROOT,ck['pde'],cfg['pde_kwargs'],cfg['n_train'],cfg['n_val'],cfg['n_test'])
            if not cache.exists():
                raise FileNotFoundError(cache)  # never regenerate data implicitly
            model=DeepONet(branch_in=pde.n_sensors,trunk_in=pde.trunk_dim,**ck['model_kwargs'])
            model.load_state_dict(ck['state_dict'])
            model.eval()
            grid=torch.as_tensor(pde.grid,dtype=torch.float32)
            with torch.no_grad():
                phi=model.trunk_features(grid).numpy().astype(np.float64)
            # Include a constant column: conservative relaxed per-sample bias bound.
            a=np.column_stack([phi,np.ones(len(phi))])
            hard=getattr(pde,'hard_ic',False)
            if hard:
                a=a*pde.grid[:,0,None]
            rows={}
            with np.load(cache) as z:
                for split in ('train','val','test'):
                    s=z[split+'_s'].T.astype(np.float64)
                    with torch.no_grad():
                        b=model.branch_coeffs(torch.from_numpy(z[split+'_u'])).numpy().astype(np.float64)
                    y=phi@b.T+float(model.bias.detach())
                    if hard:
                        lift=pde.ic_values(z[split+'_coef'],pde.grid,0)[0].T
                        # Audit lifted error with original-solution normalization below.
                        y=lift+pde.grid[:,0,None]*y
                        s0,y0=s-lift,y-lift
                    else:
                        s0,y0=s,y
                    rows[split]=projection_audit(a,s0,y0)
                    if hard:
                        rows[split]['normalization']='lift-subtracted target; not original relative L2'
            payload=dict(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         dataset=str(cache.relative_to(ROOT)),dataset_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                         git_commit=commit,semantics='full grid; full target spectrum; relaxed per-sample bias; offline oracle',splits=rows)
            tmp=target.with_suffix('.tmp')
            tmp.write_text(json.dumps(payload,indent=2),encoding='utf-8')
            tmp.replace(target)
            r=rows['test']
            print(f'{directory}/{path.stem}: projection={r["projection_relative_fro"]:.4f} total={r["actual_relative_fro"]:.4f} closure={r["closure_error"]:.2e}',flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
