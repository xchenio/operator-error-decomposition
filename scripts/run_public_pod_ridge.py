"""Training-only uncentered residual POD with mean-loss ridge; validation selection only."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.public_benchmark import atomic_json,file_hash
from src.resume import atomic_save
from scripts.run_public_benchmark import metrics,source_hash


def fit_preprocess(x,y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    mean=x.mean(axis=0); std=x.std(axis=0).clip(1e-6)
    normalized=(x-mean)/std
    residual=y-x.mean(axis=1,keepdims=True)
    q=np.linalg.svd(residual,full_matrices=False)[2].T
    u,s,vt=np.linalg.svd(normalized,full_matrices=False)
    return dict(mean=mean,std=std,q=q,u=u,s=s,vt=vt,residual=residual,n=len(x))


def fit_ridge(pre,rank,alpha):
    if alpha<=0 or rank<1 or rank>pre['q'].shape[1]: raise ValueError('invalid ridge config')
    q=pre['q'][:,:rank]
    coef=pre['residual']@q
    intercept=coef.mean(axis=0)
    factors=pre['s']/(pre['s']**2+pre['n']*alpha)
    weights=(pre['vt'].T*factors)@(pre['u'].T@(coef-intercept))
    return dict(mean=pre['mean'],std=pre['std'],q=q,weights=weights,intercept=intercept)


def predict(model,x):
    x=np.asarray(x,dtype=np.float64)
    coef=((x-model['mean'])/model['std'])@model['weights']+model['intercept']
    return coef@model['q'].T+x.mean(axis=1,keepdims=True)


def main():
    data=ROOT/'external_data/pdebench/fixed_t1_v1'
    out=ROOT/'results/public_pod_ridge_v1'; out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((data/'protocol.json').read_text())
    if file_hash(data/'pairs.npz')!=protocol['cache_sha256']: raise ValueError('cache mismatch')
    identity=dict(protocol=protocol,ranks=[16,32,64],alphas=[.0001,.001,.01,.1,1.,10.],
                  source=hashlib.sha256((file_hash(Path(__file__))+source_hash()).encode()).hexdigest(),
                  environment=dict(python=platform.python_version(),numpy=np.__version__),
                  objective='mean-sample coefficient squared error + alpha*weight Frobenius squared; unpenalized intercept')
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity: raise ValueError('identity mismatch')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if (out/'summary.json').exists():
        if json.loads((out/'summary.json').read_text())['identity']!=identity: raise ValueError('summary mismatch')
        print('[skip] completed POD ridge grid'); return
    atomic_json(manifest,dict(status='running',identity=identity,git_commit=commit))
    with np.load(data/'pairs.npz') as z: arrays={k:z[k].copy() for k in z.files if not k.startswith('test_')}
    pre=fit_preprocess(arrays['train_x'],arrays['train_y'])
    records=[]
    for rank in identity['ranks']:
        for alpha in identity['alphas']:
            name=f'r{rank}_a{alpha:g}'
            path=out/(name+'.json')
            if path.exists():
                record=json.loads(path.read_text())
                if record['identity']!=identity or record['rank']!=rank or record['alpha']!=alpha: raise ValueError('case mismatch')
            else:
                model=fit_ridge(pre,rank,alpha)
                scores={}
                for split in ('train','val'):
                    x,y=arrays[split+'_x'].astype('float64'),arrays[split+'_y'].astype('float64')
                    offset=x.mean(axis=1,keepdims=True)
                    oracle=((y-offset)@model['q'])@model['q'].T+offset
                    scores[split]=dict(prediction=metrics(predict(model,x),y),oracle=metrics(oracle,y))
                record=dict(identity=identity,git_commit=commit,rank=rank,alpha=alpha,metrics=scores,
                            selection='all grid entries reported; final model selected on validation only')
                atomic_save(dict(identity=identity,model=model,rank=rank,alpha=alpha),out/'checkpoints'/(name+'.pt'))
                atomic_json(path,record)
            records.append(record)
            atomic_json(out/'progress.json',dict(completed_cases=len(records),total_cases=18))
            print(name,'val',round(record['metrics']['val']['prediction']['mean_relative_l2'],6),flush=True)
    best=min(records,key=lambda r:r['metrics']['val']['prediction']['mean_relative_l2'])
    summary=dict(identity=identity,completed_cases=18,selected=dict(rank=best['rank'],alpha=best['alpha'],metrics=best['metrics']),
                 cases=[dict(rank=r['rank'],alpha=r['alpha'],metrics=r['metrics']) for r in records],
                 selection_metric='validation mean relative L2, tie resolved by predeclared rank/alpha iteration order',
                 limitations='deterministic linear input-to-coefficient map; validation-tuned not held-out test; not full POD-DeepONet or two-step training')
    atomic_json(out/'summary.json',summary)
    atomic_json(manifest,dict(status='completed',identity=identity,git_commit=commit))
    print('Selected',best['rank'],best['alpha'],flush=True)


if __name__=='__main__': main()
