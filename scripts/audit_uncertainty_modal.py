"""Recompute conditional uncertainty and replay frozen predictions, no fresh extraction."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_uncertainty_modal import plan,OUT,collect,compare
from scripts.uncertainty_shrink import bootstrap_moment,coefficients,matched_global
from scripts.run_public_pod_ridge import predict
from scripts.run_incremental_modal import errors
from scripts.audit_incremental_modal import branch,same
from src.public_benchmark import atomic_json,file_hash


def audit():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);p=plan();tasks={}
    for task in p['tasks']:
        folder=OUT/task;s=json.loads((folder/'development.json').read_text())
        for path,h in s['artifacts'].items():
            if file_hash(ROOT/path)!=h:raise ValueError('artifact changed')
        arrays,root,rules,nb,db=collect(task,torch.device('cuda'))
        path=folder/'checkpoints/bootstrap.pt'
        if file_hash(path)!=s['bootstrap_sha256']:raise ValueError('bootstrap hash changed')
        cp=torch.load(path,weights_only=False)
        draws=np.array([bootstrap_moment(nb,i,p['bootstrap_seed']) for i in range(p['bootstrap_replicates'])])
        np.testing.assert_array_equal(draws,np.array(cp['draws']))
        sd=draws.std(0,ddof=1);same(sd[0],s['sd_trajectory']);same(sd[1],s['sd_hierarchical'])
        n=np.mean([v.mean((0,1)) for v in nb],0);d=np.mean([v.mean((0,1)) for v in db],0)
        same(n,s['N']);same(d,s['D']);old=coefficients(n,d,np.zeros(16));same(old,rules['masks']['shrink'])
        new=coefficients(n,d,sd[1],p['kappa']);trajectory=coefficients(n,d,sd[0],p['kappa'])
        matched,g=matched_global(old,new,d)
        masks=dict(none=np.zeros(16),all=np.ones(16),old_shrink=old,uncertainty=new,trajectory_only=trajectory,matched_global=matched)
        for arm,value in masks.items():same(value,s['masks'][arm])
        same(g,s['matched_global_factor']);same(np.sum(d*matched**2),np.sum(d*new**2))
        if np.any(new>old+1e-14) or np.any(new[old>0]<=0):raise ValueError('not continuous attenuation')
        model=torch.load(root/'linear_final/checkpoints/model.pt',weights_only=False)['model'];base=predict(model,arrays['val_x'])
        for seed in p['seeds']:
            delta=branch(root/f'final_s{seed}',model,arrays['val_x'])
            for arm,mask in masks.items():same(errors(base+(delta*mask)@model['q'].T,arrays['val_y']).mean(),s['scores'][arm][seed])
        for arm,v in s['scores'].items():same(np.mean(v),s['means'][arm])
        choice='uncertainty' if s['means']['uncertainty']<=s['means']['none']*.99 else 'none'
        if choice!=s['selected']:raise ValueError('selection mismatch')
        r=dict(bootstrap_replayed=True,development_replayed=True,new_neural_runs=0,
            old_nonzero_modes=int(np.count_nonzero(old)),new_nonzero_modes=int(np.count_nonzero(new)),
            changed_modes_one_based=(np.flatnonzero(np.abs(new-old)>1e-12)+1).tolist(),matched_global_factor=g)
        if (folder/'confirmation.json').exists():
            c=json.loads((folder/'confirmation.json').read_text());lock=file_hash(OUT/'selection_lock.json')
            if c['lock']!=lock or c['ids']!=p['tasks'][task]['reserved']:raise ValueError('confirmation identity')
            chunks=[torch.load(ROOT/'external_data/uncertainty_modal_v1'/task/f'confirmation_{i}.pt',weights_only=False) for i in range(0,500,50)]
            if any(v['lock']!=lock for v in chunks) or [i for v in chunks for i in v['ids']]!=c['ids']:raise ValueError('cache identity')
            x=np.concatenate([v['x'] for v in chunks]);y=np.concatenate([v['y'] for v in chunks]);base=predict(model,x)
            for seed in p['seeds']:
                delta=branch(root/f'final_s{seed}',model,x)
                for arm,mask in masks.items():
                    pred=base+(delta*mask)@model['q'].T
                    same(errors(pred,y),c['errors'][arm][seed]);same(float(np.sqrt(np.mean((pred-y)**2))),c['rmse'][arm][seed])
            for arm in ('none','old_shrink','all','matched_global'):
                if compare(c['errors'][choice],c['errors'][arm])!=c['comparisons'][arm]:raise ValueError('inference mismatch')
            r['confirmation_replayed']=True
        tasks[task]=r
    atomic_json(OUT/'audit.json',tasks);print(json.dumps(tasks,indent=2));return tasks


if __name__=='__main__':audit()
