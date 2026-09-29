"""Frozen-branch uncertainty shrinkage, strong controls and new sealed confirmation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_field_modal import plan as field_plan,ac_arrays,model_artifacts
from scripts.run_objective_modal import arrays_for,predict_delta
from scripts.run_incremental_modal import coefficients as true_coefficients,errors,committed
from scripts.run_public_pod_ridge import predict
from scripts.run_public_ridge_residual import digest,locked_json,git_commit
from scripts.allen_cahn_data import initial,solve
from scripts.uncertainty_shrink import moments,bootstrap_moment,coefficients,matched_global
from src.public_benchmark import atomic_json,file_hash,extract_pair
from src.resume import atomic_save
OUT=ROOT/'results/public_uncertainty_modal_v1'


def plan():
    old=field_plan();parent=old['parent'];used={'burgers':set(range(128)),'advection':set()}
    historic=json.loads((ROOT/'external_data/pdebench/fixed_t1_v1/protocol.json').read_text())
    for ids in historic['sample_ids'].values():used['burgers'].update(ids)
    used['burgers'].update(json.loads((ROOT/'results/public_ridge_residual_v1/plan.json').read_text())['confirmation_ids'])
    for task,s in parent['tasks'].items():
        family='burgers' if task.startswith('burgers') else 'advection'
        used[family].update(s['train']+s['val']+s['reserved_confirmation']+old['tasks'][task]['reserved'])
    tasks={}
    for i,(task,s) in enumerate(old['tasks'].items()):
        if s['kind']=='synthetic':ids=list(range(1700,2200))
        else:
            family='burgers' if task.startswith('burgers') else 'advection'
            ids=np.random.default_rng(2026093005+i).permutation(sorted(set(range(10000))-used[family]))[:500].tolist()
            if len(ids)!=500:raise ValueError('insufficient fresh trajectories')
            used[family].update(ids)
        tasks[task]=dict(s,reserved=ids)
    p=dict(field_plan_sha256=digest(old),tasks=tasks,source=digest([old['source'],file_hash(Path(__file__)),file_hash(ROOT/'scripts/uncertainty_shrink.py')]),
        bootstrap_replicates=2000,bootstrap_seed=2026093001,kappa=1.96,seeds=list(range(5)),cv_seeds=[0,1],
        formula='a=clip(N/(D+1.96*SD_bootstrap(N)),0,1); N=mean(w*r*delta)/spatial_dim,D=mean(w*delta^2)/spatial_dim',
        interpretation='minimize D*a^2-2*N*a+1.96*SD(N)*a^2; uncertainty penalty, not a calibrated confidence constraint',
        arms=['none','all','old_shrink','uncertainty','trajectory_only','matched_global'],
        selected='uncertainty if development mean improves>=1% over none; otherwise none; no tuning or seed selection',
        inference='16 comparisons=4tasks x linear/old_shrink/all/matched_global; >=1% relative gain AND Bonferroni two-sided interval below0;10000 trajectory bootstraps seed2026093019',
        limits='No new neural fits. Bootstrap conditional on3 fitted CV folds/two probes; overlapping fold training and refit uncertainty not fully represented. Coefficients use a squared surrogate; main metric remains mean nonsquared relative L2.')
    OUT.mkdir(parents=True,exist_ok=True);locked_json(OUT/'plan_v2.json',p);return p


def data_and_root(task):
    old=field_plan()
    arrays=ac_arrays(old) if task=='allen_cahn' else arrays_for(old['parent'],task)[0]
    s=json.loads((ROOT/'results/public_field_modal_v1'/task/'development.json').read_text())
    root=ROOT/s['model_root']
    for path,h in s['artifacts'].items():
        if file_hash(ROOT/path)!=h:raise ValueError('historical artifact changed')
    return arrays,root


def collect(task,device):
    arrays,root=data_and_root(task);rules=json.loads((root/'rules.json').read_text());nb=[];db=[]
    for f,r in enumerate(rules['folds']):
        x=arrays['train_x'][r['held_rows']];y=arrays['train_y'][r['held_rows']].astype(float);deltas=[]
        for seed in (0,1):
            delta,cp=predict_delta(root/f'cv_f{f}_s{seed}',x,device);deltas.append(delta)
        b,t=true_coefficients(cp['anchor'],x,y);w=1/np.maximum(np.mean(y*y,1),r['floor'])
        n,d=moments(t-b,np.array(deltas),w,y.shape[1]);nb.append(n);db.append(d)
    return arrays,root,rules,nb,db


def develop(p,task,device):
    folder=OUT/task;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'development.json'
    if path.exists():
        s=json.loads(path.read_text())
        if s['plan']!=digest(p):raise ValueError('plan changed')
        for file,h in s['artifacts'].items():
            if file_hash(ROOT/file)!=h:raise ValueError('model changed')
        print('[skip]',task);return s
    arrays,root,rules,nb,db=collect(task,device)
    artifact=model_artifacts(root,p);identity=dict(plan=digest(p),artifacts=artifact)
    cp_path=folder/'checkpoints/bootstrap.pt'
    if cp_path.exists():
        cp=torch.load(cp_path,weights_only=False)
        if cp['identity']!=identity:raise ValueError('bootstrap changed')
        draws=cp['draws']
    else:draws=[]
    for i in range(len(draws),p['bootstrap_replicates']):
        draws.append(bootstrap_moment(nb,i,p['bootstrap_seed']))
        if (i+1)%100==0:
            atomic_save(dict(identity=identity,draws=draws),cp_path)
            atomic_json(folder/'progress.json',dict(completed=i+1,total=p['bootstrap_replicates']))
    n=np.mean([v.mean((0,1)) for v in nb],0);d=np.mean([v.mean((0,1)) for v in db],0)
    sd=np.std(np.array(draws),axis=0,ddof=1);old=coefficients(n,d,np.zeros(16))
    np.testing.assert_allclose(old,rules['masks']['shrink'],atol=1e-12,rtol=1e-12)
    uncertainty=coefficients(n,d,sd[1],p['kappa']);trajectory=coefficients(n,d,sd[0],p['kappa'])
    global_mask,factor=matched_global(old,uncertainty,d)
    masks=dict(none=np.zeros(16),all=np.ones(16),old_shrink=old,uncertainty=uncertainty,trajectory_only=trajectory,matched_global=global_mask)
    model=torch.load(root/'linear_final/checkpoints/model.pt',weights_only=False)['model'];base=predict(model,arrays['val_x'])
    scores={a:[] for a in masks}
    for seed in p['seeds']:
        delta,_=predict_delta(root/f'final_s{seed}',arrays['val_x'],device)
        for a,mask in masks.items():scores[a].append(float(errors(base+(delta*mask)@model['q'].T,arrays['val_y']).mean()))
    means={a:float(np.mean(v)) for a,v in scores.items()};selected='uncertainty' if means['uncertainty']<=means['none']*.99 else 'none'
    s=dict(plan=digest(p),model_root=str(root.relative_to(ROOT)).replace('\\','/'),artifacts=artifact,
        N=n.tolist(),D=d.tolist(),sd_trajectory=sd[0].tolist(),sd_hierarchical=sd[1].tolist(),
        masks={a:v.tolist() for a,v in masks.items()},matched_global_factor=factor,scores=scores,means=means,selected=selected,
        bootstrap_sha256=file_hash(cp_path),new_neural_runs=0)
    atomic_json(path,s);print('DEVELOPED',task,means,'selected',selected,flush=True);return s


def seal(p):
    tasks={}
    for task in p['tasks']:
        path=OUT/task/'development.json';s=json.loads(path.read_text())
        if s['plan']!=digest(p):raise ValueError('plan changed')
        for file,h in s['artifacts'].items():
            if file_hash(ROOT/file)!=h:raise ValueError('model changed')
        tasks[task]=dict(development_sha256=file_hash(path),selected=s['selected'],artifacts=s['artifacts'])
    locked_json(OUT/'selection_lock.json',dict(plan=digest(p),tasks=tasks,preseal_commit=git_commit()))


def compare(a,b):
    delta=np.mean(a,0)-np.mean(b,0);rng=np.random.default_rng(2026093019)
    draws=[delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(10000)]
    lo,hi=np.quantile(draws,[.05/32,1-.05/32]);gain=float(-delta.mean()/np.mean(b))
    return dict(difference=float(delta.mean()),relative_gain=gain,interval=[float(lo),float(hi)],success=bool(gain>=.01 and hi<0))


def confirm(p,device):
    lockpath=OUT/'selection_lock.json';committed(lockpath);committed(OUT/'plan_v2.json');lock=json.loads(lockpath.read_text())
    if lock['plan']!=digest(p):raise ValueError('lock changed')
    for task,info in p['tasks'].items():
        folder=OUT/task;entry=lock['tasks'][task];result=folder/'confirmation.json'
        if result.exists():
            if json.loads(result.read_text())['lock']!=file_hash(lockpath):raise ValueError('result changed')
            print('[skip] confirmation',task);continue
        if file_hash(folder/'development.json')!=entry['development_sha256']:raise ValueError('selection changed')
        s=json.loads((folder/'development.json').read_text())
        for file,h in entry['artifacts'].items():
            if file_hash(ROOT/file)!=h:raise ValueError('model changed')
        if info['kind']=='public' and file_hash(Path(info['source']))!=info['sha256']:raise ValueError('source changed')
        atomic_json(folder/'confirmation_opened.json',dict(lock=file_hash(lockpath),git_commit=git_commit(),consumed=True))
        chunks=[]
        for start in range(0,500,50):
            ids=info['reserved'][start:start+50];cache=ROOT/'external_data/uncertainty_modal_v1'/task/f'confirmation_{start}.pt'
            if cache.exists():
                obj=torch.load(cache,weights_only=False)
                if obj['lock']!=file_hash(lockpath) or obj['ids']!=ids:raise ValueError('cache changed')
            else:
                if info['kind']=='synthetic':x=initial(ids);y=solve(x)
                else:
                    with h5py.File(info['source'],'r') as f:x,y=extract_pair(f['tensor'],ids,100,4)
                obj=dict(ids=ids,x=x,y=y,lock=file_hash(lockpath));atomic_save(obj,cache)
            chunks.append(obj)
        x=np.concatenate([c['x'] for c in chunks]);y=np.concatenate([c['y'] for c in chunks]);root=ROOT/s['model_root']
        model=torch.load(root/'linear_final/checkpoints/model.pt',weights_only=False)['model'];base=predict(model,x)
        errs={a:[] for a in s['masks']};rmse={a:[] for a in s['masks']}
        for seed in p['seeds']:
            delta,_=predict_delta(root/f'final_s{seed}',x,device)
            for arm,mask in s['masks'].items():
                pred=base+(delta*np.array(mask))@model['q'].T
                errs[arm].append(errors(pred,y).tolist());rmse[arm].append(float(np.sqrt(np.mean((pred-y)**2))))
        comparisons={a:compare(errs[s['selected']],errs[a]) for a in ('none','old_shrink','all','matched_global')}
        r=dict(lock=file_hash(lockpath),git_commit=git_commit(),ids=info['reserved'],selected=s['selected'],errors=errs,rmse=rmse,
            means={a:float(np.mean(v)) for a,v in errs.items()},comparisons=comparisons)
        atomic_json(result,r);print('CONFIRMED',task,r['means'],comparisons,flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['plan','develop','seal','confirm']);a=a.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);p=plan();device=torch.device('cuda')
    if a.stage=='develop':
        for task in p['tasks']:develop(p,task,device)
    elif a.stage=='seal':seal(p)
    elif a.stage=='confirm':confirm(p,device)


if __name__=='__main__':main()
