"""Nested linear tuning, incremental nonlinear modal rules, then sealed confirmation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_objective_modal import (build_plan as parent_plan, arrays_for, cfg,
    predict_delta, code_hash as parent_hash)
from scripts.objective_modal_core import train_one, sample_weights
from scripts.run_public_pod_ridge import predict
from scripts.run_public_ridge_residual import digest, locked_json, git_commit
from src.public_benchmark import atomic_json, file_hash, extract_pair
from src.resume import atomic_save

OUT=ROOT/'results/public_incremental_modal_v1'
ARMS=['none','all','stable','energy','shrink']


def plan():
    p=dict(parent=parent_plan(),source=digest([file_hash(Path(__file__)),parent_hash()]),
        alphas=[0.]+[10.**i for i in range(-8,9)],objectives=['mse','normalized'],
        seeds=list(range(5)),cv_seeds=[0,1],steps=1500,
        linear_selection='3-fold mean held relative L2; shared alpha/objective across16 modes; grid-order ties',
        mask='all3 fold gains positive and mean>=.01; numerical floor1e-16 times total modal baseline risk',
        shrink='clip(pooled weighted residual-dot-delta / pooled weighted delta-square,0,1), seed-pooled',
        candidate='best development mean of stable/shrink; require>=1% improvement over none, else none; stable wins ties',
        primary='candidate versus none: >=1% relative mean improvement AND paired trajectory bootstrap upper<0',
        novelty='candidate versus energy also >=1% improvement AND bootstrap upper<0; not a literature novelty proof',
        inference='six comparisons (3 tasks x2), Bonferroni two-sided intervals at1-.05/6;5000 bootstraps;seed20260929',
        limits='Fixed rank16,width32,penalty1 and1500 steps. Not globally optimized. Linear Advection is a negative control.')
    OUT.mkdir(parents=True,exist_ok=True);locked_json(OUT/'plan.json',p);return p


def errors(yhat,y):
    return np.linalg.norm(yhat-y,axis=1)/np.maximum(np.linalg.norm(y,axis=1),1e-12)


def linear_grid(x,y,q,objective,alphas):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    mean=x.mean(0);std=x.std(0).clip(1e-6);z=(x-mean)/std
    c=(y-x.mean(1,keepdims=True))@q
    w,floor=sample_weights(y,objective);zm=np.average(z,axis=0,weights=w);cm=np.average(c,axis=0,weights=w)
    u,s,vt=np.linalg.svd((z-zm)*np.sqrt(w[:,None]),full_matrices=False)
    rhs=u.T@((c-cm)*np.sqrt(w[:,None]))
    for alpha in alphas:
        if alpha==0:
            f=np.divide(1.,s,out=np.zeros_like(s),where=s>max(s[0]*1e-10,1e-14))
        else:f=s/(s*s+len(x)*alpha)
        weights=(vt.T*f)@rhs
        yield dict(mean=mean,std=std,q=q,weights=weights,intercept=cm-zm@weights),floor


def tune_linear(x,y,q,p,folder,identity):
    folder.mkdir(parents=True,exist_ok=True);path=folder/'checkpoints/model.pt'
    if path.exists() and (folder/'selection.json').exists():
        obj=torch.load(path,map_location='cpu',weights_only=False)
        if obj['identity']!=identity:raise ValueError('linear identity changed')
        if file_hash(path)!=json.loads((folder/'selection.json').read_text())['model_sha256']:raise ValueError('linear checkpoint changed')
        return obj['model']
    folds=np.array_split(np.arange(len(x)),3);scores=[]
    for f,held in enumerate(folds):
        ids=np.concatenate([v for j,v in enumerate(folds) if j!=f])
        record=folder/f'fold{f}.json'
        if record.exists():
            saved=json.loads(record.read_text())
            if saved['identity']!=identity:raise ValueError('linear fold changed')
            scores.append(saved['scores']);continue
        values=[]
        for objective in p['objectives']:
            for alpha,(model,floor) in zip(p['alphas'],linear_grid(x[ids],y[ids],q,objective,p['alphas'])):
                values.append(dict(objective=objective,alpha=alpha,floor=floor,
                                   relative_l2=float(errors(predict(model,x[held]),y[held]).mean())))
        atomic_json(record,dict(identity=identity,train_rows=ids.tolist(),held_rows=held.tolist(),scores=values))
        scores.append(values)
    means=np.mean([[r['relative_l2'] for r in v] for v in scores],axis=0);best=int(np.argmin(means))
    chosen=scores[0][best];model,floor=next(linear_grid(x,y,q,chosen['objective'],[chosen['alpha']]))
    atomic_save(dict(identity=identity,model=model),path)
    atomic_json(folder/'selection.json',dict(identity=identity,objective=chosen['objective'],alpha=chosen['alpha'],
        floor=floor,inner_mean_relative_l2=float(means[best]),grid_means=means.tolist(),
        boundary=chosen['alpha'] in (p['alphas'][0],p['alphas'][-1]),model_sha256=file_hash(path)))
    print('LINEAR',folder.name,chosen['objective'],chosen['alpha'],flush=True);return model


def coefficients(model,x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    base=((x-model['mean'])/model['std'])@model['weights']+model['intercept']
    return base,(y-x.mean(1,keepdims=True))@model['q']


def modal_rules(folds):
    b=np.array([f['base_risk'] for f in folds]);r=np.array([f['neural_risk'] for f in folds])
    valid=b>np.maximum(b.sum(1,keepdims=True)*1e-16,1e-25)
    gains=np.divide(b-r,b,out=np.zeros_like(b),where=valid)
    stable=(valid.all(0)&(gains>0).all(0)&(gains.mean(0)>=.01)).astype(float)
    numerator=np.sum([f['numerator'] for f in folds],axis=0)
    denominator=np.sum([f['denominator'] for f in folds],axis=0)
    shrink=np.clip(np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-25),0,1)
    energy=np.zeros_like(stable);energy[:int(stable.sum())]=1
    return dict(none=np.zeros_like(stable).tolist(),all=np.ones_like(stable).tolist(),
                stable=stable.tolist(),energy=energy.tolist(),shrink=shrink.tolist()),gains.tolist()


def develop(p,task,device):
    root=OUT/task;root.mkdir(parents=True,exist_ok=True)
    arrays,cache=arrays_for(p['parent'],task)
    basepath=ROOT/'results/public_objective_modal_v1'/task/'checkpoints/bases.pt'
    q=torch.load(basepath,map_location='cpu',weights_only=False)['anchor']['q']
    identity=dict(plan=digest(p),cache=cache,basis=file_hash(basepath),task=task)
    if (root/'development.json').exists():
        obj=json.loads((root/'development.json').read_text())
        if obj['identity']!=identity:raise ValueError('development identity changed')
        print('[skip]',task,flush=True);return obj
    rows=np.array(p['parent']['correction_rows']);x=arrays['train_x'][rows];y=arrays['train_y'][rows]
    folds=np.array_split(np.arange(len(rows)),3);records=[]
    for f,held in enumerate(folds):
        ids=np.concatenate([v for j,v in enumerate(folds) if j!=f])
        model=tune_linear(x[ids],y[ids],q,p,root/f'linear_outer{f}',dict(identity,rows=rows[ids].tolist()))
        cv=dict(train_x=x[ids],train_y=y[ids],val_x=x[held],val_y=y[held])
        base,true=coefficients(model,x[held],y[held]);residual=true-base
        _,floor=sample_weights(y[ids],'normalized');w=1/np.maximum(np.mean(y[held].astype(float)**2,axis=1),floor)
        risks=[];nums=[];dens=[]
        for seed in p['cv_seeds']:
            folder=root/f'cv_f{f}_s{seed}'
            c=cfg(p['parent'],seed,'normalized',np.ones(16),fixed=True)
            train_one(c,cv,model,dict(identity,phase='cv',fold=f,linear=digest({k:v.tolist() for k,v in model.items()})),folder,device)
            delta,_=predict_delta(folder,x[held],device)
            risks.append(np.sum(w[:,None]*(delta-residual)**2,axis=0))
            nums.append(np.sum(w[:,None]*residual*delta,axis=0));dens.append(np.sum(w[:,None]*delta**2,axis=0))
        records.append(dict(train_rows=rows[ids].tolist(),held_rows=rows[held].tolist(),floor=floor,
            base_risk=np.sum(w[:,None]*residual**2,axis=0).tolist(),neural_risk=np.mean(risks,0).tolist(),
            numerator=np.mean(nums,0).tolist(),denominator=np.mean(dens,0).tolist()))
        atomic_json(root/'cv_progress.json',dict(identity=identity,folds=records))
    masks,gains=modal_rules(records)
    locked_json(root/'rules.json',dict(identity=identity,folds=records,gains=gains,masks=masks))
    model=tune_linear(x,y,q,p,root/'linear_final',dict(identity,rows=rows.tolist()))
    final=dict(train_x=x,train_y=y,val_x=arrays['val_x'],val_y=arrays['val_y'])
    base=predict(model,final['val_x']);scores={a:[] for a in ARMS}
    for seed in p['seeds']:
        folder=root/f'final_s{seed}';c=cfg(p['parent'],seed,'normalized',np.ones(16),fixed=True)
        train_one(c,final,model,dict(identity,phase='final',rules=file_hash(root/'rules.json')),folder,device)
        delta,_=predict_delta(folder,final['val_x'],device)
        for arm in ARMS:
            pred=base+(delta*np.array(masks[arm]))@q.T
            scores[arm].append(float(errors(pred,final['val_y']).mean()))
    mean={a:float(np.mean(v)) for a,v in scores.items()};choice=min(['stable','shrink'],key=mean.get)
    if mean[choice]>mean['none']*.99:choice='none'
    result=dict(identity=identity,scores=scores,means=mean,selected=choice,masks=masks,
        linear=json.loads((root/'linear_final/selection.json').read_text()),neural_records=11,
        checkpoints={str(s):file_hash(root/f'final_s{s}/checkpoints/selected.pt') for s in p['seeds']})
    atomic_json(root/'development.json',result);print('DEVELOPED',task,mean,'selected',choice,flush=True);return result


def seal(p):
    tasks={}
    for task in p['parent']['tasks']:
        root=OUT/task;s=json.loads((root/'development.json').read_text())
        if s['identity']['plan']!=digest(p):raise ValueError('plan changed')
        for seed,h in s['checkpoints'].items():
            if file_hash(root/f'final_s{seed}/checkpoints/selected.pt')!=h:raise ValueError('checkpoint changed')
        tasks[task]=dict(development_sha256=file_hash(root/'development.json'),rules_sha256=file_hash(root/'rules.json'),
            selected=s['selected'],checkpoints=s['checkpoints'],linear_sha256=s['linear']['model_sha256'])
    locked_json(OUT/'selection_lock.json',dict(plan_sha256=digest(p),tasks=tasks,preseal_commit=git_commit()))
    print('SEALED. Commit selection_lock.json before confirmation.',flush=True)


def committed(path):
    rel=path.relative_to(ROOT).as_posix()
    data=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
    if json.loads(data)!=json.loads(path.read_text()):raise ValueError('lock not committed unchanged')


def interval(a,b,seed=20260929):
    delta=np.mean(a,axis=0)-np.mean(b,axis=0);rng=np.random.default_rng(seed)
    draws=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(5000)])
    lo,hi=np.quantile(draws,[.05/12,1-.05/12]);gain=float(-delta.mean()/np.mean(b))
    return dict(difference=float(delta.mean()),relative_gain=gain,interval=[float(lo),float(hi)],
                success=bool(hi<0 and gain>=.01))


def confirm(p,device):
    lockpath=OUT/'selection_lock.json';committed(lockpath);committed(OUT/'plan.json')
    lock=json.loads(lockpath.read_text())
    if lock['plan_sha256']!=digest(p):raise ValueError('source/plan changed')
    for task,info in p['parent']['tasks'].items():
        root=OUT/task;entry=lock['tasks'][task];s=json.loads((root/'development.json').read_text())
        if file_hash(root/'development.json')!=entry['development_sha256'] or file_hash(root/'rules.json')!=entry['rules_sha256']:raise ValueError('selection changed')
        path=root/'confirmation.json'
        if path.exists():
            if json.loads(path.read_text())['lock_sha256']!=file_hash(lockpath):raise ValueError('confirmation lock changed')
            print('[skip] confirmation',task,flush=True);continue
        linearpath=root/'linear_final/checkpoints/model.pt'
        if file_hash(linearpath)!=entry['linear_sha256']:raise ValueError('linear changed')
        for seed,h in entry['checkpoints'].items():
            if file_hash(root/f'final_s{seed}/checkpoints/selected.pt')!=h:raise ValueError('branch changed')
        if file_hash(Path(info['source']))!=info['sha256']:raise ValueError('source changed')
        atomic_json(root/'confirmation_opened.json',dict(lock_sha256=file_hash(lockpath),git_commit=git_commit(),
            note='These reserved IDs are consumed once extraction begins; resume only, never retune.'))
        chunks=[]
        with h5py.File(info['source'],'r') as f:
            for start in range(0,500,50):
                ids=info['reserved_confirmation'][start:start+50]
                cache=ROOT/'external_data/pdebench/incremental_modal_v1'/task/f'confirmation_{start}.pt'
                if cache.exists():
                    chunk=torch.load(cache,weights_only=False)
                    if chunk['ids']!=ids or chunk['lock']!=file_hash(lockpath):raise ValueError('confirmation cache changed')
                else:
                    x,y=extract_pair(f['tensor'],ids,100,4);chunk=dict(ids=ids,x=x,y=y,lock=file_hash(lockpath));atomic_save(chunk,cache)
                chunks.append(chunk)
        x=np.concatenate([c['x'] for c in chunks]);y=np.concatenate([c['y'] for c in chunks])
        model=torch.load(linearpath,map_location='cpu',weights_only=False)['model'];base=predict(model,x)
        errs={a:[] for a in ARMS};rmse={a:[] for a in ARMS}
        for seed in p['seeds']:
            delta,_=predict_delta(root/f'final_s{seed}',x,device)
            for arm in ARMS:
                pred=base+(delta*np.array(s['masks'][arm]))@model['q'].T
                errs[arm].append(errors(pred,y).tolist());rmse[arm].append(float(np.sqrt(np.mean((pred-y)**2))))
        choice=entry['selected']
        result=dict(lock_sha256=file_hash(lockpath),git_commit=git_commit(),ids=info['reserved_confirmation'],
            selected=choice,errors=errs,mean_relative_l2={a:float(np.mean(v)) for a,v in errs.items()},rmse=rmse,
            primary=interval(errs[choice],errs['none']),increment_over_energy=interval(errs[choice],errs['energy']),
            limits='Trajectory bootstrap conditional on trained seeds and data regime; not independent seed/PDE replication.')
        atomic_json(path,result);print('CONFIRMED',task,result['mean_relative_l2'],result['primary'],flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['plan','develop','seal','confirm']);a.add_argument('--task',default='all');a=a.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);p=plan();device=torch.device('cuda')
    if a.stage=='develop':
        for task in (p['parent']['tasks'] if a.task=='all' else [a.task]):develop(p,task,device)
    elif a.stage=='seal':seal(p)
    elif a.stage=='confirm':confirm(p,device)


if __name__=='__main__':main()
