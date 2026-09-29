"""Whole-field relevance gates, validated synthetic Allen-Cahn, fresh sealed tests."""
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
from scripts.run_incremental_modal import (plan as prior_plan,tune_linear,modal_rules,coefficients,errors,committed)
from scripts.run_objective_modal import arrays_for as old_arrays,cfg,predict_delta
from scripts.objective_modal_core import train_one,sample_weights
from scripts.run_public_pod_ridge import fit_preprocess,predict
from scripts.run_public_ridge_residual import digest,locked_json,git_commit
from scripts.allen_cahn_data import initial,solve,PARAMS
from src.public_benchmark import atomic_json,file_hash,extract_pair
from src.resume import atomic_save
OUT=ROOT/'results/public_field_modal_v1'
OLD=ROOT/'results/public_incremental_modal_v1'


def plan():
    old=prior_plan();parent=old['parent'];tasks={};excluded={}
    for name,info in parent['tasks'].items():
        family='burgers' if name.startswith('burgers') else 'advection'
        excluded.setdefault(family,set()).update(info['train']+info['val']+info['reserved_confirmation'])
    historic=json.loads((ROOT/'external_data/pdebench/fixed_t1_v1/protocol.json').read_text())
    for ids in historic['sample_ids'].values():excluded['burgers'].update(ids)
    excluded['burgers'].update(json.loads((ROOT/'results/public_ridge_residual_v1/plan.json').read_text())['confirmation_ids'])
    excluded['burgers'].update(range(128))
    for n,(name,info) in enumerate(parent['tasks'].items()):
        family='burgers' if name.startswith('burgers') else 'advection'
        ids=np.random.default_rng(2026092910+n).permutation(sorted(set(range(10000))-excluded[family]))[:500].tolist()
        excluded[family].update(ids);tasks[name]=dict(kind='public',reserved=ids,source=info['source'],sha256=info['sha256'])
    tasks['allen_cahn']=dict(kind='synthetic',train=list(range(1000)),val=list(range(1000,1200)),reserved=list(range(1200,1700)))
    qa=json.loads((OUT/'solver_qa.json').read_text())
    if not qa['passed'] or qa['source_sha256']!=file_hash(ROOT/'scripts/allen_cahn_data.py'):raise ValueError('solver QA missing or stale')
    p=dict(prior_sha256=digest(old),parent=parent,tasks=tasks,representation_rows=parent['representation_rows'],
        correction_rows=parent['correction_rows'],source=digest([old['source'],file_hash(Path(__file__)),file_hash(ROOT/'scripts/allen_cahn_data.py')]),
        solver=PARAMS,solver_qa_sha256=file_hash(OUT/'solver_qa.json'),alphas=old['alphas'],objectives=old['objectives'],
        seeds=old['seeds'],cv_seeds=old['cv_seeds'],field_threshold=.001,
        rule='old stable AND each fold reduction>=0.1% of entire weighted baseline FIELD squared error, including projection error',
        joint_gate='OOF mean nonsquared relative L2 improves>=1% in each fold, seed averaged; gate hard and shrunken masks separately',
        selection='development best field/field_shrink with>=1% gain over none, else none; old comparator best stable/shrink with same fallback',
        confirmation='fresh500/task;12 comparisons: selected vs none, matched-k energy, old selected rule; >=1% gain and Bonferroni two-sided CI below0;10000 resamples seed2026092919',
        limits='Reused exploratory development for3 public tasks; independent synthetic Allen-Cahn, not a downloaded public benchmark; fixed architecture/rank/steps.')
    OUT.mkdir(parents=True,exist_ok=True);locked_json(OUT/'plan.json',p);return p


def ac_arrays(p):
    folder=ROOT/'external_data/allen_cahn_field_v1';chunks=[]
    for start in range(0,1200,50):
        path=folder/f'development_{start}.pt';ids=list(range(start,start+50))
        if path.exists():
            obj=torch.load(path,weights_only=False)
            if obj['plan']!=digest(p) or obj['ids']!=ids:raise ValueError('synthetic cache mismatch')
        else:
            x=initial(ids);y=solve(x);obj=dict(plan=digest(p),ids=ids,x=x,y=y);atomic_save(obj,path)
        chunks.append(obj);atomic_json(OUT/'allen_cahn/data_progress.json',dict(completed=start+50,total=1200))
    x=np.concatenate([c['x'] for c in chunks]);y=np.concatenate([c['y'] for c in chunks])
    return dict(train_x=x[:1000],train_y=y[:1000],val_x=x[1000:],val_y=y[1000:])


def ac_fit(p,arrays,device):
    root=OUT/'allen_cahn/models';rows=np.array(p['correction_rows']);x=arrays['train_x'][rows];y=arrays['train_y'][rows]
    identity=dict(plan=digest(p),task='allen_cahn');path=root/'checkpoints/basis.pt'
    if path.exists():
        cp=torch.load(path,weights_only=False)
        if cp['identity']!=identity:raise ValueError('basis identity')
        q=cp['q']
    else:
        rep=p['representation_rows'];q=fit_preprocess(arrays['train_x'][rep],arrays['train_y'][rep])['q'][:,:16]
        atomic_save(dict(identity=identity,q=q),path)
    folds=np.array_split(np.arange(600),3);records=[]
    for f,held in enumerate(folds):
        ids=np.concatenate([v for j,v in enumerate(folds) if j!=f])
        model=tune_linear(x[ids],y[ids],q,p,root/f'linear_outer{f}',dict(identity,rows=rows[ids].tolist()))
        cv=dict(train_x=x[ids],train_y=y[ids],val_x=x[held],val_y=y[held])
        b,t=coefficients(model,x[held],y[held]);res=t-b;_,floor=sample_weights(y[ids],'normalized')
        w=1/np.maximum(np.mean(y[held]**2,1),floor);risk=[];num=[];den=[]
        for seed in p['cv_seeds']:
            folder=root/f'cv_f{f}_s{seed}'
            train_one(cfg(p['parent'],seed,'normalized',np.ones(16),fixed=True),cv,model,dict(identity,fold=f),folder,device)
            d,_=predict_delta(folder,x[held],device)
            risk.append(np.sum(w[:,None]*(d-res)**2,0));num.append(np.sum(w[:,None]*res*d,0));den.append(np.sum(w[:,None]*d*d,0))
        records.append(dict(train_rows=rows[ids].tolist(),held_rows=rows[held].tolist(),floor=floor,
            base_risk=np.sum(w[:,None]*res**2,0).tolist(),neural_risk=np.mean(risk,0).tolist(),
            numerator=np.mean(num,0).tolist(),denominator=np.mean(den,0).tolist()))
        atomic_json(root/'cv_progress.json',dict(folds=records))
    masks,gains=modal_rules(records);locked_json(root/'rules.json',dict(folds=records,masks=masks,gains=gains))
    model=tune_linear(x,y,q,p,root/'linear_final',dict(identity,rows=rows.tolist()))
    final=dict(train_x=x,train_y=y,val_x=arrays['val_x'],val_y=arrays['val_y'])
    for seed in p['seeds']:
        train_one(cfg(p['parent'],seed,'normalized',np.ones(16),fixed=True),final,model,identity,root/f'final_s{seed}',device)
    return root


def field_mask(folds,risks,old_mask,threshold):
    reductions=np.array([np.array(f['base_risk'])-np.array(f['neural_risk']) for f in folds])
    shares=reductions/np.maximum(np.array(risks)[:,None],1e-25)
    mask=np.array(old_mask)*np.all(shares>=threshold,axis=0)
    return mask,shares


def model_artifacts(root,p):
    paths=[root/'rules.json',root/'linear_final/checkpoints/model.pt']
    paths += [root/f'final_s{s}/checkpoints/selected.pt' for s in p['seeds']]
    paths += [root/f'cv_f{f}_s{s}/checkpoints/selected.pt' for f in range(3) for s in p['cv_seeds']]
    return {str(v.relative_to(ROOT)).replace('\\','/'):file_hash(v) for v in paths}


def develop(p,task,device):
    dest=OUT/task;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'development.json').exists():
        s=json.loads((dest/'development.json').read_text())
        if s['plan']!=digest(p):raise ValueError('plan changed')
        for path,h in s['artifacts'].items():
            if file_hash(ROOT/path)!=h:raise ValueError('model changed')
        print('[skip]',task);return s
    arrays=ac_arrays(p) if task=='allen_cahn' else old_arrays(p['parent'],task)[0]
    root=ac_fit(p,arrays,device) if task=='allen_cahn' else OLD/task
    rules=json.loads((root/'rules.json').read_text());folds=rules['folds'];cached=[];risks=[]
    for f,record in enumerate(folds):
        x=arrays['train_x'][record['held_rows']];y=arrays['train_y'][record['held_rows']].astype(float)
        predictions=[];deltas=[];model=None
        for seed in p['cv_seeds']:
            delta,cp=predict_delta(root/f'cv_f{f}_s{seed}',x,device);model=cp['anchor'];deltas.append(delta)
        base=predict(model,x);w=1/np.maximum(np.mean(y*y,1),record['floor'])
        risk=float(np.sum(w[:,None]*(base-y)**2));risks.append(risk)
        # Independent whole-field identity check: retained coefficients + projection residual.
        b,t=coefficients(model,x,y);orth=y-(t@model['q'].T+x.mean(1,keepdims=True))
        if not np.isclose(risk,np.sum(record['base_risk'])+np.sum(w[:,None]*orth**2),rtol=1e-9,atol=1e-12):raise ValueError('field decomposition failed')
        cached.append((base,y,np.array(deltas),model['q']))
    hard,shares=field_mask(folds,risks,rules['masks']['stable'],p['field_threshold'])
    masks={k:np.array(v) for k,v in rules['masks'].items()};masks['old_energy']=masks.pop('energy')
    masks['field']=hard;masks['field_shrink']=hard*np.array(rules['masks']['shrink']);gate={}
    for arm in ('field','field_shrink'):
        gains=[]
        for base,y,deltas,q in cached:
            error=np.mean([errors(base+(d*masks[arm])@q.T,y).mean() for d in deltas])
            gains.append(float(1-error/errors(base,y).mean()))
        passed=all(v>=.01 for v in gains)
        gate[arm]=dict(fold_relative_l2_gains=gains,passed=passed)
        if not passed:masks[arm]=np.zeros(16)
    masks['energy']=np.zeros(16)
    k=max(np.count_nonzero(masks['field']),np.count_nonzero(masks['field_shrink']))
    masks['energy'][:int(k)]=1
    model=torch.load(root/'linear_final/checkpoints/model.pt',map_location='cpu',weights_only=False)['model']
    base=predict(model,arrays['val_x']);scores={a:[] for a in masks}
    for seed in p['seeds']:
        delta,_=predict_delta(root/f'final_s{seed}',arrays['val_x'],device)
        for arm,mask in masks.items():scores[arm].append(float(errors(base+(delta*mask)@model['q'].T,arrays['val_y']).mean()))
    means={a:float(np.mean(v)) for a,v in scores.items()}
    choose=lambda arms:min(arms,key=means.get) if min(means[a] for a in arms)<=means['none']*.99 else 'none'
    s=dict(plan=digest(p),model_root=str(root.relative_to(ROOT)).replace('\\','/'),artifacts=model_artifacts(root,p),
        risks=risks,fold_field_shares=shares.tolist(),pre_joint_mask=hard.tolist(),joint_gate=gate,
        masks={a:v.tolist() for a,v in masks.items()},scores=scores,means=means,
        selected=choose(['field','field_shrink']),old_selected=choose(['stable','shrink']),
        new_neural_runs=11 if task=='allen_cahn' else 0,confirmation_evaluated=False)
    atomic_json(dest/'development.json',s);print('DEVELOPED',task,means,'selected',s['selected'],flush=True);return s


def seal(p):
    tasks={}
    for task in p['tasks']:
        path=OUT/task/'development.json';s=json.loads(path.read_text())
        if s['plan']!=digest(p):raise ValueError('plan changed')
        for file,h in s['artifacts'].items():
            if file_hash(ROOT/file)!=h:raise ValueError('artifact changed')
        tasks[task]=dict(development_sha256=file_hash(path),selected=s['selected'],old_selected=s['old_selected'],artifacts=s['artifacts'])
    locked_json(OUT/'selection_lock.json',dict(plan=digest(p),tasks=tasks,preseal_commit=git_commit()))


def compare(a,b):
    delta=np.mean(a,0)-np.mean(b,0);rng=np.random.default_rng(2026092919)
    draws=[np.mean(delta[rng.integers(0,len(delta),len(delta))]) for _ in range(10000)]
    lo,hi=np.quantile(draws,[.05/24,1-.05/24]);gain=float(-np.mean(delta)/np.mean(b))
    return dict(difference=float(np.mean(delta)),relative_gain=gain,interval=[float(lo),float(hi)],success=bool(gain>=.01 and hi<0))


def confirm(p,device):
    path=OUT/'selection_lock.json';committed(path);committed(OUT/'plan.json');lock=json.loads(path.read_text())
    if lock['plan']!=digest(p):raise ValueError('lock changed')
    for task,info in p['tasks'].items():
        folder=OUT/task;entry=lock['tasks'][task];result=folder/'confirmation.json'
        if result.exists():
            if json.loads(result.read_text())['lock']!=file_hash(path):raise ValueError('confirmation lock changed')
            print('[skip] confirmation',task);continue
        if file_hash(folder/'development.json')!=entry['development_sha256']:raise ValueError('selection changed')
        s=json.loads((folder/'development.json').read_text())
        for file,h in entry['artifacts'].items():
            if file_hash(ROOT/file)!=h:raise ValueError('artifact changed')
        if info['kind']=='public' and file_hash(Path(info['source']))!=info['sha256']:raise ValueError('data changed')
        atomic_json(folder/'confirmation_opened.json',dict(lock=file_hash(path),git_commit=git_commit(),consumed=True))
        chunks=[]
        for start in range(0,500,50):
            ids=info['reserved'][start:start+50];cache=ROOT/'external_data/field_modal_v1'/task/f'confirmation_{start}.pt'
            if cache.exists():
                obj=torch.load(cache,weights_only=False)
                if obj['ids']!=ids or obj['lock']!=file_hash(path):raise ValueError('cache changed')
            else:
                if info['kind']=='synthetic':x=initial(ids);y=solve(x)
                else:
                    with h5py.File(info['source'],'r') as f:x,y=extract_pair(f['tensor'],ids,100,4)
                obj=dict(ids=ids,x=x,y=y,lock=file_hash(path));atomic_save(obj,cache)
            chunks.append(obj)
        x=np.concatenate([v['x'] for v in chunks]);y=np.concatenate([v['y'] for v in chunks]);root=ROOT/s['model_root']
        model=torch.load(root/'linear_final/checkpoints/model.pt',weights_only=False)['model'];base=predict(model,x)
        errs={a:[] for a in s['masks']}
        for seed in p['seeds']:
            delta,_=predict_delta(root/f'final_s{seed}',x,device)
            for arm,mask in s['masks'].items():errs[arm].append(errors(base+(delta*np.array(mask))@model['q'].T,y).tolist())
        comparisons={a:compare(errs[s['selected']],errs[b]) for a,b in [('linear','none'),('energy','energy'),('old_rule',s['old_selected'])]}
        r=dict(lock=file_hash(path),git_commit=git_commit(),ids=info['reserved'],selected=s['selected'],old_selected=s['old_selected'],
            errors=errs,means={a:float(np.mean(v)) for a,v in errs.items()},comparisons=comparisons)
        atomic_json(result,r);print('CONFIRMED',task,r['means'],comparisons,flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['plan','develop','seal','confirm']);a=a.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);p=plan();device=torch.device('cuda')
    if a.stage=='develop':
        for task in p['tasks']:develop(p,task,device)
    elif a.stage=='seal':seal(p)
    elif a.stage=='confirm':confirm(p,device)


if __name__=='__main__':main()
