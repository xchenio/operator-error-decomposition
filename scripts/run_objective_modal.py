"""Objective comparison and strict split-sample modal CV on pinned public tasks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.objective_modal_core import train_one,sample_weights,stable_mask
from scripts.run_public_ridge_residual import digest,locked_json,git_commit,ANCHOR,fit_fixed_basis_ridge
from scripts.run_public_pod_readout import make_branch
from scripts.run_public_pod_ridge import fit_preprocess,fit_ridge,predict
from scripts.run_public_benchmark import metrics,source_hash
from src.public_benchmark import atomic_json,file_hash,extract_pair
from src.resume import atomic_save

OUT=ROOT/'results/public_objective_modal_v1'


def code_hash():
    files=['run_objective_modal.py','objective_modal_core.py','run_public_ridge_residual.py','run_public_pod_readout.py','run_public_pod_ridge.py']
    return hashlib.sha256((''.join(file_hash(ROOT/'scripts'/f) for f in files)+source_hash()).encode()).hexdigest()


def build_plan():
    old=json.loads((ROOT/'external_data/pdebench/fixed_t1_v1/protocol.json').read_text())
    consumed=json.loads((ROOT/'results/public_ridge_residual_v1/plan.json').read_text())['confirmation_ids']
    excluded=set(range(128))|set(consumed)
    for ids in old['sample_ids'].values():excluded.update(ids)
    remaining=np.random.default_rng(2026092802).permutation(sorted(set(range(10000))-excluded)).tolist()
    adv=np.random.default_rng(2026092803).permutation(10000).tolist()
    tasks=dict(burgers_original=dict(source=old['source'],sha256=old['source_sha256'],
        train=old['sample_ids']['train'],val=old['sample_ids']['val'],reserved_confirmation=remaining[:500]),
        burgers_independent=dict(source=old['source'],sha256=old['source_sha256'],
        train=remaining[500:1500],val=remaining[1500:1700],reserved_confirmation=remaining[1700:2200]),
        advection=dict(source=str(ROOT/'external_data/pdebench/advection/1D_Advection_Sols_beta0.4.hdf5'),
        sha256='d973ff2bb3c2a5edf42957ff029b78672451532ebd6bf475848ff23cfd3ee3b6',
        train=adv[:1000],val=adv[1000:1200],reserved_confirmation=adv[1200:1700]))
    order=np.random.default_rng(2026092804).permutation(1000)
    plan=dict(source=code_hash(),tasks=tasks,time_index=100,stride=4,rank=16,alpha=10.,
        original_cache_sha256=old['cache_sha256'],original_anchor_sha256=file_hash(ANCHOR),
        representation_rows=order[:400].tolist(),correction_rows=order[400:].tolist(),
        cv_folds=3,cv_seeds=[0,1],final_seeds=list(range(5)),cv_steps=1500,final_max_steps=3000,
        objective_floor_quantile=.05,mask_rule='positive gain in all three folds and mean fold gain>=1%; seed-averaged weighted coefficient risks',
        settings='width32,one tanh layer,penalty1,AdamW wd1e-4,batch32,cosine1e-3->1e-5',
        arms=['mse_all','normalized_all','normalized_stable','normalized_random','normalized_energy'],
        limitations='All current evaluations are development/inner-CV, not confirmation. Do not load reserved targets. Advection is a linear transport negative control, not another nonlinear PDE.')
    OUT.mkdir(parents=True,exist_ok=True)
    locked_json(OUT/'plan.json',plan)
    return plan


def arrays_for(plan,task):
    info=plan['tasks'][task]
    folder=ROOT/'external_data/pdebench/objective_modal_v1'/task
    folder.mkdir(parents=True,exist_ok=True)
    identity=dict(task=task,plan_sha256=digest(plan))
    path=folder/'development.pt'
    if path.exists():
        obj=torch.load(path,map_location='cpu',weights_only=False)
        if obj['identity']!=identity:raise ValueError('data cache identity mismatch')
        return obj['arrays'],file_hash(path)
    if task=='burgers_original':
        cache=ROOT/'external_data/pdebench/fixed_t1_v1/pairs.npz'
        if file_hash(cache)!=plan['original_cache_sha256']:raise ValueError('original cache changed')
        with np.load(cache) as z:arrays={k:z[k].copy() for k in ('train_x','train_y','val_x','val_y')}
    else:
        source=Path(info['source'])
        if not source.exists():raise FileNotFoundError('Dataset not ready: '+str(source))
        print('Verifying data',task,flush=True)
        if file_hash(source)!=info['sha256']:raise ValueError('source hash mismatch')
        arrays={}
        with h5py.File(source,'r') as f:
            if f['tensor'].shape!=(10000,201,1024):raise ValueError('unexpected source shape')
            if abs(float(f['t-coordinate'][100])-1.)>1e-6:raise ValueError('unexpected target time')
            for split in ('train','val'):
                chunks=[]
                for start in range(0,len(info[split]),50):
                    ids=info[split][start:start+50]
                    cp=folder/f'{split}_{start:04d}.pt'
                    if cp.exists():
                        chunk=torch.load(cp,map_location='cpu',weights_only=False)
                        if chunk['identity']!=identity or chunk['ids']!=ids:raise ValueError('chunk mismatch')
                    else:
                        x,y=extract_pair(f['tensor'],ids,plan['time_index'],plan['stride'])
                        chunk=dict(identity=identity,ids=ids,x=x,y=y)
                        atomic_save(chunk,cp)
                    chunks.append(chunk)
                    atomic_json(OUT/task/'data_progress.json',dict(split=split,completed=start+len(ids),total=len(info[split])))
                arrays[split+'_x']=np.concatenate([c['x'] for c in chunks])
                arrays[split+'_y']=np.concatenate([c['y'] for c in chunks])
    atomic_save(dict(identity=identity,arrays=arrays),path)
    return arrays,file_hash(path)


def cfg(plan,seed,objective,mask,fixed=False):
    return dict(rank=16,kind='mlp',hidden=32,layers=1,seed=seed,objective=objective,mask=list(mask),
        fixed_final=fixed,steps=plan['cv_steps'] if fixed else plan['final_max_steps'],batch_size=32,
        lr=.001,lr_min=.00001,weight_decay=.0001,penalty=1.,eval_every=100,checkpoint_every=50,
        patience=5,min_steps=500,min_delta=.0001)


def predict_delta(folder,x,device):
    cp=torch.load(Path(folder)/'checkpoints/selected.pt',map_location='cpu',weights_only=False)
    c=cp['identity']['config'];model=cp['anchor']
    branch=make_branch(c,x.shape[1],device);branch.load_state_dict(cp['model']);branch.eval()
    with torch.no_grad():
        inp=torch.tensor((np.asarray(x,dtype=np.float64)-model['mean'])/model['std'],dtype=torch.float32,device=device)
        delta=branch(inp).cpu().numpy().astype(np.float64)*np.array(c['mask'])*cp['scale']
    return delta,cp


def panel(runs):
    return {split:{m:dict(mean=float(np.mean([r['metrics'][split]['prediction'][m] for r in runs])),
                         std=float(np.std([r['metrics'][split]['prediction'][m] for r in runs],ddof=1)))
                   for m in ('mean_relative_l2','rmse','relative_fro')} for split in ('train','val')}


def task_run(plan,task,device):
    root=OUT/task;root.mkdir(parents=True,exist_ok=True)
    arrays,cache_hash=arrays_for(plan,task)
    context=dict(plan_sha256=digest(plan),task=task,cache_sha256=cache_hash)
    if (root/'summary.json').exists():
        obj=json.loads((root/'summary.json').read_text())
        if obj['context']!=context:raise ValueError('summary mismatch')
        print('[skip]',task,flush=True);return obj
    atomic_json(root/'manifest.json',dict(context=context,status='running',git_commit=git_commit()))
    cp=root/'checkpoints/bases.pt'
    if cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=False)
        if saved['context']!=context:raise ValueError('base mismatch')
        full,anchor=saved['full'],saved['anchor']
    else:
        if task=='burgers_original':full=torch.load(ANCHOR,map_location='cpu',weights_only=False)['model']
        else:full=fit_ridge(fit_preprocess(arrays['train_x'],arrays['train_y']),16,10.)
        rows=plan['representation_rows']
        anchor=fit_ridge(fit_preprocess(arrays['train_x'][rows],arrays['train_y'][rows]),16,10.)
        atomic_save(dict(context=context,full=full,anchor=anchor),cp)
    context=dict(context,bases_sha256=file_hash(cp))
    # Full1000 matched objective comparison, separate from the reduced CV-data protocol.
    full_groups={}
    for objective in ('mse','normalized'):
        runs=[train_one(cfg(plan,s,objective,np.ones(16)),arrays,full,dict(context,phase='full'),
                        root/f'full_{objective}_s{s}',device) for s in plan['final_seeds']]
        full_groups[objective]=dict(metrics=panel(runs),best_steps=[r['best_step'] for r in runs])
        atomic_json(root/'full_progress.json',full_groups)
    # Anchor400 sees none of the600 correction labels; all folds share its fixed modal axes.
    folds=np.array_split(np.array(plan['correction_rows']),3)
    gains=[];fold_details=[]
    for fold,held in enumerate(folds):
        training=np.concatenate([v for j,v in enumerate(folds) if j!=fold])
        cv={f'{s}_{v}':arrays[f'train_{v}'][ids] for s,ids in [('train',training),('val',held)] for v in ('x','y')}
        x,y=cv['val_x'].astype(np.float64),cv['val_y'].astype(np.float64)
        base_coef=((x-anchor['mean'])/anchor['std'])@anchor['weights']+anchor['intercept']
        true_coef=(y-x.mean(axis=1,keepdims=True))@anchor['q']
        _,floor=sample_weights(cv['train_y'],'normalized')
        weights=1/np.maximum(np.mean(y*y,axis=1),floor)
        base_risk=np.sum(weights[:,None]*(base_coef-true_coef)**2,axis=0)
        risks=[]
        for seed in plan['cv_seeds']:
            folder=root/f'cv_f{fold}_s{seed}'
            train_one(cfg(plan,seed,'normalized',np.ones(16),fixed=True),cv,anchor,
                dict(context,phase='cv',fold=fold,train_rows=training.tolist(),held_rows=held.tolist()),folder,device)
            delta,_=predict_delta(folder,x,device)
            risks.append(np.sum(weights[:,None]*(base_coef+delta-true_coef)**2,axis=0))
        gain=1-np.mean(risks,axis=0)/np.maximum(base_risk,1e-20)
        gains.append(gain.tolist())
        fold_details.append(dict(fold=fold,train_rows=training.tolist(),held_rows=held.tolist(),
                                 baseline_risk=base_risk.tolist(),residual_risks=[v.tolist() for v in risks]))
        atomic_json(root/'cv_progress.json',dict(gains=gains,folds=fold_details))
    mask=stable_mask(gains);k=int(mask.sum())
    # POD modes are energy ordered from the separate representation set; no CV/val labels.
    energy_mask=np.zeros(16);energy_mask[:k]=1
    locked_json(root/'modal_selection.json',dict(context=context,gains=gains,mask=mask.tolist(),k=k,
        representation_rows=plan['representation_rows'],correction_rows=plan['correction_rows'],folds=fold_details))
    rows=plan['correction_rows']
    correction=dict(arrays,train_x=arrays['train_x'][rows],train_y=arrays['train_y'][rows])
    groups={}
    for arm in plan['arms']:
        runs=[];masks=[]
        for seed in plan['final_seeds']:
            chosen=np.ones(16)
            if arm.endswith('stable'):chosen=mask
            elif arm.endswith('energy'):chosen=energy_mask
            elif arm.endswith('random'):
                chosen=np.zeros(16);chosen[np.random.default_rng(seed+7000).permutation(16)[:k]]=1
            objective='mse' if arm=='mse_all' else 'normalized'
            runs.append(train_one(cfg(plan,seed,objective,chosen),correction,anchor,
                                  dict(context,phase='correction600',modal_selection_sha256=file_hash(root/'modal_selection.json')),
                                  root/f'{arm}_s{seed}',device))
            masks.append(chosen.tolist())
        groups[arm]=dict(metrics=panel(runs),best_steps=[r['best_step'] for r in runs],masks=masks)
        atomic_json(root/'panel_progress.json',groups)
    baseline={name:{s:metrics(predict(m,(arrays if name=='full1000' else correction)[s+'_x']),
                                        (arrays if name=='full1000' else correction)[s+'_y']) for s in ('train','val')}
              for name,m in [('full1000',full),('anchor400',anchor)]}
    linear_controls={}
    for name,m in [('full1000',full),('anchor400',anchor)]:
        fit_arrays=arrays if name=='full1000' else dict(arrays,
            train_x=arrays['train_x'][plan['representation_rows']],train_y=arrays['train_y'][plan['representation_rows']])
        grid=[]
        for alpha in (.0001,.001,.01,.1,1.,10.,100.,1000.):
            linear=fit_fixed_basis_ridge(fit_arrays,m,alpha)
            grid.append(dict(alpha=alpha,val=metrics(predict(linear,arrays['val_x']),arrays['val_y'])))
        linear_controls[name]=dict(grid=grid,selected=min(grid,key=lambda r:r['val']['mean_relative_l2']))
    atomic_json(root/'linear_controls.json',linear_controls)
    result=dict(context={k:v for k,v in context.items() if k!='bases_sha256'},bases_sha256=file_hash(cp),
        full_objective=full_groups,modal_panels=groups,k=k,selected_modes=np.flatnonzero(mask).tolist(),
        baseline=baseline,linear_controls=linear_controls,completed_neural_runs=41,limitations=plan['limitations'])
    atomic_json(root/'summary.json',result)
    atomic_json(root/'manifest.json',dict(context=context,status='completed',git_commit=git_commit()))
    print('COMPLETED',task,'stable_modes',result['selected_modes'],flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['plan','burgers_original','burgers_independent','advection','all'])
    p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    plan=build_plan()
    if a.task=='plan':print('Protocol and unused confirmation IDs locked; labels unopened');return
    for task in (list(plan['tasks']) if a.task=='all' else [a.task]):task_run(plan,task,torch.device(a.device))


if __name__=='__main__':main()
