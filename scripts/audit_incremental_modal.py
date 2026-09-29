"""Replay incremental-modal evidence from frozen checkpoints; never open new targets."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_incremental_modal import OUT,plan,modal_rules,errors,interval,coefficients,ARMS
from scripts.run_objective_modal import arrays_for,predict_delta
from scripts.objective_modal_core import sample_weights
from scripts.run_public_pod_ridge import predict
from src.public_benchmark import atomic_json,file_hash


def same(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)


def linear(folder,expected_rows):
    s=json.loads((folder/'selection.json').read_text());path=folder/'checkpoints/model.pt'
    if file_hash(path)!=s['model_sha256'] or s['identity']['rows']!=expected_rows:raise ValueError('linear provenance')
    records=[json.loads((folder/f'fold{f}.json').read_text()) for f in range(3)]
    held=[]
    for r in records:
        if r['identity']!=s['identity'] or set(r['train_rows'])&set(r['held_rows']):raise ValueError('inner fold leakage')
        if sorted(r['train_rows']+r['held_rows'])!=list(range(len(expected_rows))):raise ValueError('inner coverage')
        held+=r['held_rows']
    if sorted(held)!=list(range(len(expected_rows))):raise ValueError('held coverage')
    means=np.mean([[v['relative_l2'] for v in r['scores']] for r in records],0)
    best=records[0]['scores'][int(np.argmin(means))]
    same(means,s['grid_means'])
    if any(s[k]!=best[k] for k in ('alpha','objective')):raise ValueError('tuning choice')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if cp['identity']!=s['identity']:raise ValueError('linear identity')
    return cp['model']


def branch(folder,model,x):
    r=json.loads((folder/'result.json').read_text())
    if r['best_step']!=1500 or r['stopped_step']!=1500:raise ValueError('fixed-step rule')
    if file_hash(folder/'checkpoints/selected.pt')!=r['selected_sha256']:raise ValueError('branch hash')
    delta,cp=predict_delta(folder,x,torch.device('cpu'))
    for k in model:np.testing.assert_array_equal(model[k],cp['anchor'][k])
    # Replay on CUDA, matching training float32 operations rather than CPU GEMM ordering.
    delta,cp=predict_delta(folder,x,torch.device('cuda'))
    return delta


def audit():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);p=plan();tasks={}
    for task in p['parent']['tasks']:
        root=OUT/task
        if not (root/'development.json').exists():continue
        arrays,_=arrays_for(p['parent'],task);s=json.loads((root/'development.json').read_text())
        rules=json.loads((root/'rules.json').read_text());representation=set(p['parent']['representation_rows'])
        folds=[];held=[]
        for f,record in enumerate(rules['folds']):
            train=record['train_rows'];test=record['held_rows'];held+=test
            if set(train)&set(test) or representation&set(train+test):raise ValueError('outer leakage')
            if sorted(train+test)!=sorted(p['parent']['correction_rows']):raise ValueError('outer coverage')
            model=linear(root/f'linear_outer{f}',train);x=arrays['train_x'][test];y=arrays['train_y'][test]
            base,true=coefficients(model,x,y);res=true-base
            _,floor=sample_weights(arrays['train_y'][train],'normalized');same(floor,record['floor'])
            w=1/np.maximum(np.mean(y.astype(float)**2,1),floor);risk=[];num=[];den=[]
            for seed in p['cv_seeds']:
                delta=branch(root/f'cv_f{f}_s{seed}',model,x)
                risk.append(np.sum(w[:,None]*(delta-res)**2,0));num.append(np.sum(w[:,None]*res*delta,0));den.append(np.sum(w[:,None]*delta**2,0))
            for k,value in [('base_risk',np.sum(w[:,None]*res**2,0)),('neural_risk',np.mean(risk,0)),('numerator',np.mean(num,0)),('denominator',np.mean(den,0))]:same(value,record[k])
            folds.append(record)
        if sorted(held)!=sorted(p['parent']['correction_rows']):raise ValueError('CV coverage')
        masks,gains=modal_rules(folds)
        if masks!=rules['masks'] or masks!=s['masks']:raise ValueError('mask mismatch')
        same(gains,rules['gains']);model=linear(root/'linear_final',p['parent']['correction_rows'])
        base=predict(model,arrays['val_x']);scores={a:[] for a in ARMS}
        for seed in p['seeds']:
            delta=branch(root/f'final_s{seed}',model,arrays['val_x'])
            for arm in ARMS:scores[arm].append(float(errors(base+(delta*np.array(masks[arm]))@model['q'].T,arrays['val_y']).mean()))
        for a in ARMS:same(scores[a],s['scores'][a]);same(np.mean(scores[a]),s['means'][a])
        selected=min(['stable','shrink'],key=s['means'].get)
        if s['means'][selected]>s['means']['none']*.99:selected='none'
        if selected!=s['selected']:raise ValueError('candidate selection')
        evidence=dict(neural_records=11,development_replayed=True,selected=selected,
                      stable_modes=np.flatnonzero(masks['stable']).tolist(),linear_alpha=s['linear']['alpha'])
        if (root/'confirmation.json').exists():
            c=json.loads((root/'confirmation.json').read_text())
            if c['ids']!=p['parent']['tasks'][task]['reserved_confirmation']:raise ValueError('confirmation IDs')
            lock=json.loads((OUT/'selection_lock.json').read_text())
            if c['lock_sha256']!=file_hash(OUT/'selection_lock.json') or selected!=lock['tasks'][task]['selected']:raise ValueError('confirmation seal')
            for a in ARMS:
                if np.array(c['errors'][a]).shape!=(5,500):raise ValueError('error dimensions')
                same(np.mean(c['errors'][a]),c['mean_relative_l2'][a])
            chunks=[torch.load(ROOT/'external_data/pdebench/incremental_modal_v1'/task/f'confirmation_{i}.pt',
                               map_location='cpu',weights_only=False) for i in range(0,500,50)]
            if [v for chunk in chunks for v in chunk['ids']]!=c['ids']:raise ValueError('cached confirmation IDs')
            if any(chunk['lock']!=c['lock_sha256'] for chunk in chunks):raise ValueError('cached lock')
            x=np.concatenate([v['x'] for v in chunks]);y=np.concatenate([v['y'] for v in chunks]);base=predict(model,x)
            for seed in p['seeds']:
                delta=branch(root/f'final_s{seed}',model,x)
                for arm in ARMS:
                    prediction=base+(delta*np.array(masks[arm]))@model['q'].T
                    same(errors(prediction,y),c['errors'][arm][seed])
                    same(np.sqrt(np.mean((prediction-y)**2)),c['rmse'][arm][seed])
            if interval(c['errors'][selected],c['errors']['none'])!=c['primary']:raise ValueError('primary inference')
            if interval(c['errors'][selected],c['errors']['energy'])!=c['increment_over_energy']:raise ValueError('energy inference')
            evidence['confirmation_statistics_replayed']=True
            evidence['confirmation_predictions_replayed']=True
        tasks[task]=evidence
    result=dict(tasks=tasks,neural_records=sum(v['neural_records'] for v in tasks.values()))
    atomic_json(OUT/'audit.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':audit()
