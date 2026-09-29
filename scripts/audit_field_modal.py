"""Replay field relevance and sealed results without extracting new targets."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_field_modal import plan,OUT,ac_arrays,field_mask,compare
from scripts.run_objective_modal import arrays_for,predict_delta
from scripts.run_incremental_modal import errors,coefficients
from scripts.run_public_pod_ridge import predict
from scripts.audit_incremental_modal import linear,branch,same
from src.public_benchmark import atomic_json,file_hash


def audit():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);p=plan();result={}
    for task in p['tasks']:
        dest=OUT/task
        if not (dest/'development.json').exists():continue
        s=json.loads((dest/'development.json').read_text());root=ROOT/s['model_root']
        for path,h in s['artifacts'].items():
            if file_hash(ROOT/path)!=h:raise ValueError('artifact changed')
        arrays=ac_arrays(p) if task=='allen_cahn' else arrays_for(p['parent'],task)[0]
        rules=json.loads((root/'rules.json').read_text());risks=[];cached=[];held=[]
        for f,record in enumerate(rules['folds']):
            held+=record['held_rows'];train=record['train_rows'];test=record['held_rows']
            if set(train)&set(test) or set(train+test)&set(p['representation_rows']):raise ValueError('CV leakage')
            model=linear(root/f'linear_outer{f}',train);x=arrays['train_x'][test];y=arrays['train_y'][test].astype(float)
            base=predict(model,x);w=1/np.maximum(np.mean(y*y,1),record['floor'])
            risks.append(float(np.sum(w[:,None]*(base-y)**2)))
            delta=[branch(root/f'cv_f{f}_s{seed}',model,x) for seed in p['cv_seeds']]
            basecoef,coef=coefficients(model,x,y)
            same(np.sum(w[:,None]*(basecoef-coef)**2,0),record['base_risk'])
            same(np.mean([np.sum(w[:,None]*(basecoef+d-coef)**2,0) for d in delta],0),record['neural_risk'])
            cached.append((base,y,delta,model['q']))
        if sorted(held)!=sorted(p['correction_rows']):raise ValueError('fold coverage')
        hard,shares=field_mask(rules['folds'],risks,rules['masks']['stable'],p['field_threshold'])
        same(risks,s['risks']);same(shares,s['fold_field_shares']);same(hard,s['pre_joint_mask'])
        for arm,raw in [('field',hard),('field_shrink',hard*np.array(rules['masks']['shrink']))]:
            gains=[float(1-np.mean([errors(base+(d*raw)@q.T,y).mean() for d in ds])/errors(base,y).mean()) for base,y,ds,q in cached]
            same(gains,s['joint_gate'][arm]['fold_relative_l2_gains'])
            passed=all(g>=.01 for g in gains)
            if passed!=s['joint_gate'][arm]['passed']:raise ValueError('joint gate')
            same(raw if passed else np.zeros(16),s['masks'][arm])
        k=max(np.count_nonzero(s['masks']['field']),np.count_nonzero(s['masks']['field_shrink']))
        same(s['masks']['energy'],[1.]*k+[0.]*(16-k))
        model=linear(root/'linear_final',p['correction_rows']);base=predict(model,arrays['val_x'])
        for seed in p['seeds']:
            delta=branch(root/f'final_s{seed}',model,arrays['val_x'])
            for arm,mask in s['masks'].items():same(errors(base+(delta*mask)@model['q'].T,arrays['val_y']).mean(),s['scores'][arm][seed])
        for arm,v in s['scores'].items():same(np.mean(v),s['means'][arm])
        for key,arms in [('selected',['field','field_shrink']),('old_selected',['stable','shrink'])]:
            selected=min(arms,key=s['means'].get)
            if s['means'][selected]>s['means']['none']*.99:selected='none'
            if s[key]!=selected:raise ValueError('candidate rule')
        r=dict(development_replayed=True,new_neural_runs=s['new_neural_runs'],field_modes_one_based=(np.flatnonzero(s['masks']['field'])+1).tolist())
        if (dest/'confirmation.json').exists():
            c=json.loads((dest/'confirmation.json').read_text());lock=file_hash(OUT/'selection_lock.json')
            if c['lock']!=lock or c['ids']!=p['tasks'][task]['reserved']:raise ValueError('confirmation identity')
            chunks=[torch.load(ROOT/'external_data/field_modal_v1'/task/f'confirmation_{i}.pt',weights_only=False) for i in range(0,500,50)]
            if any(v['lock']!=lock for v in chunks) or [i for v in chunks for i in v['ids']]!=c['ids']:raise ValueError('cache provenance')
            x=np.concatenate([v['x'] for v in chunks]);y=np.concatenate([v['y'] for v in chunks]);base=predict(model,x)
            for seed in p['seeds']:
                delta=branch(root/f'final_s{seed}',model,x)
                for arm,mask in s['masks'].items():same(errors(base+(delta*mask)@model['q'].T,y),c['errors'][arm][seed])
            for key,arm in [('linear','none'),('energy','energy'),('old_rule',s['old_selected'])]:
                if compare(c['errors'][s['selected']],c['errors'][arm])!=c['comparisons'][key]:raise ValueError('inference mismatch')
            r['confirmation_replayed']=True
        result[task]=r
    atomic_json(OUT/'audit.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':audit()
