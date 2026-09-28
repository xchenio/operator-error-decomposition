"""Replay disjoint-fold modal selection and every completed panel's aggregation."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_objective_modal import OUT,code_hash,cfg
from scripts.objective_modal_core import stable_mask
from scripts.run_public_ridge_residual import digest
from src.public_benchmark import atomic_json,file_hash


def audit():
    plan=json.loads((OUT/'plan.json').read_text())
    if plan['source']!=code_hash():raise ValueError('source changed')
    historical=json.loads((ROOT/'external_data/pdebench/fixed_t1_v1/protocol.json').read_text())
    prior=json.loads((ROOT/'results/public_ridge_residual_v1/plan.json').read_text())['confirmation_ids']
    excluded=set(range(128))|set(prior)
    for ids in historical['sample_ids'].values():excluded.update(ids)
    old_reserved=set(plan['tasks']['burgers_original']['reserved_confirmation'])
    new=plan['tasks']['burgers_independent']
    new_ids=new['train']+new['val']+new['reserved_confirmation']
    if len(set(new_ids))!=1700 or set(new_ids)&(excluded|old_reserved):raise ValueError('independent split overlap')
    for task,info in plan['tasks'].items():
        if len(set(info['train']+info['val']+info['reserved_confirmation']))!=1700:raise ValueError('within-task overlap')
    representation=set(plan['representation_rows']);correction=set(plan['correction_rows'])
    if len(representation)!=400 or len(correction)!=600 or representation&correction or representation|correction!=set(range(1000)):
        raise ValueError('representation/correction split invalid')
    tasks={}
    for task in plan['tasks']:
        root=OUT/task
        if not (root/'summary.json').exists():continue
        summary=json.loads((root/'summary.json').read_text())
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['status']!='completed' or summary['context']['plan_sha256']!=digest(plan):raise ValueError('incomplete/changed task')
        base_path=root/'checkpoints/bases.pt'
        if file_hash(base_path)!=summary['bases_sha256']:raise ValueError('basis checkpoint changed')
        bases=torch.load(base_path,map_location='cpu',weights_only=False)
        selected=json.loads((root/'modal_selection.json').read_text())
        gains=[];held_all=[]
        for fold in selected['folds']:
            held=set(fold['held_rows']);train=set(fold['train_rows']);held_all.extend(held)
            if held&train or held|train!=correction or len(held)!=200 or len(train)!=400:raise ValueError('CV fold leakage')
            risk=np.mean(fold['residual_risks'],axis=0)
            # Reject a degenerate denominator rather than certifying floor-driven modal gains.
            if np.min(fold['baseline_risk'])<=1e-20:raise ValueError('modal risk below numerical floor; selection requires separate handling')
            gains.append((1-risk/np.maximum(fold['baseline_risk'],1e-20)).tolist())
        if len(held_all)!=600 or set(held_all)!=correction or gains!=selected['gains']:raise ValueError('CV coverage/risk mismatch')
        mask=stable_mask(gains).tolist();k=int(sum(mask))
        if mask!=selected['mask'] or k!=summary['k']:raise ValueError('modal rule mismatch')
        count=steps=zero=0;records={};environment=None
        for p in sorted(root.glob('*/result.json')):
            r=json.loads(p.read_text());c=r['identity']['config'];context=r['identity']['context']
            if context['plan_sha256']!=digest(plan) or context['bases_sha256']!=summary['bases_sha256']:
                raise ValueError('run provenance mismatch')
            if environment is not None and r['identity']['environment']!=environment:raise ValueError('mixed environment')
            environment=r['identity']['environment']
            name=p.parent.name;seed=c['seed'];phase=context['phase']
            objective='normalized' if phase=='cv' or 'normalized' in name else 'mse'
            expected_mask=np.ones(16)
            if name.startswith('normalized_stable'):expected_mask=np.array(mask)
            elif name.startswith('normalized_energy'):
                expected_mask=np.zeros(16);expected_mask[:k]=1
            elif name.startswith('normalized_random'):
                expected_mask=np.zeros(16);expected_mask[np.random.default_rng(seed+7000).permutation(16)[:k]]=1
            if c!=cfg(plan,seed,objective,expected_mask,fixed=phase=='cv'):raise ValueError('configuration/mask mismatch')
            cp_path=p.parent/'checkpoints/selected.pt'
            if file_hash(cp_path)!=r['selected_sha256']:raise ValueError('selected checkpoint changed')
            cp=torch.load(cp_path,map_location='cpu',weights_only=False)
            expected_base=bases['full'] if phase=='full' else bases['anchor']
            if cp['identity']!=r['identity'] or cp['step']!=r['best_step']:raise ValueError('checkpoint identity')
            for key in expected_base:
                if not np.array_equal(cp['anchor'][key],expected_base[key]):raise ValueError('frozen model mutation')
            if phase=='cv':
                if r['best_step']!=1500 or r['stopped_step']!=1500:raise ValueError('CV used early selection')
                fold=selected['folds'][context['fold']]
                if context['train_rows']!=fold['train_rows'] or context['held_rows']!=fold['held_rows']:raise ValueError('CV sample identity')
            else:
                best=r['history'][0];bad=0
                for h in r['history'][1:]:
                    if h['metrics']['val']['prediction']['mean_relative_l2']<best['metrics']['val']['prediction']['mean_relative_l2']-c['min_delta']:
                        best,bad=h,0
                    else:bad+=1
                    if h['step']>=c['min_steps'] and bad>=c['patience'] and h!=r['history'][-1]:raise ValueError('ignored stopping rule')
                if best['step']!=r['best_step'] or best['metrics']!=r['metrics']:raise ValueError('best checkpoint mismatch')
            count+=1;steps+=r['stopped_step'];zero+=r['best_step']==0;records[name]=r
        if count!=41:raise ValueError('missing or unexpected neural runs')
        for group,arms in [('full_objective',list(summary['full_objective'])),('modal_panels',plan['arms'])]:
            for arm in arms:
                prefix='full_'+arm if group=='full_objective' else arm
                runs=[records[f'{prefix}_s{s}'] for s in plan['final_seeds']]
                for split in ('train','val'):
                    for m in ('mean_relative_l2','rmse','relative_fro'):
                        v=[r['metrics'][split]['prediction'][m] for r in runs]
                        if summary[group][arm]['metrics'][split][m]!=dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=1))):raise ValueError('aggregate mismatch')
        tasks[task]=dict(completed_runs=count,total_updates=steps,initial_checkpoint_selections=zero,
                        stable_modes=summary['selected_modes'],k=k,degenerate_masks=k in (0,16),
                        frozen_models_verified=True,strict_cv_split_verified=True)
    result=dict(plan_sha256=digest(plan),tasks=tasks,completed_neural_runs=sum(v['completed_runs'] for v in tasks.values()),
                reserved_confirmation_evaluated=False,limitations='Current evidence is development-only, including the independent training split and second-PDE development split.')
    atomic_json(OUT/'audit.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':audit()
