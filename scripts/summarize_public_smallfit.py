"""Validate all 60 locked runs and report fit thresholds and paired initialization effects."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.public_benchmark import atomic_json


def stats(values):
    return dict(mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)),values=values)


def summarize(root):
    groups,paired,common,parents={},{},None,{}
    for architecture in ('deeponet_mlp','deeponet_fourier'):
        for count in (32,128,1000):
            arms={}
            for zero in (False,True):
                runs=[]
                for seed in range(5):
                    name=f'{architecture}_s{seed}_n{count}_z{int(zero)}'
                    run=json.loads((root/name/'result.json').read_text())
                    identity=run['identity']
                    expected=dict(architecture=architecture,seed=seed,n_train=count,zero_init=zero,
                                  steps=3000,lr=1e-3,checkpoint_every=50,eval_every=250)
                    if identity['configuration']!=expected: raise ValueError('unexpected configuration')
                    key=(architecture,seed)
                    parent=identity['parent_checkpoint_sha256']
                    if key in parents and parents[key]!=parent: raise ValueError('parent mismatch')
                    parents[key]=parent
                    rows=np.random.default_rng(20260923).permutation(1000)[:count].tolist()
                    if run['basis_info']['training_row_indices']!=rows: raise ValueError('subset mismatch')
                    this={k:v for k,v in identity.items() if k not in ('configuration','parent_checkpoint_sha256')}
                    if common is not None and common!=this: raise ValueError('common identity mismatch')
                    common=this
                    if set(run['metrics'])!={'train','val'}: raise ValueError('unexpected evaluation split')
                    runs.append(run)
                arms[zero]=runs
                name=f'{architecture}_n{count}_z{int(zero)}'
                group=dict(train_oracle={k:stats([r['oracles']['train'][k] for r in runs]) for k in
                                         ('mean_relative_l2','relative_fro')},budgets={},
                           fit_threshold=.01,passed_fit_seeds=sum(r['metrics']['train']['projection_excess_relative_fro']<=.01 for r in runs))
                group['first_logged_fit_step']=[next((h['step'] for h in r['history']
                    if h['training']['projection_excess_relative_fro']<=.01),None) for r in runs]
                group['ever_passed_fit_seeds']=sum(s is not None for s in group['first_logged_fit_step'])
                for budget in ('0','1500','3000'):
                    group['budgets'][budget]={split:{metric:stats([r['milestones'][budget][split][metric] for r in runs])
                                                   for metric in ('mean_relative_l2','relative_fro','projection_excess_relative_fro')}
                                             for split in ('train','val')}
                groups[name]=group
                print(name,'excess',round(group['budgets']['3000']['train']['projection_excess_relative_fro']['mean'],6),
                      'fit',group['passed_fit_seeds'],'/5','val',round(group['budgets']['3000']['val']['mean_relative_l2']['mean'],6),flush=True)
            pair={}
            for budget in ('1500','3000'):
                pair[budget]={}
                for split in ('train','val'):
                    delta=[a['milestones'][budget][split]['mean_relative_l2']-b['milestones'][budget][split]['mean_relative_l2']
                           for a,b in zip(arms[True],arms[False])]
                    pair[budget][split]={**stats(delta),'improved_seeds':sum(d<0 for d in delta)}
            paired[f'{architecture}_n{count}']=pair
    result=dict(completed_runs=60,groups=groups,paired_zero_minus_random=paired,common_identity=common,
                note='Validation-only diagnostic, not few-shot learning. Negative paired differences are better. No significance claim.')
    atomic_json(root/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for row,architecture in enumerate(('deeponet_mlp','deeponet_fourier')):
        for col,count in enumerate((32,128,1000)):
            ax=axes[row,col]
            for zero in (False,True):
                histories=[json.loads((root/f'{architecture}_s{s}_n{count}_z{int(zero)}'/'result.json').read_text())['history'] for s in range(5)]
                steps=[h['step'] for h in histories[0]]
                if any([h['step'] for h in history]!=steps for history in histories): raise ValueError('history mismatch')
                values=np.array([[h['training']['projection_excess_relative_fro'] for h in history] for history in histories])
                avg,sd=values.mean(axis=0),values.std(axis=0,ddof=1)
                line=ax.plot(steps,np.maximum(avg,1e-6),label='Zero residual' if zero else 'Random output')[0]
                ax.fill_between(steps,np.maximum(avg-sd,1e-6),np.maximum(avg+sd,1e-6),color=line.get_color(),alpha=.15)
            ax.axhline(.01,color='gray',ls=':',label='Predeclared fit threshold')
            ax.set_yscale('log'); ax.grid(alpha=.2)
            ax.set_title(f'{architecture.replace("deeponet_", "").upper()}, n={count}')
            ax.set_xlabel('Branch updates')
            if col==0: ax.set_ylabel('Train excess relative Frobenius')
    axes[0,2].legend(fontsize=8)
    fig.suptitle('Small-subset fitting: mean and seed SD; display floor 1e-6')
    fig.tight_layout(); fig.savefig(root/'fit_curves.png',dpi=160); plt.close(fig)
    return result


if __name__=='__main__': summarize(ROOT/'results/public_smallfit_v1')
