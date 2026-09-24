"""Summarize predeclared stability and budget diagnostics; do not select checkpoints."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.public_benchmark import atomic_json


def stats(values):
    return dict(mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)),values=values)


def stability(history):
    values=[h['training']['projection_excess_relative_fro'] for h in history]
    first=next((i for i,v in enumerate(values) if v<=.01),None)
    return dict(final_fit=values[-1]<=.01,stable_tail=all(v<=.01 for v in values[-4:]),
                ever_fit=first is not None,
                rebounded=first is not None and any(v>.01 for v in values[first+1:]),
                last_four_range=max(values[-4:])-min(values[-4:]))


def summarize(root):
    groups,paired,parents,common={},{},{},None
    for architecture in ('deeponet_mlp','deeponet_fourier'):
        records={}
        for schedule in ('constant','cosine'):
            for batch in (32,128):
                runs=[]
                for seed in range(5):
                    run=json.loads((root/f'{architecture}_s{seed}_{schedule}_b{batch}'/'result.json').read_text())
                    identity=run['identity']
                    expected=dict(architecture=architecture,seed=seed,n_train=128,zero_init=True,
                                  steps=3000,lr=1e-3,min_lr=1e-5,schedule=schedule,batch_size=batch,
                                  checkpoint_every=50,eval_every=250)
                    if identity['configuration']!=expected: raise ValueError('configuration mismatch')
                    key=(architecture,seed)
                    parent=identity['parent_checkpoint_sha256']
                    if key in parents and parents[key]!=parent: raise ValueError('parent mismatch')
                    parents[key]=parent
                    this={k:v for k,v in identity.items() if k not in ('configuration','parent_checkpoint_sha256')}
                    if common is not None and common!=this: raise ValueError('common identity mismatch')
                    common=this
                    if run['basis_info']['training_row_indices']!=np.random.default_rng(20260923).permutation(1000)[:128].tolist():
                        raise ValueError('subset mismatch')
                    if [h['step'] for h in run['history']]!=list(range(250,3001,250)):
                        raise ValueError('history grid mismatch')
                    if schedule=='constant' and batch==32:
                        old=json.loads((ROOT/f'results/public_smallfit_v1/{architecture}_s{seed}_n128_z1/result.json').read_text())
                        if old['metrics']!=run['metrics']: raise ValueError('previous control did not reproduce')
                    runs.append(run)
                records[(schedule,batch)]=runs
                audits=[stability(r['history']) for r in runs]
                group=dict(stability={k:sum(a[k] for a in audits) for k in ('final_fit','stable_tail','ever_fit','rebounded')},
                           tail_error_range=stats([a['last_four_range'] for a in audits]),
                           update_seconds=stats([r['train_seconds'] for r in runs]),
                           examples_seen=3000*batch,budgets={})
                for budget in ('750','1500','3000'):
                    group['budgets'][budget]={split:{metric:stats([r['milestones'][budget][split][metric] for r in runs])
                                                   for metric in ('mean_relative_l2','relative_fro','rmse','projection_excess_relative_fro')}
                                             for split in ('train','val')}
                name=f'{architecture}_{schedule}_b{batch}'
                groups[name]=group
                print(name,'excess',round(group['budgets']['3000']['train']['projection_excess_relative_fro']['mean'],6),
                      'stability',group['stability'],'val',round(group['budgets']['3000']['val']['mean_relative_l2']['mean'],6),flush=True)
        paired[architecture]={}
        for name,after,before in [('decay_b32',('cosine',32),('constant',32)),
                                  ('decay_b128',('cosine',128),('constant',128)),
                                  ('batch_constant',('constant',128),('constant',32)),
                                  ('batch_cosine',('cosine',128),('cosine',32))]:
            pair={}
            for split,metric in [('train','projection_excess_relative_fro'),('val','mean_relative_l2'),('val','rmse')]:
                delta=[a['metrics'][split][metric]-b['metrics'][split][metric] for a,b in zip(records[after],records[before])]
                pair[split+'_'+metric]={**stats(delta),'improved_seeds':sum(d<0 for d in delta)}
            paired[architecture][name]=pair
        for schedule in ('constant','cosine'):
            exposure={}
            for split,metric in [('train','projection_excess_relative_fro'),('val','mean_relative_l2')]:
                delta=[a['milestones']['750'][split][metric]-b['milestones']['3000'][split][metric]
                       for a,b in zip(records[(schedule,128)],records[(schedule,32)])]
                exposure[split+'_'+metric]=stats(delta)
            paired[architecture]['equal_examples_'+schedule]=dict(
                examples_seen=96000,full_batch_minus_minibatch=exposure,
                caveat='Different optimizer updates and decay position; descriptive, not equal-compute evidence')
    result=dict(completed_runs=40,groups=groups,paired_differences=paired,common_identity=common,
                previous_controls_exactly_reproduced=10,
                note='Five training seeds, shared data. Equal steps not equal samples/compute; validation only. No significance claim.')
    atomic_json(root/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,architecture in enumerate(('deeponet_mlp','deeponet_fourier')):
        for schedule in ('constant','cosine'):
            for batch in (32,128):
                histories=[json.loads((root/f'{architecture}_s{s}_{schedule}_b{batch}'/'result.json').read_text())['history'] for s in range(5)]
                steps=[h['step'] for h in histories[0]]
                for col,section,metric in [(0,'training','projection_excess_relative_fro'),(1,'validation','mean_relative_l2')]:
                    ax=axes[row,col]
                    values=np.array([[h[section][metric] for h in hist] for hist in histories])
                    mean,sd=values.mean(axis=0),values.std(axis=0,ddof=1)
                    line=ax.plot(steps,np.maximum(mean,1e-6),label=f'{schedule}, batch {batch}')[0]
                    ax.fill_between(steps,np.maximum(mean-sd,1e-6),np.maximum(mean+sd,1e-6),color=line.get_color(),alpha=.1)
        axes[row,0].set_yscale('log'); axes[row,0].axhline(.01,color='gray',ls=':')
        axes[row,1].axhline(.3745089947206789,color='gray',ls=':',label='Initial mean baseline')
        axes[row,0].set_ylabel('Train excess relative Frobenius')
        axes[row,1].set_ylabel('Validation mean relative L2')
        for ax in axes[row]:
            ax.set_title(architecture.replace('deeponet_','').upper()+' frozen trunk, n=128')
            ax.set_xlabel('Updates (batch 128 sees 4x samples)'); ax.grid(alpha=.2)
    axes[0,1].legend(fontsize=8)
    fig.suptitle('Zero residual: learning-rate and batch stability (5 seeds; mean +/- SD)')
    fig.tight_layout(); fig.savefig(root/'stability_curves.png',dpi=160); plt.close(fig)
    return result


if __name__=='__main__': summarize(ROOT/'results/public_stability_v1')
