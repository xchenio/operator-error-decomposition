"""Complete factorial summaries; negative paired deltas mean improvement."""
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
    groups,paired={},{}
    common=None
    parents={}
    for architecture in ('deeponet_mlp','deeponet_fourier'):
        records={}
        for white,mean in ((0,0),(1,0),(0,1),(1,1)):
            name=f'{architecture}_w{white}_m{mean}'
            runs=[]
            for seed in range(5):
                run=json.loads((root/f'{architecture}_s{seed}_w{white}_m{mean}'/'result.json').read_text())
                identity=run['identity']; cfg=identity['configuration']
                expected=dict(architecture=architecture,seed=seed,whiten=bool(white),mean_lift=bool(mean),
                              steps=3000,lr=1e-3,checkpoint_every=50,eval_every=250)
                if cfg!=expected: raise ValueError('unexpected configuration')
                key=(architecture,seed)
                parent=identity['parent_checkpoint_sha256']
                if key in parents and parents[key]!=parent: raise ValueError('unmatched parent')
                parents[key]=parent
                this={k:v for k,v in identity.items() if k not in ('configuration','parent_checkpoint_sha256')}
                if common is not None and this!=common: raise ValueError('common identity mismatch')
                common=this
                runs.append(run)
            records[(white,mean)]=runs
            group=dict(parameters=runs[0]['trainable_parameters'],
                       rank=[r['basis_info']['rank'] for r in runs],
                       retained_condition=[r['basis_info']['retained_condition'] for r in runs],
                       update_seconds=stats([r['train_seconds'] for r in runs]),budgets={})
            for budget in ('1500','3000'):
                group['budgets'][budget]={split:{metric:stats([r['milestones'][budget][split][metric] for r in runs])
                                                for metric in ('mean_relative_l2','relative_fro','rmse')}
                                          for split in ('train','val','test')}
            groups[name]=group
            print(name,'val',round(group['budgets']['3000']['val']['mean_relative_l2']['mean'],5),
                  'test',group['budgets']['3000']['test']['mean_relative_l2'],flush=True)
        pairs={}
        for label,after,before in [('whitening_without_mean',(1,0),(0,0)),
                                   ('whitening_with_mean',(1,1),(0,1)),
                                   ('mean_without_whitening',(0,1),(0,0)),
                                   ('mean_with_whitening',(1,1),(1,0))]:
            pairs[label]={}
            for split in ('val','test'):
                delta=[a['metrics'][split]['mean_relative_l2']-b['metrics'][split]['mean_relative_l2']
                       for a,b in zip(records[after],records[before])]
                pairs[label][split]={**stats(delta),'improved_seeds':sum(v<0 for v in delta)}
        pairs['budget_3000_minus_1500']={}
        for arm,runs in records.items():
            pairs['budget_3000_minus_1500'][f'w{arm[0]}_m{arm[1]}']={
                split:stats([r['milestones']['3000'][split]['mean_relative_l2']-
                             r['milestones']['1500'][split]['mean_relative_l2'] for r in runs])
                for split in ('val','test')}
        paired[architecture]=pairs
    result=dict(completed_runs=40,groups=groups,paired_differences=paired,common_identity=common,
                note='Mean +/- seed sample std; paired deltas after minus before. Negative is better. No test selection or significance claim.')
    atomic_json(root/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),sharey=True)
    for ax,architecture in zip(axes,('deeponet_mlp','deeponet_fourier')):
        for white,mean in ((0,0),(1,0),(0,1),(1,1)):
            histories=[json.loads((root/f'{architecture}_s{s}_w{white}_m{mean}'/'result.json').read_text())['history']
                       for s in range(5)]
            steps=[r['step'] for r in histories[0]]
            if any([r['step'] for r in h]!=steps for h in histories): raise ValueError('history steps mismatch')
            values=np.array([[r['validation']['mean_relative_l2'] for r in h] for h in histories])
            avg,sd=values.mean(axis=0),values.std(axis=0,ddof=1)
            label=('Whitened' if white else 'Raw scale')+(' + mean' if mean else '')
            line=ax.plot(steps,avg,label=label)[0]
            ax.fill_between(steps,avg-sd,avg+sd,color=line.get_color(),alpha=.12)
        reference=json.loads((ROOT/'results/public_fixed_t1_v1/summary.json').read_text())
        ax.axhline(reference['baselines']['val']['initial_spatial_mean'],color='gray',ls=':',label='Initial mean baseline')
        ax.set_title(architecture.replace('deeponet_','').upper()+' frozen trunk')
        ax.set_xlabel('Additional branch updates'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Validation mean relative L2 (lower is better)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Matched fixed-trunk interventions: mean and seed SD (5 seeds)')
    fig.tight_layout()
    fig.savefig(root/'validation_curves.png',dpi=160)
    plt.close(fig)
    return result


if __name__=='__main__': summarize(ROOT/'results/public_coefficients_v1')
