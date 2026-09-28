"""Identity-checked 1000-function schedule comparison and validation-tuned linear baseline."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.public_benchmark import atomic_json


def stats(values):
    return dict(mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)),values=values)


def summarize():
    root=ROOT/'results/public_scale1000_v1'
    groups,paired,parents,common={},{},{},None
    for arch in ('deeponet_mlp','deeponet_fourier'):
        arms={}
        for schedule in ('constant','cosine'):
            runs=[]
            for seed in range(5):
                run=json.loads((root/f'{arch}_s{seed}_{schedule}_b32/result.json').read_text())
                identity=run['identity']
                cfg=dict(architecture=arch,seed=seed,n_train=1000,zero_init=True,steps=3000,
                         lr=1e-3,min_lr=1e-5,schedule=schedule,batch_size=32,checkpoint_every=50,eval_every=250)
                if identity['configuration']!=cfg: raise ValueError('config mismatch')
                if run['basis_info']['training_row_indices']!=np.random.default_rng(20260923).permutation(1000).tolist():
                    raise ValueError('subset mismatch')
                key=(arch,seed); parent=identity['parent_checkpoint_sha256']
                if key in parents and parents[key]!=parent: raise ValueError('parent mismatch')
                parents[key]=parent
                shared={k:v for k,v in identity.items() if k not in ('configuration','parent_checkpoint_sha256')}
                if common is not None and common!=shared: raise ValueError('identity mismatch')
                common=shared
                if schedule=='constant':
                    old=json.loads((ROOT/f'results/public_smallfit_v1/{arch}_s{seed}_n1000_z1/result.json').read_text())
                    if old['metrics']!=run['metrics']: raise ValueError('old control mismatch')
                runs.append(run)
            arms[schedule]=runs
            group=dict(metrics={split:{metric:stats([r['metrics'][split][metric] for r in runs])
                                       for metric in ('mean_relative_l2','relative_fro','rmse','projection_excess_relative_fro')}
                                 for split in ('train','val')},
                       final_fit_seeds=sum(r['metrics']['train']['projection_excess_relative_fro']<=.01 for r in runs),
                       oracle={split:stats([r['oracles'][split]['mean_relative_l2'] for r in runs]) for split in ('train','val')})
            groups[arch+'_'+schedule]=group
            print(arch,schedule,'val',group['metrics']['val']['mean_relative_l2'],'train',group['metrics']['train']['mean_relative_l2']['mean'],flush=True)
        pairs={}
        for split,metric in [('train','projection_excess_relative_fro'),('val','mean_relative_l2'),('val','rmse')]:
            delta=[a['metrics'][split][metric]-b['metrics'][split][metric] for a,b in zip(arms['cosine'],arms['constant'])]
            pairs[split+'_'+metric]={**stats(delta),'improved_seeds':sum(d<0 for d in delta)}
        paired[arch]=pairs
    ridge=json.loads((ROOT/'results/public_pod_ridge_v1/summary.json').read_text())
    if ridge['identity']['protocol']!=common['protocol'] or ridge['completed_cases']!=18: raise ValueError('ridge protocol mismatch')
    expected=[(r,a) for r in (16,32,64) for a in (.0001,.001,.01,.1,1.,10.)]
    if [(r['rank'],r['alpha']) for r in ridge['cases']]!=expected: raise ValueError('ridge grid mismatch')
    chosen=min(ridge['cases'],key=lambda c:c['metrics']['val']['prediction']['mean_relative_l2'])
    if chosen!=ridge['selected']: raise ValueError('ridge selection mismatch')
    baseline=json.loads((ROOT/'results/public_fixed_t1_v1/summary.json').read_text())['baselines']['val']['initial_spatial_mean']
    summary=dict(completed_runs=20,common_identity=common,previous_controls_exactly_reproduced=10,
                 groups=groups,paired_cosine_minus_constant=paired,initial_mean_validation_l2=baseline,
                 pod_ridge_selected=ridge['selected'],pod_ridge_completed_cases=18,
                 limitations='Validation exploration, not new test evidence; ridge selected among 18 candidates, not budget-matched to untuned networks; no new method claim.')
    atomic_json(root/'summary.json',summary)
    print('POD selection',ridge['selected'],flush=True)
    return summary


if __name__=='__main__': summarize()
