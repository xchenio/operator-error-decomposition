"""Persist paired descriptive comparisons without opening reserved confirmation data."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.public_benchmark import atomic_json
from scripts.audit_objective_modal import audit
from scripts.run_objective_modal import OUT


def pairs(root,left,right):
    result={}
    for metric in ('mean_relative_l2','rmse'):
        delta=[]
        for seed in range(5):
            a=json.loads((root/f'{left}_s{seed}/result.json').read_text())
            b=json.loads((root/f'{right}_s{seed}/result.json').read_text())
            delta.append(a['metrics']['val']['prediction'][metric]-b['metrics']['val']['prediction'][metric])
        result[metric]=dict(values=delta,mean=float(np.mean(delta)),sample_std=float(np.std(delta,ddof=1)),
                            improved_seeds=sum(v<0 for v in delta),equal_seeds=sum(v==0 for v in delta))
    return result


def main():
    checked=audit();tasks={}
    for task in checked['tasks']:
        root=OUT/task
        s=json.loads((root/'summary.json').read_text())
        comparisons={label:pairs(root,a,b) for label,a,b in [
            ('full_normalized_minus_mse','full_normalized','full_mse'),
            ('split_normalized_minus_mse','normalized_all','mse_all'),
            ('stable_minus_all','normalized_stable','normalized_all'),
            ('stable_minus_random','normalized_stable','normalized_random'),
            ('stable_minus_energy','normalized_stable','normalized_energy')]}
        same_masks=s['modal_panels']['normalized_stable']['masks']==s['modal_panels']['normalized_energy']['masks']
        exact_repeat=0
        for seed in range(5):
            a=json.loads((root/f'normalized_stable_s{seed}/result.json').read_text())
            b=json.loads((root/f'normalized_energy_s{seed}/result.json').read_text())
            exact_repeat+=a['history']==b['history'] and a['metrics']==b['metrics']
        tasks[task]=dict(full_objective=s['full_objective'],modal_panels=s['modal_panels'],
            baseline=s['baseline'],linear_controls=s['linear_controls'],k=s['k'],
            modes_zero_based=s['selected_modes'],paired=comparisons,
            stable_equals_energy_masks=same_masks,stable_energy_exact_history_repeats=exact_repeat)
    result=dict(audit=checked,tasks=tasks,confirmation_evaluated=False,
                inference='Paired optimizer-seed descriptions on development sets; no significance, confirmation, or novel-method superiority claim.')
    atomic_json(OUT/'summary.json',result)
    for task,s in tasks.items():
        print(task,'k',s['k'],'full',[s['full_objective'][k]['metrics']['val']['mean_relative_l2'] for k in ('mse','normalized')],
              'masks',[(k,v['metrics']['val']['mean_relative_l2']) for k,v in s['modal_panels'].items()],flush=True)


if __name__=='__main__':main()
