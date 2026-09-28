"""Post-training energy strata on development data only; no model selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_objective_modal import OUT,predict_delta
from scripts.run_public_pod_ridge import predict
from scripts.diagnose_development_energy import summarize_bins
from src.public_benchmark import atomic_json,file_hash


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    result=dict(source_sha256=file_hash(Path(__file__)),tasks={},scope='Exploratory development-only strata after fixed model selection; not confirmation.')
    plan=json.loads((OUT/'plan.json').read_text())
    for task in plan['tasks']:
        folder=OUT/task
        if not (folder/'summary.json').exists():continue
        summary=json.loads((folder/'summary.json').read_text())
        cache=ROOT/'external_data/pdebench/objective_modal_v1'/task/'development.pt'
        if file_hash(cache)!=summary['context']['cache_sha256']:raise ValueError('cache changed')
        a=torch.load(cache,map_location='cpu',weights_only=False)['arrays']
        edges=np.quantile(np.mean(a['train_y'].astype(np.float64)**2,axis=1),[.2,.4,.6,.8])
        task_result=dict(edges_train_only=edges.tolist(),arms={})
        for objective in ('mse','normalized'):
            predictions=[]
            for seed in range(5):
                name=folder/f'full_{objective}_s{seed}'
                record=json.loads((name/'result.json').read_text())
                if file_hash(name/'checkpoints/selected.pt')!=record['selected_sha256']:raise ValueError('model changed')
                delta,cp=predict_delta(name,a['val_x'],torch.device('cuda'))
                baseline=predict(cp['anchor'],a['val_x'])
                predictions.append(baseline+delta@cp['anchor']['q'].T)
            bins=summarize_bins(a['val_y'].astype(np.float64),baseline,predictions,edges)
            aggregate=sum(b['n']*b.get('residual_l2',0) for b in bins)/len(a['val_y'])
            expected=summary['full_objective'][objective]['metrics']['val']['mean_relative_l2']['mean']
            if abs(aggregate-expected)>1e-12:raise ValueError('stratified aggregate mismatch')
            task_result['arms'][objective]=bins
        result['tasks'][task]=task_result
    atomic_json(OUT/'objective_energy_strata.json',result)
    for name,t in result['tasks'].items():
        print(name,[(a['bin'],a['n'],a.get('residual_l2'),b.get('residual_l2')) for a,b in zip(t['arms']['mse'],t['arms']['normalized'])])


if __name__=='__main__':main()
