"""Development-only energy stratification; never loads confirmation or old test arrays."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.public_benchmark import atomic_json,file_hash
from scripts.run_public_pod_readout import make_branch
from scripts.run_public_pod_ridge import predict
from scripts.run_public_ridge_residual import ANCHOR


def summarize_bins(y, baseline, predictions, edges):
    energy=np.mean(y*y,axis=1)
    ids=np.digitize(energy,edges)
    base_sq=np.mean((baseline-y)**2,axis=1)
    pred_sq=np.mean((np.asarray(predictions)-y[None])**2,axis=2)
    base_rel=np.sqrt(base_sq/np.maximum(energy,1e-24))
    pred_rel=np.sqrt(pred_sq/np.maximum(energy[None],1e-24)).mean(axis=0)
    records=[]
    for b in range(len(edges)+1):
        mask=ids==b
        if not mask.any():
            records.append(dict(bin=b,n=0));continue
        records.append(dict(bin=b,n=int(mask.sum()),energy_min=float(energy[mask].min()),
            energy_max=float(energy[mask].max()),baseline_l2=float(base_rel[mask].mean()),
            residual_l2=float(pred_rel[mask].mean()),delta_l2=float((pred_rel-base_rel)[mask].mean()),
            baseline_mse=float(base_sq[mask].mean()),residual_mse=float(pred_sq[:,mask].mean()),
            improved_trajectories=int(np.sum(pred_rel[mask]<base_rel[mask])),
            fraction_total_baseline_sse=float(base_sq[mask].sum()/base_sq.sum())))
    return records


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root=ROOT/'results/public_ridge_residual_v1'
    lock=json.loads((root/'selection_lock.json').read_text())
    data=ROOT/'external_data/pdebench/fixed_t1_v1'
    protocol=json.loads((data/'protocol.json').read_text())
    if file_hash(data/'pairs.npz')!=protocol['cache_sha256']:raise ValueError('cache mismatch')
    with np.load(data/'pairs.npz') as z:
        arrays={k:z[k].astype(np.float64) for k in ('train_x','train_y','val_x','val_y')}
    anchor=torch.load(ANCHOR,map_location='cpu',weights_only=False)['model']
    energies=np.mean(arrays['train_y']**2,axis=1)
    edges=np.quantile(energies,[.2,.4,.6,.8])
    result=dict(source_sha256=file_hash(Path(__file__)),cache_sha256=protocol['cache_sha256'],
        selection_lock_sha256=file_hash(root/'selection_lock.json'),energy_edges_train_only=edges.tolist(),
        proposed_denominator_floor_train_q05=float(np.quantile(energies,.05)),splits={},
        scope='Exploratory training/development diagnostics only; no confirmation/test labels or predictions loaded.')
    for split in ('train','val'):
        x,y=arrays[split+'_x'],arrays[split+'_y']
        baseline=predict(anchor,x)
        predictions=[]
        for r in lock['selected_group']['runs']:
            path=root/r['name']/'checkpoints/selected.pt'
            if file_hash(path)!=r['selected_sha256']:raise ValueError('checkpoint mismatch')
            cp=torch.load(path,map_location='cpu',weights_only=False)
            branch=make_branch(cp['identity']['config'],x.shape[1],torch.device('cuda'))
            branch.load_state_dict(cp['model']);branch.eval()
            with torch.no_grad():
                delta=branch(torch.tensor((x-anchor['mean'])/anchor['std'],dtype=torch.float32,device='cuda')).cpu().numpy().astype(np.float64)*cp['scale']
            predictions.append(baseline+delta@anchor['q'].T)
        result['splits'][split]=summarize_bins(y,baseline,predictions,edges)
    out=ROOT/'results/public_objective_modal_v1'
    atomic_json(out/'energy_diagnostic.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
