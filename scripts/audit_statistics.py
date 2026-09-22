"""Strict historical grouping; preserve sources and quarantine conflicting repeats."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config import ExpCfg
from scripts.analyze_phase import POSTFIX_DIRS


def signature(cfg, meta):
    c=dict(cfg)
    resolved=ExpCfg(**c).resolved()
    for key,value in zip(('pde_kwargs','model_kwargs','sampling_kwargs'),resolved):
        c[key]=value
    for key in ('seed','method','align_kind','eval_every','diag_every'):
        c.pop(key,None)
    # Missing historical code fingerprints cannot be reconstructed after the fact.
    c['code_hash']=meta.get('code_hash') or 'UNKNOWN'
    return json.dumps(c,sort_keys=True)


def main():
    groups=defaultdict(lambda:defaultdict(list))
    for directory in POSTFIX_DIRS:
        for path in (ROOT/'results'/directory).glob('*.json'):
            d=json.loads(path.read_text(encoding='utf-8'))
            c=d.get('meta',{}).get('cfg',{})
            if c.get('method') not in ('pi_baseline','pi_align'):
                continue
            if c.get('align_kind') not in ('none','prior'):
                continue
            groups[signature(c,d['meta'])][(c['method'],c['seed'])].append(
                dict(path=str(path.relative_to(ROOT)),value=d['test_rel_l2']))
    rows=[]
    for config,buckets in sorted(groups.items()):
        methods=defaultdict(list)
        conflicts=[]
        for (method,seed),records in sorted(buckets.items()):
            values=[r['value'] for r in records]
            if max(values)-min(values)>1e-8:
                conflicts.append(dict(method=method,seed=seed,records=records))
            else:
                methods[method].append(dict(seed=seed,value=values[0],sources=records))
        row=dict(id=hashlib.sha256(config.encode()).hexdigest()[:12],config=json.loads(config),
                 methods=dict(methods),conflicts=conflicts,
                 status='exploratory_unknown_code' if json.loads(config)['code_hash']=='UNKNOWN' else 'fingerprinted')
        b=[r['value'] for r in methods['pi_baseline']]
        a=[r['value'] for r in methods['pi_align']]
        if len(b)>=2 and len(a)>=2:
            row['comparison']=dict(n_baseline=len(b),n_align=len(a),baseline_mean=float(np.mean(b)),
                                   align_mean=float(np.mean(a)),welch_p=float(ttest_ind(b,a,equal_var=False).pvalue))
        rows.append(row)
    comparisons=[r['comparison'] for r in rows if 'comparison' in r]
    ordered=sorted(comparisons,key=lambda r:r['welch_p'])
    running=0.
    for i,r in enumerate(ordered):
        running=max(running,(len(ordered)-i)*r['welch_p'])
        r['holm_p']=min(1.,running)
    out=ROOT/'results/audit_v2/statistics.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(protocol='resolved full configuration; no averaging conflicting repeats; unknown code is exploratory',groups=rows),indent=2),encoding='utf-8')
    print(f'{len(rows)} configuration groups; {len(comparisons)} comparisons; {sum(len(r["conflicts"]) for r in rows)} conflicting seed groups quarantined',flush=True)


if __name__=='__main__':
    main()
