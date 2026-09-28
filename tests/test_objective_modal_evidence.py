import json
from pathlib import Path
import unittest
import numpy as np
from scripts.objective_modal_core import stable_mask

ROOT=Path(__file__).resolve().parents[1]


class ObjectiveModalEvidenceTest(unittest.TestCase):
    def test_disjoint_protocol(self):
        p=json.loads((ROOT/'results/public_objective_modal_v1/plan.json').read_text())
        old=json.loads((ROOT/'external_data/pdebench/fixed_t1_v1/protocol.json').read_text()) if (ROOT/'external_data/pdebench/fixed_t1_v1/protocol.json').exists() else json.loads((ROOT/'results/public_fixed_t1_v1/protocol.json').read_text())
        consumed=json.loads((ROOT/'results/public_ridge_residual_v1/plan.json').read_text())['confirmation_ids']
        excluded=set(consumed)|set(range(128))
        for ids in old['sample_ids'].values():excluded.update(ids)
        fresh=p['tasks']['burgers_independent']
        allids=fresh['train']+fresh['val']+fresh['reserved_confirmation']
        self.assertEqual(len(set(allids)),1700)
        self.assertFalse(set(allids)&excluded)
        self.assertFalse(set(allids)&set(p['tasks']['burgers_original']['reserved_confirmation']))
        self.assertFalse(set(p['representation_rows'])&set(p['correction_rows']))

    def test_available_completed_records(self):
        root=ROOT/'results/public_objective_modal_v1'
        p=json.loads((root/'plan.json').read_text())
        completed=0
        for task in p['tasks']:
            folder=root/task
            if not (folder/'summary.json').exists():continue
            s=json.loads((folder/'summary.json').read_text());completed+=1
            self.assertEqual(s['completed_neural_runs'],41)
            selection=json.loads((folder/'modal_selection.json').read_text())
            self.assertEqual(stable_mask(selection['gains']).tolist(),selection['mask'])
            held=[]
            for f in selection['folds']:
                self.assertFalse(set(f['held_rows'])&set(f['train_rows']))
                self.assertFalse(set(f['held_rows'])&set(p['representation_rows']))
                held.extend(f['held_rows'])
            self.assertEqual(sorted(held),sorted(p['correction_rows']))
            for arm,g in s['modal_panels'].items():
                values=[]
                for seed in range(5):
                    r=json.loads((folder/f'{arm}_s{seed}/result.json').read_text())
                    values.append(r['metrics']['val']['prediction']['mean_relative_l2'])
                self.assertEqual(float(np.mean(values)),g['metrics']['val']['mean_relative_l2']['mean'])
        self.assertGreaterEqual(completed,1)


if __name__=='__main__':unittest.main()
