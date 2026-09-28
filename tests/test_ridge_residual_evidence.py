import json
from pathlib import Path
import unittest
import numpy as np
from scripts.run_public_ridge_residual import paired_interval

ROOT = Path(__file__).resolve().parents[1]


class RidgeResidualEvidenceTest(unittest.TestCase):
    def test_confirmation_and_report(self):
        root = ROOT/'results/public_ridge_residual_v1'
        c = json.loads((root/'confirmation.json').read_text())
        p = json.loads((root/'plan.json').read_text())
        lock = json.loads((root/'selection_lock.json').read_text())
        residual = np.mean([r['per_trajectory_l2'] for r in c['per_seed']], axis=0)
        self.assertEqual(residual.shape, (500,))
        self.assertEqual(c['ids'], p['confirmation_ids'])
        excluded = {i for ids in p['protocol']['sample_ids'].values() for i in ids}
        self.assertFalse(excluded & set(c['ids']))
        self.assertEqual(len(set(c['ids'])), 500)
        self.assertEqual(c['use_residual'], lock['use_residual'])
        for name, values in c['baseline_per_trajectory_l2'].items():
            self.assertEqual(paired_interval(residual-np.array(values), p['bootstrap_seed'], p['bootstrap_replicates']), c['paired_comparisons'][name])
        primary = c['paired_comparisons']['frozen_ridge']
        self.assertEqual(c['primary_success'], primary['ci95'][1] < 0 and c['relative_gain_vs_frozen_ridge'] >= .01)
        self.assertFalse(c['primary_success'])
        for r in c['per_seed']:
            self.assertAlmostEqual(np.mean(r['per_trajectory_l2']), r['metrics']['mean_relative_l2'], places=14)
        mean_rmse = np.mean([r['metrics']['rmse'] for r in c['per_seed']])
        std_rmse = np.std([r['metrics']['rmse'] for r in c['per_seed']], ddof=1)
        row = f"| Ridge + selected residual | {lock['selected_group']['mean_validation_l2']:.6f} | {c['residual_mean_l2']:.6f} ± {c['residual_seed_std']:.6f} | {mean_rmse:.6f} ± {std_rmse:.6f} |"
        for file in ('docs/RIDGE_RESIDUAL_V1.md', 'paper/FOLLOWUP_RIDGE_RESIDUAL_20260928.md'):
            text = (ROOT/file).read_text(encoding='utf-8')
            self.assertIn(row, text)
            self.assertIn(f"[{primary['ci95'][0]:.6f}, {primary['ci95'][1]:.6f}]", text)


if __name__ == '__main__':
    unittest.main()
