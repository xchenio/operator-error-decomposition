"""Check fixed-POD report numbers against every versioned final record."""
import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class PodReadoutEvidenceTest(unittest.TestCase):
    def test_tables_and_shared_basis(self):
        root = ROOT/'results/public_pod_readout_v1'
        summary = json.loads((root/'summary.json').read_text())
        report = (ROOT/'docs/PUBLIC_POD_READOUT_V1.md').read_text(encoding='utf-8')
        paper = (ROOT/'paper/FOLLOWUP_POD_READOUT_20260928.md').read_text(encoding='utf-8')
        self.assertEqual(summary['completed_runs'], 30)
        self.assertEqual(len(summary['groups']), 6)
        for rank in (16, 32, 64):
            ridge = next(r for r in summary['ridge'] if r['rank'] == rank)
            groups = {g['kind']: g for g in summary['groups'] if g['rank'] == rank}
            line = f"| {rank} | {ridge['metrics']['val']['oracle']['mean_relative_l2']:.6f} | {ridge['metrics']['val']['prediction']['mean_relative_l2']:.6f} |"
            for kind in ('linear', 'mlp'):
                m = groups[kind]['metrics']['val']['mean_relative_l2']
                line += f" {m['mean']:.6f} ± {m['std']:.6f} |"
            self.assertIn(line, report)
            self.assertIn(line, paper)
            context = None
            for kind in ('linear', 'mlp'):
                values = []
                for seed in range(5):
                    r = json.loads((root/f'{kind}_r{rank}_s{seed}/result.json').read_text())
                    if context is not None:
                        self.assertEqual(context, r['identity']['context'])
                    context = r['identity']['context']
                    values.append(r['metrics']['val']['prediction']['mean_relative_l2'])
                    for split in ('train', 'val'):
                        m = r['metrics'][split]
                        self.assertEqual(m['oracle'], ridge['metrics'][split]['oracle'])
                        self.assertLess(abs(m['prediction']['relative_fro']**2-m['oracle']['relative_fro']**2-m['coefficient_relative_fro']**2), 1e-10)
                self.assertEqual(values, groups[kind]['metrics']['val']['mean_relative_l2']['values'])
                self.assertEqual(float(np.mean(values)), groups[kind]['metrics']['val']['mean_relative_l2']['mean'])


if __name__ == '__main__':
    unittest.main()
