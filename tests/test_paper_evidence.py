"""Verify active manuscript tables against versioned results (no training)."""
import json
import re
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class PaperEvidenceTests(unittest.TestCase):
    def test_current_tables_and_counts(self):
        main=(ROOT/'paper/main.md').read_text(encoding='utf-8')
        tex=(ROOT/'paper/ml4ps2026.tex').read_text(encoding='utf-8')
        fixed=json.loads((ROOT/'results/public_fixed_t1_v1/summary.json').read_text())
        small=json.loads((ROOT/'results/public_smallfit_v1/summary.json').read_text())
        stable=json.loads((ROOT/'results/public_stability_v1/summary.json').read_text())
        for name,label in [('deeponet_mlp','MLP DeepONet'),('deeponet_fourier','Fourier DeepONet'),('fno','FNO')]:
            metric=fixed['groups'][name]['test']['mean_relative_l2']
            self.assertIn(f"| {label} | {metric['mean']:.5f} ± {metric['sample_std']:.5f} |",main)
        for name,g in stable['groups'].items():
            _,arch,schedule,batch=name.split('_')
            label='MLP' if arch=='mlp' else 'Fourier'
            size=batch[1:]
            error=g['budgets']['3000']['train']['projection_excess_relative_fro']['mean']
            val=g['budgets']['3000']['val']['mean_relative_l2']
            count=g['stability']['stable_tail']
            self.assertIn(f"| {label} | {schedule} | {size} | {error:.6f} | {count}/5 | {val['mean']:.5f} ± {val['sample_std']:.5f} |",main)
            self.assertIn(f"{label} & {schedule} & {size} & {error:.6f} & {count}/5 & {val['mean']:.5f}",tex)
        # These three are the numbers the manuscript cites. They are pinned deliberately: if a
        # re-run changes them the manuscript is stale, not the test.
        wins=sum(p['3000']['val']['improved_seeds'] for p in small['paired_zero_minus_random'].values())
        self.assertEqual(wins,29,'smallfit zero-init wins changed; main.md section 5.16 cites 29/30')
        cosine_stable=sum(g['stability']['stable_tail'] for k,g in stable['groups'].items() if '_cosine_' in k)
        self.assertEqual(cosine_stable,20,'cosine stable-tail count changed; main.md section 5.10 cites 20/20')
        self.assertEqual(stable['previous_controls_exactly_reproduced'],10,
                         'stability repeat-control count changed; main.md section 4 cites 10')
        families=['public_fixed_t1_v1','public_coefficients_v1','public_smallfit_v1','public_stability_v1']
        total=sum(json.loads((ROOT/'results'/f/'summary.json').read_text())['completed_runs'] for f in families)
        self.assertEqual(total,155,'completed-run total changed; main.md section 4 cites 155')

    def test_active_links_citations_and_archive(self):
        for filename in ('main.md','short_paper.md','ml4ps2026_4page.md','ml4ps2026.tex'):
            path=ROOT/'paper'/filename
            text=path.read_text(encoding='utf-8')
            self.assertNotIn('\ufffd',text)
            self.assertNotIn('Submitted to the 9th',text)
            self.assertTrue((ROOT/'paper/archive/pre_audit_20260924'/filename).is_file())
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if not target.startswith(('http:','https:','#')):
                    self.assertTrue((path.parent/target).is_file(),target)
        tex=(ROOT/'paper/ml4ps2026.tex').read_text(encoding='utf-8')
        citations={key for group in re.findall(r'\\cite\{([^}]+)\}',tex) for key in group.split(',')}
        bibliography=set(re.findall(r'\\bibitem\{([^}]+)\}',tex))
        self.assertLessEqual(citations,bibliography)
        self.assertEqual(tex.count('{'),tex.count('}'))
        self.assertEqual(tex.count('\\begin{'),tex.count('\\end{'))
        for target in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',tex):
            self.assertTrue((ROOT/'paper'/target).is_file(),target)


if __name__=='__main__': unittest.main()
