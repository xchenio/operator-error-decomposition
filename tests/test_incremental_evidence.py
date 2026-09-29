import json
from pathlib import Path
import subprocess
import unittest
import numpy as np
from scripts.run_incremental_modal import interval,modal_rules

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/public_incremental_modal_v1'


class IncrementalEvidenceTest(unittest.TestCase):
    def test_seal_precedes_confirmation(self):
        lock=json.loads((OUT/'selection_lock.json').read_text())
        committed=json.loads(subprocess.check_output(['git','show','ad94e69:results/public_incremental_modal_v1/selection_lock.json'],cwd=ROOT))
        self.assertEqual(lock,committed)
        p=json.loads((OUT/'plan.json').read_text())
        for task in lock['tasks']:
            c=json.loads((OUT/task/'confirmation.json').read_text())
            self.assertTrue(c['git_commit'].startswith('ad94e69'))
            self.assertEqual(c['ids'],p['parent']['tasks'][task]['reserved_confirmation'])
            self.assertEqual(c['selected'],lock['tasks'][task]['selected'])

    def test_statistics_and_paper(self):
        paper=(ROOT/'paper/FOLLOWUP_INCREMENTAL_MODAL_20260929.md').read_text(encoding='utf-8')
        for task in ('burgers_original','burgers_independent','advection'):
            root=OUT/task;s=json.loads((root/'development.json').read_text());c=json.loads((root/'confirmation.json').read_text())
            rules=json.loads((root/'rules.json').read_text());masks,_=modal_rules(rules['folds'])
            self.assertEqual(masks,s['masks'])
            precision=12 if task=='advection' else 6
            for a,v in s['means'].items():self.assertIn(f'{v:.{precision}f}',paper)
            for a,v in c['mean_relative_l2'].items():
                self.assertEqual(v,float(np.mean(c['errors'][a])))
                self.assertIn(f'{v:.{precision}f}',paper)
            self.assertEqual(interval(c['errors'][c['selected']],c['errors']['none']),c['primary'])
            self.assertEqual(interval(c['errors'][c['selected']],c['errors']['energy']),c['increment_over_energy'])
            self.assertEqual(c['primary']['success'],task!='advection')
            self.assertFalse(c['increment_over_energy']['success'])


if __name__=='__main__':unittest.main()
