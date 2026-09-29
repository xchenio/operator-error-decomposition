import json
from pathlib import Path
import subprocess
import unittest
import numpy as np
from scripts.run_field_modal import compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/public_field_modal_v1'


class FieldEvidenceTest(unittest.TestCase):
    def test_seal_and_tables(self):
        locked=json.loads(subprocess.check_output(['git','show','8a4020b:results/public_field_modal_v1/selection_lock.json'],cwd=ROOT))
        self.assertEqual(locked,json.loads((OUT/'selection_lock.json').read_text()))
        paper=(ROOT/'paper/FOLLOWUP_FIELD_MODAL_20260929.md').read_text(encoding='utf-8')
        for task in locked['tasks']:
            s=json.loads((OUT/task/'development.json').read_text());c=json.loads((OUT/task/'confirmation.json').read_text())
            self.assertTrue(c['git_commit'].startswith('8a4020b'))
            for r in (s,c):
                for arm in ('none',s['selected'],'energy',s['old_selected']):
                    self.assertIn(f"{r['means'][arm]:.{12 if task=='advection' else 6}f}",paper)
            for label,arm in [('linear','none'),('energy','energy'),('old_rule',s['old_selected'])]:
                self.assertEqual(compare(c['errors'][s['selected']],c['errors'][arm]),c['comparisons'][label])
            self.assertEqual(c['comparisons']['linear']['success'],task!='advection')
            self.assertFalse(c['comparisons']['energy']['success']);self.assertFalse(c['comparisons']['old_rule']['success'])
            for arm,values in c['errors'].items():
                self.assertEqual(np.array(values).shape,(5,500));self.assertEqual(float(np.mean(values)),c['means'][arm])


if __name__=='__main__':unittest.main()
