import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.summarize_public_stability import stability


class StabilitySummaryTests(unittest.TestCase):
    def audit(self,values):
        return stability([{'training':{'projection_excess_relative_fro':v}} for v in values])

    def test_rebound_not_hidden_by_final_recovery(self):
        result=self.audit([.02,.009,.011,.005])
        self.assertTrue(result['final_fit'])
        self.assertTrue(result['rebounded'])
        self.assertFalse(result['stable_tail'])

    def test_stable_tail_and_threshold_boundary(self):
        result=self.audit([.03,.01,.005,.004,.003])
        self.assertTrue(result['stable_tail'])
        self.assertFalse(result['rebounded'])
        self.assertAlmostEqual(result['last_four_range'],.007)

    def test_never_fit(self):
        result=self.audit([.02,.015,.014,.013])
        self.assertFalse(result['ever_fit'])
        self.assertFalse(result['rebounded'])


if __name__=='__main__': unittest.main()
