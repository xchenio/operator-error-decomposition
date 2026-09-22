import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.projection_audit import projection_audit


class ProjectionTest(unittest.TestCase):
    def test_known_omitted_mode(self):
        s = np.diag([2., 1.])
        a = np.array([[0.], [1.]])
        y = a @ np.array([[0., 1.]])
        r = projection_audit(a, s, y)
        self.assertAlmostEqual(r['representation_sq'], .8)
        self.assertAlmostEqual(r['coefficient_sq'], 0.)

    def test_bias_and_pythagoras(self):
        rng = np.random.default_rng(4)
        a = np.column_stack([rng.normal(size=(30,4)), np.ones(30)])
        s = rng.normal(size=(30,9))
        y = a @ rng.normal(size=(5,9))
        r = projection_audit(a,s,y)
        self.assertLess(abs(r['closure_error']), 1e-12)
        self.assertLessEqual(r['projection_mean_relative_l2'], r['actual_mean_relative_l2'])

    def test_rank_deficient_and_zero(self):
        a = np.ones((6,3))
        self.assertEqual(projection_audit(a,a,a)['rank'], 1)
        self.assertEqual(projection_audit(a*0,a,a*0)['rank'], 0)


if __name__ == '__main__':
    unittest.main()
