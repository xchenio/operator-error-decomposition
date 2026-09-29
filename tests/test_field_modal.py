import unittest
import numpy as np
from scripts.run_field_modal import field_mask,compare
from scripts.allen_cahn_data import initial,solve,reaction


class FieldModalTest(unittest.TestCase):
    def test_projection_floor_matters(self):
        fold=dict(base_risk=[10.,.001],neural_risk=[9.,0.])
        mask,shares=field_mask([fold]*3,[100.]*3,[1.,1.],.001)
        np.testing.assert_array_equal(mask,[1.,0.])
        self.assertAlmostEqual(shares[0,1],.00001)
        # Both percentages look attractive; a large uncorrectable field floor rejects both.
        mask,_=field_mask([fold]*3,[100000.]*3,[1.,1.],.001)
        np.testing.assert_array_equal(mask,[0.,0.])

    def test_strang_special_cases_and_chunk_identity(self):
        x=initial([22,23]);np.testing.assert_array_equal(x,np.concatenate([initial([22]),initial([23])]))
        np.testing.assert_allclose(initial([22],512)[:,::2],initial([22]),atol=1e-14)
        c=np.full((1,32),.25)
        np.testing.assert_allclose(solve(c,dt=.001,final_time=.01),reaction(c,.01),atol=1e-13)
        np.testing.assert_allclose(solve(c,rho=0,final_time=.01),c,atol=1e-13)

    def test_comparison_degenerate(self):
        a=np.ones((5,20));self.assertFalse(compare(a,a)['success'])
        self.assertTrue(compare(.9*a,a)['success'])


if __name__=='__main__':unittest.main()
