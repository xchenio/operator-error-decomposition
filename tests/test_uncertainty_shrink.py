import unittest
import numpy as np
from scripts.uncertainty_shrink import coefficients,matched_global,bootstrap_moment,moments


class UncertaintyShrinkTest(unittest.TestCase):
    def test_continuity_and_no_new_zero(self):
        n=np.array([.2,2.,-.1,0.]);d=np.ones(4);zero=np.zeros(4)
        old=coefficients(n,d,zero);new=coefficients(n,d,np.ones(4)*.2)
        np.testing.assert_allclose(old,[.2,1,0,0]);self.assertTrue(np.all(new<=old))
        self.assertTrue(np.all(new[old>0]>0))
        a=coefficients(n,d,np.ones(4)*.2);b=coefficients(n,d,np.ones(4)*(.2+1e-8))
        self.assertLess(np.max(np.abs(a-b)),1e-7)
        # Stationarity at an interior optimum for the explicitly penalized objective.
        self.assertAlmostEqual((d[0]+1.96*.2)*a[0]-n[0],0.)

    def test_matched_energy(self):
        old=np.array([1.,.3,0.]);new=np.array([.8,.1,0.]);d=np.array([2.,3.,4.])
        matched,factor=matched_global(old,new,d)
        self.assertAlmostEqual(float(np.sum(d*matched**2)),float(np.sum(d*new**2)))
        self.assertGreater(factor,0);self.assertLess(factor,1)

    def test_bootstrap_replicate_resume_and_dependence(self):
        blocks=[np.stack([np.ones((20,3))*i,np.ones((20,3))*(i+1)]) for i in range(3)]
        uninterrupted=np.array([bootstrap_moment(blocks,i) for i in range(20)])
        resumed=np.array([bootstrap_moment(blocks,i) for i in range(8)]+[bootstrap_moment(blocks,i) for i in range(8,20)])
        np.testing.assert_array_equal(uninterrupted,resumed)
        self.assertEqual(float(uninterrupted[:,0].std()),0.)
        self.assertGreater(float(uninterrupted[:,1].std()),0.)
        n,d=moments(np.ones((20,3)),np.ones((2,20,3)),np.ones(20),4)
        np.testing.assert_array_equal(n,d);self.assertEqual(float(n.mean()),.25)


if __name__=='__main__':unittest.main()
