import unittest
import json
from pathlib import Path
import numpy as np
from scripts.uncertainty_shrink import coefficients,matched_global,bootstrap_moment,moments


class UncertaintyShrinkTest(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]
        needed=['results/public_uncertainty_modal_v1/plan_v2.json',
                'results/public_field_modal_v1/plan.json',
                'results/public_ridge_residual_v1/plan.json',
                'external_data/pdebench/fixed_t1_v1/protocol.json']
        missing=[p for p in needed if not (root/p).exists()]
        if missing:
            self.skipTest(f'artifacts absent (data is git-ignored): {missing}')

    def test_fresh_ids_and_preflight_metadata(self):
        root=Path(__file__).resolve().parents[1]
        p=json.loads((root/'results/public_uncertainty_modal_v1/plan_v2.json').read_text())
        old=json.loads((root/'results/public_field_modal_v1/plan.json').read_text())
        self.assertEqual(p['cv_seeds'],[0,1])
        used={'burgers':set(range(128)),'advection':set()}
        hist=json.loads((root/'external_data/pdebench/fixed_t1_v1/protocol.json').read_text())
        for ids in hist['sample_ids'].values():used['burgers'].update(ids)
        used['burgers'].update(json.loads((root/'results/public_ridge_residual_v1/plan.json').read_text())['confirmation_ids'])
        for task,s in old['parent']['tasks'].items():
            family='burgers' if task.startswith('burgers') else 'advection'
            used[family].update(s['train']+s['val']+s['reserved_confirmation']+old['tasks'][task]['reserved'])
        for task,s in p['tasks'].items():
            ids=set(s['reserved']);self.assertEqual(len(ids),500)
            if task=='allen_cahn':self.assertFalse(ids&set(range(1700)));self.assertFalse(ids&set(range(100000,100016)))
            else:
                family='burgers' if task.startswith('burgers') else 'advection'
                self.assertFalse(ids&used[family]);used[family].update(ids)

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
