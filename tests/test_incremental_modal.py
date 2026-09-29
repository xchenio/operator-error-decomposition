import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts.run_incremental_modal import linear_grid,modal_rules,tune_linear,interval,errors
from scripts.run_public_pod_ridge import predict
from scripts.objective_modal_core import sample_weights


class IncrementalModalTest(unittest.TestCase):
    def test_weighted_ridge_stationarity_and_ols(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(40,5));q=np.eye(5)[:,:3]
        y=x@rng.normal(size=(5,5))+2
        for objective in ('mse','normalized'):
            m,_=next(linear_grid(x,y,q,objective,[.1]));z=(x-m['mean'])/m['std']
            c=(y-x.mean(1,keepdims=True))@q;w,_=sample_weights(y,objective)
            e=z@m['weights']+m['intercept']-c
            np.testing.assert_allclose(np.mean(w[:,None]*e,0),0,atol=1e-12)
            np.testing.assert_allclose(z.T@(w[:,None]*e)/len(x)+.1*m['weights'],0,atol=1e-12)
        m,_=next(linear_grid(x,y,np.eye(5),'mse',[0.]))
        np.testing.assert_allclose(predict(m,x),y,atol=1e-11)

    def test_modes_degenerate_and_shrink(self):
        f=dict(base_risk=[1.,1.,0.],neural_risk=[.8,1.1,0.],numerator=[.3,-.2,0.],denominator=[1.,1.,0.])
        masks,gains=modal_rules([f,f,f]);self.assertEqual(masks['stable'],[1.,0.,0.])
        np.testing.assert_allclose(masks['shrink'],[.3,0.,0.]);self.assertEqual(gains[0][2],0.)
        a=np.ones((5,20));r=interval(a,a);self.assertFalse(r['success']);self.assertEqual(r['interval'],[0.,0.])
        self.assertTrue(interval(.9*a,a)['success'])

    def test_inner_tuning_resume(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(36,4));y=x*.5
        p=dict(objectives=['mse','normalized'],alphas=[0.,.1,1.])
        with tempfile.TemporaryDirectory() as d:
            m=tune_linear(x,y,np.eye(4),p,Path(d),{'test':1})
            n=tune_linear(x,y,np.eye(4),p,Path(d),{'test':1})
            self.assertLess(errors(predict(m,x),y).mean(),1e-10)
            for k in m:np.testing.assert_array_equal(m[k],n[k])
            with self.assertRaisesRegex(ValueError,'identity'):tune_linear(x,y,np.eye(4),p,Path(d),{'test':2})


if __name__=='__main__':unittest.main()
