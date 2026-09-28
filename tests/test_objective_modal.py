import os
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from scripts.objective_modal_core import sample_weights,stable_mask,train_one
from scripts.run_public_pod_ridge import fit_preprocess,fit_ridge
from scripts.diagnose_development_energy import summarize_bins


class ObjectiveModalTest(unittest.TestCase):
    def test_weights_masks_and_bins(self):
        y=np.arange(1,121,dtype=float).reshape(20,6)
        w,floor=sample_weights(y,'normalized')
        self.assertAlmostEqual(w.mean(),1.)
        self.assertEqual(floor,np.quantile(np.mean(y*y,axis=1),.05))
        self.assertGreater(w[0],w[-1])
        np.testing.assert_array_equal(sample_weights(y,'mse')[0],np.ones(20))
        gains=np.array([[.1,.1,0.,.001],[.2,-.1,0.,.002],[.3,.1,0.,.001]])
        np.testing.assert_array_equal(stable_mask(gains),[1,0,0,0])
        bins=summarize_bins(y,np.zeros_like(y),[y],np.quantile(np.mean(y*y,axis=1),[.2,.4,.6,.8]))
        self.assertEqual(sum(b['n'] for b in bins),20)
        self.assertTrue(all(b['residual_l2']==0 for b in bins))

    def test_resume_and_no_cv_selection_leakage(self):
        device=torch.device(os.environ.get('PUBLIC_TEST_DEVICE','cpu'))
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(43)
        x,y=rng.normal(size=(40,12)),rng.normal(size=(40,12))
        model=fit_ridge(fit_preprocess(x[:15],y[:15]),4,10.)
        arrays=dict(train_x=x[15:35],train_y=y[15:35],val_x=x[35:],val_y=y[35:])
        for objective in ('mse','normalized'):
            cfg=dict(rank=4,kind='mlp',hidden=8,layers=1,seed=0,objective=objective,
                mask=[1.,0.,1.,0.],fixed_final=True,steps=6,batch_size=8,lr=.001,lr_min=.00001,
                weight_decay=.0001,penalty=1.,eval_every=2,checkpoint_every=2,patience=1,min_steps=2,min_delta=10.)
            with tempfile.TemporaryDirectory() as d:
                a,b,c=[Path(d)/v for v in ('a','b','c')]
                full=train_one(cfg,arrays,model,{'test':True},a,device)
                train_one(cfg,arrays,model,{'test':True},b,device,stop_after=3)
                resumed=train_one(cfg,arrays,model,{'test':True},b,device)
                altered=train_one(cfg,dict(arrays,val_y=arrays['val_y']+5),model,{'test':True},c,device)
                self.assertEqual(full['history'],resumed['history'])
                self.assertEqual(full['best_step'],6)
                self.assertEqual(altered['best_step'],6)
                states=[torch.load(v/'checkpoints/selected.pt',weights_only=False,map_location='cpu')['model'] for v in (a,b,c)]
                for key in states[0]:
                    self.assertTrue(torch.equal(states[0][key],states[1][key]))
                    self.assertTrue(torch.equal(states[0][key],states[2][key]))
                with self.assertRaisesRegex(ValueError,'identity mismatch'):
                    train_one(dict(cfg,mask=[1.,1.,1.,1.]),arrays,model,{'test':True},a,device)


if __name__=='__main__':unittest.main()
