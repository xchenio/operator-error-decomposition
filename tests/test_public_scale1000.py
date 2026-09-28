import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_public_scale1000 import train_one
from scripts.run_public_smallfit import train_one as old_train
from scripts.run_public_pod_ridge import fit_preprocess,fit_ridge,predict
from src.models import DeepONet


class ScaleTests(unittest.TestCase):
    def test_ridge_matches_normal_equation(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(50,8)); y=rng.normal(size=(50,8))
        pre=fit_preprocess(x,y); model=fit_ridge(pre,4,.1)
        z=(x-pre['mean'])/pre['std']
        coef=pre['residual']@model['q']; intercept=coef.mean(axis=0)
        expected=np.linalg.solve(z.T@z+50*.1*np.eye(8),z.T@(coef-intercept))
        np.testing.assert_allclose(model['weights'],expected,atol=1e-12)
        np.testing.assert_allclose(model['q'].T@model['q'],np.eye(4),atol=1e-12)
        before=model['weights'].copy(); output=predict(model,rng.normal(size=(12,8)))
        self.assertEqual(output.shape,(12,8))
        np.testing.assert_array_equal(before,model['weights'])

    def test_scale_resume_and_control(self):
        torch.set_num_threads(1)
        device=torch.device(os.environ.get('PUBLIC_TEST_DEVICE','cpu'))
        rng=np.random.default_rng(5)
        arrays={'x':((np.arange(80)+.5)/80).astype('float32')}
        for split,n in (('train',1000),('val',8)):
            arrays[split+'_x']=rng.normal(size=(n,80)).astype('float32')
            arrays[split+'_y']=arrays[split+'_x']*.2+.5
        protocol={'normalization':dict(input_mean=0.,input_std=1.,target_mean=.5,target_std=1.)}
        for schedule in ('constant','cosine'):
            with self.subTest(schedule=schedule),tempfile.TemporaryDirectory() as d:
                root=Path(d); torch.manual_seed(19)
                parent=DeepONet(branch_in=80,trunk_in=1,width=64,trunk_type='fourier')
                source=root/'parent.pt'
                torch.save(dict(model=parent.state_dict(),identity=dict(data_protocol=protocol,
                           configuration=dict(architecture='deeponet_fourier',seed=0))),source)
                cfg=dict(architecture='deeponet_fourier',seed=0,n_train=1000,zero_init=True,
                         steps=4,lr=1e-3,min_lr=1e-5,schedule=schedule,batch_size=32,
                         checkpoint_every=1,eval_every=1)
                full=train_one(cfg,arrays,protocol,source,root/'full',device)
                train_one(cfg,arrays,protocol,source,root/'resume',device,stop_after=2)
                resumed=train_one(cfg,arrays,protocol,source,root/'resume',device)
                a=torch.load(root/'full/checkpoints/latest.pt',weights_only=False)
                b=torch.load(root/'resume/checkpoints/latest.pt',weights_only=False)
                for key in a['model']: self.assertTrue(torch.equal(a['model'][key],b['model'][key]))
                self.assertEqual(full['milestones'],resumed['milestones'])
                self.assertEqual(full['schedule_state'],resumed['schedule_state'])
                with self.assertRaises(ValueError):
                    train_one({**cfg,'min_lr':2e-5},arrays,protocol,source,root/'resume',device)
                if schedule=='constant':
                    self.assertEqual(full['metrics'],old_train(cfg,arrays,protocol,source,root/'old',device)['metrics'])


if __name__=='__main__': unittest.main()
