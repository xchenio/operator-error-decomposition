"""Validate the actual matched benchmark runner, including its resume path."""
import sys
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_public_benchmark import train_one


class MatchedResumeTest(unittest.TestCase):
    def test_both_model_families_resume_and_match_metric(self):
        torch.set_num_threads(1)
        device=torch.device(os.environ.get('PUBLIC_TEST_DEVICE','cpu'))
        rng=np.random.default_rng(42)
        arrays={'x':((np.arange(32)+.5)/32).astype('float32')}
        for name in ('train','val','test'):
            arrays[name+'_x']=rng.normal(size=(8,32)).astype('float32')
            arrays[name+'_y']=(arrays[name+'_x']*.3+1).astype('float32')
        protocol={'normalization':{'input_mean':0.,'input_std':1.,'target_mean':0.,'target_std':1.}}
        for architecture in ('deeponet_mlp','fno'):
            with self.subTest(architecture=architecture),tempfile.TemporaryDirectory() as d:
                root=Path(d)
                cfg=dict(architecture=architecture,seed=3,steps=4,lr=1e-3,batch_size=3,
                         checkpoint_every=1,eval_every=2)
                expected=train_one(cfg,arrays,protocol,root/'continuous',device)
                train_one(cfg,arrays,protocol,root/'resumed',device,stop_after=2)
                actual=train_one(cfg,arrays,protocol,root/'resumed',device)
                a=torch.load(root/'continuous/checkpoints/latest.pt',weights_only=False)
                b=torch.load(root/'resumed/checkpoints/latest.pt',weights_only=False)
                for key in a['model']:
                    self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)
                self.assertEqual(expected['metrics'],actual['metrics'])
                with self.assertRaises(ValueError):
                    train_one({**cfg,'steps':5},arrays,protocol,root/'resumed',device)


if __name__=='__main__': unittest.main()
