import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_public_coefficients import paired_bases, train_one
from src.models import DeepONet


class CoefficientTests(unittest.TestCase):
    def test_same_span_and_scale(self):
        rng=np.random.default_rng(7)
        features=rng.normal(size=(80,6))*np.array([1,2,3,4,5,1e-10])
        raw,white,info=paired_bases(features)
        rank=info['rank']
        qr=np.linalg.svd(raw.astype('float64'),full_matrices=False)[0][:,:rank]
        qw=np.linalg.svd(white.astype('float64'),full_matrices=False)[0][:,:rank]
        np.testing.assert_allclose(qr@qr.T,qw@qw.T,atol=1e-7)
        self.assertEqual(raw.shape,white.shape)
        self.assertAlmostEqual(float(np.mean(raw**2)),1.,places=6)
        self.assertAlmostEqual(float(np.mean(white**2)),1.,places=6)
        singular=np.linalg.svd(white.astype('float64'),compute_uv=False)[:rank]
        self.assertLess(singular[0]/singular[-1],1.000001)

    def test_resume(self):
        torch.set_num_threads(1)
        device=torch.device(os.environ.get('PUBLIC_TEST_DEVICE','cpu'))
        rng=np.random.default_rng(5)
        arrays={'x':((np.arange(80)+.5)/80).astype('float32')}
        for split in ('train','val','test'):
            arrays[split+'_x']=rng.normal(size=(8,80)).astype('float32')
            arrays[split+'_y']=arrays[split+'_x']*.2+.5
        protocol={'normalization':dict(input_mean=0.,input_std=1.,target_mean=.5,target_std=1.)}
        for whiten,mean_lift in ((False,False),(True,True)):
            with self.subTest(whiten=whiten),tempfile.TemporaryDirectory() as d:
                root=Path(d)
                parent=DeepONet(branch_in=80,trunk_in=1,width=64,trunk_type='fourier')
                source=root/'parent.pt'
                torch.save(dict(model=parent.state_dict(),identity=dict(data_protocol=protocol,
                           configuration=dict(architecture='deeponet_fourier',seed=0))),source)
                cfg=dict(architecture='deeponet_fourier',seed=0,whiten=whiten,mean_lift=mean_lift,
                         steps=4,lr=1e-3,checkpoint_every=1,eval_every=2)
                full=train_one(cfg,arrays,protocol,source,root/'full',device)
                train_one(cfg,arrays,protocol,source,root/'resume',device,stop_after=2)
                resumed=train_one(cfg,arrays,protocol,source,root/'resume',device)
                a=torch.load(root/'full/checkpoints/latest.pt',weights_only=False)
                b=torch.load(root/'resume/checkpoints/latest.pt',weights_only=False)
                for key in a['model']: self.assertTrue(torch.equal(a['model'][key],b['model'][key]))
                self.assertEqual(full['metrics'],resumed['metrics'])
                self.assertEqual(full['milestones'],resumed['milestones'])
                with self.assertRaises(ValueError):
                    train_one({**cfg,'steps':5},arrays,protocol,source,root/'resume',device)


if __name__=='__main__': unittest.main()
