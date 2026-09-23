import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_public_smallfit import train_one
from scripts.run_public_benchmark import metrics
from src.models import DeepONet


class SmallfitTests(unittest.TestCase):
    def test_initialization_subset_and_resume(self):
        torch.set_num_threads(1)
        device=torch.device(os.environ.get('PUBLIC_TEST_DEVICE','cpu'))
        rng=np.random.default_rng(5)
        arrays={'x':((np.arange(80)+.5)/80).astype('float32')}
        for split in ('train','val'):
            arrays[split+'_x']=rng.normal(size=(8,80)).astype('float32')
            arrays[split+'_y']=arrays[split+'_x']*.2+.5
        protocol={'normalization':dict(input_mean=0.,input_std=1.,target_mean=.5,target_std=1.)}
        for zero in (False,True):
            with self.subTest(zero=zero),tempfile.TemporaryDirectory() as d:
                root=Path(d)
                torch.manual_seed(19)
                parent=DeepONet(branch_in=80,trunk_in=1,width=64,trunk_type='fourier')
                source=root/'parent.pt'
                torch.save(dict(model=parent.state_dict(),identity=dict(data_protocol=protocol,
                           configuration=dict(architecture='deeponet_fourier',seed=0))),source)
                cfg=dict(architecture='deeponet_fourier',seed=0,n_train=4,zero_init=zero,
                         steps=4,lr=1e-3,checkpoint_every=1,eval_every=2)
                full=train_one(cfg,arrays,protocol,source,root/'full',device)
                train_one(cfg,arrays,protocol,source,root/'resume',device,stop_after=2)
                resumed=train_one(cfg,arrays,protocol,source,root/'resume',device)
                a=torch.load(root/'full/checkpoints/latest.pt',weights_only=False)
                b=torch.load(root/'resume/checkpoints/latest.pt',weights_only=False)
                for key in a['model']: self.assertTrue(torch.equal(a['model'][key],b['model'][key]))
                self.assertEqual(full['milestones'],resumed['milestones'])
                self.assertEqual(full['oracles'],resumed['oracles'])
                expected=np.random.default_rng(20260923).permutation(8)[:4].tolist()
                self.assertEqual(full['basis_info']['training_row_indices'],expected)
                if zero:
                    offset=arrays['val_x'].mean(axis=1,keepdims=True)
                    baseline=metrics(np.broadcast_to(offset,arrays['val_y'].shape),arrays['val_y'])
                    for key,value in baseline.items(): self.assertEqual(full['milestones']['0']['val'][key],value)
                    self.assertLess(full['oracles']['val']['relative_fro'],baseline['relative_fro'])
                with self.assertRaises(ValueError):
                    train_one({**cfg,'n_train':5},arrays,protocol,source,root/'resume',device)


if __name__=='__main__': unittest.main()
