import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_public_stability import train_one, learning_rate
from scripts.run_public_smallfit import train_one as previous_train
from src.models import DeepONet


class StabilityTests(unittest.TestCase):
    def test_schedule_endpoints(self):
        cfg=dict(schedule='cosine',lr=1e-3,min_lr=1e-5,steps=3000)
        self.assertAlmostEqual(learning_rate(cfg,0),1e-3)
        self.assertAlmostEqual(learning_rate(cfg,2999),1e-5)
        self.assertGreater(learning_rate(cfg,1000),learning_rate(cfg,2000))

    def test_resume_and_previous_control(self):
        torch.set_num_threads(1)
        device=torch.device(os.environ.get('PUBLIC_TEST_DEVICE','cpu'))
        rng=np.random.default_rng(5)
        arrays={'x':((np.arange(80)+.5)/80).astype('float32')}
        for split in ('train','val'):
            arrays[split+'_x']=rng.normal(size=(128,80)).astype('float32')
            arrays[split+'_y']=arrays[split+'_x']*.2+.5
        protocol={'normalization':dict(input_mean=0.,input_std=1.,target_mean=.5,target_std=1.)}
        for schedule in ('constant','cosine'):
            for batch in (32,128):
                with self.subTest(schedule=schedule,batch=batch),tempfile.TemporaryDirectory() as d:
                    root=Path(d)
                    torch.manual_seed(19)
                    parent=DeepONet(branch_in=80,trunk_in=1,width=64,trunk_type='fourier')
                    source=root/'parent.pt'
                    torch.save(dict(model=parent.state_dict(),identity=dict(data_protocol=protocol,
                               configuration=dict(architecture='deeponet_fourier',seed=0))),source)
                    cfg=dict(architecture='deeponet_fourier',seed=0,n_train=128,zero_init=True,
                             steps=4,lr=1e-3,min_lr=1e-5,schedule=schedule,batch_size=batch,
                             checkpoint_every=1,eval_every=1)
                    full=train_one(cfg,arrays,protocol,source,root/'full',device)
                    train_one(cfg,arrays,protocol,source,root/'resume',device,stop_after=2)
                    resumed=train_one(cfg,arrays,protocol,source,root/'resume',device)
                    a=torch.load(root/'full/checkpoints/latest.pt',weights_only=False)
                    b=torch.load(root/'resume/checkpoints/latest.pt',weights_only=False)
                    for key in a['model']: self.assertTrue(torch.equal(a['model'][key],b['model'][key]))
                    self.assertEqual(full['milestones'],resumed['milestones'])
                    self.assertEqual(full['schedule_state'],resumed['schedule_state'])
                    self.assertEqual([h['learning_rate'] for h in full['history']],
                                     [h['learning_rate'] for h in resumed['history']])
                    with self.assertRaises(ValueError):
                        train_one({**cfg,'min_lr':2e-5},arrays,protocol,source,root/'resume',device)
                    if schedule=='constant' and batch==32:
                        old=previous_train(cfg,arrays,protocol,source,root/'old',device)
                        self.assertEqual(full['metrics'],old['metrics'])


if __name__=='__main__': unittest.main()
