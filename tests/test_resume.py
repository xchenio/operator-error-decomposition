"""An interrupted Adam trajectory must equal uninterrupted training, bit for bit."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models import DeepONet
from src.pdes import make_problem
from src.data import Split
from src.trainer import Trainer, StageCfg
from src.utils import set_seed


class Interrupted(Exception):
    pass


class InterruptTrainer(Trainer):
    def _save_progress(self,opt,iteration):
        super()._save_progress(opt,iteration)
        if iteration==2:
            raise Interrupted()


class ResumeTest(unittest.TestCase):
    def test_exact_continuation(self):
        torch.set_num_threads(1)
        p=make_problem('dr',Nx=5,Nt=3,m=5)
        rng=np.random.default_rng(42)
        split=Split(rng.normal(size=(4,p.n_coef)).astype('float32'),
                    rng.normal(size=(4,p.n_sensors)).astype('float32'),
                    rng.normal(size=(4,len(p.grid))).astype('float32'))
        def trainer(cls=Trainer):
            set_seed(11)
            model=DeepONet(p.n_sensors,p.trunk_dim,width=3,branch_hidden=4,
                          trunk_hidden=4,branch_layers=1,trunk_layers=1)
            return cls(model,p,{k:split for k in ('train','val','test')},torch.device('cpu'),
                       seed=8,diag_U=np.eye(len(p.grid))[:,:3],diag_w=np.ones(3))
        stages=[StageCfg(kind='data',steps=5,n_pts=4)]
        continuous=trainer()
        expected=continuous.fit(stages,bs=2,eval_every=1,diag_every=1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resume.pt'
            with self.assertRaises(Interrupted):
                trainer(InterruptTrainer).fit(stages,bs=2,eval_every=1,diag_every=1,
                                             checkpoint_path=path,checkpoint_every=1)
            restored=trainer()
            actual=restored.fit(stages,bs=2,eval_every=1,diag_every=1,
                                checkpoint_path=path,checkpoint_every=1,resume=True)
            for k,v in continuous.model.state_dict().items():
                self.assertTrue(torch.equal(v,restored.model.state_dict()[k]),k)
            self.assertEqual(expected['history'],actual['history'])
            self.assertEqual(expected['diag_history'],actual['diag_history'])


if __name__=='__main__':
    unittest.main()
