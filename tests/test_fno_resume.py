import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_fno_pdebench as run


class FNOResumeTest(unittest.TestCase):
    def test_interrupted_matches_continuous(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=root/'fixture.h5'
            with h5py.File(data,'w') as f:
                f['tensor']=np.random.default_rng(22).normal(size=(8,3,16)).astype('float32')
            base=['probe','--data',str(data),'--steps','4','--checkpoint-every','1',
                  '--device','cpu','--width','4','--layers','1','--modes','3']
            def invoke(name,resume=False):
                with patch.object(sys,'argv',base+['--out',str(root/name)]+(['--resume'] if resume else [])):
                    run.main()
            invoke('continuous.json')
            original=run.atomic_save
            def stop(state,path):
                original(state,path)
                if state['step']==2: raise InterruptedError('simulated interruption')
            with patch.object(run,'atomic_save',stop):
                with self.assertRaises(InterruptedError): invoke('resumed.json')
            invoke('resumed.json',True)
            a=torch.load(root/'continuous.pt',weights_only=False)
            b=torch.load(root/'resumed.pt',weights_only=False)
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)


if __name__=='__main__': unittest.main()
