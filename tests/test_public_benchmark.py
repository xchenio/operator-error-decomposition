import sys
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.public_benchmark import split_indices, extract_pair


class ProtocolTest(unittest.TestCase):
    def test_splits_disjoint_reproducible(self):
        a=split_indices(100,(50,10,10),123)
        b=split_indices(100,(50,10,10),123)
        self.assertEqual(len(set(np.concatenate(list(a.values())))),70)
        for k in a: np.testing.assert_array_equal(a[k],b[k])

    def test_sample_time_space_semantics(self):
        values=np.arange(7*5*12,dtype='float32').reshape(7,5,12)
        with tempfile.TemporaryDirectory() as d:
            with h5py.File(Path(d)/'test.h5','w') as f:
                ds=f.create_dataset('tensor',data=values)
                x,y=extract_pair(ds,[5,1,3],2,3)
                np.testing.assert_array_equal(x,values[[5,1,3],0,::3])
                np.testing.assert_array_equal(y,values[[5,1,3],2,::3])


if __name__=='__main__': unittest.main()
