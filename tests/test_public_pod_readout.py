import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from scripts.run_public_pod_ridge import fit_preprocess, fit_ridge
from scripts.run_public_pod_readout import prepare, make_branch, train_one, learning_rate


class PodReadoutTest(unittest.TestCase):
    def test_same_basis_and_objective(self):
        rng = np.random.default_rng(19)
        x, y = rng.normal(size=(40, 12)), rng.normal(size=(40, 12))
        model = fit_ridge(fit_preprocess(x, y), 4, 1.)
        arrays = dict(train_x=x, train_y=y, val_x=x[:5], val_y=y[:5])
        prepared, scale = prepare(arrays, model)
        np.testing.assert_array_equal(prepared['train']['x'], (x-model['mean'])/model['std'])
        coef = rng.normal(size=(40, 4))
        pred = coef@model['q'].T+x.mean(axis=1, keepdims=True)
        residual = y-(prepared['train']['coef']@model['q'].T+x.mean(axis=1, keepdims=True))
        self.assertAlmostEqual(np.mean((pred-y)**2), np.sum((coef-prepared['train']['coef'])**2)/(40*12)+np.mean(residual**2), places=12)
        changed = dict(arrays, val_y=arrays['val_y']+100)
        changed_prepared, changed_scale = prepare(changed, model)
        self.assertEqual(scale, changed_scale)
        np.testing.assert_array_equal(prepared['train']['coef'], changed_prepared['train']['coef'])

    def test_resume_and_zero(self):
        device = torch.device(os.environ.get('PUBLIC_TEST_DEVICE', 'cpu'))
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(7)
        x, y = rng.normal(size=(40, 12)), rng.normal(size=(40, 12))
        model = fit_ridge(fit_preprocess(x, y), 4, 1.)
        arrays = dict(train_x=x, train_y=y, val_x=x[:5], val_y=y[:5])
        for kind in ('linear', 'mlp'):
            cfg = dict(rank=4, kind=kind, seed=2, hidden=8, layers=2, steps=6,
                       batch_size=8, lr=.001, lr_min=.00001, eval_every=2, checkpoint_every=2)
            branch = make_branch(cfg, 12, device)
            self.assertEqual(torch.count_nonzero(branch(torch.ones(3, 12, device=device))).item(), 0)
            self.assertAlmostEqual(learning_rate(cfg, 0), cfg['lr'])
            self.assertAlmostEqual(learning_rate(cfg, 5), cfg['lr_min'])
            with tempfile.TemporaryDirectory() as d:
                a, b = Path(d)/'a', Path(d)/'b'
                full = train_one(cfg, arrays, model, {'synthetic': True}, a, device)
                self.assertIsNone(train_one(cfg, arrays, model, {'synthetic': True}, b, device, stop_after=3))
                resumed = train_one(cfg, arrays, model, {'synthetic': True}, b, device)
                self.assertEqual(full['history'], resumed['history'])
                ca = torch.load(a/'checkpoints/latest.pt', weights_only=False, map_location='cpu')
                cb = torch.load(b/'checkpoints/latest.pt', weights_only=False, map_location='cpu')
                for key in ca['model']:
                    self.assertTrue(torch.equal(ca['model'][key], cb['model'][key]))
                self.assertEqual(train_one(cfg, arrays, model, {'synthetic': True}, b, device), resumed)
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    train_one(dict(cfg, lr=.002), arrays, model, {'synthetic': True}, b, device)


if __name__ == '__main__':
    unittest.main()
