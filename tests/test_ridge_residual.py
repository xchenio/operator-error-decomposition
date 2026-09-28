import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts.run_public_pod_ridge import fit_preprocess, fit_ridge, predict
from scripts.run_public_ridge_residual import train_one, choose_ids, paired_interval, digest, confirmation
from src.public_benchmark import atomic_json, file_hash


class RidgeResidualTest(unittest.TestCase):
    def test_disjoint_ids_and_bootstrap(self):
        protocol = dict(sample_ids=dict(train=list(range(100, 300)), val=list(range(300, 340)), test=list(range(340, 380))))
        ids = choose_ids(protocol, total=1000, count=100)
        self.assertEqual(ids, choose_ids(protocol, total=1000, count=100))
        self.assertEqual(len(set(ids)), 100)
        self.assertFalse(set(ids) & set(range(380)))
        with self.assertRaises(ValueError):
            choose_ids(protocol, total=400, count=100)
        self.assertEqual(paired_interval(np.zeros(20)), dict(mean=0., ci95=[0., 0.]))
        self.assertTrue(paired_interval(np.full(20, -.1))['ci95'][1] < 0)

    def test_zero_frozen_resume_and_patience(self):
        device = torch.device(os.environ.get('PUBLIC_TEST_DEVICE', 'cpu'))
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(12)
        x = rng.normal(size=(40, 12))
        y = .5*x+.2*np.sin(x)
        arrays = dict(train_x=x, train_y=y, val_x=x[:8], val_y=y[:8])
        anchor = fit_ridge(fit_preprocess(x, y), 4, 1.)
        original = copy.deepcopy(anchor)
        for force_stop in (False, True):
            cfg = dict(rank=4, kind='mlp', seed=1, hidden=8, layers=1, steps=8, batch_size=8,
                lr=.001, lr_min=.00001, weight_decay=.0001, penalty=.1, eval_every=2,
                checkpoint_every=2, patience=2, min_steps=4, min_delta=10. if force_stop else .000001)
            with tempfile.TemporaryDirectory() as d:
                a, b = Path(d)/'a', Path(d)/'b'
                full = train_one(cfg, arrays, anchor, {'test': True}, a, device)
                self.assertIsNone(train_one(cfg, arrays, anchor, {'test': True}, b, device, stop_after=3))
                resumed = train_one(cfg, arrays, anchor, {'test': True}, b, device)
                self.assertEqual(full['history'], resumed['history'])
                self.assertEqual(full['metrics'], resumed['metrics'])
                for filename in ('latest.pt', 'selected.pt'):
                    ca = torch.load(a/'checkpoints'/filename, weights_only=False, map_location='cpu')
                    cb = torch.load(b/'checkpoints'/filename, weights_only=False, map_location='cpu')
                    for k in ca['model']:
                        self.assertTrue(torch.equal(ca['model'][k], cb['model'][k]))
                pred = predict(anchor, x[:8])
                from scripts.run_public_benchmark import metrics
                self.assertEqual(full['history'][0]['metrics']['val']['prediction'], metrics(pred, y[:8]))
                self.assertEqual(full['history'][0]['metrics']['val']['correction_relative_fro'], 0.)
                if force_stop:
                    self.assertEqual(full['stopped_step'], 4)
                    self.assertEqual(full['best_step'], 0)
                    self.assertEqual(full['stop_reason'], 'patience')
                self.assertEqual(train_one(cfg, arrays, anchor, {'test': True}, b, device), resumed)
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    train_one(dict(cfg, penalty=1.), arrays, anchor, {'test': True}, b, device)
        for k in anchor:
            np.testing.assert_array_equal(anchor[k], original[k])

    def test_confirmation_commit_gate_and_completed_skip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lock = dict(selected='synthetic')
            plan = dict(test=True)
            atomic_json(root/'selection_lock.json', lock)
            with patch('scripts.run_public_ridge_residual.OUT', root), patch('scripts.run_public_ridge_residual.ROOT', root), \
                 patch('scripts.run_public_ridge_residual.validate_lock'), \
                 patch('scripts.run_public_ridge_residual.subprocess.check_output') as command, \
                 patch('scripts.run_public_ridge_residual.h5py.File') as h5:
                command.return_value = '{}'
                with self.assertRaisesRegex(ValueError, 'committed before confirmation'):
                    confirmation(plan, torch.device('cpu'))
                h5.assert_not_called()
                command.return_value = json.dumps(lock)
                result = dict(identity=dict(plan_sha256=digest(plan), selection_lock_sha256=file_hash(root/'selection_lock.json')), test_result=True)
                atomic_json(root/'confirmation.json', result)
                self.assertEqual(confirmation(plan, torch.device('cpu')), result)
                h5.assert_not_called()


if __name__ == '__main__':
    unittest.main()
