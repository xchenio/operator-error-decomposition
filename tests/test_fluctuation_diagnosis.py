import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_fluctuation import decompose  # noqa: E402


class DecomposeTest(unittest.TestCase):
    def test_perfect_prediction(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(size=(20, 64))
        d = decompose(Y, Y)
        self.assertAlmostEqual(d["mean_relative_l2"], 0.0)
        self.assertAlmostEqual(d["fluctuation_r2"], 1.0)
        self.assertAlmostEqual(d["normalized_mean_error"], 0.0)

    def test_zero_fluctuation_predictors_all_give_zero_r2(self):
        """R^2 isolates fluctuation skill: any predictor with zero fluctuation scores 0.

        Both the exact-mean predictor and the all-zero predictor have P_c = 0, so the
        measure is invariant to the mean and must not be read as total accuracy.
        """
        rng = np.random.default_rng(1)
        Y = rng.normal(size=(16, 32)) + 3.0
        mean_only = np.repeat(Y.mean(axis=1)[:, None], Y.shape[1], axis=1)
        for tag, P in (("mean-only", mean_only), ("zero", np.zeros_like(Y))):
            with self.subTest(tag):
                self.assertAlmostEqual(decompose(P, Y)["fluctuation_r2"], 0.0)
        # the two differ only in the mean term, which R^2 does not see
        self.assertNotAlmostEqual(decompose(mean_only, Y)["mean_relative_l2"],
                                  decompose(np.zeros_like(Y), Y)["mean_relative_l2"])

    def test_negative_r2_reported_not_clipped(self):
        rng = np.random.default_rng(2)
        Y = rng.normal(size=(16, 32))
        Yc = Y - Y.mean(axis=1, keepdims=True)
        worse = -2.0 * Yc                      # anti-correlated fluctuation
        self.assertLess(decompose(worse, Y)["fluctuation_r2"], 0.0)


class PublicDataTest(unittest.TestCase):
    data = ROOT / "external_data/pdebench/fixed_t1_v1"

    def setUp(self):
        if not (self.data / "pairs.npz").exists():
            self.skipTest("public PDEBench cache not present")

    def test_mean_is_conserved_and_fluctuation_energy_is_recorded(self):
        with np.load(self.data / "pairs.npz") as z:
            x, y = z["test_x"], z["test_y"]
        mx, my = x.mean(axis=1), y.mean(axis=1)
        self.assertGreater(np.corrcoef(mx, my)[0, 1], 0.9999)
        self.assertLess(np.abs(my - mx).mean(), 1e-3 * np.abs(my).mean())
        d = decompose(np.repeat(mx[:, None], y.shape[1], axis=1), y)
        self.assertAlmostEqual(d["fluctuation_r2"], 0.0)
        self.assertGreater(d["fluctuation_energy_fraction"], 0.0)
        self.assertLess(d["fluctuation_energy_fraction"], 1.0)


class ReconstructionTest(unittest.TestCase):
    """Recomputed predictions must reproduce the metric stored by the original run."""

    def test_two_runs_reproduce_stored_metrics(self):
        base = ROOT / "results/public_fixed_t1_v1"
        if not (base / "deeponet_mlp_s0" / "checkpoints" / "latest.pt").exists():
            self.skipTest("public family checkpoints not present")
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts/diagnose_fluctuation.py"),
                 "--families", "public_fixed_t1_v1", "--limit", "2",
                 "--device", "cpu", "--out", tmp],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            payload = json.loads((Path(tmp) / "public_fixed_t1_v1.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["runs"])
            self.assertEqual(payload["faithfulness_failures"], [])
            self.assertLess(payload["max_abs_metric_difference"], payload["tolerance"])
            for run in payload["runs"]:
                for split, score in run["splits"].items():
                    self.assertIn("stored_mean_relative_l2", score)
                    self.assertIn("fluctuation_r2", score)


if __name__ == "__main__":
    unittest.main()
