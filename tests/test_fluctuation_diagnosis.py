import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_fluctuation import decompose, parse_family  # noqa: E402


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

    def test_squared_error_split_is_an_exact_identity(self):
        """The constant vector and the zero-mean residual are orthogonal, so the per-sample
        squared relative error splits exactly. This is an identity, not an attribution."""
        rng = np.random.default_rng(3)
        for trial in range(5):
            Y = rng.normal(size=(12, 24)) * rng.uniform(0.5, 3.0) + rng.uniform(-2, 2)
            P = Y * rng.uniform(0.0, 1.5) + rng.normal(size=(12, 24)) * 0.4
            d = decompose(P, Y)
            self.assertAlmostEqual(
                d["squared_error_mean_share"] + d["squared_error_fluctuation_share"],
                d["relative_l2_rms"] ** 2, places=12)
            self.assertAlmostEqual(d["squared_error_closure"], 0.0, places=12)

    def test_per_sample_r2_closes_against_the_squared_fluctuation_share(self):
        """fluctuation share = (1 - per-sample R^2) * per-sample fluctuation energy.

        The global energy-weighted R^2 does NOT satisfy this identity, which is why both
        are reported: they weight samples differently.
        """
        rng = np.random.default_rng(4)
        Y = rng.normal(size=(24, 16)) * rng.uniform(0.2, 4.0, size=(24, 1)) + rng.uniform(-3, 3)
        P = Y * rng.uniform(0.0, 1.4, size=(24, 1)) + rng.normal(size=(24, 16)) * 0.5
        d = decompose(P, Y)
        self.assertAlmostEqual(
            d["squared_error_fluctuation_share"],
            (1.0 - d["fluctuation_r2_per_sample"]) * d["fluctuation_energy_fraction_per_sample"],
            places=12)

    def test_negative_r2_reported_not_clipped(self):
        rng = np.random.default_rng(2)
        Y = rng.normal(size=(16, 32))
        Yc = Y - Y.mean(axis=1, keepdims=True)
        worse = -2.0 * Yc                      # anti-correlated fluctuation
        self.assertLess(decompose(worse, Y)["fluctuation_r2"], 0.0)


class ParseFamilyTest(unittest.TestCase):
    def test_builtin_names_resolve(self):
        self.assertEqual(parse_family("public_fixed_t1_v1"),
                         ("public_fixed_t1_v1", "endtoend", ("test", "val")))
        self.assertEqual(parse_family("public_stability_v1"),
                         ("public_stability_v1", "branch", ("val",)))

    def test_explicit_spec_for_new_families(self):
        self.assertEqual(parse_family("public_fixed_t05_v1=endtoend:test,val"),
                         ("public_fixed_t05_v1", "endtoend", ("test", "val")))

    def test_unknown_input_rejected(self):
        with self.assertRaises(ValueError):
            parse_family("public_nope_v1")
        with self.assertRaises(ValueError):
            parse_family("x=magic:test")


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


class ManuscriptTableTest(unittest.TestCase):
    """The manuscript's fluctuation tables must equal the recomputed JSON, not hand copies."""

    main = ROOT / "paper/main.md"

    def setUp(self):
        if not self.main.exists():
            self.skipTest("manuscript not present")

    @staticmethod
    def _stats(rows, key):
        v = [r[key] for r in rows]
        return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)

    def _groups(self, path, split="test"):
        payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
        groups = {}
        for run in payload["runs"]:
            groups.setdefault(run["configuration"]["architecture"], []).append(run["splits"][split])
        return groups

    def test_section_5_2_fluctuation_table(self):
        text = self.main.read_text(encoding="utf-8")
        label = {"deeponet_mlp": "MLP DeepONet", "deeponet_fourier": "Fourier DeepONet", "fno": "FNO"}
        for key, rows in self._groups("results/fluctuation_v1/public_fixed_t1_v1.json").items():
            rel, _ = self._stats(rows, "mean_relative_l2")
            g, gs = self._stats(rows, "fluctuation_r2")
            p, ps = self._stats(rows, "fluctuation_r2_per_sample")
            ms, _ = self._stats(rows, "squared_error_mean_share")
            fs, _ = self._stats(rows, "squared_error_fluctuation_share")
            self.assertIn(f"| {label[key]} | {rel:.5f} | {g:+.4f} ± {gs:.4f} | "
                          f"**{p:+.4f} ± {ps:.4f}** | {ms:.4f} | {fs:.4f} |", text)

    def test_section_5_4_step_ladder_table(self):
        text = self.main.read_text(encoding="utf-8")
        label = {"deeponet_mlp": "MLP DeepONet", "deeponet_fourier": "Fourier DeepONet", "fno": "FNO"}
        paths = {1500: "results/fluctuation_v1/public_fixed_t1_v1.json",
                 5000: "results/fluctuation_steps_v1/public_steps5k_v1.json",
                 20000: "results/fluctuation_steps_v1/public_steps20k_v1.json",
                 50000: "results/fluctuation_steps_v1/public_steps50k_v1.json"}
        missing = [p for p in paths.values() if not (ROOT / p).exists()]
        if missing:
            self.skipTest(f"step ladder artifacts absent: {missing}")
        for budget, path in paths.items():
            for key, rows in self._groups(path).items():
                rel, _ = self._stats(rows, "mean_relative_l2")
                g, gs = self._stats(rows, "fluctuation_r2")
                p, ps = self._stats(rows, "fluctuation_r2_per_sample")
                self.assertIn(f"| {budget} | {label[key]} | {rel:.5f} | {g:+.4f} ± {gs:.4f} | "
                              f"**{p:+.4f} ± {ps:.4f}** |", text)


    def test_section_5_5_horizon_ladder_table(self):
        text = self.main.read_text(encoding="utf-8")
        label = {"deeponet_mlp": "MLP DeepONet", "deeponet_fourier": "Fourier DeepONet", "fno": "FNO"}
        paths = {10: "results/fluctuation_horizon_v1/t010/public_horizon_t010_v1.json",
                 25: "results/fluctuation_horizon_v1/t025/public_horizon_t025_v1.json",
                 50: "results/fluctuation_horizon_v1/t050/public_horizon_t050_v1.json",
                 75: "results/fluctuation_horizon_v1/t075/public_horizon_t075_v1.json",
                 100: "results/fluctuation_v1/public_fixed_t1_v1.json",
                 150: "results/fluctuation_horizon_v1/t150/public_horizon_t150_v1.json",
                 200: "results/fluctuation_horizon_v1/t200/public_horizon_t200_v1.json"}
        missing = [p for p in paths.values() if not (ROOT / p).exists()]
        if missing:
            self.skipTest(f"horizon artifacts absent: {missing}")
        for tag, path in paths.items():
            rows = {k: v for k, v in self._groups(path).items()}
            energy, _ = self._stats(rows["deeponet_mlp"], "fluctuation_energy_fraction")
            cells = []
            for key in ("deeponet_mlp", "deeponet_fourier", "fno"):
                p, ps = self._stats(rows[key], "fluctuation_r2_per_sample")
                cells.append(f"{p:+.4f} ± {ps:.4f}")
            self.assertIn(f"| {tag/100:.2f} | {energy:.4f} | " + " | ".join(cells) + " |", text)


    def test_section_5_6_advection_control_table(self):
        text = self.main.read_text(encoding="utf-8")
        paths = {10: "results/fluctuation_advection_v1/t010/advection_horizon_t010_v1.json",
                 25: "results/fluctuation_advection_v1/t025/advection_horizon_t025_v1.json",
                 50: "results/fluctuation_advection_v1/t050/advection_horizon_t050_v1.json",
                 75: "results/fluctuation_advection_v1/t075/advection_horizon_t075_v1.json",
                 100: "results/fluctuation_advection_v1/t100/advection_horizon_t100_v1.json",
                 150: "results/fluctuation_advection_v1/t150/advection_horizon_t150_v1.json",
                 200: "results/fluctuation_advection_v1/t200/advection_horizon_t200_v1.json"}
        missing = [p for p in paths.values() if not (ROOT / p).exists()]
        if missing:
            self.skipTest(f"advection artifacts absent: {missing}")
        for tag, path in paths.items():
            rows = self._groups(path)
            energy, _ = self._stats(rows["deeponet_mlp"], "fluctuation_energy_fraction")
            cells = []
            for key in ("deeponet_mlp", "deeponet_fourier", "fno"):
                p, ps = self._stats(rows[key], "fluctuation_r2_per_sample")
                cells.append(f"{p:+.4f} ± {ps:.4f}")
            self.assertIn(f"| {tag/100:.2f} | {energy:.4f} | " + " | ".join(cells) + " |", text)


    def test_section_5_6_2_diffusion_sorption_table(self):
        text = self.main.read_text(encoding="utf-8")
        paths = {125: "results/fluctuation_diffsorp_v1/t025/diffsorp_in020_t025_v1.json",
                 200: "results/fluctuation_diffsorp_v1/t040/diffsorp_in020_t040_v1.json",
                 275: "results/fluctuation_diffsorp_v1/t055/diffsorp_in020_t055_v1.json",
                 350: "results/fluctuation_diffsorp_v1/t070/diffsorp_in020_t070_v1.json",
                 425: "results/fluctuation_diffsorp_v1/t085/diffsorp_in020_t085_v1.json",
                 500: "results/fluctuation_diffsorp_v1/t100/diffsorp_in020_t100_v1.json"}
        missing = [p for p in paths.values() if not (ROOT / p).exists()]
        if missing:
            self.skipTest(f"diff-sorp artifacts absent: {missing}")
        for horizon, path in paths.items():
            rows = self._groups(path)
            energy, _ = self._stats(rows["deeponet_mlp"], "fluctuation_energy_fraction")
            cells = []
            for key in ("deeponet_mlp", "deeponet_fourier", "fno"):
                p, ps = self._stats(rows[key], "fluctuation_r2_per_sample")
                cells.append(f"{p:+.4f} ± {ps:.4f}")
            self.assertIn(f"| {horizon} | {energy:.4f} | " + " | ".join(cells) + " |", text)


if __name__ == "__main__":
    unittest.main()
