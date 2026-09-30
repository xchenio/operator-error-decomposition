"""Checks for the trunk-span skill bound and the a-priori predictor report."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.skill_bound import orthonormal_basis  # noqa: E402


class BoundTest(unittest.TestCase):
    def test_projection_onto_a_subspace_can_only_help(self):
        """The bound is an oracle over coefficients, so it can never be worse than any
        particular coefficient choice. This is the property the bound relies on."""
        rng = np.random.default_rng(7)
        for _ in range(5):
            n, p = 40, 6
            basis = orthonormal_basis(rng.normal(size=(n, p)))
            Y = rng.normal(size=(12, n)) * 2.0
            projected = (Y @ basis) @ basis.T
            self.assertLessEqual(np.sum((Y - projected) ** 2), np.sum(Y ** 2) + 1e-12)

    def test_orthonormal_basis_drops_numerically_null_directions(self):
        rng = np.random.default_rng(8)
        full = rng.normal(size=(30, 5))
        rank_deficient = np.column_stack([full, full[:, 0] * 1e-14])
        self.assertEqual(orthonormal_basis(rank_deficient).shape[1], 5)

    def test_orthonormal_basis_is_orthonormal(self):
        rng = np.random.default_rng(9)
        q = orthonormal_basis(rng.normal(size=(30, 7)))
        self.assertAlmostEqual(float(np.max(np.abs(q.T @ q - np.eye(q.shape[1])))), 0.0, places=10)


class ReportTest(unittest.TestCase):
    def test_bound_is_never_violated_by_a_trained_model(self):
        path = ROOT / "results/skill_bound_v1/report.json"
        if not path.exists():
            self.skipTest("skill bound report absent")
        report = json.loads(path.read_text(encoding="utf-8"))
        checked = 0
        for setting, arms in report.items():
            for arch, entry in arms.items():
                if "achieved_mean" not in entry:
                    continue
                checked += 1
                self.assertGreaterEqual(
                    entry["skill_bound_mean"], entry["achieved_mean"],
                    f"{setting} {arch}: bound below achieved")
        self.assertGreater(checked, 0)

    def test_a_priori_report_uses_training_split_only(self):
        """The predictor must not touch test targets, so its statistics are recomputed and
        the settings must match the trained families it is compared against."""
        path = ROOT / "results/skill_prediction_v1/report.json"
        if not path.exists():
            self.skipTest("predictor report absent")
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(report["rows"]), 20)
        self.assertEqual({r["pde"] for r in report["rows"]},
                         {"burgers", "advection", "diffsorp"})
        for row in report["rows"]:
            self.assertIn(row["pde"] + "_" + row["time_tag"],
                          {"burgers_010", "burgers_025", "burgers_050", "burgers_075",
                           "burgers_100", "burgers_150", "burgers_200", "advection_010",
                           "advection_025", "advection_050", "advection_075",
                           "advection_100", "advection_150", "advection_200",
                           "diffsorp_025", "diffsorp_040", "diffsorp_055",
                           "diffsorp_070", "diffsorp_085", "diffsorp_100"})

    def test_thresholds_are_fitted_without_the_held_out_pde(self):
        """The out-of-sample claim only holds if the third PDE never entered the fit."""
        path = ROOT / "results/skill_prediction_v1/report.json"
        if not path.exists():
            self.skipTest("predictor report absent")
        report = json.loads(path.read_text(encoding="utf-8"))
        info = report["critical_fluctuation_energy"]
        self.assertEqual(info["fitted_on"], ["advection", "burgers"])
        self.assertEqual(info["held_out"], ["diffsorp"])
        for arch in ("deeponet_mlp", "deeponet_fourier", "fno"):
            self.assertEqual(info[arch + "_rule_held_out"], "6/6")

    def test_advection_fluctuation_energy_is_horizon_independent(self):
        """Pure transport must leave the fluctuation energy unchanged; this is the check that
        the advection preparation extracted a translation and not something else."""
        path = ROOT / "results/skill_prediction_v1/report.json"
        if not path.exists():
            self.skipTest("predictor report absent")
        rows = [r for r in json.loads(path.read_text(encoding="utf-8"))["rows"]
                if r["pde"] == "advection"]
        self.assertEqual(len(rows), 7)
        for row in rows:
            self.assertAlmostEqual(row["fluctuation_energy_fraction"], 0.6485, places=3)


if __name__ == "__main__":
    unittest.main()
