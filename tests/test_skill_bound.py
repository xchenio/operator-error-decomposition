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

    def test_physics_arm_never_produces_positive_fluctuation_skill(self):
        """The preregistered prediction was that physics-informed training would NOT give
        positive skill; if this ever fails, section 5.7's branch-bottleneck claim is
        contradicted and the manuscript must change rather than the test."""
        path = ROOT / "results/fluctuation_physics_v1/physics_nu001_t100_v1.json"
        if not path.exists():
            self.skipTest("physics artifacts absent")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["faithfulness_failures"], [])
        for run in payload["runs"]:
            self.assertLessEqual(run["splits"]["test"]["fluctuation_r2_per_sample"], 0.5)

    def test_physics_arm_at_the_short_horizon_falsified_the_strict_prediction(self):
        """The preregistered prediction was that physics-informed training would NOT reach
        positive skill at T=0.5. It did (+0.0088, +0.0516), so the test asserts the corrected
        record: positive but far below supervised, with the deficit in the mean share."""
        pi = ROOT / "results/fluctuation_physics_v1/t050/physics_nu001_t050_v1.json"
        sup = ROOT / "results/fluctuation_burgers_nu001_v1/t050/burgers_nu001_t050_v1.json"
        if not pi.exists() or not sup.exists():
            self.skipTest("t=0.5 artifacts absent")

        def by_arch(path):
            payload = json.loads(path.read_text(encoding="utf-8"))
            out = {}
            for run in payload["runs"]:
                out.setdefault(run["configuration"]["architecture"], []).append(
                    run["splits"]["test"])
            return {k: {m: float(np.mean([r[m] for r in v]))
                        for m in ("fluctuation_r2_per_sample", "squared_error_mean_share")}
                    for k, v in out.items()}

        p, s = by_arch(pi), by_arch(sup)
        for arch in ("deeponet_mlp", "deeponet_fourier"):
            self.assertGreater(p[arch]["fluctuation_r2_per_sample"], 0.0,
                               "the falsification record no longer applies")
            self.assertLess(p[arch]["fluctuation_r2_per_sample"],
                            s[arch]["fluctuation_r2_per_sample"] / 2.0)
            self.assertGreater(p[arch]["squared_error_mean_share"],
                               s[arch]["squared_error_mean_share"])

    def test_cfd_system_falsifies_the_positive_predictions(self):
        """The preregistered falsification criterion: on the three-field system the rule
        predicted positive skill for density (MLP) and for both trunks on Vx, and it was
        wrong. If a future refit makes these pass, this test records that the claim changed
        rather than the evidence."""
        predicted = {"density": {"deeponet_mlp": True, "deeponet_fourier": False},
                     "Vx": {"deeponet_mlp": True, "deeponet_fourier": True},
                     "pressure": {"deeponet_mlp": False, "deeponet_fourier": False}}
        wrong = 0
        checked = 0
        for comp, per_arch in predicted.items():
            path = ROOT / f"results/fluctuation_cfd_v1/{comp}/cfd_{comp}_v1.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["faithfulness_failures"], [])
            for arch, positive in per_arch.items():
                values = [r["splits"]["test"]["fluctuation_r2_per_sample"]
                          for r in payload["runs"]
                          if r["configuration"]["architecture"] == arch]
                if not values:
                    continue
                checked += 1
                wrong += int((float(np.mean(values)) > 0) != positive)
        self.assertEqual(checked, 6)
        self.assertEqual(wrong, 3)

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
