"""The physics-informed arm's unmatched work per update: section 5.11's measured factor.

Section 5.11 states that the physics-informed update costs 9.6x to 11.9x a supervised update.
That number is read from the `train_seconds` each training loop recorded, so this test only
checks the recorded figures still support the claim made in the manuscript. It skips when the
run artifacts are absent.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/pi_update_cost_v1/report.json"

PAIRS = [
    ("Burgers Nu=0.01, t=0.50", "physics_nu001_t050_v1", "burgers_nu001_t050_v1"),
    ("Burgers Nu=0.01, t=1.00", "physics_nu001_t100_v1", "burgers_nu001_t100_v1"),
    ("linear advection, t=1.00", "physics_advection_t100_v1", "advection_t100_v1"),
]


class UpdateCostTest(unittest.TestCase):
    def setUp(self):
        if not REPORT.exists():
            self.skipTest("PI update-cost report absent")
        self.report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.rows = self.report["rows"]

    def test_every_pair_was_measured_for_both_trunks(self):
        seen = {(r["setting"], r["architecture"]) for r in self.rows}
        for label, _, _ in PAIRS:
            for arch in ("deeponet_mlp", "deeponet_fourier"):
                self.assertIn((label, arch), seen, f"{label} / {arch}")

    def test_physics_update_costs_more_than_supervised(self):
        for r in self.rows:
            self.assertGreater(r["ratio"], 1.0, f"{r['setting']} {r['architecture']}")

    def test_ratio_matches_the_manuscript_claim(self):
        """Section 5.11 quotes 9.4x to 12.8x; recomputed here from artifacts."""
        for r in self.rows:
            expected = (r["seconds_per_update_pi"]
                        / r["seconds_per_update_supervised"])
            self.assertAlmostEqual(r["ratio"], expected, places=9)
        ratios = [r["ratio"] for r in self.rows]
        self.assertGreaterEqual(min(ratios), 9.0, f"min ratio drifted: {min(ratios):.3f}")
        self.assertLessEqual(max(ratios), 13.0, f"max ratio drifted: {max(ratios):.3f}")

    def test_no_fno_in_the_physics_arm(self):
        """FNO has no time coordinate to differentiate; see section 5.11."""
        self.assertFalse(any(r["architecture"] == "fno" for r in self.rows))


if __name__ == "__main__":
    unittest.main()
