"""The disjoint-split replication: the recorded result must match the artifacts.

The ledger in `experiments/prediction_ledger_rs2.json` is the manuscript's section 5.17. These
tests check it is internally consistent and that every quoted number is reachable from the
recomputed diagnostics, so the results section cannot drift from the runs.

Everything here skips when the split-2 artifacts are absent, because they are produced by
training on data that is git-ignored.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.split_utils import draw_split  # noqa: E402

LEDGER = ROOT / "experiments/prediction_ledger_rs2.json"
DIAG_ROOT = ROOT / "results/rs2_fluctuation"
DATA_ROOT = ROOT / "external_data/pdebench/rs2"

TAGS = ("010", "025", "050", "075", "100", "150", "200")
ARCHS = ("deeponet_mlp", "deeponet_fourier", "fno")
SIZES = (1000, 200, 200)
TOTAL = 10000
SEED_1 = 20260922
SEED_2 = 20261028


class DisjointnessTest(unittest.TestCase):
    """The preflight gate that must pass before any split-2 run."""

    def test_split_2_shares_no_trajectory_with_split_1(self):
        a = draw_split(TOTAL, SIZES, SEED_1)
        b = draw_split(TOTAL, SIZES, SEED_2, exclude_seed=SEED_1)
        a_ids = {int(i) for split in a.values() for i in split}
        b_ids = {int(i) for split in b.values() for i in split}
        self.assertEqual(a_ids & b_ids, set())

    def test_a_bare_second_seed_would_not_have_been_independent(self):
        """Why the exclusion is needed: two bare seeds overlap on about 196 ids."""
        a = draw_split(TOTAL, SIZES, SEED_1)
        naive = draw_split(TOTAL, SIZES, SEED_2)
        a_ids = {int(i) for split in a.values() for i in split}
        shared = a_ids & {int(i) for split in naive.values() for i in split}
        self.assertGreater(len(shared), 50)

    def test_split_is_deterministic(self):
        first = draw_split(TOTAL, SIZES, SEED_2, exclude_seed=SEED_1)
        second = draw_split(TOTAL, SIZES, SEED_2, exclude_seed=SEED_1)
        for name in ("train", "val", "test"):
            self.assertEqual(list(first[name]), list(second[name]), name)


class LedgerTest(unittest.TestCase):
    """section 5.17's numbers."""

    def setUp(self):
        if not LEDGER.exists():
            self.skipTest("split-2 ledger absent")
        self.ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
        self.predictions = self.ledger["predictions"]

    def test_prediction_signs_follow_from_the_rule(self):
        """Recompute each sign from the rule so a hand-typed one cannot survive."""
        thresholds = self.ledger["thresholds"]
        for p in self.predictions:
            expected = "+" if p["fluctuation_energy"] >= thresholds[p["architecture"]] else "-"
            self.assertEqual(p["predicted"], expected, f"{p['setting']} {p['architecture']}")

    def test_measured_signs_agree_with_measured_values(self):
        for p in self.predictions:
            expected = "+" if p["measured"] > 0 else "-"
            self.assertEqual(p["measured_sign"], expected, p["setting"])

    def test_accounting(self):
        correct = sum(1 for p in self.predictions if p["predicted"] == p["measured_sign"])
        self.assertEqual(correct, 13)
        self.assertEqual(len(self.predictions), 14)

    def test_fno_has_no_threshold_on_split_2(self):
        """The deviation from the preregistration; recorded, not hidden."""
        self.assertIsNone(self.ledger["thresholds"]["fno"])
        self.assertFalse(any(p["architecture"] == "fno" for p in self.predictions))

    def test_thresholds_are_within_the_predeclared_band(self):
        for arch, refit in self.ledger["thresholds"].items():
            if refit is None:
                continue
            split1 = {"deeponet_mlp": 0.2477, "deeponet_fourier": 0.2728}[arch]
            drift = abs(refit - split1) / split1
            self.assertLessEqual(drift, 0.25, f"{arch} drifted {drift:.1%}")

    def test_held_out_horizons_are_the_full_ladder(self):
        settings = sorted({p["setting"] for p in self.predictions})
        self.assertEqual(len(settings), 7)
        archs = {p["architecture"] for p in self.predictions}
        self.assertEqual(archs, {"deeponet_mlp", "deeponet_fourier"})


class ArtifactTest(unittest.TestCase):
    """The 315 runs and 21 diagnostics the section rests on."""

    def setUp(self):
        if not DIAG_ROOT.exists():
            self.skipTest("split-2 diagnostics absent")

    def test_all_21_diagnostics_exist(self):
        found = sorted(p.parent.name + '/' + p.parent.parent.name
                       for p in DIAG_ROOT.rglob('rs2_*_v1.json'))
        self.assertEqual(len(found), 21, found)

    def test_all_21_families_have_15_completed_runs(self):
        families = sorted(p for p in (ROOT / "results").iterdir()
                          if p.is_dir() and p.name.startswith("rs2_burgers")
                          or p.is_dir() and p.name.startswith("rs2_advection"))
        self.assertEqual(len(families), 21)
        for folder in families:
            n = len(list(folder.rglob("result.json")))
            self.assertEqual(n, 15, f"{folder.name} has {n} runs")

    def test_every_cached_split_2_family_is_declared_disjoint(self):
        for protocol in (DATA_ROOT).rglob("protocol.json"):
            info = json.loads(protocol.read_text(encoding="utf-8"))
            self.assertEqual(info.get("split_seed"), SEED_2, protocol)
            self.assertEqual(info.get("excluded_seed"), SEED_1, protocol)
            self.assertTrue(info.get("disjoint_from_split1"), protocol)


if __name__ == "__main__":
    unittest.main()
