"""The second-split machinery: disjointness, determinism, and above all no change to split 1."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.split_utils import draw_split, split_fields  # noqa: E402
from src.public_benchmark import split_indices  # noqa: E402

TOTAL = 10000
SIZES = (1000, 200, 200)
SEED_1 = 20260922
SEED_2 = 20261028


class DefaultPathUnchangedTest(unittest.TestCase):
    """If this breaks, every committed cache silently changed and every run identity moved."""

    def test_no_exclusion_reproduces_the_original_helper(self):
        ids = draw_split(TOTAL, SIZES, SEED_1)
        reference = split_indices(TOTAL, SIZES, SEED_1)
        for name in ("train", "val", "test"):
            self.assertEqual(list(ids[name]), list(reference[name]), name)

    def test_recorded_caches_still_match_what_the_code_produces(self):
        """Existing prepared splits must still be exactly what draw_split would write."""
        candidates = [ROOT / "external_data/pdebench/fixed_t1_v1",
                      ROOT / "external_data/pdebench/fixed_t100_v1",
                      ROOT / "external_data/pdebench/burgers_nu001/fixed_t100_v1"]
        found = [p for p in candidates if (p / "protocol.json").exists()]
        if not found:
            self.skipTest("no prepared cache present (external_data is git-ignored)")
        reference = split_indices(TOTAL, SIZES, SEED_1)
        for folder in found:
            info = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
            recorded = {k: [int(i) for i in v] for k, v in info["sample_ids"].items()}
            for name in ("train", "val", "test"):
                self.assertEqual(recorded[name], list(reference[name]), f"{folder.name}/{name}")

    def test_protocol_fields_record_how_the_split_was_drawn(self):
        ids = draw_split(TOTAL, SIZES, SEED_2, exclude_seed=SEED_1)
        fields = split_fields(ids, SEED_2, SIZES, SEED_1)
        self.assertEqual(fields["split_seed"], SEED_2)
        self.assertEqual(fields["excluded_seed"], SEED_1)
        self.assertTrue(fields["disjoint_from_split1"])
        self.assertEqual(fields["split_sizes"], list(SIZES))
        # The two entries the old format wrote must still be present, since downstream
        # scripts read them.
        self.assertIn("sample_ids", fields)
        self.assertEqual(fields["split_seed"], SEED_2)


class DisjointnessTest(unittest.TestCase):
    def setUp(self):
        self.a = draw_split(TOTAL, SIZES, SEED_1)
        self.b = draw_split(TOTAL, SIZES, SEED_2, exclude_seed=SEED_1)

    def test_no_shared_trajectory_at_all(self):
        a = {int(i) for split in self.a.values() for i in split}
        b = {int(i) for split in self.b.values() for i in split}
        self.assertEqual(a & b, set())

    def test_no_shared_trajectory_within_each_split(self):
        for name in ("train", "val", "test"):
            a = {int(i) for i in self.a[name]}
            b = {int(i) for i in self.b[name]}
            self.assertEqual(a & b, set(), name)

    def test_a_bare_second_seed_would_not_have_been_enough(self):
        """Documents why the exclusion is needed: ~196 shared ids otherwise."""
        naive = draw_split(TOTAL, SIZES, SEED_2)
        a = {int(i) for split in self.a.values() for i in split}
        shared = a & {int(i) for split in naive.values() for i in split}
        self.assertGreater(len(shared), 50)

    def test_split_2_sizes_are_complete(self):
        for name in ("train", "val", "test"):
            self.assertEqual(len(self.b[name]), dict(zip(("train", "val", "test"),
                                                         SIZES))[name])

    def test_split_2_ids_are_unique_and_in_range(self):
        flat = [int(i) for split in self.b.values() for i in split]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertGreaterEqual(min(flat), 0)
        self.assertLess(max(flat), TOTAL)


class DeterminismTest(unittest.TestCase):
    def test_repeated_calls_agree(self):
        first = draw_split(TOTAL, SIZES, SEED_2, exclude_seed=SEED_1)
        second = draw_split(TOTAL, SIZES, SEED_2, exclude_seed=SEED_1)
        for name in ("train", "val", "test"):
            self.assertEqual(list(first[name]), list(second[name]), name)

    def test_different_seeds_give_different_splits(self):
        a = draw_split(TOTAL, SIZES, SEED_2, exclude_seed=SEED_1)
        b = draw_split(TOTAL, SIZES, SEED_2 + 1, exclude_seed=SEED_1)
        self.assertNotEqual(list(a["test"]), list(b["test"]))


class GuardTest(unittest.TestCase):
    def test_invalid_sizes_rejected(self):
        with self.assertRaises(ValueError):
            draw_split(TOTAL, (0, 200, 200))
        with self.assertRaises(ValueError):
            draw_split(100, SIZES)

    def test_exclusion_that_leaves_too_few_trajectories_is_rejected(self):
        """Rather than silently producing a smaller split."""
        with self.assertRaises(ValueError):
            draw_split(1500, SIZES, SEED_2, exclude_seed=SEED_1)


if __name__ == "__main__":
    unittest.main()
