"""Pin the frozen source hashes so the refactor cannot silently change a run identity."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path_inserted = False
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys_path_inserted = True

from src.source_hash import (  # noqa: E402
    FROZEN_FILE_LISTS, file_bytes, source_files, source_hash)


class FrozenValueTest(unittest.TestCase):
    """Values recorded by scripts/_source_hash_baseline.py before the lists were centralised."""

    def setUp(self):
        self.baseline = json.loads(
            (ROOT / "results" / "source_hash_baseline.json").read_text(encoding="utf-8"))

    def test_each_kind_reproduces_its_pre_refactor_value(self):
        for kind, expected in self.baseline.items():
            self.assertEqual(source_hash(kind), expected, kind)

    def test_no_kind_was_added_or_dropped(self):
        self.assertEqual(set(FROZEN_FILE_LISTS), set(self.baseline))


class ContractTest(unittest.TestCase):
    def test_unknown_kind_raises(self):
        """A typo must not produce a plausible-looking identity."""
        with self.assertRaises(KeyError):
            source_hash("public_benchmark_typo")

    def test_every_listed_file_exists(self):
        for kind in FROZEN_FILE_LISTS:
            for path in source_files(kind):
                self.assertTrue(path.is_file(), f"{kind}: missing {path}")

    def test_first_entry_is_the_entry_script(self):
        for kind, rels in FROZEN_FILE_LISTS.items():
            self.assertTrue(rels[0].startswith("scripts/"), kind)

    def test_hash_is_independent_of_checkout_location(self):
        """Only the file name is hashed, not the path, so moving the repo must not change it."""
        import hashlib
        manual = hashlib.sha256()
        for path in source_files("public_benchmark"):
            manual.update(path.name.encode())
            manual.update(file_bytes(path))
        self.assertEqual(manual.hexdigest(), source_hash("public_benchmark"))


if __name__ == "__main__":
    unittest.main()
