"""Structural checks on the manuscript that a section-level read-through misses.

Each of these defects is invisible when reading any single section, because the
statement is locally correct; it only becomes wrong once the document changes
around it.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper/main.md"
METADATA = ROOT / "paper/arxiv_metadata.md"

# JCP Guide for Authors: the abstract "does not exceed 250 words".
ABSTRACT_CAP = 250
# Elsevier: 3 to 5 highlights of at most 85 characters each, spaces included.
HIGHLIGHT_MIN, HIGHLIGHT_MAX = 3, 5
HIGHLIGHT_CAP = 85


def _abstract(text):
    """The abstract of the manuscript, delimited by the following section."""
    start = text.find("## Abstract")
    end = text.find("## 1. Research question and contribution")
    assert 0 <= start < end, "abstract block not found"
    return text[start:end]


def _metadata_abstract(text):
    """Same text, but the metadata file's next heading is "## Keywords"."""
    return text.split("## Abstract", 1)[1].split("## Keywords", 1)[0]


def _cited_numbers(text, limit=30):
    """Reference numbers actually cited, tolerating grouped form "[13, 14]"."""
    found = set()
    for grp in re.findall(r"\[([\d.,\s]+)\]", text):
        if "." in grp:  # a decimal such as Nu = 0.001, not a citation
            continue
        for part in re.split(r"[,\s]+", grp.strip()):
            if part.isdigit() and 1 <= int(part) <= limit:
                found.add(int(part))
    return found


class AbstractLengthTest(unittest.TestCase):
    def setUp(self):
        if not MANUSCRIPT.exists():
            self.skipTest("manuscript not present")
        self.text = MANUSCRIPT.read_text(encoding="utf-8")

    def test_abstract_respects_the_journal_cap(self):
        """Bracketed [n] markers are citations, not prose; exclude them from the count."""
        prose = re.sub(r"\[[0-9,\s]+\]", " ", _abstract(self.text))
        n = len(re.findall(r"[A-Za-z0-9]+", prose))
        self.assertLessEqual(n, ABSTRACT_CAP,
                             f"abstract is {n} words, over JCP's {ABSTRACT_CAP}-word cap")


class ReferenceListTest(unittest.TestCase):
    def setUp(self):
        if not MANUSCRIPT.exists():
            self.skipTest("manuscript not present")
        self.text = MANUSCRIPT.read_text(encoding="utf-8")

    def _reference_block(self):
        start = self.text.find("\n## References")
        if start < 0:
            self.fail("no '## References' section")
        block = self.text[start:]
        stop = block.find("\n## ", 1)
        return block[:stop] if stop > 0 else block

    def test_numbering_is_contiguous_from_one(self):
        nums = [int(m.group(1))
                for m in re.finditer(r"^(\d+)\.\s", self._reference_block(), flags=re.M)]
        self.assertEqual(nums, list(range(1, len(nums) + 1)), "reference numbers are not 1..N")

    def test_every_entry_is_cited(self):
        nums = {int(m.group(1))
                for m in re.finditer(r"^(\d+)\.\s", self._reference_block(), flags=re.M)}
        cited = _cited_numbers(self.text, limit=max(nums) if nums else 30)
        self.assertEqual(sorted(nums - cited), [],
                         "reference-list entries never cited in the text")

    def test_no_citation_points_beyond_the_list(self):
        nums = {int(m.group(1))
                for m in re.finditer(r"^(\d+)\.\s", self._reference_block(), flags=re.M)}
        cited = _cited_numbers(self.text, limit=max(nums) if nums else 30)
        self.assertEqual(sorted(cited - nums), [], "citation to a non-existent entry")


class CrossReferenceTest(unittest.TestCase):
    def setUp(self):
        if not MANUSCRIPT.exists():
            self.skipTest("manuscript not present")
        self.text = MANUSCRIPT.read_text(encoding="utf-8")

    def test_every_section_pointer_resolves(self):
        headings = set()
        for m in re.finditer(r"^#{2,4}\s+(.+?)\s*$", self.text, flags=re.M):
            num = re.match(r"(\d+(?:\.\d+)*)[\.\s]", m.group(1))
            if num:
                headings.add(num.group(1))
        dangling = sorted({m.group(1) for m in
                           re.finditer(r"section\s+(\d+\.\d+)", self.text)} - headings)
        self.assertEqual(dangling, [], "cross-reference to a heading that does not exist")


class FigureFileTest(unittest.TestCase):
    def setUp(self):
        if not MANUSCRIPT.exists():
            self.skipTest("manuscript not present")
        self.text = MANUSCRIPT.read_text(encoding="utf-8")

    def test_linked_images_exist(self):
        missing = []
        for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", self.text):
            target = m.group(1)
            if target.startswith(("http://", "https://")):
                continue
            if not (MANUSCRIPT.parent / target).exists():
                missing.append(target)
        self.assertEqual(missing, [], "manuscript links to a missing image")


class MetadataTest(unittest.TestCase):
    """The arXiv metadata is written to the same caps, so the two cannot drift."""

    def setUp(self):
        if not METADATA.exists():
            self.skipTest("metadata file not present")
        self.meta = METADATA.read_text(encoding="utf-8")

    def test_primary_and_cross_list_are_declared(self):
        self.assertIn("cs.LG", self.meta)
        self.assertIn("math.NA", self.meta)

    def test_highlights_respect_the_character_cap(self):
        block = self.meta.split("## Highlights", 1)[-1].split("\n## ", 1)[0]
        bullets = [ln[2:].strip() for ln in block.splitlines()
                   if ln.strip().startswith("- ")]
        self.assertTrue(HIGHLIGHT_MIN <= len(bullets) <= HIGHLIGHT_MAX,
                        f"expected 3-5 highlights, found {len(bullets)}")
        over = [b for b in bullets if len(b) > HIGHLIGHT_CAP]
        self.assertEqual(over, [], "highlight exceeds the 85-character cap")

    def test_metadata_abstract_matches_the_manuscript(self):
        """Both documents carry the same abstract, so quoting one cannot contradict the other."""
        if not MANUSCRIPT.exists():
            self.skipTest("manuscript not present")
        main = MANUSCRIPT.read_text(encoding="utf-8")
        a = _metadata_abstract(self.meta).strip()
        b = _abstract(main).split("## Abstract", 1)[-1].strip()
        self.assertEqual(a, b, "arXiv metadata abstract differs from the manuscript")


if __name__ == "__main__":
    unittest.main()
