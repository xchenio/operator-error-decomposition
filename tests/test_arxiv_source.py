"""Guard the Markdown-to-LaTeX conversion.

A converter can drop a paragraph and still produce a file that compiles, so
compiling is not evidence of a faithful conversion. These tests compare the
generated source against the manuscript structurally: every section,
subsection, table and reference must survive, and the editorial notes that
belong to the repository must not.

They also pin the two things that break the build on arXiv, which uses
pdflatex: characters with no pdflatex glyph, and environments left unbalanced.
"""
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper/arxiv/main.tex"
MD = ROOT / "paper/main.md"


def _regenerate() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/md_to_tex.py")],
                   check=True, capture_output=True)


class ConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MD.exists():
            raise unittest.SkipTest("manuscript not present")
        _regenerate()
        cls.tex = TEX.read_text(encoding="utf-8")
        cls.md = MD.read_text(encoding="utf-8")

    def test_every_section_survives(self):
        md = [s for s in re.findall(r"^##\s+(.+)$", self.md, flags=re.M)
              if not s.startswith(("Abstract", "References"))]
        tex = re.findall(r"\\section\{", self.tex)
        self.assertEqual(len(tex), len(md), "a section was lost in conversion")

    def test_every_subsection_survives(self):
        md = re.findall(r"^###\s+(.+)$", self.md, flags=re.M)
        tex = re.findall(r"\\subsection\{", self.tex)
        self.assertEqual(len(tex), len(md), "a subsection was lost in conversion")

    def test_every_table_survives(self):
        md = len(re.findall(r"^\|\s*---", self.md, flags=re.M))
        tex = len(re.findall(r"\\begin\{tabular\}", self.tex))
        self.assertEqual(tex, md, "a table was lost in conversion")

    def test_every_reference_survives(self):
        tail = self.md[self.md.find("\n## References"):]
        md = len(re.findall(r"^\d+\.\s", tail, flags=re.M))
        tex = len(re.findall(r"\\bibitem\{", self.tex))
        self.assertEqual(tex, md, "a reference was lost in conversion")

    def test_editorial_notes_are_dropped(self):
        for needle in ("Evidence-audited research draft",
                       "Not a submitted manuscript",
                       "Retitled 2026-10-28"):
            self.assertNotIn(needle, self.tex,
                             f"repository note leaked into the submission: {needle}")

    def test_thesis_disclosure_is_kept(self):
        """The disclosure is a submission section, not repository metadata."""
        self.assertIn("Relation to the author", self.tex)

    def test_no_character_pdflatex_cannot_typeset(self):
        bad = sorted({ch for ch in self.tex if ord(ch) > 127})
        self.assertEqual(bad, [], f"non-ascii left in source: {bad}")

    def test_environments_are_balanced(self):
        for env in ("abstract", "frontmatter", "document", "thebibliography"):
            opens = len(re.findall(r"\\begin\{%s\}" % env, self.tex))
            closes = len(re.findall(r"\\end\{%s\}" % env, self.tex))
            self.assertEqual(opens, closes, f"unbalanced {env}")

    def test_abstract_is_closed(self):
        self.assertIn(r"\begin{abstract}", self.tex)
        self.assertIn(r"\end{abstract}", self.tex)


if __name__ == "__main__":
    unittest.main()
