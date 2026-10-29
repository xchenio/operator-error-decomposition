"""Check the generated LaTeX against the manuscript for lost or broken content.

A converter can silently drop or mangle a paragraph, and the result still
compiles, so compiling is not evidence that the conversion is faithful. This
compares structural counts between the two documents and reports anything that
went missing.

Writes results/tex_check.txt because the Windows console cannot print some of
the characters involved.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper/arxiv/main.tex"
MD = ROOT / "paper/main.md"
OUT = ROOT / "results/tex_check.txt"


def main() -> None:
    if not TEX.exists():
        print("no generated tex; run scripts/md_to_tex.py first")
        return

    tex = TEX.read_text(encoding="utf-8")
    md = MD.read_text(encoding="utf-8")

    lines = []
    fail = 0

    # Editorial notes that must not appear in the submission.
    for needle in ("Evidence-audited research draft", "Not a submitted manuscript",
                   "Retitled 2026-10-28"):
        if needle in tex:
            lines.append(f"FAIL  editorial note survived: {needle}")
            fail += 1

    # Structural counts: the LaTeX must carry every section of the manuscript.
    # "Abstract" and "References" become environments rather than \section, so
    # they are expected to be absent from the section count.
    md_sections = [s for s in re.findall(r"^##\s+(.+)$", md, flags=re.M)
                   if not s.startswith(("Abstract", "References"))]
    md_subs = re.findall(r"^###\s+(.+)$", md, flags=re.M)
    tex_sections = re.findall(r"\\section\{", tex)
    tex_subs = re.findall(r"\\subsection\{", tex)
    lines.append(f"sections: manuscript {len(md_sections)}, tex {len(tex_sections)}")
    lines.append(f"subsections: manuscript {len(md_subs)}, tex {len(tex_subs)}")
    if len(tex_sections) != len(md_sections):
        lines.append(f"FAIL  section count differs by "
                     f"{len(md_sections) - len(tex_sections)}")
        fail += 1
    if len(tex_subs) != len(md_subs):
        lines.append(f"FAIL  subsection count differs by "
                     f"{len(md_subs) - len(tex_subs)}")
        fail += 1

    # Tables.
    md_tables = len([m for m in re.finditer(r"^\|\s*---", md, flags=re.M)])
    tex_tables = len(re.findall(r"\\begin\{tabular\}", tex))
    lines.append(f"tables: manuscript {md_tables}, tex {tex_tables}")
    if md_tables != tex_tables:
        lines.append(f"FAIL  table count differs")
        fail += 1

    # Reference entries.
    md_refs = len(re.findall(r"^\d+\.\s", md[md.find("\n## References"):], flags=re.M))
    tex_refs = len(re.findall(r"\\bibitem\{", tex))
    lines.append(f"references: manuscript {md_refs}, tex {tex_refs}")
    if md_refs != tex_refs:
        lines.append("FAIL  reference count differs")
        fail += 1

    # Front matter.
    for needle, label in ((r"\begin{abstract}", "abstract opened"),
                          (r"\end{abstract}", "abstract closed"),
                          (r"\end{frontmatter}", "frontmatter closed"),
                          (r"\includegraphics", "figure"),
                          (r"\title{", "title"),
                          (r"\author{", "author")):
        ok = needle in tex
        lines.append(f"{'ok  ' if ok else 'FAIL'} {label}")
        if not ok:
            fail += 1

    # The thesis disclosure is a real section and must be present.
    if "Relation to the author" not in tex:
        lines.append("FAIL  thesis disclosure section missing")
        fail += 1
    else:
        lines.append("ok   thesis disclosure present")

    # Leftover Unicode that pdflatex cannot typeset.
    bad = sorted({ch for ch in tex if ord(ch) > 127})
    lines.append("")
    lines.append("non-ascii remaining: " + (", ".join(f"{c!r} U+{ord(c):04X}" for c in bad)
                                            or "none"))

    # Unbalanced environments would break the build.
    for env in ("abstract", "frontmatter", "document", "thebibliography"):
        o = len(re.findall(r"\\begin\{%s\}" % env, tex))
        c = len(re.findall(r"\\end\{%s\}" % env, tex))
        if o != c:
            lines.append(f"FAIL  {env}: {o} begin vs {c} end")
            fail += 1

    lines.append("")
    lines.append(f"{fail} failure(s)" if fail else "all checks passed")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
