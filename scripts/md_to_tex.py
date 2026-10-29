"""Convert the manuscript from Markdown to an elsarticle LaTeX source.

One source serves both purposes: the same file can be uploaded to arXiv and,
later, submitted to the journal, which asks for elsarticle anyway.

Two things make this more than a mechanical replacement:

1. LaTeX special characters. The text is full of "97.4%" and
   "fluctuation_r2_per_sample", and a bare % or _ is a compile error. Math
   blocks and inline code are therefore pulled out first and restored after
   escaping, so that escaping never touches a formula.
2. Citations are written literally in the manuscript, "[10]", rather than as
   \\cite{}, so the reference list is emitted as a thebibliography of numbered
   \\bibitem entries, which numbers itself and stays consistent with the text.

Usage: python scripts/md_to_tex.py [-o paper/arxiv/main.tex]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "paper/main.md"
DEFAULT_OUT = ROOT / "paper/arxiv/main.tex"

# Sections that are editorial metadata for the repository, not part of the paper.
# The thesis disclosure is deliberately not here: it is a real submission section.
SKIP_HEADINGS: set[str] = set()
# Paragraph markers that open with these strings are dropped wholesale.
SKIP_PREFIXES = (
    "**Evidence-audited research draft",
    "*Retitled 2026-10-28",
)


# pdflatex, which arXiv uses by default, has no glyphs for several characters
# the manuscript uses. Replacing them here keeps the source compilable there
# rather than only locally under xelatex.
UNICODE_MAP = {
    "\u00b1": r"$\pm$",        # PLUS-MINUS SIGN
    "\u2014": "---",           # EM DASH
    "\u2013": "--",            # EN DASH
    "\u220e": r"$\blacksquare$",  # END OF PROOF
    "\u00b2": r"$^2$",         # SUPERSCRIPT TWO
    "\u2192": r"$\to$",        # RIGHTWARDS ARROW
    "\u2248": r"$\approx$",    # ALMOST EQUAL TO
    "\u2019": "'",
    "\u2018": "`",
    "\u201c": "``",
    "\u201d": "''",
    "\u00fc": r'\"u',         # u WITH DIAERESIS, in a cited author name
}


def deunicode(text: str) -> str:
    for src, dst in UNICODE_MAP.items():
        text = text.replace(src, dst)
    return text


def looks_math(s: str) -> bool:
    """Decide whether a backticked span is inline math or a code identifier.

    The manuscript uses backticks for both, so the distinction has to be read
    off the content: a LaTeX command, a superscript or a brace means math, and
    a file path or a multi-letter snake_case token means code.
    """
    if "\\" in s:
        return True
    if "/" in s:
        return False
    if "^" in s or "{" in s or "}" in s:
        return True
    if "=" in s and re.search(r"[A-Za-z]", s):
        return True
    if "_" in s:
        # Math subscripts are short ("y_j"); identifiers are words.
        return all(len(part) <= 3 for part in s.split("_"))
    return False


def protect(text: str):
    """Pull math and inline code out, returning (text, slots)."""
    slots: list[str] = []

    def stash(rendered: str) -> str:
        slots.append(rendered)
        return f"\x00{len(slots) - 1}\x00"

    # Display math: \[ ... \]
    def disp(m):
        return stash("\\[" + m.group(1) + "\\]")

    text = re.sub(r"\\\[(.*?)\\\]", disp, text, flags=re.S)

    # Inline code: `x`, which in this manuscript may equally be inline math.
    def code(m):
        body = m.group(1)
        if looks_math(body):
            return stash("$" + body + "$")
        return stash(r"\texttt{" + body.replace("_", r"\_").replace("^", r"\^{}") + "}")

    text = re.sub(r"`([^`]+)`", code, text)

    # Inline math already written as $ ... $
    def inl(m):
        return stash("$" + m.group(1) + "$")

    text = re.sub(r"(?<!\\)\$([^$]+)\$", inl, text)

    # Superscripts written literally in the prose, "R^2": a bare ^ is invalid in
    # text mode. Stashed so that escaping leaves the dollars alone. Math blocks
    # are already stashed by now, so this only touches prose.
    def sup(m):
        return stash("$" + m.group(0) + "$")

    text = re.sub(r"R\^2", sup, text)
    return text, slots


def escape(text: str) -> str:
    """Escape characters that are special in LaTeX text mode."""
    out = []
    for ch in text:
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        else:
            out.append(ch)
    return "".join(out)


def inline(text: str, slots: list[str]) -> str:
    """Apply inline Markdown markup to one paragraph's text."""
    text = escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\\emph{\1}", text)
    # Markdown links: keep the label, drop repository-relative targets.
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: (r"\href{%s}{%s}" % (m.group(2), m.group(1))
                   if m.group(2).startswith("http") else m.group(1)),
        text,
    )
    return restore(text, slots)


def restore(text: str, slots: list[str]) -> str:
    def back(m):
        return slots[int(m.group(1))]

    return re.sub("\x00(\\d+)\x00", back, text)


def to_tabular(rows: list[str], slots: list[str]) -> str:
    """Convert a Markdown table block to a booktabs tabular."""
    cells = []
    for r in rows:
        if re.match(r"^\|[\s:|-]+\|$", r.strip()):
            continue
        parts = [c.strip() for c in r.strip().strip("|").split("|")]
        cells.append(parts)
    if not cells:
        return ""
    ncols = max(len(c) for c in cells)
    # A right-aligned column in the separator row becomes an r column.
    sep = next((r for r in rows if re.match(r"^\|[\s:|-]+\|$", r.strip())), None)
    aligns = []
    if sep:
        for c in sep.strip().strip("|").split("|"):
            c = c.strip()
            aligns.append("r" if c.endswith("-:") else
                          ("c" if c.startswith(":-") and c.endswith("-") else "l"))
    aligns += ["l"] * (ncols - len(aligns))
    spec = "".join(aligns[:ncols])

    out = [r"\begin{center}",
           r"\begin{tabular}{%s}" % spec, r"\toprule"]
    head, *body = cells
    out.append(" & ".join(inline(c, slots) for c in head) + r" \\")
    out.append(r"\midrule")
    for row in body:
        padded = row + [""] * (ncols - len(row))
        out.append(" & ".join(inline(c, slots) for c in padded) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(out)


def convert(md: str) -> str:
    lines = md.splitlines()

    # Drop the editorial status notes at the top.
    while lines and (not lines[0].strip() or
                     lines[0].lstrip().startswith(SKIP_PREFIXES)):
        lines.pop(0)

    md = "\n".join(lines)
    md, slots = protect(md)
    lines = md.splitlines()

    body: list[str] = []
    i = 0
    title_done = False
    orcid = ""
    abstract_open = False
    keywords_open = False
    in_refs = False
    refs: list[str] = []
    ref_buf: list[str] = []

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        # --- repository editorial notes, not part of the paper ---
        if raw.strip().startswith(SKIP_PREFIXES):
            while i < len(lines) and lines[i].strip():
                i += 1
            continue

        # --- title ---
        if line.startswith("# ") and not title_done:
            body.append(r"\title{" + inline(line[2:].strip(), slots) + "}")
            title_done = True
            i += 1
            continue

        # --- author block: bold name, then ORCID, then affiliation ---
        if raw.strip().startswith("**Xiaochen Yan**"):
            body.append(r"\author{Xiaochen Yan}")
            i += 1
            continue
        # elsarticle has no \note; the ORCID goes into the address line.
        if raw.strip().startswith("ORCID:"):
            orcid = raw.strip()
            i += 1
            continue
        if raw.strip().startswith("Independent Researcher"):
            addr = raw.strip().split("·")[0].strip().rstrip(".")
            body.append(r"\address{%s. %s}" % (addr, orcid) if orcid
                        else r"\address{%s}" % addr)
            i += 1
            continue

        # --- headings ---
        m = re.match(r"^(#{2,4})\s+(.*)$", line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            title = re.sub(r"^(\d+(\.\d+)*)\.?\s*", r"\1 ", title)
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
            if in_refs:
                in_refs = False  # a new heading closes the reference list
            if title in SKIP_HEADINGS:
                skip_depth = level
                i += 1
                while i < len(lines):
                    nxt = re.match(r"^(#{2,4})\s+", lines[i])
                    if nxt and len(nxt.group(1)) <= skip_depth:
                        break
                    i += 1
                continue
            if title.startswith("References"):
                in_refs = True
                i += 1
                continue
            if title.startswith("Abstract"):
                body.append(r"\begin{abstract}")
                abstract_open = True
                i += 1
                continue
            if title.startswith("Keywords"):
                body.append(r"\begin{keyword}")
                keywords_open = True
                i += 1
                continue
            body.append(f"\\{cmd}{{{inline(title, slots)}}}")
            i += 1
            continue

        # --- reference entries ---
        if in_refs and re.match(r"^\d+\.\s", line):
            if ref_buf:
                refs.append(" ".join(ref_buf))
                ref_buf = []
            ref_buf.append(re.sub(r"^\d+\.\s*", "", line))
            i += 1
            continue
        if in_refs and line.strip() and not re.match(r"^\d+\.\s", line):
            if ref_buf:
                ref_buf.append(line.strip())
            i += 1
            continue

        # --- tables ---
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            body.append(to_tabular(block, slots))
            body.append("")
            continue

        # --- images ---
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if m:
            src = m.group(2)
            name = Path(src).name
            body.append(r"\begin{figure}[htbp]")
            body.append(r"\centering")
            body.append(r"\includegraphics[width=0.85\linewidth]{%s}" % name)
            if m.group(1):
                body.append(r"\caption{" + inline(m.group(1), slots) + "}")
            body.append(r"\end{figure}")
            i += 1
            continue

        # --- lists ---
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", raw)
        if m:
            indent, marker, item = m.groups()
            ordered = marker[0].isdigit()
            env = "enumerate" if ordered else "itemize"
            items = []
            while i < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if not mm or (mm.group(2)[0].isdigit() != ordered):
                    break
                items.append(inline(mm.group(3), slots))
                i += 1
            body.append(f"\\begin{{{env}}}")
            for it in items:
                body.append(r"\item " + it)
            body.append(f"\\end{{{env}}}")
            body.append("")
            continue

        # --- blank line ---
        if not line.strip():
            if ref_buf:
                refs.append(" ".join(ref_buf))
                ref_buf = []
            body.append("")
            i += 1
            continue

        # --- ordinary paragraph ---
        if abstract_open and re.match(r"^##\s", line):
            abstract_open = False
        body.append(inline(line, slots))
        i += 1

    if ref_buf:
        refs.append(" ".join(ref_buf))

    text = "\n".join(body)

    # The abstract is closed by the next section, and frontmatter closes with it.
    if r"\begin{abstract}" in text and r"\end{abstract}" not in text:
        first = text.find("\\section")
        if first > 0:
            text = (text[:first] + "\\end{abstract}\n\n\\end{frontmatter}\n\n"
                    + text[first:])
        else:
            text += "\n\\end{abstract}\n\n\\end{frontmatter}\n"

    # Emit the reference list.
    if refs:
        text += "\n\n\\begin{thebibliography}{%d}\n" % len(refs)
        for r in refs:
            entry = restore(r, slots)
            entry = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", entry)
            text += "\\bibitem{}\n" + entry + "\n\n"
        text += "\\end{thebibliography}\n"

    return text


PREAMBLE = r"""\documentclass[preprint,12pt]{elsarticle}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}

% Left empty: the arXiv version is not tied to a journal.
\journal{}

\begin{document}

\begin{frontmatter}
"""

POSTMATTER = r"""
\end{frontmatter}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", default=str(DEFAULT_IN))
    ap.add_argument("-o", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    md = deunicode(Path(args.i).read_text(encoding="utf-8"))
    body = convert(md)

    out = Path(args.o)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PREAMBLE + body + "\n\\end{document}\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
