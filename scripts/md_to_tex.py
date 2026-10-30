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
    # Change-log entries addressed to readers of the repository, not of the
    # paper. They report what changed on a given date, which has no place in a
    # submitted manuscript.
    "**Latest 2026-09-29 follow-up",
    "**Earlier 2026-09-29",
    "**2026-09-24 audit",
)

# JCP wants keywords and so does the front matter; the metadata file is the
# single place they are defined, so the two documents cannot diverge.
def _load_keywords() -> list[str]:
    meta = ROOT / "paper/arxiv_metadata.md"
    try:
        block = meta.read_text(encoding="utf-8").split("## Keywords", 1)[1]
        block = block.split("## Highlights", 1)[0]
        keys = [k.strip() for k in block.replace("\n", " ").split(";") if k.strip()]
        return [k for k in keys if not k.startswith("(")]
    except (IndexError, OSError):
        return []


KEYWORDS = _load_keywords()

# Back matter: unnumbered, and outside the paper's own section numbering.
BACKMATTER = {
    "Declaration of competing interest",
    "Funding",
    "Data availability",
    "Author contribution",
    "Relation to the author's prior thesis",
    "Acknowledgments",
}


# pdflatex, which arXiv uses by default, has no glyphs for several characters
# the manuscript uses. Replacing them here keeps the source compilable there
# rather than only locally under xelatex.
# U+00B1 becomes a sentinel rather than "$\pm$" straight away: the backslash
# would make looks_math() treat a plain numeric span as a formula and wrap it
# in a second pair of dollars. The sentinel is expanded per context instead.
PM = "@@PM@@"

UNICODE_MAP = {
    "\u00b1": PM,              # PLUS-MINUS SIGN
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
        return stash("\\[" + m.group(1).replace(PM, r"\pm") + "\\]")

    text = re.sub(r"\\\[(.*?)\\\]", disp, text, flags=re.S)

    # Inline code: `x`, which in this manuscript may equally be inline math.
    def code(m):
        body = m.group(1)
        if looks_math(body):
            return stash("$" + body.replace(PM, r"\pm") + "$")
        body = body.replace(PM, r"$\pm$")
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

    # Bare URLs. Escaping would mangle them and they do not break across lines,
    # so they are stashed as \url, which hyperref allows to break. The
    # lookbehind keeps URLs that are the target of a Markdown link untouched;
    # those become \href later.
    def url(m):
        return stash(r"\url{" + m.group(1) + "}")

    text = re.sub(r"(?<!\()\b(https?://[^\s\)\]]+)", url, text)
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
           # Shrink to fit only when the table is wider than the text block;
           # narrow tables are left at natural size.
           r"\begin{adjustbox}{max width=\linewidth}",
           r"\small",
           r"\begin{tabular}{%s}" % spec, r"\toprule"]
    head, *body = cells
    out.append(" & ".join(inline(c, slots) for c in head) + r" \\")
    out.append(r"\midrule")
    for row in body:
        padded = row + [""] * (ncols - len(row))
        out.append(" & ".join(inline(c, slots) for c in padded) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{center}"]
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
    preface: list[str] = []
    i = 0
    title_done = False
    orcid = ""
    in_pre = False            # between the author block and the abstract
    first_section_done = False
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

        # --- block quote, used for the status note in section 5.8 ---
        if raw.lstrip().startswith(">"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                block.append(lines[i].lstrip()[1:].strip())
                i += 1
            body.append(r"\begin{quote}")
            body.append(inline(" ".join(block), slots))
            body.append(r"\end{quote}")
            body.append("")
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
            parts = [p.strip() for p in raw.strip().split("·")]
            addr = parts[0].rstrip(".")
            body.append(r"\address{%s. %s}" % (addr, orcid) if orcid
                        else r"\address{%s}" % addr)
            if len(parts) > 1 and "@" in parts[1]:
                body.append(r"\ead{%s}" % parts[1])
            in_pre = True
            i += 1
            continue

        # --- headings ---
        m = re.match(r"^(#{2,4})\s+(.*)$", line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            # LaTeX numbers sections itself; keeping the manuscript's manual
            # numbers would double them up ("2. 2 Related work and overlap").
            num = re.match(r"^(\d+(?:\.\d+)*)\.?\s", title)
            title = re.sub(r"^(\d+(\.\d+)*)\.?\s*", "", title)
            # Heading depth comes from the numbering, not from the number of
            # hashes: the manuscript writes both "5.6" and "5.6.1" as ###, and
            # taking the hashes would turn 5.6.1 into 5.7, leaving every cross
            # reference in the text pointing at a section that does not exist.
            depth = len(num.group(1).split(".")) if num else {2: 1, 3: 2, 4: 3}[level]
            cmd = {1: "section", 2: "subsection", 3: "subsubsection"}.get(depth, "subsubsection")
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
                in_pre = False
                body.append(r"\begin{abstract}")
                abstract_open = True
                i += 1
                continue

            # The first numbered section ends the front matter. Anything read
            # before the abstract (the scope paragraph and the changelog notes)
            # belongs to the body, not to front matter, where it would suppress
            # the title block entirely.
            if not first_section_done:
                if abstract_open:
                    body.append(r"\end{abstract}")
                    abstract_open = False
                body.append(r"\begin{keyword}")
                body.append(r" \sep ".join(KEYWORDS))
                body.append(r"\end{keyword}")
                body.append(r"\end{frontmatter}")
                body.extend(preface)
                preface.clear()
                first_section_done = True
            if title.startswith("Keywords"):
                body.append(r"\begin{keyword}")
                keywords_open = True
                i += 1
                continue
            # Declarations after the references are back matter, not numbered
            # sections. Numbering them also collides with "section 9 of the
            # protocol", which means the protocol document, not this paper.
            marker = "*" if title in BACKMATTER else ""
            body.append(f"\\{cmd}{marker}{{{inline(title, slots)}}}")
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
                if mm and (mm.group(2)[0].isdigit() == ordered):
                    items.append([mm.group(3)])
                    i += 1
                    continue
                # A list entry runs over several source lines. Without absorbing
                # the continuation each entry becomes its own list and every one
                # is numbered 1, which is what the seven contributions in
                # section 1 did.
                # Lazy continuation: Markdown counts an unindented line as part
                # of the item, so requiring indentation leaves the rest of the
                # entry outside it and its markup unhandled.
                if (items and lines[i].strip()
                        and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i])
                        and not re.match(r"^\s*(#{1,4}\s|\||!\[|>)", lines[i])):
                    items[-1].append(lines[i].strip())
                    i += 1
                    continue
                break
            body.append(f"\\begin{{{env}}}")
            # Joined before conversion, so emphasis spanning two source lines
            # is converted rather than left as literal asterisks.
            for it in items:
                body.append(r"\item " + inline(" ".join(it), slots))
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
        # Collected as a whole rather than line by line: emphasis in this
        # manuscript routinely spans two source lines, and a per-line pass
        # leaves the asterisks in the output as literal text.
        para = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip():
                break
            if re.match(r"^\s*(#{1,4}\s|\||!\[|>|- |\* |\d+\.\s)", nxt):
                break
            para.append(nxt.rstrip())
            i += 1
        target = preface if in_pre else body
        target.append(inline(" ".join(para), slots))

    if ref_buf:
        refs.append(" ".join(ref_buf))

    text = "\n".join(body)

    # Safety net: if the document somehow ends without a numbered section, the
    # front matter must still be closed or nothing will render.
    if not first_section_done:
        if r"\begin{abstract}" in text and r"\end{abstract}" not in text:
            text += "\n\\end{abstract}"
        text += "\n\\begin{keyword}\n" + r" \sep ".join(KEYWORDS) \
            + "\n\\end{keyword}\n\\end{frontmatter}\n" + "\n".join(preface)

    # Whatever sentinel is left sits in ordinary prose.
    text = text.replace(PM, r"$\pm$")

    # Emit the reference list.
    if refs:
        text += "\n\n\\begin{thebibliography}{%d}\n" % len(refs)
        for n, r in enumerate(refs, 1):
            entry = restore(r, slots)
            entry = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", entry)
            # "(added 2026-10-08; ...)" is a note to co-readers of the
            # repository about when an entry was checked. It is editorial and
            # does not belong in a reference list.
            entry = re.sub(r"\((?:added|checked)\s+[\d-]{10}[^)]*\)", "", entry)
            entry = re.sub(r"\s{2,}", " ", entry).strip()
            # Citations are written literally in the text, "[10]", rather than
            # through \cite, so the keys are labels only. Leaving them empty
            # made natbib report one "multiply defined" warning per entry.
            text += "\\bibitem{ref%d}\n" % n + entry + "\n\n"
        text += "\\end{thebibliography}\n"

    return text


PREAMBLE = r"""\documentclass[preprint,12pt]{elsarticle}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}
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
