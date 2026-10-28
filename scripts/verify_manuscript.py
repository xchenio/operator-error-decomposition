"""Check the manuscript for internal inconsistencies before submission.

Run this after any edit to paper/main.md. It catches the class of defect that a
section-level read-through misses, because each statement looks correct in
isolation:

1. a "section X.Y" cross-reference that no longer resolves after renumbering;
2. a reference list that is not contiguous 1..N, or an entry that is never cited,
   or a citation to a number that does not exist;
3. an abstract over JCP's hard 250-word cap;
4. an image link to a file that is not in the repository.

Exit status is non-zero if any check fails. Output is written to
results/manuscript_check.txt as well as stdout, because the Windows console
defaults to a GBK code page that cannot print U+00B1 or U+00B2.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper/main.md"
METADATA = ROOT / "paper/arxiv_metadata.md"
REPORT = ROOT / "results/manuscript_check.txt"

# JCP: "The abstract should be ... concise and factual ... not exceed 250 words."
ABSTRACT_CAP = 250
# Elsevier: highlights are 3 to 5 bullet points of at most 85 characters each.
HIGHLIGHT_CAP = 85


def check(text: str) -> tuple[list[str], list[str]]:
    """Return (informational lines, failure lines)."""
    info: list[str] = []
    fail: list[str] = []

    # --- 1. cross-references -------------------------------------------------
    headings = set()
    for m in re.finditer(r"^(#{2,4})\s+(.+?)\s*$", text, flags=re.M):
        title = m.group(2)
        num = re.match(r"(\d+(?:\.\d+)*)[\.\s]", title)
        if num:
            headings.add(num.group(1))
    for m in re.finditer(r"section\s+(\d+\.\d+)", text):
        if m.group(1) not in headings:
            fail.append(f"dangling cross-reference to section {m.group(1)}")
    targets = set(re.findall(r"section (\d+\.\d+)", text))
    info.append(f"cross-references checked: {len(targets)} distinct targets")
    info.append(f"headings found: {len(headings)}")

    # --- 2. reference list ---------------------------------------------------
    refs_start = text.find("\n## References")
    if refs_start < 0:
        fail.append("no '## References' section found")
        return info, fail

    block = text[refs_start:]
    stop = block.find("\n## ", 1)
    if stop > 0:
        block = block[:stop]

    numbered = [int(m.group(1)) for m in re.finditer(r"^(\d+)\.\s", block, flags=re.M)]
    if numbered != list(range(1, len(numbered) + 1)):
        fail.append(f"reference numbers are not contiguous 1..{len(numbered)}: {numbered}")

    # Citations appear both singly, "[2]", and grouped, "[13, 14]". Bracketed
    # decimals such as "[0.001]" or intervals such as "[0, 500]" are not
    # citations, so keep only plain integers in the plausible reference range.
    MAX_REF = 30
    cited = set()
    for grp in re.findall(r"\[([\d.,\s]+)\]", text):
        if "." in grp:
            continue
        for part in re.split(r"[,\s]+", grp.strip()):
            if part.isdigit() and 1 <= int(part) <= MAX_REF:
                cited.add(int(part))
    unknown = sorted(cited - set(numbered))
    if unknown:
        fail.append(f"cited but not in the reference list: {unknown}")
    uncited = sorted(set(numbered) - cited)
    if uncited:
        fail.append(f"in the reference list but never cited: {uncited}")
    info.append(f"references: {len(numbered)} numbered, {len(cited)} distinct citations")

    # --- 3. abstract length --------------------------------------------------
    a_start = text.find("## Abstract")
    a_end = text.find("## 1. Research question and contribution")
    if 0 <= a_start < a_end:
        abstract = text[a_start:a_end]
        # Every bracketed [n] is a citation, not prose.
        prose = re.sub(r"\[[0-9,\s]+\]", " ", abstract)
        n = len(re.findall(r"[A-Za-z0-9]+", prose))
        info.append(f"abstract: {n} words (cap {ABSTRACT_CAP})")
        if n > ABSTRACT_CAP:
            fail.append(f"abstract is {n} words, over the {ABSTRACT_CAP}-word cap")
    else:
        fail.append("could not locate the abstract block")

    # --- 4. image links -------------------------------------------------------
    # Links in main.md are relative to paper/, not to the repository root.
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
        target = m.group(1)
        if target.startswith(("http://", "https://")):
            continue
        if not (MANUSCRIPT.parent / target).exists():
            fail.append(f"image file not found: {target}")

    # --- 5. metadata highlights -----------------------------------------------
    if METADATA.exists():
        meta = METADATA.read_text(encoding="utf-8")
        hl = meta.split("## Highlights", 1)[-1].split("\n## ", 1)[0]
        bullets = [ln[2:].strip() for ln in hl.splitlines() if ln.strip().startswith("- ")]
        info.append(f"highlights: {len(bullets)}")
        if not 3 <= len(bullets) <= 5:
            fail.append(f"highlights must number 3 to 5, found {len(bullets)}")
        for b in bullets:
            if len(b) > HIGHLIGHT_CAP:
                fail.append(f"highlight over {HIGHLIGHT_CAP} chars ({len(b)}): {b}")

    return info, fail


def main() -> None:
    if not MANUSCRIPT.exists():
        print("manuscript not found:", MANUSCRIPT)
        sys.exit(1)

    text = MANUSCRIPT.read_text(encoding="utf-8")
    info, fail = check(text)

    lines = list(info)
    lines.append("")
    if fail:
        for f in fail:
            lines.append(f"FAIL  {f}")
        lines.append("")
        lines.append(f"{len(fail)} check(s) failed")
    else:
        lines.append("all checks passed")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
