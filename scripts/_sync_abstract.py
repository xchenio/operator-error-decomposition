"""One-off: regenerate the metadata abstract from the manuscript.

Two documents carry the abstract and a test asserts they agree, so the
metadata file is derived rather than maintained by hand.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "paper/main.md").read_text(encoding="utf-8")
abstract = text.split("## Abstract", 1)[1].split(
    "## 1. Research question and contribution", 1)[0].strip()

meta_path = ROOT / "paper/arxiv_metadata.md"
meta = meta_path.read_text(encoding="utf-8")
head, _, tail = meta.partition("## Abstract")
tail = tail.split("## Keywords", 1)[1]
meta_path.write_text(head + "## Abstract\n\n" + abstract + "\n\n## Keywords" + tail,
                     encoding="utf-8", newline="")

check = meta_path.read_text(encoding="utf-8").split("## Abstract", 1)[1].split(
    "## Keywords", 1)[0].strip()
print("identical:", check == abstract)
