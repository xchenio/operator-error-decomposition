"""One-off: state plainly what arXiv's log does and does not contain."""
import subprocess
import sys
from pathlib import Path


def pypdf():
    try:
        import pypdf  # noqa
        return pypdf
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "--target", str(Path.home() / "pdftools"), "pypdf"], check=True)
        sys.path.insert(0, str(Path.home() / "pdftools"))
        import pypdf  # noqa
        return pypdf


src = Path(r"F:\Desktop\File Processing Result.pdf")
if not src.exists():
    print("missing", src)
    raise SystemExit(1)

pypdf = pypdf()
reader = pypdf.PdfReader(str(src))
text = "\n".join((p.extract_text() or "") for p in reader.pages)

counts = {}
for key in ("overfull", "underfull", "undefined", "missing", "error",
            "multiply defined", "failed", "hyphenat", "font warning"):
    counts[key] = sum(1 for ln in text.splitlines() if key in ln.lower())

out = [f"pages: {len(reader.pages)}", ""]
out.append("=== verdict ===")
for k, v in counts.items():
    out.append(f"{k:18s} {v}")

out.append("")
out.append("=== every overfull/underfull line, verbatim ===")
found = [ln.strip() for ln in text.splitlines()
         if "overfull" in ln.lower() or "underfull" in ln.lower()]
out.extend(found or ["(none)"])

out.append("")
out.append("=== status ===")
for ln in text.splitlines():
    s = ln.strip()
    if s and ("SUCCEEDED" in s or "processed OK" in s or "Output written" in s):
        out.append(s)

dest = Path(r"F:\Desktop\PINN-主账号\pinode_align\results\arxiv_result4.txt")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text("\n".join(out), encoding="utf-8")
print("wrote", dest)
