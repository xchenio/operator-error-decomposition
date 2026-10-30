"""One-off: read arXiv's processing result and list every flagged line."""
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

out = [f"pages: {len(reader.pages)}", ""]
out.append("=== status lines ===")
for ln in text.splitlines():
    s = ln.strip()
    if s and ("SUCCEEDED" in s or "processed OK" in s or "Output written" in s
              or "failed" in s.lower()):
        out.append(s)

out.append("")
out.append("=== flagged lines ===")
KEYS = ("overfull", "underfull", "undefined", "missing", "error", "warning",
        "not found", "multiply", "failed", "font", "figure", "image")
for ln in text.splitlines():
    s = ln.strip()
    low = s.lower()
    if s and any(k in low for k in KEYS) and "Key for Highlighted" not in s:
        out.append(s[:200])

dest = Path(r"F:\Desktop\PINN-主账号\pinode_align\results\arxiv_result3.txt")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text("\n".join(out), encoding="utf-8")
print("wrote", dest)
