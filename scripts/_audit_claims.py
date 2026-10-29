"""One-off: check the numeric claims a reviewer flagged against the artifacts."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
md = (ROOT / "paper/main.md").read_text(encoding="utf-8")
out = []


def grab(pattern, label, width=160):
    hits = [m.group(0).replace("\n", " ") for m in re.finditer(pattern, md)]
    out.append(f"--- {label} ({len(hits)}) ---")
    for h in hits[:6]:
        out.append("  " + h[:width])


# 1. Monotone decay claim and the Fourier values at 1.5 / 2.0.
grab(r"monotonic[^.]{0,120}", "monotonic claims")
grab(r"\|\s*1\.50\s*\|[^|]*\|[^|]*\|[^|]*\|", "horizon rows t=1.50")
grab(r"\|\s*2\.00\s*\|[^|]*\|[^|]*\|[^|]*\|", "horizon rows t=2.00")

# 2. CFD component comparison.
grab(r"worse[^.]{0,140}", "'worse' claims")
grab(r"\| density \|[^|]*\|[^|]*\|[^|]*\|", "CFD density row")

# 3. FLOP typo.
grab(r"192\.68e8|2\.68e8", "FLOP value")
grab(r"FLOPs for 1500 steps", "FLOP column header")

# 4. Fluctuation energy values.
grab(r"0\.2200", "energy 0.2200")
grab(r"\|\s*1\.00\s*\|\s*0\.2367", "energy 0.2367 row")

# 5. Trunk counts.
grab(r"50 trained trunks|five settings|45 ", "trunk counts")

# 6. Run counts.
grab(r"175 recon|175 runs|185", "run counts")

# 7. Darcy.
grab(r"two `?beta`? values|only setting|beta = 10|beta 10", "Darcy beta")

# 8. Section 5.19 cross reference.
grab(r"section 5\.4\)", "5.19 cross-ref")

# 9. relL2 range.
grab(r"0\.66 to 1\.10|best relL2", "relL2 range claim")

# 10. Data source counts.
grab(r"remaining seven|DaRUS", "data source counts", 200)

dest = ROOT / "results/claim_audit.txt"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text("\n".join(out), encoding="utf-8")
print("wrote", dest)
