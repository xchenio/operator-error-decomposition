"""论文图 2：对齐增益 vs 基线跨种子标准差（r = 0.985）。

用法：python scripts/plot_gain_vs_std.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})

# 数据源：scripts/analyze_phase.py 的输出
d = json.load(open(ROOT / "results/phase_analysis.json", encoding="utf-8"))
rows = [{"setting": t["setting"], "baseline_std": t["base_std"],
         "gain_pct": t["gain_pct"], "welch_p": t["welch_p"],
         "baseline_mean": t["base_mean"], "align_mean": t["align_mean"],
         "align_std": t["align_std"], "width": 0, "nu": 0} for t in d["table"]]
sig = np.array([r["baseline_std"] for r in rows])
gain = np.array([r["gain_pct"] for r in rows])
pval = np.array([r["welch_p"] for r in rows])
names = [r["setting"] for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

ax = axes[0]
ax.scatter(sig, gain, s=90, c="#2A6F97", zorder=3, edgecolor="k", linewidth=0.5)
for x, y, n, p in zip(sig, gain, names, pval):
    ax.annotate(f"{n}\n(p={p:.3f})", (x, y), fontsize=7,
                xytext=(6, -6), textcoords="offset points")
coef = np.polyfit(sig, gain, 1)
xs = np.linspace(0, sig.max() * 1.15, 50)
ax.plot(xs, np.polyval(coef, xs), "--", color="#B0413E", lw=1.3)
r = np.corrcoef(sig, gain)[0, 1]
ax.set_xlabel("baseline cross-seed std (optimization instability)")
ax.set_ylabel("alignment gain (%)")
ax.set_title(f"(a) gain is predicted by instability\nPearson r = {r:.3f}  (n = 4 settings)")
ax.set_xlim(0, sig.max() * 1.15)
ax.set_ylim(0, gain.max() * 1.35)

ax = axes[1]
x = np.arange(len(rows))
w = 0.36
ax.bar(x - w / 2, [float(r_["baseline_mean"]) for r_ in rows], w,
       yerr=sig, capsize=3, color="#B0413E", alpha=0.9, label="baseline")
ax.bar(x + w / 2, [float(r_["align_mean"]) for r_ in rows], w,
       yerr=[float(r_["align_std"]) for r_ in rows], capsize=3,
       color="#2A6F97", alpha=0.9, label="aligned")
ax.set_xticks(x)
ax.set_xticklabels([n.replace("burgers|", "").replace("burgers2d|", "2D|")
                    for n in names], fontsize=7)
ax.set_ylabel("test relative $L_2$ error")
ax.set_title("(b) three seeds per setting, all p < 0.05")
ax.legend(fontsize=7)
ax.set_ylim(0, 1.05)

fig.tight_layout()
out = ROOT / "results" / "gain_vs_std.png"
fig.savefig(out)
print(f"[saved] {out}\nPearson r = {r:.4f}")
