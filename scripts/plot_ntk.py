"""把 NTK / 曲率分析结果画成论文的机制图。

用法：
  python scripts/plot_ntk.py --json results/ntk.json --out results/ntk_fig
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})
C_BASE, C_ALIGN = "#B0413E", "#2A6F97"


def label_of(r):
    kind = "aligned" if r["method"] != "pi_baseline" else "baseline"
    return f"{kind} s{r['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/ntk.json")
    ap.add_argument("--out", default="results/ntk_fig")
    args = ap.parse_args()
    recs = json.load(open(ROOT / args.json, encoding="utf-8"))
    if not recs:
        print("empty"); return
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    # --- A: 输出 NTK 特征值谱 ---
    ax = axes[0]
    for r in recs:
        ev = np.array(r["output_ntk"]["ev_top"])
        c = C_ALIGN if r["method"] != "pi_baseline" else C_BASE
        ls = "-" if r["method"] != "pi_baseline" else "--"
        ax.semilogy(np.arange(1, len(ev) + 1), np.maximum(ev, 1e-6),
                    c=c, ls=ls, lw=1.4, alpha=0.9, label=label_of(r))
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel(r"$\lambda_i/\lambda_1$")
    ax.set_title("(a) Output NTK spectrum")
    ax.legend(fontsize=7)
    ax.set_ylim(1e-6, 2)

    # --- B: trunk 奇异值谱 ---
    ax = axes[1]
    for r in recs:
        sv = np.array(r["trunk_ntk"]["sv_top"])
        c = C_ALIGN if r["method"] != "pi_baseline" else C_BASE
        ls = "-" if r["method"] != "pi_baseline" else "--"
        ax.semilogy(np.arange(1, len(sv) + 1), np.maximum(sv, 1e-6),
                    c=c, ls=ls, lw=1.4, alpha=0.9, label=label_of(r))
    ax.set_xlabel("trunk basis index"); ax.set_ylabel(r"$\sigma_i/\sigma_1$")
    ax.set_title("(b) Trunk spectrum")
    ax.legend(fontsize=7)

    # --- C: 尖锐度与 NTK 有效秩 ---
    ax = axes[2]
    base = [r for r in recs if r["method"] == "pi_baseline"]
    alig = [r for r in recs if r["method"] != "pi_baseline"]

    def vals(rs, f):
        return [f(r) for r in rs]

    x = np.arange(2)
    for i, (name, f) in enumerate([
            ("NTK eff. rank", lambda r: r["output_ntk"]["eff_rank_entropy"]),
            (r"sharpness $\lambda_{max}/2L$", lambda r: r["hessian"]["sharpness"]),
            (r"$\lambda_{max}$", lambda r: r["hessian"]["hess_lambda_max"])]):
        pass
    # 用两个归一化子图表达更清楚：左侧原始值散点
    metrics = [("NTK eff. rank", lambda r: r["output_ntk"]["eff_rank_entropy"]),
               ("sharpness\n(×1e-2)", lambda r: r["hessian"]["sharpness"] * 1e-2)]
    w = 0.32
    for i, (nm, f) in enumerate(metrics):
        vb = vals(base, f); va = vals(alig, f)
        ax.bar(i - w / 2 - 0.02, np.mean(vb), w, color=C_BASE, alpha=0.85,
               label="baseline" if i == 0 else None)
        ax.scatter([i - w / 2 - 0.02] * len(vb), vb, color="k", s=14, zorder=3)
        if va:
            ax.bar(i + w / 2 + 0.02, np.mean(va), w, color=C_ALIGN, alpha=0.85,
                   label="aligned" if i == 0 else None)
            ax.scatter([i + w / 2 + 0.02] * len(va), va, color="k", s=14, zorder=3)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_title("(c) NTK rank & loss sharpness")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out / "ntk_mechanism.png")
    print(f"[saved] {out/'ntk_mechanism.png'}")

    # 数字汇总
    print(f"\n{'ckpt':46s} {'relL2':>7s} {'trunkER':>8s} {'ntkER':>7s} "
          f"{'top1':>6s} {'lmax':>9s} {'sharp':>8s}")
    for r in sorted(recs, key=lambda x: (x["method"], x["seed"])):
        rl = r.get("rel_l2")
        rls = f"{rl:.4f}" if isinstance(rl, (int, float)) else "  n/a"
        print(f"{r['ckpt'][:46]:46s} {rls:>7s} "
              f"{r['trunk_ntk']['eff_rank']:8.2f} "
              f"{r['output_ntk']['eff_rank_entropy']:7.2f} "
              f"{r['output_ntk']['top1_share']:6.3f} "
              f"{r['hessian']['hess_lambda_max']:9.1f} "
              f"{r['hessian']['sharpness']:8.1f}")


if __name__ == "__main__":
    main()
