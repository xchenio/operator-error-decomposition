"""结果分析：汇总表 + 四张关键图。

用法：
  python scripts/analyze.py --out results/pilot_aw
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
sys.path.insert(0, str(ROOT))
from src.utils import pearson, spearman  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def load_runs(out_dir: Path):
    runs = []
    for f in sorted(out_dir.glob("*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if "meta" not in d:
            continue
        d["_file"] = f.stem
        runs.append(d)
    return runs


def short_label(r):
    c = r["meta"]["cfg"]
    m = c["method"]
    aw = c["align_weight"]
    kind = c["align_kind"]
    if kind == "none":
        return m
    return f"{m} [{kind}×{aw:g}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pilot_aw")
    ap.add_argument("--smooth", type=int, default=1)
    args = ap.parse_args()

    out_dir = ROOT / args.out
    runs = load_runs(out_dir)
    if not runs:
        print(f"no runs found in {out_dir}")
        return
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 汇总表 ----------
    print("\n" + "=" * 112)
    print(f"{'run':40s} {'testRelL2':>9s} {'LB(表示下界)':>12s} {'opt_gap':>8s} "
          f"{'capt_final':>10s} {'eff_rank':>8s} {'angle':>6s} {'time(s)':>7s}")
    print("=" * 112)
    for r in sorted(runs, key=lambda x: x["test_rel_l2"]):
        lb = r.get("lb_final", float("nan"))
        gap = r.get("opt_gap", float("nan"))
        print(f"{short_label(r):40s} {r['test_rel_l2']:9.4f} {lb:12.4f} {gap:8.4f} "
              f"{r['final_diag']['captured_energy']:10.4f} "
              f"{r['final_diag']['eff_rank']:8.2f} "
              f"{r['final_diag']['angle_mean_deg']:6.1f} "
              f"{r['meta']['wall_time']:7.0f}")
    print("=" * 112)
    print("LB = sqrt(1 - captured_energy)：给定 trunk 时，所有 branch 中最好的相对误差（硬下界）")
    print("opt_gap = testRelL2 - LB：优化差距。LB 大 = 表示受限；opt_gap 大 = 优化受限\n")

    # ---------- 图 1: val relL2 曲线 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in runs:
        h = r["history"]
        ax[0].plot(h["step"], h["val_rel_l2"], label=short_label(r), lw=1.6)
    ax[0].set_xlabel("step"); ax[0].set_ylabel("val rel L2")
    ax[0].set_title("generalization"); ax[0].legend(fontsize=7)

    # ---------- 图 2: captured energy 曲线 ----------
    for r in runs:
        dh = r["diag_history"]
        ax[1].plot(dh["step"], dh["captured_energy"], label=short_label(r), lw=1.6)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("captured energy")
    ax[1].set_title("trunk basis vs true solution subspace")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "curves.png"); plt.close(fig)

    # ---------- 图 3: 相关性散点 ----------
    xs = [r["final_diag"]["captured_energy"] for r in runs]
    ys = [r["test_rel_l2"] for r in runs]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, s=60, c=range(len(runs)), cmap="tab10")
    for i, r in enumerate(runs):
        ax.annotate(short_label(r), (xs[i], ys[i]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    if len(runs) >= 3:
        ax.set_title(f"captured vs test relL2\npearson={pearson(xs, ys):.3f} "
                     f"spearman={spearman(xs, ys):.3f}")
    ax.set_xlabel("captured energy (final)"); ax.set_ylabel("test rel L2")
    fig.tight_layout(); fig.savefig(fig_dir / "corr.png"); plt.close(fig)

    # ---------- 图 4: trunk 奇异值谱 ----------
    fig, ax = plt.subplots(figsize=(5, 4))
    for r in runs:
        sv = np.array(r["final_sv"])
        ax.semilogy(np.arange(1, len(sv) + 1), np.maximum(sv, 1e-8),
                    label=short_label(r), lw=1.4)
    ax.set_xlabel("trunk basis index"); ax.set_ylabel("normalized singular value")
    ax.set_title("trunk singular value spectrum (steeper = more collapsed)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "spectrum.png"); plt.close(fig)

    # ---------- 图 5: 误差归因（表示下界 vs 实际误差） ----------
    labels = [short_label(r) for r in runs]
    lbs = np.array([r.get("lb_final", np.nan) for r in runs])
    acts = np.array([r["test_rel_l2"] for r in runs])
    order = np.argsort(-acts)
    y = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(runs) + 2.2))
    ax.barh(y, lbs[order], color="#4C72B0", label="LB (representation limit)")
    ax.barh(y, (acts - lbs)[order], left=lbs[order], color="#DD8452",
            label="optimization gap")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order], fontsize=7)
    ax.set_xlabel("test relative L2 error")
    ax.set_title("Error attribution: representation limit vs optimization gap")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(fig_dir / "attribution.png"); plt.close(fig)

    print(f"\nfigures -> {fig_dir}")


if __name__ == "__main__":
    main()
