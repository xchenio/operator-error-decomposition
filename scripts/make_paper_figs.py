"""生成主论文 (paper/main.md) 需要的全部统计图。

只读 results/*.json，不训练。输出到 results/paper_figs/。

  python scripts/make_paper_figs.py

图清单：
  fig1_attribution    三项分解堆叠条（LB / opt_gap_train / gen_gap）
  fig2_2d_controls    2D 六组干预的 relL2 条形图（5 seed 误差棒）
  fig3_bug            静默广播 bug 修复前后对比
  fig4_varfloor       方差地板：sigma_baseline vs sigma_aligned（含 ~0.01 带与重跑噪声线）
  fig5_mechanism      NTK 有效秩 / 尖锐度：baseline vs aligned
  fig6_1d_ablation    1D 宽度 64、10000 步的六方法对照
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

OUT = ROOT / "results" / "paper_figs"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


# ---------------- 数据加载 ----------------

def load(dirs):
    recs = []
    for d in dirs:
        for f in glob.glob(str(ROOT / "results" / d / "**" / "*.json"), recursive=True):
            if "phase_analysis" in f or "attribution" in f or "ntk" in f:
                continue
            try:
                j = json.load(open(f, encoding="utf-8"))
            except Exception:                          # noqa: BLE001
                continue
            if "meta" not in j or "test_rel_l2" not in j:
                continue
            c = j["meta"]["cfg"]
            mkw = c.get("model_kwargs") or {}
            pk = c.get("pde_kwargs") or {}
            recs.append({
                "dir": d, "pde": c["pde"], "method": c["method"],
                "align_kind": c.get("align_kind"), "seed": c["seed"],
                "width": mkw.get("width") or pk.get("width"),
                "nu": pk.get("nu"), "steps": c["steps"],
                "trunk": mkw.get("trunk_type", "mlp"),
                "ortho": float(c.get("ortho_weight") or 0.0),
                "icw": (c.get("lambdas_override") or {}).get("ic"),
                "test": j["test_rel_l2"], "train": j.get("train_rel_l2"),
                "lb": j.get("lb_final"), "captured": (j.get("final_diag") or {}).get("captured_energy"),
                "eff_rank": (j.get("final_diag") or {}).get("eff_rank"),
            })
    # 去重：同一配置可能被重复跑过（如 expU / expV 各有一份 2D 结果），
    # 按 (pde,nu,width,steps,trunk,method,align_kind,ortho,icw,seed) 取平均
    dedup = {}
    for r in recs:
        k = (r["pde"], r["nu"], r["width"], r["steps"], r["trunk"], r["method"],
             r["align_kind"], r["ortho"], r["icw"], r["seed"])
        dedup.setdefault(k, []).append(r)
    out = []
    for rs in dedup.values():
        base = dict(rs[0])
        for key in ("test", "train", "lb", "captured", "eff_rank"):
            vals = [x[key] for x in rs if x[key] is not None]
            base[key] = sum(vals) / len(vals) if vals else None
        base["_n_dup"] = len(rs)
        out.append(base)
    return out


def arm(r):
    if r["trunk"] != "mlp":
        return "Fourier-feature trunk"
    if r["ortho"] > 0:
        return "orthogonalization reg."
    if r["method"] == "data_only":
        return "supervised (data only)"
    if r["icw"]:
        return f"10x IC weight"
    if r["method"] == "pi_baseline":
        return "baseline"
    if r["align_kind"] in (None, "none"):
        return r["method"]
    return {"prior": "align: spectral prior",
            "random": "align: random smooth subspace"}.get(r["align_kind"],
                                                           f"align: {r['align_kind']}")


def cell(r):
    if r["nu"] is not None:
        nu = r["nu"]
    elif r["pde"] == "burgers":
        nu = 0.02                     # PDE 默认粘性；旧 run 的 pde_kwargs 为空
    else:
        nu = "default"
    return f"{r['pde']}|nu={nu}|w={r['width']}|{r['steps']}"


def agg(recs, keyfn):
    g = {}
    for r in recs:
        g.setdefault(keyfn(r), []).append(r)
    return g


def mstd(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


# ---------------- 图 1：三项分解 ----------------

def fig1(recs):
    groups = agg(recs, lambda r: (cell(r), arm(r)))
    wanted = [("1D Burgers\nnu=0.02, w64", "burgers|nu=0.02|w=64|6000", "baseline"),
              ("1D Burgers\nnu=0.02, w64", "burgers|nu=0.02|w=64|6000", "align: spectral prior"),
              ("1D Burgers\nnu=0.005, w64", "burgers|nu=0.005|w=64|6000", "baseline"),
              ("1D Burgers\nnu=0.005, w64", "burgers|nu=0.005|w=64|6000", "align: spectral prior"),
              ("2D Burgers\nw32", "burgers2d|nu=default|w=32|6000", "baseline"),
              ("2D Burgers\nw32", "burgers2d|nu=default|w=32|6000", "align: spectral prior"),
              ("2D Burgers\nw32", "burgers2d|nu=default|w=32|6000", "Fourier-feature trunk")]
    labels, lbs, opts, gens = [], [], [], []
    for lab, ck, ak in wanted:
        rs = [r for r in groups.get((ck, ak), []) if r["train"] is not None and r["lb"] is not None]
        if not rs:
            continue
        lb = np.mean([r["lb"] for r in rs])
        tr = np.mean([r["train"] for r in rs])
        te = np.mean([r["test"] for r in rs])
        labels.append(f"{lab}\n{ak}")
        lbs.append(lb)
        opts.append(max(tr - lb, 0.0))
        gens.append(te - tr)
    if not labels:
        print("  [skip] fig1: no runs with train_rel_l2")
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x, lbs, 0.62, label="LB (representation)", color="#4C72B0")
    ax.bar(x, opts, 0.62, bottom=lbs, label="opt_gap_train (optimization)", color="#DD8452")
    ax.bar(x, gens, 0.62, bottom=np.array(lbs) + np.array(opts),
           label="gen_gap (generalization)", color="#55A868")
    for i, (lb, op, ge) in enumerate(zip(lbs, opts, gens)):
        ax.text(i, lb + op + ge + 0.02, f"{lb+op+ge:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("test relative $L_2$")
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=7.5, loc="upper left", ncol=3, framealpha=0.9)
    ax.set_title("Three-way error attribution:  test = LB + opt_gap_train + gen_gap")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_attribution.png", dpi=200)
    plt.close(fig)
    print("  [ok] fig1_attribution.png")


# ---------------- 图 2：2D 干预对照 ----------------

def fig2(recs):
    rs = [r for r in recs if r["pde"] == "burgers2d" and r["steps"] == 6000]
    groups = agg(rs, arm)
    order = ["baseline", "orthogonalization reg.", "10x IC weight",
             "align: random smooth subspace", "align: spectral prior",
             "Fourier-feature trunk", "supervised (data only)"]
    names, means, sds, cols = [], [], [], []
    palette = {"baseline": "#C44E52", "orthogonalization reg.": "#C44E52",
               "10x IC weight": "#C44E52", "align: random smooth subspace": "#8172B3",
               "align: spectral prior": "#8172B3", "Fourier-feature trunk": "#55A868",
               "supervised (data only)": "#937860"}
    for k in order:
        if k not in groups:
            continue
        m, s = mstd([r["test"] for r in groups[k]])
        names.append(k)
        means.append(m)
        sds.append(s)
        cols.append(palette.get(k, "#4C72B0"))
    if not names:
        return
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x, means, 0.62, yerr=sds, capsize=3, color=cols,
           error_kw={"elinewidth": 1, "alpha": 0.9})
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=8)
    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.text(len(names) - 0.4, 1.01, "trivial $u\\equiv0$", fontsize=7, color="grey", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("orthogonalization reg.", "ortho reg.")
                        .replace("align: random smooth subspace", "align:\nrandom smooth")
                        .replace("align: spectral prior", "align:\nspectral prior")
                        .replace("supervised (data only)", "supervised\n(data only)")
                        .replace("Fourier-feature trunk", "Fourier\ntrunk")
                        .replace("10x IC weight", "10x IC\nweight")
                        for n in names], fontsize=7.5)
    ax.set_ylabel("test relative $L_2$")
    ax.set_ylim(0, 1.2)
    ax.set_title("2D Burgers, width 32, 6000 steps, 5 seeds (mean $\\pm$ sd)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_2d_controls.png", dpi=200)
    plt.close(fig)
    print("  [ok] fig2_2d_controls.png")


# ---------------- 图 3：bug 前后 ----------------

def fig3():
    data = {"diffusion–reaction": (0.9857, 0.1737, 0.9774, 0.9999),
            "Burgers (1D)": (0.9992, 0.7213, 0.6653, 0.9471)}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    names = list(data)
    for ax, idx, title in zip(axes, (0, 1), ("test relative $L_2$", "trunk captured energy")):
        if idx == 0:
            b = [data[n][0] for n in names]
            a = [data[n][1] for n in names]
            ax.set_ylim(0, 1.1)
        else:
            b = [data[n][2] for n in names]
            a = [data[n][3] for n in names]
            ax.set_ylim(0, 1.1)
        x = np.arange(len(names))
        ax.bar(x - 0.18, b, 0.34, label="buggy", color="#C44E52")
        ax.bar(x + 0.18, a, 0.34, label="fixed", color="#55A868")
        for i, (bb, aa) in enumerate(zip(b, a)):
            ax.text(i - 0.18, bb + 0.03, f"{bb:.3f}", ha="center", fontsize=7)
            ax.text(i + 0.18, aa + 0.03, f"{aa:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.5)
        ax.set_title(title)
        ax.legend(fontsize=7.5)
    fig.suptitle("Effect of fixing the one-line broadcasting bug", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_bug.png", dpi=200)
    plt.close(fig)
    print("  [ok] fig3_bug.png")


# ---------------- 图 4：方差地板 ----------------

def fig4(repeat_sigma=None):
    p = ROOT / "results" / "phase_analysis.json"
    if not p.exists():
        print("  [skip] fig4: no phase_analysis.json")
        return
    tab = json.load(open(p, encoding="utf-8"))["table"]
    tb = np.array([t["base_std"] for t in tab])
    ta = np.array([t["align_std"] for t in tab])
    gain = np.array([t["gain_pct"] for t in tab])
    is2d = np.array([t["setting"].startswith("burgers2d") for t in tab])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axes[0]
    m = ~is2d
    ax.scatter(tb[m], ta[m], s=28, color="#4C72B0", zorder=3)
    for x, y in zip(tb[m], ta[m]):
        ax.plot([x, x], [y, y], color="grey", lw=0.4, zorder=1)
    lo = np.logspace(np.log10(max(tb.min(), 1e-3) * 0.7), np.log10(tb.max() * 1.4), 50)
    ax.plot(lo, lo, "k--", lw=0.8, label="$\\sigma_{align}=\\sigma_{base}$")
    ax.axhspan(0.006, 0.014, color="#55A868", alpha=0.18,
               label="aligned band $\\approx0.006$–$0.014$")
    if repeat_sigma is not None:
        if repeat_sigma > 0:
            ax.axhline(repeat_sigma, color="#C44E52", ls=":", lw=1.2,
                       label=f"rerun noise $\\sigma_{{repeat}}$={repeat_sigma:.4f}")
        else:
            # σ_repeat = 0 无法画在对数轴上，改为文字标注
            ax.text(0.02, 0.06, "rerun noise $\\sigma_{repeat}=0$\n(bit-identical reruns)",
                    transform=ax.transAxes, fontsize=7, color="#C44E52")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("baseline cross-seed sd")
    ax.set_ylabel("aligned cross-seed sd")
    ax.legend(fontsize=6.8, loc="upper left")
    ax.set_title("(a) variance clamp")

    ax = axes[1]
    ax.scatter(tb[m], gain[m], s=28, color="#8172B3")
    ax.set_xscale("log")
    ax.set_xlabel("baseline cross-seed sd")
    ax.set_ylabel("accuracy gain (%)")
    r = (np.corrcoef(tb[m], gain[m])[0, 1] if len(tb[m]) > 2 else float("nan"))
    ax.set_title(f"(b) gain vs baseline sd: $r$={r:.2f} (retracted law)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_varfloor.png", dpi=200)
    plt.close(fig)
    print("  [ok] fig4_varfloor.png")


# ---------------- 图 5：机制 ----------------

def fig5():
    p = ROOT / "results" / "ntk.json"
    if not p.exists():
        print("  [skip] fig5: no ntk.json")
        return
    rows = json.load(open(p, encoding="utf-8"))
    rows = [r for r in rows if r.get("align_kind") in ("none", "prior")]
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    for ax, (key, sub, title) in zip(axes, [
            ("output_ntk", "eff_rank_entropy", "output-NTK effective rank"),
            ("output_ntk", "top1_share", "NTK top-eigenvalue share"),
            ("hessian", "hess_lambda_max", "$\\lambda_{max}(\\nabla^2 L)$")]):
        base = [r[key][sub] for r in rows if r["align_kind"] == "none"]
        al = [r[key][sub] for r in rows if r["align_kind"] == "prior"]
        if not base or not al:
            continue
        ax.bar([0], [np.mean(base)], 0.5, color="#C44E52", yerr=np.std(base, ddof=1) if len(base) > 1 else 0)
        ax.bar([1], [np.mean(al)], 0.5, color="#55A868", yerr=np.std(al, ddof=1) if len(al) > 1 else 0)
        for i, v in enumerate(base):
            ax.scatter([0], [v], color="k", s=12, zorder=3)
        for i, v in enumerate(al):
            ax.scatter([1], [v], color="k", s=12, zorder=3)
        if key == "hessian":
            ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["baseline", "aligned"], fontsize=8)
        ax.set_title(title, fontsize=8.5)
    fig.suptitle("Alignment flattens the landscape and widens the NTK spectrum", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_mechanism.png", dpi=200)
    plt.close(fig)
    print("  [ok] fig5_mechanism.png")


# ---------------- 图 6：1D 方法对照 ----------------

def fig6(recs):
    rs = [r for r in recs if r["pde"] == "burgers" and r["width"] == 64
          and r["steps"] == 10000 and r["nu"] in (0.02, None)]
    groups = agg(rs, arm)
    if not groups:
        print("  [skip] fig6: no 10000-step 1D runs")
        return
    order = sorted(groups, key=lambda k: np.mean([r["test"] for r in groups[k]]))
    means = [np.mean([r["test"] for r in groups[k]]) for k in order]
    sds = [np.std([r["test"] for r in groups[k]], ddof=1) if len(groups[k]) > 1 else 0.0
           for k in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.barh(x, means, 0.62, xerr=sds, capsize=3, color="#4C72B0")
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(m + s + 0.005, i, f"{m:.4f}", va="center", fontsize=7.5)
    ax.set_yticks(x)
    ax.set_yticklabels(order, fontsize=7.5)
    ax.set_xlabel("test relative $L_2$")
    ax.set_xlim(0.6, 0.78)
    ax.set_title("1D Burgers, width 64, 10000 steps (single seed; §5.3)")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_1d_ablation.png", dpi=200)
    plt.close(fig)
    print("  [ok] fig6_1d_ablation.png")


def fig7(recs):
    """预算对比：物理 vs 监督，随步数的变化（§5.8 核心图）。"""
    def series(pde, arm_name, steps_list, **kw):
        xs, ys, ss = [], [], []
        for st in steps_list:
            rs = [r for r in recs if r["pde"] == pde and arm(r) == arm_name
                  and r["steps"] == st and kw.get("nu", r["nu"]) == r["nu"]]
            if pde == "burgers":
                rs = [r for r in rs if (r["nu"] or 0.02) == 0.02 and r["width"] == 64]
            else:
                rs = [r for r in rs if r["width"] == 32]
            if not rs:
                continue
            m, s = mstd([r["test"] for r in rs])
            xs.append(st)
            ys.append(m)
            ss.append(s)
        return xs, ys, ss

    steps_list = [6000, 20000, 30000, 50000]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
    for ax, pde, title in zip(axes, ("burgers", "burgers2d"),
                              ("1D Burgers, $\\nu$=0.02, w64", "2D Burgers, w32")):
        for arm_name, col, mk in (
                (("Fourier-feature trunk" if pde == "burgers2d" else "baseline"),
                 "#55A868", "o"),
                ("supervised (data only)", "#937860", "s")):
            xs, ys, ss = series(pde, arm_name, steps_list)
            if not xs:
                continue
            lab = ("physics-informed" if "Fourier" in arm_name or arm_name == "baseline"
                   else "fully supervised")
            ax.errorbar(xs, ys, yerr=ss, marker=mk, color=col, label=lab,
                        capsize=3, lw=1.4, ms=5)
        ax.set_xscale("log")
        ax.set_xticks([6000, 20000, 50000])
        ax.set_xticklabels(["6k", "20k", "50k"])
        ax.minorticks_off()
        ax.set_xlabel("training steps")
        ax.set_ylabel("test relative $L_2$")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7.5)
    fig.suptitle("Budget-matched comparison: physics-informed vs fully supervised", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_budget.png", dpi=200)
    plt.close(fig)
    print("  [ok] fig7_budget.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="expV,expW,expQ,expS,expL,expN,expT,expU,"
                                       "expDO,expDOlong,expDO50k,expS20k,expS50k")
    ap.add_argument("--repeat-sigma", type=float, default=None,
                    help="同 seed 重跑的标准差，画进 fig4 作为噪声地板参考线")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load([d.strip() for d in args.dirs.split(",") if d.strip()])
    print(f"[data] {len(recs)} runs")
    fig1(recs)
    fig2(recs)
    fig3()
    fig4(args.repeat_sigma)
    fig5()
    fig6(recs)
    fig7(recs)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
