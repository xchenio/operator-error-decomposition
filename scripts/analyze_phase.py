"""相图与 r 的统计收尾。

关键点：直接把「基线跨种子 std」当作自变量去拟合 r 会**高估**相关性，
因为 std 本身是由 3 个 seed 估出来的、相对误差约 40% 的噪声量。
本脚本用 **seed 级参数化 bootstrap**：
  对每个设置，分别从该设置的 3 个 baseline / 3 个 aligned 结果中有放回重采样，
  重算 mean/std/gain，再跨设置重算 r。重复 N 次得到 r 的置信区间。

用法：
  python scripts/analyze_phase.py --root results --out results/phase_analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_WANTED = ["pi_baseline", "pi_align"]
NBOOT = 4000
# 只使用 2026-09-16 广播 bug 修复【之后】跑的目录；
# 修复前的 run（expA..expJ 等）结论已被证伪，绝不能混入统计。
POSTFIX_DIRS = ["expL", "expL2", "expN", "expQ", "expQ2", "expS", "expT",
                "expU", "expV"]


def load_runs(root: Path):
    recs = []
    for d in POSTFIX_DIRS:
        for f in glob.glob(str(root / d / "**" / "*.json"), recursive=True):
            rec = _read_one(f)
            if rec is not None:
                recs.append(rec)
    return recs


def _read_one(f):
    if "phase_analysis" in Path(f).name or "ntk" in Path(f).name:
        return None
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception:
        return None
    if "meta" not in d or "test_rel_l2" not in d:
        return None
    c = d["meta"]["cfg"]
    if c.get("method") not in SETTINGS_WANTED:
        return None
    if c.get("align_kind") not in ("none", "prior"):
        return None
    pk = c.get("pde_kwargs") or {}
    mkw = c.get("model_kwargs") or {}
    if mkw.get("trunk_type", "mlp") != "mlp":
        return None                                  # Fourier trunk 是另一个方法，不能混入
    if abs(float(mkw.get("init_gain", 1.0)) - 1.0) > 1e-9:
        return None
    if abs(float(c.get("ortho_weight") or 0.0)) > 1e-9:
        return None
    if abs(float(c.get("causal") or 0.0)) > 1e-9:
        return None
    return {
        "file": Path(f).name,
        "dir": Path(f).parent.name,
        "pde": c.get("pde"),
        "method": c.get("method"),
        "align_kind": c.get("align_kind"),
        "seed": c.get("seed"),
        "width": mkw.get("width") or pk.get("width"),
        "nu": pk.get("nu"),
        "steps": c.get("steps"),
        "relL2": d["test_rel_l2"],
    }


def group_key(r):
    pde = r["pde"]
    if pde == "burgers2d":
        return f"{pde}|nu={r['nu']}|w={r['width']}"
    nu = r["nu"] if r["nu"] is not None else 0.02
    return f"{pde}|nu={nu}|w={r['width']}"


def build_table(recs):
    # 同一 (设置, 方法, seed) 可能被重复跑过（且 CUDA 有非确定性），取平均以降噪
    acc = {}
    for r in recs:
        key = (group_key(r), r["method"], r["seed"])
        acc.setdefault(key, []).append(r["relL2"])
    groups = {}
    for (gk, m, s), vals in acc.items():
        groups.setdefault(gk, {"baseline": {}, "align": {}})
        kind = "baseline" if m == "pi_baseline" else "align"
        groups[gk][kind][s] = float(np.mean(vals))
    table = []
    for k, g in sorted(groups.items()):
        b, a = g["baseline"], g["align"]
        if len(b) < 2 or len(a) < 2:
            continue
        bv, av = np.array(list(b.values())), np.array(list(a.values()))
        mb, ma = bv.mean(), av.mean()
        table.append({
            "setting": k, "n_seed_base": len(bv), "n_seed_align": len(av),
            "base_mean": float(mb), "base_std": float(bv.std(ddof=1)),
            "align_mean": float(ma), "align_std": float(av.std(ddof=1)),
            "gain_pct": float((mb - ma) / mb * 100.0),
            "_bv": bv, "_av": av,
        })
    return table


def welch_p(x, y):
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(vx / nx + vy / ny)
    if se <= 0:
        return float("nan")
    t = (x.mean() - y.mean()) / se
    from scipy import stats
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return float(2 * stats.t.sf(abs(t), df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="results/phase_analysis.json")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="打印每个设置纳入的逐 seed 原始值")
    args = ap.parse_args()

    recs = load_runs(ROOT / args.root)
    if args.verbose:
        print("[raw]")
        for r in sorted(recs, key=lambda x: (group_key(x), x["method"], x["seed"])):
            print(f"  {r['dir']:8s} {group_key(r):28s} {r['method']:12s} "
                  f"s{r['seed']} steps={r['steps']:>6} relL2={r['relL2']:.4f}  {r['file'][:52]}")
    table = build_table(recs)
    if len(table) < 3:
        print(f"not enough settings yet ({len(table)})")
        for t in table:
            print("  ", t["setting"])
        return

    print(f"\n{'setting':46s} {'nB':>2s} {'nA':>2s} {'base mean±std':>22s} "
          f"{'align mean±std':>22s} {'gain%':>7s} {'p':>7s}")
    print("-" * 118)
    for t in table:
        p = welch_p(t["_bv"], t["_av"])
        t["welch_p"] = p
        print(f"{t['setting']:46s} {t['n_seed_base']:2d} {t['n_seed_align']:2d} "
              f"{t['base_mean']:10.4f} ± {t['base_std']:7.4f}   "
              f"{t['align_mean']:10.4f} ± {t['align_std']:7.4f}   "
              f"{t['gain_pct']:7.2f} {p:7.4f}")

    std = np.array([t["base_std"] for t in table])
    gain = np.array([t["gain_pct"] for t in table])
    r = float(np.corrcoef(std, gain)[0, 1])
    print(f"\n设置数 n = {len(table)}")
    print(f"Pearson r (base_std vs gain) = {r:.4f}")

    # ---- seed 级参数化 bootstrap ----
    rng = np.random.default_rng(args.seed)
    rs = []
    for _ in range(args.nboot):
        s_, g_ = [], []
        ok = True
        for t in table:
            bv = rng.choice(t["_bv"], size=len(t["_bv"]), replace=True)
            av = rng.choice(t["_av"], size=len(t["_av"]), replace=True)
            mb, ma = bv.mean(), av.mean()
            if mb <= 0:
                ok = False
                break
            s_.append(bv.std(ddof=1))
            g_.append((mb - ma) / mb * 100.0)
        if not ok:
            continue
        s_, g_ = np.array(s_), np.array(g_)
        if s_.std() < 1e-12 or g_.std() < 1e-12:
            continue
        rs.append(np.corrcoef(s_, g_)[0, 1])
    rs = np.array(rs)
    lo, hi = np.percentile(rs, [2.5, 97.5])
    print(f"seed 级 bootstrap (n={len(rs)}): r 的 95% CI = [{lo:.3f}, {hi:.3f}]  "
          f"(median {np.median(rs):.3f})")

    out = {
        "n_settings": len(table),
        "pearson_r": r,
        "pearson_r_ci95": [float(lo), float(hi)],
        "bootstrap_n": int(len(rs)),
        "table": [{k: v for k, v in t.items() if not k.startswith("_")}
                  for t in table],
    }
    outp = ROOT / args.out
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
