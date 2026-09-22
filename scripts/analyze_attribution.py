"""三项误差归因：把 test relL2 拆成「表示 / 优化 / 泛化」三部分。

    test_relL2 = LB + opt_gap_train + gen_gap

    LB            = sqrt(1 - captured_energy)  表示下界（任何 branch 都跨不过去）
    opt_gap_train = train_relL2 - LB          优化 / branch 容量差距
    gen_gap       = test_relL2 - train_relL2   对新输入函数的泛化差距

用法：
  python scripts/analyze_attribution.py --dirs expQ,expS,expT,expV
  python scripts/analyze_attribution.py --dirs expV --pde burgers2d --verbose
  python scripts/analyze_attribution.py --dirs expV --out results/attribution.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAN = float("nan")


def read_one(f: str):
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception:
        return None
    if "meta" not in d or "test_rel_l2" not in d:
        return None
    c = d["meta"]["cfg"]
    mkw = c.get("model_kwargs") or {}
    pk = c.get("pde_kwargs") or {}
    tr = d.get("train_rel_l2")
    return {
        "file": Path(f).name,
        "dir": Path(f).parent.name,
        "pde": c.get("pde"),
        "method": c.get("method"),
        "align_kind": c.get("align_kind"),
        "seed": c.get("seed"),
        "width": mkw.get("width") or pk.get("width"),
        "nu": pk.get("nu"),
        "trunk_type": mkw.get("trunk_type", "mlp"),
        "hard_ic": bool(pk.get("hard_ic", False)),
        "ortho": float(c.get("ortho_weight") or 0.0),
        "steps": c.get("steps"),
        "lb": d.get("lb_final"),
        "train": tr,
        "test": d["test_rel_l2"],
        "captured": (d.get("final_diag") or {}).get("captured_energy"),
        # 旧 run（2026-09-21 之前）没有 train_rel_l2，无法做三项分解
        "complete": tr is not None and d.get("lb_final") is not None,
    }


def arm_name(r) -> str:
    """把 (方法 + trunk 架构 + 是否硬约束 IC) 归并成一个「干预手段」名。"""
    if r["trunk_type"] != "mlp":
        return f"fourier-trunk/{r['method']}"
    if r["ortho"] > 0:
        return "ortho"
    if r["method"] == "pi_baseline":
        return "baseline" + ("+hardIC" if r["hard_ic"] else "")
    kind = r["align_kind"]
    if kind in (None, "none"):
        return r["method"] + ("+hardIC" if r["hard_ic"] else "")
    if r["hard_ic"]:
        return f"align {kind}+hardIC"
    return f"align {kind}"


def setting_name(r) -> str:
    nu = r["nu"] if r["nu"] is not None else "default"
    return f"{r['pde']}|nu={nu}|w={r['width']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="expV",
                    help="逗号分隔的 results 子目录（只放 bug 修复之后的目录）")
    ap.add_argument("--pde", default="", help="只保留该 PDE（如 burgers2d）")
    ap.add_argument("--out", default="", help="可选：把汇总写成 JSON")
    ap.add_argument("--verbose", action="store_true", help="打印逐 run 原始值")
    args = ap.parse_args()

    recs = []
    for d in [x.strip() for x in args.dirs.split(",") if x.strip()]:
        for f in glob.glob(str(ROOT / "results" / d / "**" / "*.json"), recursive=True):
            r = read_one(f)
            if r is None:
                continue
            if args.pde and r["pde"] != args.pde:
                continue
            recs.append(r)

    if not recs:
        print("no runs found")
        return

    if args.verbose:
        print("[raw]  (train=nan 表示该 run 早于三项分解，无法参与归因)")
        for r in sorted(recs, key=lambda x: (setting_name(x), arm_name(x), x["seed"])):
            print(f"  {r['dir']:6s} {setting_name(r):28s} {arm_name(r):16s} "
                  f"s{r['seed']} steps={r['steps']} "
                  f"LB={fmt(r['lb'])} train={fmt(r['train'])} test={r['test']:.4f}"
                  f"{'' if r['complete'] else '   [incomplete]'}")

    # 同一 (setting, arm, seed) 可能被重复跑过（如 expU / expV 两份 2D 结果），
    # 也与 analyze_phase.py 保持一致：先按 seed 取平均去重，避免 n 被重复计数
    dedup: dict = {}
    for r in recs:
        dedup.setdefault((setting_name(r), arm_name(r), r["seed"]), []).append(r)
    recs = []
    for k, rs in dedup.items():
        base = dict(rs[0])
        for key in ("lb", "train", "test", "captured"):
            vals = [x[key] for x in rs if x[key] is not None]
            base[key] = sum(vals) / len(vals) if vals else None
        recs.append(base)

    # 按 (setting, arm) 聚合
    groups: dict = {}
    for r in recs:
        groups.setdefault((setting_name(r), arm_name(r)), []).append(r)

    print(f"\n{'setting':30s} {'arm':18s} {'n':>2s} "
          f"{'LB':>7s} {'train':>7s} {'test':>7s} "
          f"{'opt(LB->train)':>15s} {'gen(train->test)':>16s}")
    print("-" * 108)
    rows = []
    for (sk, ak), rs in sorted(groups.items()):
        ok = [r for r in rs if r["complete"]]
        if not ok:
            print(f"{sk:30s} {ak:18s} {len(rs):2d}   （缺 train_rel_l2，跳过）")
            continue
        lb = float(np.mean([r["lb"] for r in ok]))
        tr = float(np.mean([r["train"] for r in ok]))
        te = float(np.mean([r["test"] for r in ok]))
        opt = tr - lb
        gen = te - tr
        sd = float(np.std([r["test"] for r in ok], ddof=1)) if len(ok) > 1 else 0.0
        print(f"{sk:30s} {ak:18s} {len(ok):2d} "
              f"{lb:7.4f} {tr:7.4f} {te:7.4f} "
              f"{opt:15.4f} {gen:16.4f}   test_std={sd:.4f}")
        rows.append({"setting": sk, "arm": ak, "n": len(ok), "lb": lb,
                     "train": tr, "test": te, "opt_gap_train": opt,
                     "gen_gap": gen, "test_std": sd,
                     "n_incomplete": len(rs) - len(ok)})

    if rows:
        w = np.array([r["test"] for r in rows])          # 用 test 误差做权重
        w = w / w.sum()
        agg = {k: float(np.sum(w * np.array([r[k] for r in rows])))
               for k in ("lb", "opt_gap_train", "gen_gap")}
        tot = sum(agg.values())
        print(f"\n[aggregate] 按 test 误差加权平均（n={len(rows)} 组）")
        for k in ("lb", "opt_gap_train", "gen_gap"):
            print(f"   {k:14s} {agg[k]:.4f}   ({agg[k] / tot * 100:5.1f}% of total)")
        print(f"   {'total':14s} {tot:.4f}")
        print("\n判读：gen_gap 大 ⇒ 对新输入函数的泛化受限（不是优化问题）；"
              "\n      opt_gap_train 大 ⇒ 优化/branch 容量受限；"
              "\n      LB 大 ⇒ 表示受限。")

    if args.out:
        outp = ROOT / args.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"rows": rows}, open(outp, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        print(f"\n[saved] {outp}")


def fmt(x):
    return "   nan" if x is None else f"{x:6.4f}"


if __name__ == "__main__":
    main()
