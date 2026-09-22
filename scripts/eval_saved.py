"""给已保存的 checkpoint 补算 train / val / test relL2 与 LB（无需重训）。

背景：三项分解（LB / opt_gap_train / gen_gap）是 2026-09-21 才加进训练器的，
更早的 run 只有 val / test。本脚本用存档的 .pt + 同一份缓存数据集把 train 误差补上，
从而把旧结果也纳入归因表。

为保证与训练时完全一致，这里复用 Trainer.evaluate / Trainer.diagnostics，
并且 probe 点集用与训练相同的 make_probe_idx(seed)。

用法：
  python scripts/eval_saved.py --dir results/expV
  python scripts/eval_saved.py --dir results/expV --out results/attribution_2d.json
  python scripts/eval_saved.py --dir results/expV --write-back   # 回写进同名 .json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.basis import build_data_subspace            # noqa: E402
from src.config import make_probe_idx                # noqa: E402
from src.data import build_dataset, dataset_cache_path  # noqa: E402
from src.models import DeepONet                      # noqa: E402
from src.pdes import make_problem                    # noqa: E402
from src.trainer import Trainer                      # noqa: E402
from src.utils import get_device                     # noqa: E402

MAX_PROBE = 4096          # 必须与 scripts/run.py 保持一致


def eval_one(ckpt: Path, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    pkw = ck.get("pde_kwargs") or {}          # 已解析（含默认值）→ 用于重建 PDE
    pde = make_problem(ck["pde"], **pkw)
    model = DeepONet(branch_in=pde.n_sensors, trunk_in=pde.trunk_dim,
                     **ck["model_kwargs"])
    model.load_state_dict(ck["state_dict"])

    # 缓存路径必须按训练时的 CLI kwargs（cfg["pde_kwargs"]）来算，否则会命中不到
    # 已有缓存、把同一份数据重新解一遍（2D 约 885s）
    ckkw = cfg.get("pde_kwargs") or {}
    splits = build_dataset(pde, cfg["n_train"], cfg["n_val"], cfg["n_test"],
                           seed=0,
                           cache=str(dataset_cache_path(
                               ROOT, ck["pde"], ckkw, cfg["n_train"],
                               cfg["n_val"], cfg["n_test"], seed=0)))

    # 与训练时同一套 probe / 诊断目标，保证 LB 与存档的 lb_final 一致
    probe_idx = make_probe_idx(pde.grid.shape[0], MAX_PROBE, cfg["seed"])
    width = ck["model_kwargs"].get("width", 64)
    U, w = build_data_subspace(splits["val"].s[:, probe_idx].T, width)

    tr = Trainer(model, pde, splits, device=device, seed=cfg["seed"],
                 diag_U=U, diag_w=w, probe_coords=pde.grid[probe_idx], n_data=0)
    train = tr.evaluate("train")
    val = tr.evaluate("val")
    test = tr.evaluate("test")
    m = tr.diagnostics()
    lb = math.sqrt(max(0.0, 1.0 - m["captured_energy"]))

    return {
        "ckpt": ckpt.stem,
        "pde": ck["pde"],
        "method": cfg["method"],
        "align_kind": cfg.get("align_kind"),
        "seed": cfg["seed"],
        "width": width,
        "lb": lb,
        "train_rel_l2": train,
        "val_rel_l2": val,
        "test_rel_l2": test,
        "opt_gap_train": train - lb,
        "gen_gap": test - train,
        "captured": m["captured_energy"],
        "eff_rank": m["eff_rank"],
    }


def stored(ckpt: Path):
    """同名 .json 里存档的值，用于一致性核对（probe/数据集是否对得上）。"""
    jf = ckpt.with_suffix(".json")
    if not jf.exists():
        return None
    try:
        d = json.load(open(jf, encoding="utf-8"))
        return {"lb": d.get("lb_final"), "val": d.get("val_rel_l2"),
                "test": d.get("test_rel_l2")}
    except Exception:                                # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="results 下的子目录（含 .pt）")
    ap.add_argument("--out", default="", help="可选：汇总 JSON 输出路径")
    ap.add_argument("--write-back", action="store_true",
                    help="把 train/分解字段回写进同名 .json（会加 train_rel_l2_source 标记）")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = (get_device(True) if args.device == "auto" else torch.device(args.device))
    ckpts = sorted((ROOT / "results" / args.dir).glob("*.pt"))
    if not ckpts:
        print("no checkpoints found")
        return

    rows = []
    for p in ckpts:
        try:
            r = eval_one(p, device)
        except Exception as e:                      # noqa: BLE001
            print(f"  [FAILED] {p.name}: {e}")
            continue
        rows.append(r)
        s = stored(p)
        chk = ""
        if s and s["test"] is not None:
            chk = (f"   [存档 test={s['test']:.4f} LB={s['lb']:.4f} "
                   f"Δtest={r['test_rel_l2'] - s['test']:+.2e}]")
        print(f"  {r['ckpt'][:52]:52s} LB={r['lb']:.4f} train={r['train_rel_l2']:.4f} "
              f"val={r['val_rel_l2']:.4f} test={r['test_rel_l2']:.4f} "
              f"opt={r['opt_gap_train']:+.4f} gen={r['gen_gap']:+.4f}{chk}")

        if args.write_back:
            jf = p.with_suffix(".json")
            if jf.exists():
                d = json.load(open(jf, encoding="utf-8"))
                d["train_rel_l2"] = r["train_rel_l2"]
                d["opt_gap_train"] = r["opt_gap_train"]
                d["gen_gap"] = r["gen_gap"]
                d["train_rel_l2_source"] = "eval_saved.py (recomputed from checkpoint)"
                json.dump(d, open(jf, "w", encoding="utf-8"), indent=2)
                print(f"       -> wrote back into {jf.name}")

    if rows:
        print(f"\n[n={len(rows)}] 均值：")
        for k in ("lb", "train_rel_l2", "test_rel_l2", "opt_gap_train", "gen_gap"):
            v = sum(r[k] for r in rows) / len(rows)
            print(f"   {k:14s} {v:.4f}")

    if args.out:
        outp = ROOT / args.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"rows": rows}, open(outp, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
