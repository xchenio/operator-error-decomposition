"""实验入口：按 (pde × method × n_data × seed) 网格跑，并保存结果。

示例：
  python scripts/run.py --pde dr --methods pi_baseline,pi_align --n-data 0 --seeds 0,1 --steps 20000
  python scripts/run.py --pde burgers --methods pi_baseline,pi_align,two_stage_freeze --n-data 0,8 --seeds 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (MODEL_DEFAULTS, ExpCfg, build_targets, make_probe_idx,  # noqa: E402
                        make_stages, pick_data_idx)
from src.data import build_dataset, dataset_cache_path  # noqa: E402
from src.models import DeepONet  # noqa: E402
from src.pdes import make_problem  # noqa: E402
from src.trainer import Trainer  # noqa: E402
from src.utils import (Logger, code_fingerprint, count_params,  # noqa: E402
                       ensure_dir, env_fingerprint, get_device, set_seed)

MAX_PROBE = 4096

# 复现性指纹（论文 §5.9）：由 main() 填充后写进每个 run 的 meta。
# 同配置两次结果不一致时，先比对 code_hash 再怀疑数值非确定性。
CODE_HASH = ""
ENV = {}


def run_one(cfg: ExpCfg, pde, splits, device, out_dir: Path, logger,
            save_model: bool = False) -> dict:
    pkw, mkw, skw = cfg.resolved()
    set_seed(cfg.seed)

    model = DeepONet(branch_in=pde.n_sensors, trunk_in=pde.trunk_dim, **mkw)

    probe_idx = make_probe_idx(pde.grid.shape[0], MAX_PROBE, cfg.seed)
    data_idx = pick_data_idx(len(splits["train"]), cfg.n_data, cfg.seed)
    align_U, align_w, diag_U, diag_w = build_targets(
        pde, splits, probe_idx, data_idx, mkw["width"], cfg.align_kind)

    trainer = Trainer(model, pde, splits, device=device, seed=cfg.seed,
                      align_U=align_U, align_w=align_w,
                      diag_U=diag_U, diag_w=diag_w,
                      probe_coords=pde.grid[probe_idx],
                      n_data=cfg.n_data, data_split=cfg.data_split,
                      grad_clip=cfg.grad_clip, logger=logger)

    stages = make_stages(cfg, skw)
    t0 = time.time()
    res = trainer.fit(stages, bs=skw["bs"], eval_every=cfg.eval_every,
                      diag_every=cfg.diag_every)
    res["meta"] = {
        "cfg": {k: (v if not isinstance(v, dict) else dict(v))
                for k, v in vars(cfg).items()},
        "n_params": count_params(model),
        "n_probe": int(len(probe_idx)),
        "n_data_used": int(len(data_idx)),
        "align_rank": int(align_U.shape[1]) if align_U is not None else 0,
        "diag_rank": int(diag_U.shape[1]),
        "stages": [{"kind": s.kind, "steps": s.steps, "align": s.align,
                    "freeze_trunk": s.freeze_trunk, "trunk_lr_mult": s.trunk_lr_mult,
                    "lambdas": s.lambdas} for s in stages],
        "wall_time": time.time() - t0,
        # 复现性指纹（论文 §5.9）：同配置两次结果不一致时，先比对 code_hash
        "code_hash": CODE_HASH,
        "env": ENV,
    }
    tag = cfg.tag()
    with open(out_dir / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    if save_model:
        torch.save({"state_dict": model.state_dict(),
                    "model_kwargs": mkw,
                    "cfg": {k: (v if not isinstance(v, dict) else dict(v))
                            for k, v in vars(cfg).items()},
                    "pde": cfg.pde,
                    "pde_kwargs": pkw},
                   out_dir / f"{tag}.pt")
    logger.log(f"[done] {tag} | test relL2={res['test_rel_l2']:.4f} "
               f"captured={res['final_diag']['captured_energy']:.4f} "
               f"({res['meta']['wall_time']:.0f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="dr",
                    choices=["dr", "burgers", "burgers2d", "darcy"])
    ap.add_argument("--methods", default="pi_baseline")
    ap.add_argument("--n-data", default="0", help="逗号分隔，例如 0,8,32")
    ap.add_argument("--align-kind", default="auto",
                    help="prior|data|hybrid|none；auto=按 n_data 自动选")
    ap.add_argument("--align-weight", default="1.0", help="逗号分隔")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--diag-every", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--out", default="results/exp1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pde-kwargs", default="{}", help='JSON，例如 {"nu":0.005}')
    ap.add_argument("--hard-ic", action="store_true",
                    help="Burgers：u=u0+t*NN 的硬约束 IC 抬升")
    ap.add_argument("--nu", type=float, default=0.0, help="Burgers 粘性系数（覆盖默认）")
    ap.add_argument("--nx", type=int, default=0, help="Burgers 求解网格点数（覆盖默认）")
    ap.add_argument("--nx-store", type=int, default=0, help="Burgers 存储网格点数")
    ap.add_argument("--width", type=int, default=0, help="0=用默认值")
    ap.add_argument("--steps-col", type=int, default=0)
    ap.add_argument("--bs", type=int, default=0)
    ap.add_argument("--n-ic", type=int, default=0)
    ap.add_argument("--n-bc", type=int, default=0)
    ap.add_argument("--ic-weight", default="", help="逗号分隔，初始/边界项权重，如 1,10,100")
    ap.add_argument("--init-gain", type=float, default=1.0)
    ap.add_argument("--ortho-weight", default="0.0", help="逗号分隔，trunk 去相关正则权重")
    ap.add_argument("--trunk-type", default="mlp", choices=["mlp", "fourier"])
    ap.add_argument("--causal", default="0.0", help="逗号分隔，因果加权 eps（0=关闭）")
    ap.add_argument("--causal-M", type=int, default=20, help="时间方向分块数")
    ap.add_argument("--lbfgs-steps", type=int, default=200, help="L-BFGS 外层步数")
    ap.add_argument("--save-model", action="store_true", help="保存最终 state_dict 供 NTK 分析")
    ap.add_argument("--skip-existing", action="store_true",
                    help="若该配置的 JSON 已存在则跳过（使重跑幂等，适合长批量中断后续跑）")
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    device = (get_device(True) if args.device == "auto" else torch.device(args.device))
    global CODE_HASH, ENV
    CODE_HASH = code_fingerprint(ROOT)
    ENV = env_fingerprint(device)
    out_dir = ensure_dir(ROOT / args.out)
    logger = Logger(out_dir / "log.txt", verbose=not args.quiet)
    logger.log(f"[fingerprint] code_hash={CODE_HASH} env={ENV}")

    pkw = json.loads(args.pde_kwargs)
    if args.hard_ic:
        pkw["hard_ic"] = True
    if args.nu:
        pkw["nu"] = args.nu
    if args.nx:
        pkw["Nx"] = args.nx
    if args.nx_store:
        pkw["Nx_store"] = args.nx_store
    pde = make_problem(args.pde, **pkw)
    logger.log(f"[pde] {pde.describe()}  kwargs={pkw}")
    logger.log(f"[device] {device}")

    cache = dataset_cache_path(ROOT, args.pde, pkw, args.n_train, args.n_val,
                               args.n_test, seed=0)
    splits = build_dataset(pde, args.n_train, args.n_val, args.n_test, seed=0,
                           cache=str(cache), force_regen=args.regen)

    grid_t = torch.as_tensor(pde.grid, dtype=torch.float32)
    logger.log(f"[data] grid={pde.grid.shape}, train={len(splits['train'])}, "
               f"val={len(splits['val'])}, test={len(splits['test'])}")

    extra_model = {"init_gain": args.init_gain, "trunk_type": args.trunk_type}
    if args.width:
        extra_model["width"] = args.width
    extra_sampling = {}
    if args.steps_col:
        extra_sampling["n_col"] = args.steps_col
    if args.bs:
        extra_sampling["bs"] = args.bs
    if args.n_ic:
        extra_sampling["n_ic"] = args.n_ic
    if args.n_bc:
        extra_sampling["n_bc"] = args.n_bc

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    nds = [int(x) for x in args.n_data.split(",")]
    aks = [x.strip() for x in args.align_kind.split(",")]
    aws = [float(x) for x in args.align_weight.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    icws = [float(x) for x in args.ic_weight.split(",")] if args.ic_weight else [None]
    ows = [float(x) for x in args.ortho_weight.split(",")]
    caus = [float(x) for x in args.causal.split(",")]

    all_res = []
    for nd in nds:
        for method in methods:
            for icw in icws:
                for ow in ows:
                    for cz in caus:
                        for ak in aks:
                            for aw in aws:
                                for sd in seeds:
                                    kind = ak
                                    if kind == "auto":
                                        kind = "prior" if nd == 0 else "data"
                                    if method in ("pi_baseline", "data_only", "pi_ortho",
                                                  "pi_lbfgs", "two_stage_freeze",
                                                  "two_stage_slow", "two_stage_data",
                                                  "pi_data"):
                                        kind = "none"
                                    # 各方法只跑自己关心的参数组合，避免重复
                                    if kind == "none" and (aw != aws[0] or ak != aks[0]):
                                        continue
                                    if method not in ("pi_ortho", "pi_align_ortho") \
                                            and ow != ows[0]:
                                        continue
                                    if method in ("pi_ortho", "pi_align_ortho") and ow == 0.0:
                                        continue
                                    causal_m = method.startswith("pi_causal")
                                    if not causal_m and cz != caus[0]:
                                        continue
                                    if causal_m and cz == 0.0:
                                        continue
                                    lam = {}
                                    if icw is not None:
                                        lam = {"pde": 1.0, "ic": icw, "bc": icw}
                                    cfg = ExpCfg(pde=args.pde, method=method, n_data=nd,
                                                 seed=sd, align_kind=kind, align_weight=aw,
                                                 ortho_weight=ow, causal=cz,
                                                 causal_M=args.causal_M,
                                                 lbfgs_steps=args.lbfgs_steps,
                                                 steps=args.steps, lr=args.lr,
                                                 eval_every=args.eval_every,
                                                 diag_every=args.diag_every,
                                                 n_train=args.n_train, n_val=args.n_val,
                                                 n_test=args.n_test,
                                                 pde_kwargs=pkw, model_kwargs=extra_model,
                                                 sampling_kwargs=extra_sampling,
                                                 lambdas_override=lam)
                                    # tag 必须唯一区分一次运行（含损失权重），否则
                                    # --skip-existing 失效、不同超参会互相覆盖同名 JSON
                                    tag_show = cfg.tag()
                                    if args.skip_existing and (out_dir / f"{tag_show}.json").exists():
                                        logger.log(f"\n=== {tag_show} ===")
                                        logger.log("  [skip] 已有结果")
                                        try:
                                            d = json.load(open(out_dir / f"{tag_show}.json",
                                                               encoding="utf-8"))
                                            all_res.append({
                                                "tag": tag_show, "pde": args.pde,
                                                "method": method, "n_data": nd,
                                                "align_kind": kind,
                                                "align_weight": aw, "ortho_weight": ow,
                                                "causal": cz, "ic_weight": icw,
                                                "seed": sd,
                                                "width": extra_model.get(
                                                    "width", MODEL_DEFAULTS[args.pde]["width"]),
                                                "test_rel_l2": d["test_rel_l2"],
                                                "val_rel_l2": d["val_rel_l2"],
                                                "train_rel_l2": d.get(
                                                    "train_rel_l2", float("nan")),
                                                "lb_init": d["lb_init"],
                                                "lb_final": d["lb_final"],
                                                "opt_gap": d["opt_gap"],
                                                "opt_gap_train": d.get(
                                                    "opt_gap_train", float("nan")),
                                                "gen_gap": d.get(
                                                    "gen_gap", float("nan")),
                                                "captured_init": d["init_diag"]["captured_energy"],
                                                "captured_final": d["final_diag"]["captured_energy"],
                                                "eff_rank_final": d["final_diag"]["eff_rank"],
                                                "angle_mean_final": d["final_diag"]["angle_mean_deg"],
                                                "wall_time": d["meta"]["wall_time"],
                                            })
                                        except Exception:
                                            pass
                                        continue
                                    logger.log(f"\n=== {tag_show} ===")
                                    try:
                                        r = run_one(cfg, pde, splits, device, out_dir,
                                                    logger, save_model=args.save_model)
                                        r["_tag"] = tag_show
                                        all_res.append({
                                            "tag": tag_show, "pde": args.pde,
                                            "method": method, "n_data": nd,
                                            "align_kind": kind, "align_weight": aw,
                                            "ortho_weight": ow, "causal": cz,
                                            "ic_weight": icw, "seed": sd,
                                            "width": extra_model.get(
                                                "width", MODEL_DEFAULTS[args.pde]["width"]),
                                            "test_rel_l2": r["test_rel_l2"],
                                            "val_rel_l2": r["val_rel_l2"],
                                            "train_rel_l2": r["train_rel_l2"],
                                            "lb_init": r["lb_init"],
                                            "lb_final": r["lb_final"],
                                            "opt_gap": r["opt_gap"],
                                            "opt_gap_train": r["opt_gap_train"],
                                            "gen_gap": r["gen_gap"],
                                            "captured_init": r["init_diag"]["captured_energy"],
                                            "captured_final": r["final_diag"]["captured_energy"],
                                            "eff_rank_final": r["final_diag"]["eff_rank"],
                                            "angle_mean_final": r["final_diag"]["angle_mean_deg"],
                                            "wall_time": r["meta"]["wall_time"],
                                        })
                                    except Exception as e:
                                        import traceback
                                        logger.log(f"[FAILED] {tag_show}: {e}")
                                        traceback.print_exc()

    if all_res:
        import csv
        keys = list(all_res[0].keys())
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_res)
        logger.log(f"\n[saved] {out_dir/'summary.csv'}")
        for r in all_res:
            logger.log("  " + " | ".join(
                f"{k}={r[k]:.4f}" if isinstance(r[k], float) else f"{k}={r[k]}"
                for k in ("tag", "train_rel_l2", "test_rel_l2",
                          "captured_final", "eff_rank_final")))
    logger.close()


if __name__ == "__main__":
    main()
