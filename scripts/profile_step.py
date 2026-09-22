"""把单步训练的时间拆开，回答"GPU 利用率为什么不高"。

分别计时：
  sample   : pde.sample_points() + 搬到 GPU（纯 CPU + H2D 拷贝，会强制同步）
  forward  : 构造 loss（含 functorch 求导，GPU）
  backward : loss.backward()（GPU）
  step     : optimizer.step()（GPU）
  eval     : 在验证集上评估一次（GPU + CPU）
  diag     : 基函数诊断（trunk 前向 GPU + SVD 在 CPU）

用法：python scripts/profile_step.py --pde burgers --width 64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExpCfg, build_targets, make_probe_idx, make_stages, pick_data_idx  # noqa: E402
from src.data import build_dataset   # noqa: E402
from src.models import DeepONet      # noqa: E402
from src.pdes import make_problem    # noqa: E402
from src.trainer import Trainer, _to_device   # noqa: E402
from src.utils import get_device     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="burgers")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n-col", type=int, default=64)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n-repeat", type=int, default=30)
    args = ap.parse_args()

    dev = get_device(True)
    pde = make_problem(args.pde)
    key = "default"
    splits = build_dataset(pde, 200, 50, 50, seed=0,
                           cache=str(ROOT / "data" / f"{args.pde}_{key}_n200_50_50_s0.npz"))
    cfg = ExpCfg(pde=args.pde, method="pi_baseline", seed=0,
                 model_kwargs={"width": args.width},
                 sampling_kwargs={"n_col": args.n_col, "bs": args.bs})
    _, mkw, skw = cfg.resolved()
    model = DeepONet(branch_in=pde.n_sensors, trunk_in=pde.trunk_dim, **mkw).to(dev)

    probe_idx = make_probe_idx(pde.grid.shape[0], 4096, 0)
    data_idx = pick_data_idx(len(splits["train"]), 0, 0)
    U, w, diag_U, diag_w = build_targets(pde, splits, probe_idx, data_idx, mkw["width"], "none")
    tr = Trainer(model, pde, splits, device=dev, seed=0, align_U=U, align_w=w,
                 diag_U=diag_U, diag_w=diag_w, probe_coords=pde.grid[probe_idx])
    stage = make_stages(cfg, skw)[0]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    sp = splits["train"]

    def one(do_backward=True):
        idx = tr._sample_train_batch(skwn["bs"])
        coef = sp.coef[idx]
        t0 = time.perf_counter()
        pts_np = pde.sample_points(coef, stage.n_col, stage.n_ic, stage.n_bc, tr.rng)
        pts = _to_device(pts_np, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_sample = time.perf_counter() - t0

        u_b = torch.as_tensor(sp.u[idx], dtype=torch.float32, device=dev)
        t0 = time.perf_counter()
        terms = tr.physics_terms(idx, stage)
        loss = sum(v for v in terms.values()
                   if torch.is_tensor(v) and v.dim() == 0)
        t_fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        if do_backward:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t0
        return t_sample, t_fwd, t_bwd

    skwn = skw
    # warmup
    for _ in range(5):
        one()

    ts = np.array([one() for _ in range(args.n_repeat)])
    n = args.n_repeat

    # 一次性开销
    t0 = time.perf_counter(); tr.evaluate("val"); 
    if dev.type == "cuda": torch.cuda.synchronize()
    t_eval = time.perf_counter() - t0
    t0 = time.perf_counter(); tr.diagnostics(); t_diag = time.perf_counter() - t0
    t0 = time.perf_counter()
    pde.sample_points(sp.coef[:skw["bs"]], stage.n_col, stage.n_ic, stage.n_bc, tr.rng)
    t_gen = time.perf_counter() - t0

    print(f"\npde={args.pde}  width={args.width}  n_col={args.n_col}  bs={args.bs}")
    print(f"  采样+搬GPU (CPU/H2D): {ts[:,0].mean()*1000:8.2f} ms   ({ts[:,0].mean()/ts.sum(1).mean()*100:5.1f}%)")
    print(f"  前向 (GPU,含求导)  : {ts[:,1].mean()*1000:8.2f} ms   ({ts[:,1].mean()/ts.sum(1).mean()*100:5.1f}%)")
    print(f"  反向+更新 (GPU)    : {ts[:,2].mean()*1000:8.2f} ms   ({ts[:,2].mean()/ts.sum(1).mean()*100:5.1f}%)")
    print(f"  ---- 单步合计      : {ts.sum(1).mean()*1000:8.2f} ms")
    print(f"  evaluate('val')    : {t_eval*1000:8.2f} ms   (每 eval_every 步一次)")
    print(f"  diagnostics()      : {t_diag*1000:8.2f} ms   (每 diag_every 步一次，SVD 在 CPU)")
    print(f"                       其中纯采样: {t_gen*1000:8.2f} ms")

    # 估算纯 GPU 时间占比
    gpu = ts[:,1] + ts[:,2]
    print(f"\n  => 单步内真正在 GPU 上的时间占比 ≈ {gpu.sum()/ts.sum()*100:.1f}%")
    print(f"  => 其余 {100-gpu.sum()/ts.sum()*100:.1f}% 是 CPU 采样 + H2D + 同步开销")
    print("  => 想提高 GPU 利用率：增大 n_col/bs（摊薄 launch），或减少每步 CPU 采样（预生成配点）")


if __name__ == "__main__":
    main()
