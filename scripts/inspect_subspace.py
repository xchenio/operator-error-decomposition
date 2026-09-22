"""可行性体检：在训练之前，回答三个问题。

Q1. 解本身的"有效秩"是多少？（解空间有几个重要维度）
Q2. 无数据的解析先验基（sin/Chebyshev）与真实解子空间的重合度如何？
     —— 如果重合度极低，那么"零数据对齐"这条路就不成立。
Q3. 随机初始化的 trunk 基与真实解子空间的重合度如何？（基线起点）

用法：
  python scripts/inspect_subspace.py --pde dr
  python scripts/inspect_subspace.py --pde burgers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.basis import build_prior_basis  # noqa: E402
from src.config import PDE_DEFAULTS, make_probe_idx  # noqa: E402
from src.data import build_dataset  # noqa: E402
from src.diagnostics import (captured_energy_np, effective_rank,  # noqa: E402
                             principal_angles, singular_values, subspace_metrics)
from src.models import DeepONet  # noqa: E402
from src.pdes import make_problem  # noqa: E402

MAX_PROBE = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="dr", choices=["dr", "burgers", "darcy"])
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()

    pde = make_problem(args.pde, **PDE_DEFAULTS[args.pde])
    cache = ROOT / "data" / f"{args.pde}_n{args.n_train}_{args.n_val}_{args.n_test}_s0.npz"
    splits = build_dataset(pde, args.n_train, args.n_val, args.n_test, seed=0,
                           cache=str(cache))
    probe_idx = make_probe_idx(pde.grid.shape[0], MAX_PROBE, 0)
    probe = pde.grid[probe_idx]
    print(f"\n=== {pde.describe()} | probe={len(probe_idx)} ===")

    # ---- Q1 解的本征谱 ----
    S = splits["train"].s[:, probe_idx].T                       # (Np, n)
    sv = singular_values(S)
    print("\n[Q1] 解矩阵归一化奇异值（前 12 个）:")
    print("   ", np.array2string(sv[:12], precision=4, suppress_small=True))
    for k in (5, 10, 20, 40, 80):
        if k <= len(sv):
            energy = (sv[:k] ** 2).sum() / (sv ** 2).sum()
            print(f"    前 {k:3d} 个模态累计能量 = {energy:.4f}")
    print(f"    解矩阵有效秩 = {effective_rank(S):.2f}")

    # 真实子空间（用 train 解，仅用于分析）
    from src.basis import build_data_subspace, energy_rank
    r_true = max(1, min(args.width, energy_rank(sv, 0.999)))
    U_true, w_true = build_data_subspace(S, r_true)
    print(f"    99.9% 能量所需模态数 = {energy_rank(sv, 0.999)}"
          f" -> 取真实子空间维度 r = {U_true.shape[1]}")

    # ---- Q2 解析先验基 ----
    U_prior, w_prior = build_prior_basis(probe, pde.prior_spec, pde.ranges, args.width)
    m = subspace_metrics(U_prior, U_true, w_true)
    print("\n[Q2] 解析先验基（零数据） vs 真实解子空间:")
    print(f"    captured_energy = {m['captured_energy']:.4f}")
    print(f"    angle mean/max  = {m['angle_mean_deg']:.1f} / {m['angle_max_deg']:.1f} deg")
    print(f"    先验基维度      = {U_prior.shape[1]}")
    # 用不同数量的先验基
    for nk in (10, 20, 40, 60, 100):
        Up_n, _ = build_prior_basis(probe, pde.prior_spec, pde.ranges, nk)
        ce = captured_energy_np(Up_n, U_true, w_true)
        print(f"    取前 {nk:3d} 个先验基 -> captured = {ce:.4f}")

    # ---- Q3 随机初始化 trunk ----
    print("\n[Q3] 随机初始化 trunk 基 vs 真实解子空间:")
    for sd in [int(x) for x in args.seeds.split(",")]:
        torch.manual_seed(sd)
        model = DeepONet(branch_in=pde.n_sensors, trunk_in=pde.trunk_dim,
                         width=args.width, branch_hidden=128, branch_layers=3,
                         trunk_hidden=128, trunk_layers=3)
        with torch.no_grad():
            Phi = model.trunk_features(torch.as_tensor(probe, dtype=torch.float32))
        mm = subspace_metrics(Phi.numpy().astype(np.float64), U_true, w_true)
        print(f"    seed {sd}: captured={mm['captured_energy']:.4f} "
              f"eff_rank={mm['eff_rank']:.2f} angle_mean={mm['angle_mean_deg']:.1f}deg")

    # ---- 参考：用 train 解自己当目标（上界） ----
    print(f"\n[ref] 真实子空间自对齐 captured = "
          f"{captured_energy_np(U_true, U_true, w_true):.4f}（应为 1.0）")


if __name__ == "__main__":
    main()
