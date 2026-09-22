"""决定性检查：用监督训练拟合真解后，物理残差是否真的很小？

若残差远小于 E[u^2]（平凡解 s≡0 的残差水平），说明 PDE 公式/实现正确，
纯物理学不会是优化问题；否则说明残差实现有 bug。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PDE_DEFAULTS  # noqa: E402
from src.data import build_dataset  # noqa: E402
from src.models import DeepONet  # noqa: E402
from src.pdes import make_problem  # noqa: E402
from src.trainer import Trainer, StageCfg  # noqa: E402
from src.utils import get_device, rel_l2, set_seed  # noqa: E402


def main():
    pde = make_problem("dr", **PDE_DEFAULTS["dr"])
    splits = build_dataset(pde, 200, 50, 50, seed=0,
                           cache=str(ROOT / "data/dr_default_n200_50_50_s0.npz"))
    dev = get_device(True)
    set_seed(0)
    model = DeepONet(branch_in=pde.n_sensors, trunk_in=2, width=100,
                     branch_hidden=128, branch_layers=3,
                     trunk_hidden=128, trunk_layers=3)

    from src.basis import build_data_subspace
    U_diag, w_diag = build_data_subspace(splits["val"].s.T, 100)
    tr = Trainer(model, pde, splits, device=dev, seed=0,
                 probe_coords=pde.grid, n_data=0,
                 diag_U=U_diag, diag_w=w_diag)
    # 只做监督训练
    tr.fit([StageCfg(kind="data", steps=4000, lr=1e-3, n_pts=256)],
           bs=16, eval_every=1000, diag_every=0)
    r2 = tr.evaluate("test")
    print(f"\n[supervised] test relL2 = {r2:.4f}")

    # ---- 用 autograd 计算物理残差 ----
    rng = np.random.default_rng(0)
    sp = splits["train"]
    idx = rng.choice(len(sp), 16, replace=False)
    coef = sp.coef[idx]
    u_b = torch.as_tensor(sp.u[idx], dtype=torch.float32, device=dev)
    pts = pde.sample_points(coef, 2000, 200, 200, rng)
    # 真解（插值到配点：这里直接在配点上重新解一次不现实，改用网格上的残差）
    evalr = model.evaluator(u_b)
    coords = torch.as_tensor(pts["col"]["coords"], dtype=torch.float32, device=dev)
    s, ds, d2s = evalr(coords)
    forcing = torch.as_tensor(pts["col"]["forcing"], dtype=torch.float32, device=dev)
    res = ds[..., 0] - pde.D * d2s[..., 1, 1] - pde.k * s ** 2 - forcing
    print(f"[autograd residual on trained model] mean(res^2) = {float((res**2).mean()):.3e}")
    print(f"   |s| mean = {float(s.abs().mean()):.3f}, |res| mean = {float(res.abs().mean()):.3e}")

    # ---- 参照：平凡解 s≡0 的残差 = E[u^2] ----
    print(f"[reference] s≡0 时残差 = E[u^2] = {float((forcing**2).mean()):.3e}")
    print(f"[reference] 真解 |s| 量级: std={splits['train'].s.std():.3f}")

    # ---- 用有限差分估计真解的残差（粗网格，仅作量级参考）----
    Nt, Nx = pde.grid_shape
    S = splits["train"].s[:8].reshape(8, Nt, Nx)
    x = np.linspace(0, 1, Nx)
    dt = pde.T / (Nt - 1)
    dx = x[1] - x[0]
    s_t = np.gradient(S, dt, axis=1)
    s_xx = np.gradient(np.gradient(S, dx, axis=2), dx, axis=2)
    u_at = pde.input_at(splits["train"].coef[:8], x)[:, None, :]
    res_fd = s_t - pde.D * s_xx - pde.k * S ** 2 - u_at
    print(f"[FD check on ground truth] mean(res^2) = {float((res_fd**2).mean()):.3e} "
          f"（粗网格 dt={dt:.4f}，含离散误差）")
    print(f"   |S| mean={np.abs(S).mean():.3f}, |u| mean={np.abs(u_at).mean():.3f}")


if __name__ == "__main__":
    main()
