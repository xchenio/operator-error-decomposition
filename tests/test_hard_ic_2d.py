"""2D 硬约束 IC（u = u0 + t*NN）的回归测试。

覆盖三类最容易静默出错的地方：
  1. ic_values 的 u0 必须与 input_at、与求解器存储的第 0 个时间层完全一致；
  2. ic_values 的解析偏导必须与有限差分一致；
  3. _losses_hard 里的 PDE 残差必须与 torch autograd 独立算出的残差一致。

第 3 条就是本项目 README §6.5 那类"静默错误"的检测配方本身：
只用"真解的有限差分残差"检查不出来，必须拿 autograd 残差对账。

运行：
  python tests/test_hard_ic_2d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import DeepONet          # noqa: E402
from src.pdes import make_problem        # noqa: E402
from src.trainer import _to_device       # noqa: E402


def _u0_torch(coef_b, coords, K):
    """用 torch 独立实现 u0（与 numpy 的 ic_values 完全不同的代码路径）。"""
    x, y = coords[:, 1], coords[:, 2]
    k = torch.as_tensor(2 * np.pi * np.arange(1, K + 1), dtype=coords.dtype)
    A = torch.as_tensor(coef_b[: K * K].reshape(K, K), dtype=coords.dtype)
    B = torch.as_tensor(coef_b[K * K:].reshape(K, K), dtype=coords.dtype)
    Sx, Cx = torch.sin(x[:, None] * k), torch.cos(x[:, None] * k)
    Sy, Cy = torch.sin(y[:, None] * k), torch.cos(y[:, None] * k)
    return (torch.einsum("ij,ni,nj->n", A, Sx, Cy)
            + torch.einsum("ij,ni,nj->n", B, Cx, Sy))


def test_u0_matches_input_at():
    p = make_problem("burgers2d")
    coef = p.sample_coeffs(3, np.random.default_rng(0))

    # (a) 共享网格 (M, 3)
    M = np.stack([np.full(5, 0.4), np.linspace(0, 1, 5), np.linspace(0.2, 0.9, 5)],
                 axis=1)
    a = p.input_at(coef, M[:, 1:])          # (3, 5)
    b = p.ic_values(coef, M, 0)[0]          # (3, 5)
    assert a.shape == b.shape == (3, 5), (a.shape, b.shape)
    assert np.abs(a - b).max() < 1e-6, np.abs(a - b).max()

    # (b) 逐样本配对 (B, N, 3)
    B, N = 3, 7
    rng = np.random.default_rng(1)
    Pb = np.stack([rng.uniform(0, 1, (B, N)) for _ in range(3)], axis=2)
    a = p.input_at(coef, Pb[:, :, 1:])      # (3, 7)
    b = p.ic_values(coef, Pb, 0)[0]         # (3, 7)
    assert a.shape == b.shape == (B, N), (a.shape, b.shape)
    assert np.abs(a - b).max() < 1e-6, np.abs(a - b).max()
    print("  [ok] u0 == input_at（共享网格 + 逐样本配对）")


def test_u0_matches_solver_t0():
    """解析抬升项必须等于求解器存下来的第 0 个时间层。"""
    p = make_problem("burgers2d")
    coef = p.sample_coeffs(2, np.random.default_rng(3))
    s = p.solve(coef)                                  # (2, Nt*Ns*Ns)
    n0 = p.Ns * p.Ns
    s0 = s[:, :n0]                                     # t = 0 那一层
    u0 = p.ic_values(coef, p.grid[:n0], 0)[0]          # (2, n0)
    err = np.abs(s0 - u0).max()
    assert err < 1e-5, err
    print(f"  [ok] 解析 u0 == 求解器 t=0 层（max err {err:.2e}）")


def test_derivatives_vs_finite_difference():
    p = make_problem("burgers2d", hard_ic=True)
    coef = p.sample_coeffs(1, np.random.default_rng(5))
    h = 1e-5

    def f(x, y):
        return float(p.input_at(coef, np.array([[x, y]]))[0, 0])

    x0, y0 = 0.31, 0.77
    C = np.array([[0.0, x0, y0]])
    # 共享网格约定：输出 (n, M, 2)，n=1, M=1
    _, d1, d2 = p.ic_values(coef, C, order=2)
    u0x, u0y = d1[0, 0, 0], d1[0, 0, 1]
    u0xx, u0yy = d2[0, 0, 0], d2[0, 0, 1]

    fdx = (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h)
    fdy = (f(x0, y0 + h) - f(x0, y0 - h)) / (2 * h)
    fdxx = (f(x0 + h, y0) - 2 * f(x0, y0) + f(x0 - h, y0)) / h ** 2
    fdyy = (f(x0, y0 + h) - 2 * f(x0, y0) + f(x0, y0 - h)) / h ** 2

    for name, ana, fd in (("u_x", u0x, fdx), ("u_y", u0y, fdy),
                          ("u_xx", u0xx, fdxx), ("u_yy", u0yy, fdyy)):
        rel = abs(ana - fd) / max(abs(fd), 1e-6)
        assert rel < 1e-4, f"{name}: analytic={ana:.8f} fd={fd:.8f} rel={rel:.2e}"
    print("  [ok] 解析偏导 == 有限差分（u_x, u_y, u_xx, u_yy）")


def test_hard_residual_matches_autograd():
    """_losses_hard 的残差必须与 torch autograd 算出的残差一致。"""
    torch.manual_seed(0)
    p = make_problem("burgers2d", hard_ic=True)
    B, N = 2, 12
    rng = np.random.default_rng(7)
    coef = p.sample_coeffs(B, rng)
    u_b = torch.as_tensor(p.branch_input(coef), dtype=torch.float32)
    model = DeepONet(branch_in=p.n_sensors, trunk_in=3, width=8,
                     branch_hidden=8, branch_layers=1,
                     trunk_hidden=8, trunk_layers=1)

    pts = p.sample_points(coef, N, 6, 6, rng)
    terms = p._losses_hard(model.evaluator(u_b), _to_device(pts, torch.device("cpu")))
    res_fast = terms["pde_pts"]                       # (B, N)

    ref = []
    for b in range(B):
        cb = torch.as_tensor(pts["col"]["coords"][b],
                             dtype=torch.float32).requires_grad_(True)
        t = cb[:, 0]
        net = model(u_b[b:b + 1], cb)[0]              # (N,)
        u = _u0_torch(coef[b], cb, p.K) + t * net
        g1 = torch.autograd.grad(u.sum(), cb, create_graph=True)[0]
        u_t, u_x, u_y = g1[:, 0], g1[:, 1], g1[:, 2]
        u_xx = torch.autograd.grad(u_x.sum(), cb, retain_graph=True)[0][:, 1]
        u_yy = torch.autograd.grad(u_y.sum(), cb, retain_graph=True)[0][:, 2]
        ref.append(u_t + u * u_x + u * u_y - p.nu * (u_xx + u_yy))
    res_ref = torch.stack(ref)

    diff = (res_fast - res_ref).abs().max().item()
    scale = res_ref.abs().max().item()
    assert diff < 1e-4 * max(scale, 1.0), f"residual mismatch: {diff:.3e} (scale {scale:.3e})"
    print(f"  [ok] hard-IC 残差 == autograd 残差（max diff {diff:.2e}, scale {scale:.2e}）")


def test_ic_is_exactly_satisfied():
    """硬约束下 t=0 时 u ≡ u0，与网络输出无关。"""
    p = make_problem("burgers2d", hard_ic=True)
    coef = p.sample_coeffs(2, np.random.default_rng(11))
    u_b = torch.as_tensor(p.branch_input(coef), dtype=torch.float32)
    torch.manual_seed(1)
    model = DeepONet(branch_in=p.n_sensors, trunk_in=3, width=8,
                     branch_hidden=8, branch_layers=1,
                     trunk_hidden=8, trunk_layers=1)
    pts = p.sample_points(coef, 8, 8, 6, np.random.default_rng(13))
    coords = torch.as_tensor(pts["ic"]["coords"], dtype=torch.float32)
    net, _, _ = model.evaluator(u_b)(coords, order=0)
    t = coords[..., 0]
    assert float(t.abs().max()) == 0.0, float(t.abs().max())
    u0 = torch.as_tensor(p.ic_values(coef, pts["ic"]["coords"], 0)[0],
                         dtype=torch.float32)
    u_pred = u0 + t * net                       # u(t=0) = u0 + 0*net
    err = float((u_pred - u0).abs().max().detach())
    assert err == 0.0, err
    print("  [ok] t=0 时 u ≡ u0（与网络参数无关）")


if __name__ == "__main__":
    print("[test] 2D hard-IC 回归测试")
    test_u0_matches_input_at()
    test_u0_matches_solver_t0()
    test_derivatives_vs_finite_difference()
    test_hard_residual_matches_autograd()
    test_ic_is_exactly_satisfied()
    print("[test] 全部通过")
