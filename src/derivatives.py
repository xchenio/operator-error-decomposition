"""trunk 网络的导数工具。

DeepONet 的输出为  u(y) = sum_k b_k * phi_k(y) + bias，
因此对 y 的导数只需要对 trunk 特征 phi 求导，然后与 branch 系数线性组合即可。
这是 PI-DeepONet 能高效计算 PDE 残差的关键。

优先使用 torch.func 的 vmap+jacrev（快），失败时回退到显式循环（稳）。
"""
from __future__ import annotations

from typing import Callable

import torch

_FUNCTCH_OK = True
try:
    from torch.func import jacfwd, jacrev, vmap
except Exception:  # pragma: no cover
    _FUNCTCH_OK = False


def trunk_derivs(trunk_fn: Callable, y: torch.Tensor, width: int, order: int = 2):
    """计算 phi 及其 1/2 阶导数。

    trunk_fn: 单个点 (d,) -> (p,)
    y: (M, d)  查询坐标（无需 requires_grad）
    返回:
        phi  : (M, p)
        dphi : (M, p, d)      order>=1
        d2phi: (M, p, d, d)   order>=2，否则 None
    """
    if _FUNCTCH_OK:
        try:
            return _derivs_functorch(trunk_fn, y, order)
        except Exception:
            pass
    return _derivs_loop(trunk_fn, y, width, order)


def _derivs_functorch(trunk_fn, y: torch.Tensor, order: int):
    # 二阶导数的实现选择（实测，1024 点 / p=100 / d=2，CPU）：
    #   jacrev(jacrev) : 爆内存（>10GB）且最慢
    #   jacrev(jacfwd) : ~345 ms
    #   jacfwd(jacrev) : ~107 ms
    #   jacfwd(jacfwd) : ~ 18 ms   ← 输入维 d 很小(1~3)时最优：只需 d² 次前向
    # 四个结果数值一致（差异 ~5e-8）。
    phi = torch.vmap(trunk_fn)(y)                     # (M, p)
    if order >= 1:
        dphi = vmap(jacrev(trunk_fn))(y)              # (M, p, d)
    else:
        dphi = None
    if order >= 2:
        d2phi = vmap(jacfwd(jacfwd(trunk_fn)))(y)     # (M, p, d, d)
    else:
        d2phi = None
    return phi, dphi, d2phi


def _derivs_loop(trunk_fn, y: torch.Tensor, width: int, order: int):
    """回退实现：对每个 trunk 分量做一次 backward。慢但总是能用。"""
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    y = y.detach().clone().requires_grad_(True)
    phi = trunk_fn(y)                                  # (M, p)
    M, d = y.shape
    dev, dt = y.device, y.dtype

    if order >= 1:
        dphi = torch.zeros(M, width, d, device=dev, dtype=dt)
        for k in range(width):
            g = torch.autograd.grad(
                phi[:, k], y, grad_outputs=torch.ones(M, device=dev, dtype=dt),
                create_graph=True, retain_graph=True, allow_unused=False,
            )[0]
            dphi[:, k, :] = g
    else:
        dphi = None

    if order >= 2:
        d2phi = torch.zeros(M, width, d, d, device=dev, dtype=dt)
        for k in range(width):
            for j in range(d):
                g1 = torch.autograd.grad(
                    phi[:, k], y, grad_outputs=torch.ones(M, device=dev, dtype=dt),
                    create_graph=True, retain_graph=True,
                )[0][:, j]
                g2 = torch.autograd.grad(
                    g1, y, grad_outputs=torch.ones(M, device=dev, dtype=dt),
                    create_graph=True, retain_graph=True,
                )[0]
                d2phi[:, k, j, :] = g2
    else:
        d2phi = None

    return phi, dphi, d2phi
