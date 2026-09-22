"""DeepONet 模型定义。"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from .derivatives import trunk_derivs


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "sin":
        return _Sin()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unknown activation {name}")


class _Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    """输入 (..., in_dim) -> (..., out_dim)；无归一化层，保证 torch.func 可用。"""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, n_layers: int,
                 activation: str = "tanh", init_gain: float = 1.0):
        super().__init__()
        self.init_gain = init_gain
        layers = []
        dim = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(dim, hidden), make_activation(activation)]
            dim = hidden
        layers += [nn.Linear(dim, out_dim)]
        self.net = nn.Sequential(*layers)
        self._reset()

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=self.init_gain)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class FourierFeature(nn.Module):
    """固定随机傅里叶特征（不可训练），用于对抗谱偏差的对照基线。"""

    def __init__(self, in_dim: int, n_feat: int, sigma: float = 1.0):
        super().__init__()
        B = torch.randn(in_dim, n_feat) * sigma
        self.register_buffer("B", B)

    def forward(self, y):
        z = 2.0 * torch.pi * (y @ self.B)
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)


class DeepONet(nn.Module):
    """标准（非偏置增强）DeepONet：G(u)(y) = <b(u), phi(y)> + bias"""

    def __init__(self, branch_in: int, trunk_in: int, width: int = 100,
                 branch_hidden: int = 128, branch_layers: int = 3,
                 trunk_hidden: int = 128, trunk_layers: int = 3,
                 activation: str = "tanh", init_gain: float = 1.0,
                 trunk_type: str = "mlp", n_ff: int = 64, ff_sigma: float = 1.0):
        super().__init__()
        self.width = width
        self.trunk_in = trunk_in
        self.trunk_type = trunk_type
        self.branch = MLP(branch_in, width, branch_hidden, branch_layers,
                          activation, init_gain)
        if trunk_type == "mlp":
            self.trunk = MLP(trunk_in, width, trunk_hidden, trunk_layers,
                             activation, init_gain)
        elif trunk_type == "fourier":
            ff = FourierFeature(trunk_in, n_ff, ff_sigma)
            self.trunk = nn.Sequential(
                ff, MLP(2 * n_ff, width, trunk_hidden, trunk_layers,
                        activation, init_gain))
        else:
            raise ValueError(f"unknown trunk_type {trunk_type}")
        self.bias = nn.Parameter(torch.zeros(1))

    # ---- 基本前向 ----
    def branch_coeffs(self, u: torch.Tensor) -> torch.Tensor:
        return self.branch(u)                    # (B, p)

    def trunk_features(self, y: torch.Tensor) -> torch.Tensor:
        return self.trunk(y)                     # (M, p)

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """u: (B, m)  branch 输入；y: (M, d)  trunk 坐标 -> (B, M)"""
        b = self.branch_coeffs(u)                # (B, p)
        phi = self.trunk_features(y)             # (M, p)
        return torch.einsum("bp,mp->bm", b, phi) + self.bias

    # ---- 带导数的求值器 ----
    def evaluator(self, coeffs: torch.Tensor, order: int = 2
                  ) -> Callable[[torch.Tensor], tuple]:
        """返回 evalr(coords)。coords: (B, N, d) -> (u, du, d2u)。

        u  : (B, N)
        du : (B, N, d)
        d2u: (B, N, d, d) 或 None
        """
        b = self.branch_coeffs(coeffs)           # (B, p)
        B, p = b.shape

        def evalr(coords: torch.Tensor, order: int = order):
            Bd, N = coords.shape[0], coords.shape[1]
            flat = coords.reshape(Bd * N, -1)
            phi, dphi, d2phi = trunk_derivs(self.trunk, flat, p, order=order)
            u = torch.einsum("bp,bnp->bn", b, phi.view(Bd, N, p)) + self.bias
            if order >= 1:
                du = torch.einsum("bp,bnpd->bnd", b, dphi.view(Bd, N, p, -1))
            else:
                du = None
            if order >= 2:
                d = d2phi.shape[-1]
                d2u = torch.einsum("bp,bnpde->bnde", b, d2phi.view(Bd, N, p, d, d))
            else:
                d2u = None
            return u, du, d2u

        return evalr

    def predict_on_grid(self, u: torch.Tensor, grid: torch.Tensor,
                        chunk: Optional[int] = None) -> torch.Tensor:
        """u: (B, m), grid: (Ng, d) -> (B, Ng)"""
        b = self.branch_coeffs(u)
        phi = self.trunk_features(grid)               # (Ng, p)
        return torch.einsum("bp,np->bn", b, phi) + self.bias
