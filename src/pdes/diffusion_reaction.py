"""1D 扩散-反应方程（PI-DeepONet 经典基准，Wang et al. Sci. Adv. 2021）。

    s_t = D s_xx + k s^2 + u(x),   x ∈ (0,1), t ∈ (0,1]
    s(0,x) = 0,  s(t,0) = s(t,1) = 0

branch 输入: 源项 u(x) 在 m 个传感器上的取值
trunk 输入 : (t, x)
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from .base import PDEProblem


class DiffusionReactionProblem(PDEProblem):
    name = "dr"
    trunk_dim = 2

    def __init__(self, D: float = 0.01, k: float = 0.01, Nx: int = 101,
                 T: float = 1.0, Nt: int = 41, dt: float = 1e-3,
                 K: int = 12, m: int = 101, amp: float = 0.5, decay: float = 1.2):
        self.D, self.k, self.Nx = D, k, Nx
        self.T, self.Nt, self.dt = T, Nt, dt
        self.K = K
        self.n_coef = K
        self.amp = amp
        self.decay = decay

        self.x = np.linspace(0.0, 1.0, Nx)
        self.dx = self.x[1] - self.x[0]

        self.sensor_coords = np.linspace(0.0, 1.0, m)
        self.n_sensors = m

        ts = np.linspace(0.0, T, Nt)
        TT, XX = np.meshgrid(ts, self.x, indexing="ij")
        self.grid = np.stack([TT.ravel(), XX.ravel()], axis=1).astype(np.float64)
        self.grid_shape = (Nt, Nx)

        self.ranges = [(0.0, T), (0.0, 1.0)]
        self.prior_spec = [("cheb", 14), ("sin", 18)]
        self.time_dim = 0
        self.t_range = (0.0, T)

        # 内部节点的 sin 基（用于源项）
        self._sin_int = np.sin(np.pi * np.outer(self.x[1:-1], np.arange(1, K + 1)))

    # ---------------- 输入函数 ----------------
    def sample_coeffs(self, n, rng):
        i = np.arange(1, self.K + 1, dtype=np.float64)
        sig = self.amp * i ** (-self.decay)
        return rng.normal(0, 1, (n, self.K)) * sig[None, :]

    def input_at(self, coef, x):
        """x 为 (B,N) 且 B == coef.shape[0] 时逐样本配对，否则共享网格。"""
        x = np.asarray(x, dtype=np.float64)
        k = np.arange(1, self.K + 1)
        if x.ndim >= 2 and x.shape[0] == coef.shape[0]:
            S = np.sin(np.pi * x[..., None] * k)                  # (B, N, K)
            return np.einsum("bk,bnk->bn", coef, S)
        S = np.sin(np.pi * np.outer(x.ravel(), k))
        out = coef @ S.T
        return out.reshape(coef.shape[0], *x.shape)

    def branch_input(self, coef):
        return self.input_at(coef, self.sensor_coords)

    # ---------------- 求解（有限差分 + 半隐式） ----------------
    def solve(self, coef, verbose=False):
        n = coef.shape[0]
        m = self.Nx - 2
        r = self.D * self.dt / self.dx ** 2
        ab = np.zeros((3, m))
        ab[0, 1:] = -r
        ab[1, :] = 1.0 + 2.0 * r
        ab[2, :-1] = -r

        forcing = (self._sin_int @ coef.T)                 # (m, n)
        s = np.zeros((m, n))
        nsteps = int(round(self.T / self.dt))
        every = max(1, nsteps // (self.Nt - 1))
        out = [np.zeros((n, self.Nx))]
        for step in range(nsteps):
            rhs = s + self.dt * (self.k * s ** 2 + forcing)
            s = solve_banded((1, 1), ab, rhs)
            if (step + 1) % every == 0 and len(out) < self.Nt:
                full = np.zeros((n, self.Nx))
                full[:, 1:-1] = s.T
                out.append(full.copy())
        while len(out) < self.Nt:
            full = np.zeros((n, self.Nx))
            full[:, 1:-1] = s.T
            out.append(full.copy())
        return np.stack(out, axis=1).reshape(n, -1)

    # ---------------- 采样 ----------------
    def sample_points(self, coef, n_col, n_ic, n_bc, rng):
        B = coef.shape[0]
        t = rng.uniform(0, self.T, (B, n_col, 1))
        x = rng.uniform(0, 1, (B, n_col, 1))
        col = {"coords": np.concatenate([t, x], axis=2)}
        # 残差中需要源项 u(x)
        col["forcing"] = self.input_at(coef, x[:, :, 0]).astype(np.float32)

        xic = rng.uniform(0, 1, (B, n_ic, 1))
        tic = np.zeros_like(xic)
        ic = {"coords": np.concatenate([tic, xic], axis=2),
              "target": np.zeros((B, n_ic), dtype=np.float32)}

        tbc = rng.uniform(0, self.T, (B, n_bc, 1))
        side = rng.integers(0, 2, (B, n_bc, 1)).astype(np.float64)
        bc = {"coords": np.concatenate([tbc, side], axis=2),
              "target": np.zeros((B, n_bc), dtype=np.float32)}
        return {"col": col, "ic": ic, "bc": bc}

    # ---------------- 损失 ----------------
    def losses(self, evalr, pts):
        s, ds, d2s = evalr(pts["col"]["coords"])
        s_t = ds[..., 0]
        s_xx = d2s[..., 1, 1]
        forcing = pts["col"]["forcing"]
        res = s_t - self.D * s_xx - self.k * s ** 2 - forcing

        sic, _, _ = evalr(pts["ic"]["coords"], order=0)
        ic = sic - pts["ic"]["target"]

        sbc, _, _ = evalr(pts["bc"]["coords"], order=0)
        bc = sbc - pts["bc"]["target"]

        return {"pde": (res ** 2).mean(),
                "pde_pts": res,
                "col_coords": pts["col"]["coords"],
                "ic": (ic ** 2).mean(),
                "bc": (bc ** 2).mean()}
