"""1D 周期粘性 Burgers 方程。

    u_t + u u_x - nu u_xx = 0,   x ∈ [0,1) 周期, t ∈ [0, T]
    u(0, x) = u0(x)  ~ 周期高斯随机场（截断 Fourier 级数）

branch 输入: u0 在 m 个等距传感器上的取值
trunk 输入 : (t, x)
"""
from __future__ import annotations

import numpy as np

from .base import PDEProblem


class BurgersProblem(PDEProblem):
    name = "burgers"
    trunk_dim = 2

    def __init__(self, nu: float = 0.02, Nx: int = 256, Nx_store: int = 128,
                 T: float = 1.0, Nt: int = 41, dt: float = 5e-4,
                 K: int = 8, m: int = 128, ell: float = 3.0,
                 hard_ic: bool = False):
        # hard_ic: 令 u(t,x) = u0(x) + t * NN(t,x)，初值精确满足。
        # 目的：消除 u≡0 这个"PDE+周期边界的精确解"陷阱——
        # 平凡解使 PDE 残差恒为 0，只有 IC 提供梯度，优化器难以逃离。
        self.hard_ic = hard_ic
        self.nu = nu
        self.Nx = Nx
        self.Nx_store = Nx_store
        self.T = T
        self.Nt = Nt
        self.dt = dt
        self.K = K
        self.n_coef = 2 * K
        self.ell = ell

        self.x_full = np.arange(Nx) / Nx
        self.k = 2 * np.pi * np.fft.fftfreq(Nx, d=1.0 / Nx)
        self.k2 = self.k ** 2

        self.sensor_coords = np.arange(m) / m
        self.n_sensors = m

        step = Nx // Nx_store
        self.store_idx = np.arange(0, Nx, step)[:Nx_store]
        xs = np.arange(Nx_store) / Nx_store
        ts = np.linspace(0.0, T, Nt)
        TT, XX = np.meshgrid(ts, xs, indexing="ij")
        self.grid = np.stack([TT.ravel(), XX.ravel()], axis=1).astype(np.float64)
        self.grid_shape = (Nt, Nx_store)

        self.ranges = [(0.0, T), (0.0, 1.0)]
        self.prior_spec = [("cheb", 14), ("fourier", 8)]
        self.time_dim = 0                 # trunk 第 0 维是时间
        self.t_range = (0.0, T)

    # ---------------- 输入函数 ----------------
    def sample_coeffs(self, n, rng):
        k = np.arange(1, self.K + 1)
        sig = np.exp(-0.5 * (k / self.ell) ** 2)
        a = rng.normal(0, 1, (n, self.K)) * sig[None, :]
        b = rng.normal(0, 1, (n, self.K)) * sig[None, :]
        return np.concatenate([a, b], axis=1)

    def input_at(self, coef, x):
        """两种约定：
        - x 为 (B, N) 且 B == coef.shape[0]：逐样本配对，返回 (B, N)
        - 否则：共享网格，返回 (n, *x.shape)
        """
        x = np.asarray(x, dtype=np.float64)
        k = 2 * np.pi * np.arange(1, self.K + 1)
        if x.ndim >= 2 and x.shape[0] == coef.shape[0]:
            S = np.sin(x[..., None] * k)              # (B, N, K)
            C = np.cos(x[..., None] * k)
            return (np.einsum("bk,bnk->bn", coef[:, : self.K], S)
                    + np.einsum("bk,bnk->bn", coef[:, self.K:], C))
        xf = x.ravel()
        S = np.sin(np.outer(xf, k))                   # (M, K)
        C = np.cos(np.outer(xf, k))
        out = coef[:, : self.K] @ S.T + coef[:, self.K:] @ C.T   # (n, M)
        return out.reshape(coef.shape[0], *x.shape)

    def branch_input(self, coef):
        return self.input_at(coef, self.sensor_coords)

    # ---------------- 硬约束 IC 所需的 u0 及其导数（解析） ----------------
    def ic_values(self, coef, coords, order: int = 0):
        """u0 及其 x 导数在 coords 的 x 坐标处。返回 list，长度 = order+1，各 (n, *shape)。

        u0(x) = Σ a_i sin(k_i x) + b_i cos(k_i x),  k_i = 2πi  （周期）
        """
        x = np.asarray(coords[..., 1], dtype=np.float64)
        k = 2 * np.pi * np.arange(1, self.K + 1)          # (K,)
        A, B = coef[:, : self.K], coef[:, self.K:]        # (n, K)
        out = []

        if x.ndim == 1:
            # 共享网格 coords (M,2) -> 输出 (n, M)
            S = np.sin(np.outer(x, k))                     # (M, K)
            C = np.cos(np.outer(x, k))
            out.append(A @ S.T + B @ C.T)
            if order >= 1:
                out.append(A @ (k[None, :] * C).T - B @ (k[None, :] * S).T)
            if order >= 2:
                k2 = k ** 2
                out.append(-A @ (k2[None, :] * S).T - B @ (k2[None, :] * C).T)
        else:
            # coords (B,N,2)，第 b 行配 coef[b] -> 输出 (B, N)
            S = np.sin(x[..., None] * k)                   # (B, N, K)
            C = np.cos(x[..., None] * k)
            out.append(np.einsum("bk,bnk->bn", A, S)
                       + np.einsum("bk,bnk->bn", B, C))
            if order >= 1:
                kk = k[None, None, :]
                out.append(np.einsum("bk,bnk->bn", A, kk * C)
                           - np.einsum("bk,bnk->bn", B, kk * S))
            if order >= 2:
                k2 = (k ** 2)[None, None, :]
                out.append(-np.einsum("bk,bnk->bn", A, k2 * S)
                           - np.einsum("bk,bnk->bn", B, k2 * C))
        return out

    # ---------------- 求解（Strang 分裂 + 谱方法） ----------------
    def solve(self, coef, verbose=False):
        n = coef.shape[0]
        u0 = self.input_at(coef, self.x_full)          # (n, Nx)
        nsteps = int(round(self.T / self.dt))
        every = max(1, nsteps // (self.Nt - 1))
        E_half = np.exp(-self.nu * self.k2 * self.dt / 2)

        uh = np.fft.fft(u0, axis=1)
        out = [u0[:, self.store_idx].copy()]
        for s in range(nsteps):
            # 扩散半步（精确）
            uh = uh * E_half
            u = np.fft.ifft(uh, axis=1).real
            # 对流整步（RK4）
            k1 = self._conv(u)
            k2 = self._conv(u + 0.5 * self.dt * k1)
            k3 = self._conv(u + 0.5 * self.dt * k2)
            k4 = self._conv(u + self.dt * k3)
            u = u + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            uh = np.fft.fft(u, axis=1) * E_half
            if (s + 1) % every == 0 and len(out) < self.Nt:
                out.append(u[:, self.store_idx].copy())
        while len(out) < self.Nt:
            out.append(u[:, self.store_idx].copy())
        return np.stack(out, axis=1).reshape(n, -1)     # (n, Nt*Nx_store)

    def _conv(self, u):
        """u_t = -u u_x"""
        uh = np.fft.fft(u, axis=1)
        ux = np.fft.ifft(1j * self.k * uh, axis=1).real
        return -u * ux

    # ---------------- 采样 ----------------
    def sample_points(self, coef, n_col, n_ic, n_bc, rng):
        B = coef.shape[0]
        t = rng.uniform(0, self.T, (B, n_col, 1))
        x = rng.uniform(0, 1, (B, n_col, 1))
        col = {"coords": np.concatenate([t, x], axis=2)}

        xic = rng.uniform(0, 1, (B, n_ic, 1))
        tic = np.zeros_like(xic)
        ic = {"coords": np.concatenate([tic, xic], axis=2),
              "target": self.input_at(coef, xic[:, :, 0]).astype(np.float32)}

        tbc = rng.uniform(0, self.T, (B, n_bc, 1))
        zero = np.zeros_like(tbc)
        one = np.ones_like(tbc)
        bc = {"a": np.concatenate([tbc, zero], axis=2),
              "b": np.concatenate([tbc, one], axis=2)}
        if self.hard_ic:
            # u = u0 + t*NN 的抬升项；u0 周期 -> u0(0)=u0(1), u0'(0)=u0'(1)
            cv = self.ic_values(coef, col["coords"], order=2)
            col["u0"], col["u0x"], col["u0xx"] = [c.astype(np.float32) for c in cv]
            for key in ("a", "b"):
                v = self.ic_values(coef, bc[key], order=1)
                bc[key + "_u0"], bc[key + "_u0x"] = [c.astype(np.float32) for c in v]
        return {"col": col, "ic": ic, "bc": bc}

    # ---------------- 损失 ----------------
    def losses(self, evalr, pts):
        if self.hard_ic:
            return self._losses_hard(evalr, pts)
        u, du, d2u = evalr(pts["col"]["coords"])          # coords (B,N,2): (t,x)
        u_t = du[..., 0]
        u_x = du[..., 1]
        u_xx = d2u[..., 1, 1]
        res = u_t + u * u_x - self.nu * u_xx

        uic, _, _ = evalr(pts["ic"]["coords"], order=0)
        tgt = pts["ic"]["target"]
        ic = uic - tgt

        ua, da, _ = evalr(pts["bc"]["a"], order=1)
        ub, db, _ = evalr(pts["bc"]["b"], order=1)
        bc_v = ua - ub
        bc_d = da[..., 1] - db[..., 1]

        return {
            "pde": (res ** 2).mean(),
            "pde_pts": res,                      # 因果加权需要逐点残差
            "col_coords": pts["col"]["coords"],
            "ic": (ic ** 2).mean(),
            "bc": (bc_v ** 2).mean() + (bc_d ** 2).mean(),
        }

    def _losses_hard(self, evalr, pts):
        """硬约束 IC 版本：u(t,x) = u0(x) + t * net(t,x)。

        由此 u_t = net + t*net_t，u_x = u0_x + t*net_x，u_xx = u0_xx + t*net_xx。
        IC 精确满足，不再有 ic 损失项；平凡解 u≡0 也不可达（u0 非零）。
        """
        col = pts["col"]
        coords = col["coords"]
        net, dnet, d2net = evalr(coords)
        t = coords[..., 0]
        u0 = col["u0"]
        u0x = col["u0x"]
        u0xx = col["u0xx"]

        u = u0 + t * net
        u_t = net + t * dnet[..., 0]
        u_x = u0x + t * dnet[..., 1]
        u_xx = u0xx + t * d2net[..., 1, 1]
        res = u_t + u * u_x - self.nu * u_xx

        na, da, _ = evalr(pts["bc"]["a"], order=1)
        nb, db, _ = evalr(pts["bc"]["b"], order=1)
        ta = pts["bc"]["a"][..., 0]
        # u0 周期，故抬升项在 x=0 与 x=1 相同，相减后抵消
        bc_v = ta * (na - nb)
        bc_d = ta * (da[..., 1] - db[..., 1])

        return {
            "pde": (res ** 2).mean(),
            "pde_pts": res,
            "col_coords": coords,
            "bc": (bc_v ** 2).mean() + (bc_d ** 2).mean(),
        }
