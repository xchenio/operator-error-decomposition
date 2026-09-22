"""2D Darcy 流（稳态）。

    -∇·(a(x,y) ∇p) = f,   (x,y) ∈ (0,1)^2,   p|∂Ω = 0

a(x,y) = exp(g),  g = Σ_{i,j=1..3} c_ij sin(π i x) sin(π j y)   （解析，可求任意点值与导数）

branch 输入: a 在 m×m 传感器网格上的取值
trunk 输入 : (x, y)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import PDEProblem


class DarcyProblem(PDEProblem):
    name = "darcy"
    trunk_dim = 2

    def __init__(self, N: int = 41, nmode: int = 3, m: int = 21,
                 f: float = 10.0, sigma: float = 0.5):
        self.N = N
        self.nmode = nmode
        self.f = f
        self.sigma = sigma
        self.n_coef = nmode * nmode

        xs = np.linspace(0.0, 1.0, N)
        self.xs = xs
        Xg, Yg = np.meshgrid(xs, xs, indexing="ij")
        self.grid = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float64)
        self.grid_shape = (N, N)

        sg = np.linspace(0.0, 1.0, m)
        SX, SY = np.meshgrid(sg, sg, indexing="ij")
        self.sensor_coords = np.stack([SX.ravel(), SY.ravel()], axis=1)
        self.n_sensors = self.sensor_coords.shape[0]

        self.ranges = [(0.0, 1.0), (0.0, 1.0)]
        self.prior_spec = [("sin", 16), ("sin", 16)]
        self._nm = int(np.sqrt(self.n_coef))

    def _basis(self, coords):
        """返回 Sx, Cx, Sy, Cy，各 (M, nmode)"""
        x = coords[..., 0].ravel()
        y = coords[..., 1].ravel()
        n = np.arange(1, self.nmode + 1)
        Sx = np.sin(np.pi * np.outer(x, n))
        Cx = np.pi * n[None, :] * np.cos(np.pi * np.outer(x, n))
        Sy = np.sin(np.pi * np.outer(y, n))
        Cy = np.pi * n[None, :] * np.cos(np.pi * np.outer(y, n))
        return Sx, Cx, Sy, Cy

    # ---------------- 输入函数 ----------------
    def sample_coeffs(self, n, rng):
        return rng.normal(0, self.sigma, (n, self.n_coef))

    def input_at(self, coef, coords):
        """a(x,y)，coords (..., 2) -> (n, *coords.shape[:-1])"""
        Sx, _, Sy, _ = self._basis(coords)
        C = coef.reshape(coef.shape[0], self._nm, self._nm)
        g = np.einsum("mi,nij,mj->nm", Sx, C, Sy)      # (n, M)
        a = np.exp(g)
        return a.reshape(coef.shape[0], *coords.shape[:-1])

    def input_and_grad(self, coef, coords):
        """coords 为 (B, N, 2) 且 B == coef.shape[0] 时逐样本配对，否则共享网格。"""
        Sx, Cx, Sy, Cy = self._basis(coords)
        C = coef.reshape(coef.shape[0], self._nm, self._nm)
        n = self.nmode
        per_batch = (coords.ndim >= 3 and coords.shape[0] == coef.shape[0])
        if per_batch:
            B, N = coords.shape[0], coords.shape[1]
            Sx = Sx.reshape(B, N, n); Cx = Cx.reshape(B, N, n)
            Sy = Sy.reshape(B, N, n); Cy = Cy.reshape(B, N, n)
            g = np.einsum("bnk,bkl,bnl->bn", Sx, C, Sy)
            gx = np.einsum("bnk,bkl,bnl->bn", Cx, C, Sy)
            gy = np.einsum("bnk,bkl,bnl->bn", Sx, C, Cy)
            a = np.exp(g)
            return a, a * gx, a * gy
        g = np.einsum("mi,nij,mj->nm", Sx, C, Sy)
        gx = np.einsum("mi,nij,mj->nm", Cx, C, Sy)
        gy = np.einsum("mi,nij,mj->nm", Sx, C, Cy)
        a = np.exp(g)
        shp = (coef.shape[0], *coords.shape[:-1])
        return (a.reshape(shp), (a * gx).reshape(shp), (a * gy).reshape(shp))

    def branch_input(self, coef):
        return self.input_at(coef, self.sensor_coords)

    # ---------------- 求解（5 点有限差分） ----------------
    def solve(self, coef, verbose=False):
        N, n = self.N, coef.shape[0]
        h = self.xs[1] - self.xs[0]
        m = N - 2
        ii, jj = np.meshgrid(np.arange(1, N - 1), np.arange(1, N - 1), indexing="ij")
        idx = ((ii - 1) * m + (jj - 1)).ravel()
        ii_f, jj_f = ii.ravel(), jj.ravel()
        out = np.zeros((n, N * N))
        for s in range(n):
            a = self.input_at(coef[s: s + 1], self.grid).reshape(N, N)
            ax = 0.5 * (a[:-1, :] + a[1:, :])       # (N-1, N): i 与 i+1 之间
            ay = 0.5 * (a[:, :-1] + a[:, 1:])       # (N, N-1): j 与 j+1 之间
            center = (ax[ii_f - 1, jj_f] + ax[ii_f, jj_f]
                      + ay[ii_f, jj_f - 1] + ay[ii_f, jj_f]) / h ** 2
            rows, cols, vals = [idx], [idx], [center]
            # east / west
            msk = ii_f <= N - 3
            rows.append(idx[msk]); cols.append((ii_f[msk]) * m + (jj_f[msk] - 1))
            vals.append(-ax[ii_f[msk], jj_f[msk]] / h ** 2)
            msk = ii_f >= 2
            rows.append(idx[msk]); cols.append((ii_f[msk] - 2) * m + (jj_f[msk] - 1))
            vals.append(-ax[ii_f[msk] - 1, jj_f[msk]] / h ** 2)
            # north / south
            msk = jj_f <= N - 3
            rows.append(idx[msk]); cols.append((ii_f[msk] - 1) * m + jj_f[msk])
            vals.append(-ay[ii_f[msk], jj_f[msk]] / h ** 2)
            msk = jj_f >= 2
            rows.append(idx[msk]); cols.append((ii_f[msk] - 1) * m + (jj_f[msk] - 2))
            vals.append(-ay[ii_f[msk], jj_f[msk] - 1] / h ** 2)

            A = sp.coo_matrix((np.concatenate(vals),
                               (np.concatenate(rows), np.concatenate(cols))),
                              shape=(m * m, m * m)).tocsr()
            rhs = np.full(m * m, self.f)
            p = spla.spsolve(A, rhs)
            full = np.zeros((N, N))
            full[1:-1, 1:-1] = p.reshape(m, m)
            out[s] = full.ravel()
        return out

    # ---------------- 采样 ----------------
    def sample_points(self, coef, n_col, n_ic, n_bc, rng):
        B = coef.shape[0]
        xy = rng.uniform(0, 1, (B, n_col, 2))
        a, ax, ay = self.input_and_grad(coef, xy)
        col = {"coords": xy, "a": a.astype(np.float32),
               "ax": ax.astype(np.float32), "ay": ay.astype(np.float32)}

        # 边界：四条边随机取点
        e = rng.integers(0, 4, (B, n_bc))
        v = rng.uniform(0, 1, (B, n_bc))
        z = np.zeros((B, n_bc))
        o = np.ones((B, n_bc))
        coords = np.zeros((B, n_bc, 2))
        coords[..., 0] = np.where((e == 0) | (e == 1), v, np.where(e == 2, z, o))
        coords[..., 1] = np.where((e == 2) | (e == 3), v, np.where(e == 0, z, o))
        bc = {"coords": coords, "target": np.zeros((B, n_bc), dtype=np.float32)}
        return {"col": col, "ic": None, "bc": bc}

    # ---------------- 损失 ----------------
    def losses(self, evalr, pts):
        p, dp, d2p = evalr(pts["col"]["coords"])
        p_x, p_y = dp[..., 0], dp[..., 1]
        p_xx, p_yy = d2p[..., 0, 0], d2p[..., 1, 1]
        a = pts["col"]["a"]
        ax = pts["col"]["ax"]
        ay = pts["col"]["ay"]
        res = ax * p_x + a * p_xx + ay * p_y + a * p_yy + self.f

        pbc, _, _ = evalr(pts["bc"]["coords"], order=0)
        bc = pbc - pts["bc"]["target"]
        return {"pde": (res ** 2).mean(), "bc": (bc ** 2).mean()}
