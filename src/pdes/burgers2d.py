"""2D 周期粘性 Burgers 方程。

    u_t + u u_x + u u_y = nu (u_xx + u_yy),   (x,y) ∈ [0,1)^2 周期,  t ∈ [0,T]
    u(0,x,y) = u0(x,y)  ~ 截断二维 Fourier 级数（解析，可在任意点求值）

branch 输入: u0 在 m×m 传感器网格上的取值
trunk 输入 : (t, x, y)

注意：所有 input_at 类函数都必须区分「共享网格」与「逐样本配对」两种约定
（见 README §6.5 记录的那个静默广播 bug）。
"""
from __future__ import annotations

import numpy as np

from .base import PDEProblem


class Burgers2DProblem(PDEProblem):
    name = "burgers2d"
    trunk_dim = 3

    def __init__(self, nu: float = 0.05, Nx: int = 64, Ny: int = 64,
                 Ns: int = 32, T: float = 1.0, Nt: int = 21, dt: float = 5e-4,
                 K: int = 4, m: int = 16, ell: float = 2.5,
                 hard_ic: bool = False):
        # hard_ic: 令 u(t,x,y) = u0(x,y) + t * NN(t,x,y)，初值精确满足。
        # 目的与 1D 相同：消除 u≡0 这个"PDE + 周期边界的精确解"陷阱。
        # 2D 上 baseline 会确定性塌到 u≡0（relL2≈1.003），硬约束 IC 是检验
        # "塌缩只是因为 IC 是软约束"这一假设的最直接对照。
        self.hard_ic = hard_ic
        self.nu = nu
        self.Nx, self.Ny, self.Ns = Nx, Ny, Ns
        self.T, self.Nt, self.dt = T, Nt, dt
        self.K = K
        self.ell = ell
        self.n_coef = 2 * K * K

        self.x_full = np.arange(Nx) / Nx
        self.y_full = np.arange(Ny) / Ny

        sg = np.linspace(0.0, 1.0, m, endpoint=False)
        SX, SY = np.meshgrid(sg, sg, indexing="ij")
        self.sensor_coords = np.stack([SX.ravel(), SY.ravel()], axis=1)
        self.n_sensors = self.sensor_coords.shape[0]

        xs = np.linspace(0.0, 1.0, Ns, endpoint=False)
        ts = np.linspace(0.0, T, Nt)
        TT, XX, YY = np.meshgrid(ts, xs, xs, indexing="ij")
        self.grid = np.stack([TT.ravel(), XX.ravel(), YY.ravel()],
                             axis=1).astype(np.float64)
        self.grid_shape = (Nt, Ns, Ns)

        self.ranges = [(0.0, T), (0.0, 1.0), (0.0, 1.0)]
        self.prior_spec = [("cheb", 10), ("fourier", 4), ("fourier", 4)]
        self.time_dim = 0
        self.t_range = (0.0, T)

    # ---------------- 输入函数 ----------------
    def sample_coeffs(self, n, rng):
        i = np.arange(1, self.K + 1)[:, None]
        j = np.arange(1, self.K + 1)[None, :]
        sig = np.exp(-0.5 * ((i + j) / self.ell) ** 2)          # (K, K)
        a = rng.normal(0, 1, (n, self.K, self.K)) * sig[None]
        b = rng.normal(0, 1, (n, self.K, self.K)) * sig[None]
        return np.concatenate([a.reshape(n, -1), b.reshape(n, -1)], axis=1)

    def _pair(self, coef):
        K = self.K
        A = coef[:, : K * K].reshape(-1, K, K)
        B = coef[:, K * K:].reshape(-1, K, K)
        return A, B

    def input_at(self, coef, coords):
        """u0 在 coords (..., 2)=(x,y) 处。

        coords 为 (B,N,2) 且 B == coef.shape[0] 时逐样本配对 -> (B,N)；
        否则共享网格 -> (n, *coords.shape[:-1])。
        """
        coords = np.asarray(coords, dtype=np.float64)
        x = coords[..., 0]
        y = coords[..., 1]
        k = 2 * np.pi * np.arange(1, self.K + 1)
        A, B = self._pair(coef)
        per_batch = (coords.ndim >= 3 and coords.shape[0] == coef.shape[0])
        if per_batch:
            Sx = np.sin(x[..., None] * k)      # (B,N,K)
            Cx = np.cos(x[..., None] * k)
            Sy = np.sin(y[..., None] * k)
            Cy = np.cos(y[..., None] * k)
            u = (np.einsum("bni,bij,bnj->bn", Sx, A, Cy)
                 + np.einsum("bni,bij,bnj->bn", Cx, B, Sy))
            return u
        Sx = np.sin(np.outer(x.ravel(), k))
        Cx = np.cos(np.outer(x.ravel(), k))
        Sy = np.sin(np.outer(y.ravel(), k))
        Cy = np.cos(np.outer(y.ravel(), k))
        u = (np.einsum("mi,nij,mj->nm", Sx, A, Cy)
             + np.einsum("mi,nij,mj->nm", Cx, B, Sy))
        return u.reshape(coef.shape[0], *coords.shape[:-1])

    def branch_input(self, coef):
        return self.input_at(coef, self.sensor_coords)

    # ---------------- 硬约束 IC 所需的 u0 及其偏导（解析） ----------------
    def ic_values(self, coef, coords, order: int = 0):
        """u0 及其偏导在 coords 的 (x, y) 处。返回 list：

            [0] u0                (n, M) 共享网格 / (B, N) 逐样本
            [1] [u0_x, u0_y]      (..., 2)      （order >= 1）
            [2] [u0_xx, u0_yy]    (..., 2)      （order >= 2）

        u0(x,y) = Σ_ij A_ij sin(k_i x) cos(k_j y) + Σ_ij B_ij cos(k_i x) sin(k_j y)
        """
        coords = np.asarray(coords, dtype=np.float64)
        x = coords[..., 1]
        y = coords[..., 2]
        k = 2 * np.pi * np.arange(1, self.K + 1)
        A, B = self._pair(coef)                        # (n, K, K)
        per_batch = coords.ndim >= 3 and coords.shape[0] == coef.shape[0]

        Sx = np.sin(x[..., None] * k)
        Cx = np.cos(x[..., None] * k)
        Sy = np.sin(y[..., None] * k)
        Cy = np.cos(y[..., None] * k)

        def quad(M, Fx, Fy):
            """Σ_ij M_ij Fx_i Fy_j"""
            return (np.einsum("bij,bni,bnj->bn", M, Fx, Fy) if per_batch
                    else np.einsum("nij,mi,mj->nm", M, Fx, Fy))

        ki = k[None, :, None]          # 沿 i（x 方向波数）
        kj = k[None, None, :]          # 沿 j（y 方向波数）

        out = [quad(A, Sx, Cy) + quad(B, Cx, Sy)]
        if order >= 1:
            u0x = quad(A * ki, Cx, Cy) - quad(B * ki, Sx, Sy)
            u0y = -quad(A * kj, Sx, Sy) + quad(B * kj, Cx, Cy)
            out.append(np.stack([u0x, u0y], axis=-1))
        if order >= 2:
            u0xx = -quad(A * ki ** 2, Sx, Cy) - quad(B * ki ** 2, Cx, Sy)
            u0yy = -quad(A * kj ** 2, Sx, Cy) - quad(B * kj ** 2, Cx, Sy)
            out.append(np.stack([u0xx, u0yy], axis=-1))
        return out

    # ---------------- 求解（2D Strang 分裂 + 谱方法） ----------------
    def solve(self, coef, verbose=False):
        n = coef.shape[0]
        Nx, Ny = self.Nx, self.Ny
        Xg, Yg = np.meshgrid(self.x_full, self.y_full, indexing="ij")
        pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
        u0 = self.input_at(coef, pts).reshape(n, Nx, Ny)

        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=1.0 / Nx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=1.0 / Ny)
        K2 = (kx ** 2)[:, None] + (ky ** 2)[None, :]
        E_half = np.exp(-self.nu * K2 * self.dt / 2)

        uh = np.fft.fft2(u0, axes=(1, 2))
        nsteps = int(round(self.T / self.dt))
        every = max(1, nsteps // (self.Nt - 1))
        step = Nx // self.Ns
        out = [self._store(u0, step)]
        for s in range(nsteps):
            uh = uh * E_half
            u = np.fft.ifft2(uh, axes=(1, 2)).real
            k1 = self._conv(u, kx, ky)
            k2 = self._conv(u + 0.5 * self.dt * k1, kx, ky)
            k3 = self._conv(u + 0.5 * self.dt * k2, kx, ky)
            k4 = self._conv(u + self.dt * k3, kx, ky)
            u = u + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            uh = np.fft.fft2(u, axes=(1, 2)) * E_half
            if (s + 1) % every == 0 and len(out) < self.Nt:
                out.append(self._store(u, step))
        while len(out) < self.Nt:
            out.append(self._store(u, step))
        return np.stack(out, axis=1).reshape(n, -1)

    def _store(self, u, step):
        return u[:, ::step, ::step][:, : self.Ns, : self.Ns].copy()

    def _conv(self, u, kx, ky):
        uh = np.fft.fft2(u, axes=(1, 2))
        ux = np.fft.ifft2(1j * kx[:, None] * uh, axes=(1, 2)).real
        uy = np.fft.ifft2(1j * ky[None, :] * uh, axes=(1, 2)).real
        return -(u * ux + u * uy)

    # ---------------- 采样 ----------------
    def sample_points(self, coef, n_col, n_ic, n_bc, rng):
        B = coef.shape[0]
        t = rng.uniform(0, self.T, (B, n_col, 1))
        x = rng.uniform(0, 1, (B, n_col, 1))
        y = rng.uniform(0, 1, (B, n_col, 1))
        col = {"coords": np.concatenate([t, x, y], axis=2)}

        xi = rng.uniform(0, 1, (B, n_ic, 1))
        yi = rng.uniform(0, 1, (B, n_ic, 1))
        ti = np.zeros_like(xi)
        ic_xy = np.concatenate([xi, yi], axis=2)
        ic = {"coords": np.concatenate([ti, xi, yi], axis=2),
              "target": self.input_at(coef, ic_xy).astype(np.float32)}

        tb = rng.uniform(0, self.T, (B, n_bc, 1))
        v = rng.uniform(0, 1, (B, n_bc, 1))
        z, o = np.zeros_like(v), np.ones_like(v)
        # x 方向配对：x=0 与 x=1
        bxa = {"coords": np.concatenate([tb, z, v], axis=2)}
        bxb = {"coords": np.concatenate([tb, o, v], axis=2)}
        # y 方向配对：y=0 与 y=1
        bya = {"coords": np.concatenate([tb, v, z], axis=2)}
        byb = {"coords": np.concatenate([tb, v, o], axis=2)}
        if self.hard_ic:
            # u = u0 + t*NN 的抬升项；u0 双周期，故抬升项在对边完全相同
            u0, du0, d2u0 = self.ic_values(coef, col["coords"], order=2)
            col["u0"] = u0.astype(np.float32)
            col["du0"] = du0.astype(np.float32)
            col["d2u0"] = d2u0.astype(np.float32)
        return {"col": col, "ic": ic,
                "bc": {"ax": bxa, "bx": bxb, "ay": bya, "by": byb}}

    # ---------------- 损失 ----------------
    def losses(self, evalr, pts):
        if self.hard_ic:
            return self._losses_hard(evalr, pts)
        coord_keys = ("ax", "bx", "ay", "by")
        used = {k: pts["bc"][k]["coords"] for k in coord_keys}
        ua = evalr(pts["col"]["coords"], order=2)
        u = ua[0]
        du, d2u = ua[1], ua[2]
        u_t = du[..., 0]
        u_x, u_y = du[..., 1], du[..., 2]
        u_xx = d2u[..., 1, 1]
        u_yy = d2u[..., 2, 2]
        res = u_t + u * u_x + u * u_y - self.nu * (u_xx + u_yy)

        uic, _, _ = evalr(pts["ic"]["coords"], order=0)
        ic = uic - pts["ic"]["target"]

        bc_terms = []
        for ka, kb, comp in (("ax", "bx", 1), ("ay", "by", 2)):
            va, da, _ = evalr(used[ka], order=1)
            vb, db, _ = evalr(used[kb], order=1)
            bc_terms.append((va - vb) ** 2)
            bc_terms.append((da[..., comp] - db[..., comp]) ** 2)
        bc = sum(t.mean() for t in bc_terms)

        return {"pde": (res ** 2).mean(),
                "pde_pts": res,
                "col_coords": pts["col"]["coords"],
                "ic": (ic ** 2).mean(),
                "bc": bc}

    def _losses_hard(self, evalr, pts):
        """硬约束 IC 版本：u(t,x,y) = u0(x,y) + t * net(t,x,y)。

        由此 u_t = net + t*net_t，u_x = u0_x + t*net_x，u_xx = u0_xx + t*net_xx（y 同理）。
        IC 精确满足，不再有 ic 损失项；平凡解 u≡0 也不可达（u0 非零）。
        """
        col = pts["col"]
        coords = col["coords"]
        net, dnet, d2net = evalr(coords)
        t = coords[..., 0]
        u0 = col["u0"]
        du0 = col["du0"]
        d2u0 = col["d2u0"]

        u = u0 + t * net
        u_t = net + t * dnet[..., 0]
        u_x = du0[..., 0] + t * dnet[..., 1]
        u_y = du0[..., 1] + t * dnet[..., 2]
        u_xx = d2u0[..., 0] + t * d2net[..., 1, 1]
        u_yy = d2u0[..., 1] + t * d2net[..., 2, 2]
        res = u_t + u * u_x + u * u_y - self.nu * (u_xx + u_yy)

        bc_terms = []
        for ka, kb, comp in (("ax", "bx", 1), ("ay", "by", 2)):
            ca = pts["bc"][ka]["coords"]
            cb = pts["bc"][kb]["coords"]
            va, da, _ = evalr(ca, order=1)
            vb, db, _ = evalr(cb, order=1)
            ta = ca[..., 0]
            # u0 双周期，抬升项在对边完全相同，相减后抵消
            bc_terms.append((ta * (va - vb)) ** 2)
            bc_terms.append((ta * (da[..., comp] - db[..., comp])) ** 2)

        return {"pde": (res ** 2).mean(),
                "pde_pts": res,
                "col_coords": coords,
                "bc": sum(t.mean() for t in bc_terms)}
