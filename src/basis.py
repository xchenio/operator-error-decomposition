"""目标子空间（target subspace）的构造。

两类来源：
  1) prior  —— 解析谱基（Chebyshev / sin / Fourier），**完全不需要数据**；
  2) data   —— 少量真实解做 SVD 得到的主子空间。

两者都返回 (Q, w)：
  Q : (N, r) 列正交的基
  w : (r,)   每个方向的重要性权重（已归一化，和为 1）
"""
from __future__ import annotations

import numpy as np


def chebyshev_basis(x: np.ndarray, K: int) -> np.ndarray:
    """x in [0,1] -> (N, K)，T_0..T_{K-1}(2x-1)"""
    z = 2.0 * x - 1.0
    B = np.ones((x.shape[0], K))
    if K >= 2:
        B[:, 1] = z
    for k in range(2, K):
        B[:, k] = 2.0 * z * B[:, k - 1] - B[:, k - 2]
    return B


def sin_basis(x: np.ndarray, K: int) -> np.ndarray:
    """x in [0,1] -> (N, K)，sin(pi k x), k=1..K（满足零 Dirichlet）"""
    k = np.arange(1, K + 1)
    return np.sin(np.pi * np.outer(x, k))


def fourier_basis(x: np.ndarray, K: int) -> np.ndarray:
    """x in [0,1] 周期 -> (N, 2K+1)：1, cos(2pi k x), sin(2pi k x)"""
    cols = [np.ones_like(x)]
    deg = [0]
    for k in range(1, K + 1):
        cols.append(np.cos(2 * np.pi * k * x))
        cols.append(np.sin(2 * np.pi * k * x))
        deg += [k, k]
    return np.stack(cols, axis=1), np.array(deg, dtype=float)


def _dim_basis(x: np.ndarray, kind: str, K: int):
    if kind == "cheb":
        return chebyshev_basis(x, K), np.arange(K, dtype=float)
    if kind == "sin":
        return sin_basis(x, K), np.arange(1, K + 1, dtype=float)
    if kind == "fourier":
        return fourier_basis(x, K)
    raise ValueError(f"unknown basis kind: {kind}")


def orthonormalize(A: np.ndarray) -> np.ndarray:
    """A: (N, r) -> 列正交 Q (N, r')，丢弃数值上线性相关的列。"""
    Q, R = np.linalg.qr(A)
    diag = np.abs(np.diag(R)) if R.ndim == 2 else np.abs(R)
    keep = diag > 1e-10 * max(diag[0], 1e-300)
    return Q[:, keep]


def build_prior_basis(coords: np.ndarray, spec, ranges, n_basis: int,
                      weight_power: float = 2.0, max_degree: int = 40):
    """张量积解析基，按总次数升序取前 n_basis 个。

    coords : (N, d)
    spec   : 每个维度一个 (kind, K)，kind ∈ {cheb, sin, fourier}
    ranges : 每个维度 (lo, hi)，用于把坐标归一化到 [0,1]
    """
    coords = np.asarray(coords, dtype=np.float64)
    d = coords.shape[1]
    bases, degs = [], []
    for j in range(d):
        lo, hi = ranges[j]
        xj = (coords[:, j] - lo) / (hi - lo)
        kind, K = spec[j]
        B, dg = _dim_basis(xj, kind, K)
        bases.append(B)
        degs.append(dg)

    # 张量积，按总次数排序
    idx = np.indices(tuple(b.shape[1] for b in bases)).reshape(d, -1).T
    total_deg = sum(degs[j][idx[:, j]] for j in range(d))
    order = np.argsort(total_deg, kind="stable")
    idx_deg = idx[order]
    deg_sorted = total_deg[order]
    idx_deg = idx_deg[deg_sorted <= max_degree]
    deg_sorted = deg_sorted[deg_sorted <= max_degree]
    if len(idx_deg) > n_basis:
        idx_deg = idx_deg[:n_basis]
        deg_sorted = deg_sorted[:n_basis]

    cols = [np.prod([bases[j][:, idx_deg[i, j]] for j in range(d)], axis=0)
            for i in range(len(idx_deg))]
    A = np.stack(cols, axis=1) if cols else np.zeros((coords.shape[0], 0))
    Q = orthonormalize(A)
    w = 1.0 / np.power(1.0 + deg_sorted[:Q.shape[1]], weight_power)
    w = w / w.sum()
    return Q.astype(np.float32), w.astype(np.float32)


def energy_rank(sv: np.ndarray, thr: float = 0.999) -> int:
    """累计能量达到 thr 所需的模态数。"""
    e = np.cumsum(sv ** 2)
    e = e / max(e[-1], 1e-300)
    return int(np.searchsorted(e, thr) + 1)


def build_random_subspace(probe: np.ndarray, spec, ranges, n_basis: int,
                          seed: int = 0, pool_factor: int = 4):
    """对照用：在"同一类光滑函数"里随机取一个 n_basis 维子空间。

    做法：先构造 pool_factor 倍大的解析基池，再用随机高斯矩阵混合出一个
    n_basis 维子空间。它保留了"光滑函数"这一先验，但**打乱了谱结构/排序**，
    用来回答：对齐的收益来自"谱先验本身"还是"只要逼 trunk 张成高维子空间就行"。
    """
    pool, _ = build_prior_basis(probe, spec, ranges, max(n_basis * pool_factor, n_basis + 8))
    rng = np.random.default_rng(seed + 991)
    A = pool @ rng.normal(0, 1, (pool.shape[1], n_basis))
    Q = orthonormalize(A)
    w = 1.0 / np.arange(1, Q.shape[1] + 1, dtype=np.float64) ** 2
    w = w / w.sum()
    return Q.astype(np.float32), w.astype(np.float32)


def build_random_gauss_subspace(n_points: int, n_basis: int, seed: int = 0):
    """更强的对照：纯高斯随机子空间（非光滑）。"""
    rng = np.random.default_rng(seed + 991)
    A = rng.normal(0, 1, (n_points, n_basis))
    Q = orthonormalize(A)
    w = np.ones(Q.shape[1], dtype=np.float32) / Q.shape[1]
    return Q.astype(np.float32), w


def build_data_subspace(solutions: np.ndarray, n_basis: int):
    """solutions: (N, n_samples)，每列一个真实解在固定网格上的取值。

    返回 (Q, w)：Q 为前 r 个左奇异向量，w 为能量权重 σ²/Σσ²。
    """
    S = np.asarray(solutions, dtype=np.float64)
    # 去掉均值列（不中心化，因为解本身有结构；但去掉全零列）
    keep = np.linalg.norm(S, axis=0) > 1e-14
    S = S[:, keep]
    if S.shape[1] == 0:
        raise ValueError("empty solution matrix")
    U, s, _ = np.linalg.svd(S, full_matrices=False)
    r = int(min(n_basis, len(s)))
    Q = U[:, :r]
    w = s[:r] ** 2
    w = w / w.sum()
    return Q.astype(np.float32), w.astype(np.float32)


def build_combined_subspace(prior_Q: np.ndarray, prior_w: np.ndarray,
                            data_Q: np.ndarray, data_w: np.ndarray,
                            n_basis: int):
    """把 prior 基与 data 基按权重拼接后重新正交化，取前 n_basis 个方向。"""
    A = np.concatenate([prior_Q * np.sqrt(prior_w)[None, :],
                        data_Q * np.sqrt(data_w)[None, :]], axis=1)
    Q, R = np.linalg.qr(A)
    diag = np.abs(np.diag(R)) if R.ndim == 2 else np.abs(R)
    # 按 QR 得到的"重要性"排序（对角元绝对值）
    order = np.argsort(-diag)
    Q = Q[:, order][:, :n_basis]
    w = 1.0 / np.arange(1, Q.shape[1] + 1, dtype=np.float64) ** 2
    w = w / w.sum()
    return Q.astype(np.float32), w.astype(np.float32)
