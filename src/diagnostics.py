"""基函数（子空间）诊断。

核心对象：trunk 特征矩阵 Phi ∈ R^{N×p}，其列张成模型"可表示的解空间" span(Phi)。
目标子空间 U ∈ R^{N×r}（真实解 / 解析谱基）的主方向。

指标：
  - captured_energy : 目标子空间被 span(Phi) 捕获的能量比例，∈[0,1]，越大越好
  - principal angles: 两个子空间的主角度（min / mean / max），越小越好
  - grassmann       : 主角度的二范数（Grassmann 距离），越小越好
  - eff_rank        : Phi 的有效秩（奇异值熵），衡量 trunk 实际用了多少"独立基"
  - cond            : Phi 的条件数，衡量基的病态程度
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# ---------------- numpy 侧（分析用） ----------------

def _orth(A: np.ndarray) -> np.ndarray:
    Q, R = np.linalg.qr(A)
    diag = np.abs(np.diag(R)) if R.ndim == 2 else np.abs(R)
    keep = diag > 1e-10 * max(diag[0], 1e-300)
    return Q[:, keep]


def effective_rank(A: np.ndarray, eps: float = 1e-12) -> float:
    s = np.linalg.svd(np.asarray(A, dtype=np.float64), compute_uv=False)
    s = s[s > eps * max(s[0], 1e-300)]
    if len(s) == 0:
        return 0.0
    p = s ** 2
    p = p / p.sum()
    ent = float(-(p * np.log(p + 1e-300)).sum())
    return float(np.exp(ent))


def singular_values(A: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(np.asarray(A, dtype=np.float64), compute_uv=False)
    s0 = max(s[0], 1e-300)
    return s / s0


def principal_angles(Q1: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    """返回主角度（弧度），升序。"""
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def captured_energy_np(Phi: np.ndarray, U: np.ndarray,
                       w: Optional[np.ndarray] = None) -> float:
    """目标子空间 U 被 span(Phi) 捕获的能量比例（加权）。"""
    Q = _orth(np.asarray(Phi, dtype=np.float64))
    Un = np.asarray(U, dtype=np.float64)
    C = Q.T @ Un                                  # (rankQ, r)
    per_col = np.sum(C ** 2, axis=0)              # 每个目标方向被捕获的比例
    per_col = np.clip(per_col, 0.0, 1.0)
    if w is None:
        return float(per_col.mean())
    w = np.asarray(w, dtype=np.float64)
    w = w[: len(per_col)]
    return float((per_col * w).sum() / max(w.sum(), 1e-300))


def subspace_metrics(Phi: np.ndarray, U: np.ndarray,
                     w: Optional[np.ndarray] = None) -> dict:
    """一次算全套诊断指标。"""
    Phi = np.asarray(Phi, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    Q = _orth(Phi)
    Un = U / np.clip(np.linalg.norm(U, axis=0, keepdims=True), 1e-300, None)
    ang = principal_angles(Q, Un)
    out = {
        "captured_energy": captured_energy_np(Phi, Un, w),
        "angle_min_deg": float(np.degrees(ang.min())) if len(ang) else float("nan"),
        "angle_mean_deg": float(np.degrees(ang.mean())) if len(ang) else float("nan"),
        "angle_max_deg": float(np.degrees(ang.max())) if len(ang) else float("nan"),
        "grassmann": float(np.sqrt((ang ** 2).sum())) if len(ang) else float("nan"),
        "eff_rank": effective_rank(Phi),
        "cond": float(np.linalg.cond(Phi)) if min(Phi.shape) > 0 else float("nan"),
    }
    return out


def spectral_centroid(Phi: np.ndarray, U: np.ndarray, grid_shape, axis: int) -> dict:
    """沿某个网格轴做 DCT，比较 trunk 基与解的频率重心（谱偏差证据）。

    grid_shape: 例如 (Nt, Nx)，要求 prod(grid_shape) == N
    axis: 对哪个轴做频谱（如 x 轴 = 1）
    """
    from scipy.fft import dct

    def centroid(M):
        # M: (N, k)
        A = M.reshape(*grid_shape, -1)
        A = np.moveaxis(A, axis, 1)                # (n_axis, ...)
        A = A.reshape(A.shape[0], -1)
        C = dct(A, axis=0, norm="ortho")
        e = (C ** 2).sum(axis=1)
        e = e / max(e.sum(), 1e-300)
        idx = np.arange(len(e))
        return float((e * idx).sum() / len(e))

    return {
        "trunk_centroid": centroid(np.asarray(Phi, dtype=np.float64)),
        "target_centroid": centroid(np.asarray(U, dtype=np.float64)),
    }


# ---------------- torch 侧（可微，用于对齐损失） ----------------

def alignment_loss(phi: torch.Tensor, U: torch.Tensor,
                   w: Optional[torch.Tensor] = None,
                   ridge_rel: float = 1e-4) -> torch.Tensor:
    """可微的"未被捕获能量"：L = 1 - ||P_phi U||_W^2。

    phi: (N, p) trunk 特征（需要梯度）
    U  : (N, r) 目标子空间基（固定 buffer）
    w  : (r,)   方向权重（已归一化）；None 表示等权
    用岭正则化的 Gram 矩阵避免求显式正交基，便于反向传播。
    """
    N, p = phi.shape
    G = phi.transpose(0, 1) @ phi                       # (p, p)
    ridge = ridge_rel * (torch.diagonal(G).mean() + 1e-12)
    A = G + ridge * torch.eye(p, device=phi.device, dtype=phi.dtype)
    Z = torch.linalg.solve(A, phi.transpose(0, 1) @ U)  # (p, r)
    per = torch.einsum("nr,np,pr->r", U, phi, Z)        # diag(U^T P U)
    per = per.clamp(0.0, 1.0)
    if w is None:
        cap = per.mean()
    else:
        w = w.to(phi.dtype)
        cap = (per * w).sum() / w.sum().clamp_min(1e-12)
    return 1.0 - cap
