"""数据集构建与缓存。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pdes import PDEProblem


# 不影响 (coef, u, s) 生成的 PDE 参数：不进数据缓存文件名，否则会为同一份数据
# 重复求解（burgers2d 生成一次约 550s）。hard_ic 只改变损失的参数化。
DATASET_AGNOSTIC_KWARGS = {"hard_ic"}


def dataset_cache_path(root, pde: str, pde_kwargs: dict | None,
                       n_train: int, n_val: int, n_test: int, seed: int = 0):
    """数据缓存文件名：所有脚本必须用同一个函数，否则会各自生成一份副本。"""
    key = "_".join(f"{k}{v}" for k, v in sorted((pde_kwargs or {}).items())
                   if k not in DATASET_AGNOSTIC_KWARGS) or "default"
    return Path(root) / "data" / f"{pde}_{key}_n{n_train}_{n_val}_{n_test}_s{seed}.npz"


@dataclass
class Split:
    coef: np.ndarray   # (n, n_coef)
    u: np.ndarray      # (n, m)     branch 输入
    s: np.ndarray      # (n, Ng)    解（与 pde.grid 对应）

    def __len__(self):
        return self.u.shape[0]


def generate_split(pde: PDEProblem, n: int, rng: np.random.Generator) -> Split:
    coef = pde.sample_coeffs(n, rng)
    u = pde.branch_input(coef)
    t0 = time.time()
    s = pde.solve(coef)
    if n > 0:
        print(f"  solve {n} samples in {time.time()-t0:.1f}s "
              f"(|s|max={np.abs(s).max():.3f}, std={s.std():.3f})")
    return Split(coef.astype(np.float32), u.astype(np.float32), s.astype(np.float32))


def build_dataset(pde: PDEProblem, n_train: int, n_val: int, n_test: int,
                  seed: int = 0, cache: str | None = None,
                  force_regen: bool = False) -> dict:
    if cache and not force_regen and Path(cache).exists():
        z = np.load(cache)
        out = {}
        for k in ("train", "val", "test"):
            out[k] = Split(z[f"{k}_coef"], z[f"{k}_u"], z[f"{k}_s"])
        print(f"[data] loaded cache {cache}")
        return out

    rng = np.random.default_rng(seed)
    out = {}
    for name, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        print(f"[data] generating {name} ({n}) ...")
        out[name] = generate_split(pde, n, rng)

    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{f"{k}_coef": v.coef for k, v in out.items()},
                            **{f"{k}_u": v.u for k, v in out.items()},
                            **{f"{k}_s": v.s for k, v in out.items()})
        print(f"[data] saved cache {cache}")
    return out
