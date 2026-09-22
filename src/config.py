"""实验配置：问题默认参数、方法（训练阶段）预设、目标子空间构造。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .basis import (build_combined_subspace, build_data_subspace, build_prior_basis,
                    build_random_gauss_subspace, build_random_subspace)
from .trainer import StageCfg

PDE_DEFAULTS = {
    "burgers": dict(nu=0.02, Nx=256, Nx_store=128, T=1.0, Nt=41, dt=5e-4,
                    K=8, m=128, ell=3.0),
    "burgers2d": dict(nu=0.05, Nx=64, Ny=64, Ns=32, T=1.0, Nt=21, dt=5e-4,
                      K=4, m=16, ell=2.5),
    "dr": dict(D=0.01, k=0.01, Nx=101, T=1.0, Nt=41, dt=1e-3,
               K=12, m=101, amp=0.5, decay=1.2),
    "darcy": dict(N=41, nmode=3, m=21, f=10.0, sigma=0.5),
}

MODEL_DEFAULTS = {
    "burgers": dict(width=100, branch_hidden=128, branch_layers=3,
                    trunk_hidden=128, trunk_layers=3),
    "burgers2d": dict(width=64, branch_hidden=128, branch_layers=3,
                      trunk_hidden=128, trunk_layers=3),
    "dr": dict(width=100, branch_hidden=128, branch_layers=3,
               trunk_hidden=128, trunk_layers=3),
    "darcy": dict(width=100, branch_hidden=256, branch_layers=3,
                  trunk_hidden=128, trunk_layers=3),
}

SAMPLING_DEFAULTS = {
    "burgers": dict(n_col=64, n_ic=64, n_bc=32, bs=16, n_pts=256),
    "burgers2d": dict(n_col=64, n_ic=48, n_bc=24, bs=16, n_pts=256),
    "dr": dict(n_col=64, n_ic=64, n_bc=32, bs=16, n_pts=256),
    "darcy": dict(n_col=64, n_ic=0, n_bc=64, bs=16, n_pts=256),
}


@dataclass
class ExpCfg:
    pde: str = "dr"
    method: str = "pi_baseline"
    n_data: int = 0
    seed: int = 0
    align_kind: str = "prior"      # prior | data | hybrid | random | randomgauss | none
    align_weight: float = 1.0
    ortho_weight: float = 0.0      # trunk 去相关正则（无目标子空间的对照）
    causal: float = 0.0            # 因果加权强度 eps（0=关闭）
    causal_M: int = 20             # 时间方向分块数
    lbfgs_steps: int = 200         # L-BFGS 精调外层步数（每步 max_iter=20）
    steps: int = 20000
    lr: float = 1e-3
    pretrain_frac: float = 0.25
    n_train: int = 200
    n_val: int = 50
    n_test: int = 50
    eval_every: int = 500
    diag_every: int = 500
    grad_clip: Optional[float] = None
    data_split: str = "train"
    pde_kwargs: dict = field(default_factory=dict)
    model_kwargs: dict = field(default_factory=dict)
    sampling_kwargs: dict = field(default_factory=dict)
    lambdas_override: dict = field(default_factory=dict)   # 覆盖 pde/ic/bc/data 权重

    def resolved(self):
        pkw = dict(PDE_DEFAULTS[self.pde]); pkw.update(self.pde_kwargs)
        mkw = dict(MODEL_DEFAULTS[self.pde]); mkw.update(self.model_kwargs)
        skw = dict(SAMPLING_DEFAULTS[self.pde]); skw.update(self.sampling_kwargs)
        return pkw, mkw, skw

    def tag(self) -> str:
        extra = "_".join(f"{k}{v}" for k, v in sorted(self.pde_kwargs.items()))
        name = f"{self.pde}_{extra}" if extra else self.pde
        m = self.model_kwargs or {}
        if m.get("trunk_type", "mlp") != "mlp":
            name += f"-{m['trunk_type']}"
        if m.get("width"):
            name += f"-w{m['width']}"
        if m.get("init_gain", 1.0) != 1.0:
            name += f"-g{m['init_gain']:g}"
        if self.causal:
            name += f"-causal{self.causal:g}"
        if self.ortho_weight:
            name += f"-ortho{self.ortho_weight:g}"
        # 损失权重必须进 tag：否则同一次批量里 ic/bc 权重不同的 run 会写同一个文件名
        # 互相覆盖，且 --skip-existing 永远匹配不上（与 README §6.5 同类的静默问题）
        if self.lambdas_override:
            name += "_lam" + "_".join(f"{k}{v:g}"
                                      for k, v in sorted(self.lambdas_override.items()))
        return (f"{name}_{self.method}_nd{self.n_data}"
                f"_a{self.align_kind}{self.align_weight}_s{self.seed}")


def make_stages(cfg: ExpCfg, skw: dict) -> list:
    """把方法名翻译成训练阶段列表。"""
    m = cfg.method
    n_col, n_ic, n_bc = skw["n_col"], skw["n_ic"], skw["n_bc"]
    n_pts = skw["n_pts"]
    aw = cfg.align_weight if cfg.align_kind != "none" else 0.0
    ow = cfg.ortho_weight
    pre = int(cfg.steps * cfg.pretrain_frac)

    def phys(steps, **kw):
        lamb = dict(cfg.lambdas_override) if cfg.lambdas_override else {}
        if "lambdas" in kw:
            if kw["lambdas"]:
                lamb.update(kw["lambdas"])
            kw = {k: v for k, v in kw.items() if k != "lambdas"}
        base = dict(kind="physics", steps=steps, lr=cfg.lr, n_col=n_col,
                    n_ic=n_ic, n_bc=n_bc, n_pts=n_pts, align=aw, ortho=ow,
                    causal=cfg.causal, causal_M=cfg.causal_M, lambdas=lamb)
        base.update(kw)
        return StageCfg(**base)

    def dat(steps, **kw):
        base = dict(kind="data", steps=steps, lr=cfg.lr, n_col=n_col,
                    n_ic=n_ic, n_bc=n_bc, n_pts=n_pts, align=aw, ortho=ow)
        base.update(kw)
        return StageCfg(**base)

    if m == "pi_baseline":
        return [phys(cfg.steps, lambdas={})]
    if m == "pi_align":
        return [phys(cfg.steps, lambdas={}, align=aw)]
    if m == "pi_ortho":
        return [phys(cfg.steps, lambdas={}, align=0.0, ortho=ow)]
    def lbf(steps, **kw):
        # L-BFGS 用固定训练集：必须足够大，否则会过拟合固定样本、泛化崩掉
        # （实测用 bs=16×64 点会把 relL2 从 0.72 炸到 2.00）
        base = dict(kind="lbfgs", steps=steps, lr=1.0, n_col=32, n_ic=32, n_bc=16,
                    n_pts=max(n_pts, 512), lbfgs_bs=64, align=aw, ortho=0.0,
                    causal=0.0, causal_M=cfg.causal_M,
                    lambdas=dict(cfg.lambdas_override or {}))
        base.update(kw)
        return StageCfg(**base)

    if m == "pi_lbfgs":
        return [phys(cfg.steps, lambdas={}, align=0.0), lbf(cfg.lbfgs_steps)]
    if m == "pi_align_lbfgs":
        return [phys(cfg.steps, lambdas={}, align=aw), lbf(cfg.lbfgs_steps, align=aw)]
    if m == "pi_causal":
        return [phys(cfg.steps, lambdas={}, align=0.0, ortho=0.0,
                     causal=cfg.causal, causal_M=cfg.causal_M)]
    if m == "pi_causal_align":
        return [phys(cfg.steps, lambdas={}, align=aw, ortho=0.0,
                     causal=cfg.causal, causal_M=cfg.causal_M)]
    if m == "pi_causal_fourier":
        # 配合 --trunk-type fourier
        return [phys(cfg.steps, lambdas={}, align=0.0, ortho=0.0,
                     causal=cfg.causal, causal_M=cfg.causal_M)]
    if m == "pi_align_ortho":
        return [phys(cfg.steps, lambdas={}, align=aw, ortho=ow)]
    if m == "pi_data":
        return [phys(cfg.steps, lambdas={"data": 1.0}, align=0.0)]
    if m == "pi_data_align":
        return [phys(cfg.steps, lambdas={"data": 1.0}, align=aw)]
    if m == "two_stage_freeze":
        return [dat(pre, align=0.0), phys(cfg.steps, freeze_trunk=True, align=0.0)]
    if m == "two_stage_slow":
        return [dat(pre, align=0.0), phys(cfg.steps, trunk_lr_mult=0.1, align=0.0)]
    if m == "two_stage_align":
        return [dat(pre, align=aw), phys(cfg.steps, freeze_trunk=True, align=aw)]
    if m == "two_stage_data":
        return [dat(pre, align=0.0),
                phys(cfg.steps, lambdas={"data": 1.0}, freeze_trunk=True, align=0.0)]
    if m == "data_only":
        return [dat(cfg.steps, align=0.0)]
    raise ValueError(f"unknown method {m}")


def pick_data_idx(n_avail: int, n_data: int, seed: int) -> np.ndarray:
    if n_data <= 0:
        return np.zeros(0, dtype=int)
    return np.random.default_rng(seed + 1234).choice(
        n_avail, size=min(n_data, n_avail), replace=False)


def make_probe_idx(n_grid: int, max_probe: int, seed: int) -> np.ndarray:
    if n_grid <= max_probe:
        return np.arange(n_grid)
    return np.random.default_rng(seed + 77).choice(n_grid, size=max_probe, replace=False)


def build_targets(pde, splits, probe_idx, data_idx, width: int, kind: str):
    """返回 (align_U, align_w, diag_U, diag_w)。

    align_* : 参与对齐损失的目标子空间
    diag_*  : 只用 val 解构造，仅用于诊断，绝不参与训练
    """
    probe = pde.grid[probe_idx]
    S_val = splits["val"].s[:, probe_idx].T                      # (Np, n_val)
    diag_U, diag_w = build_data_subspace(S_val, width)

    if kind == "none" or (kind == "data" and len(data_idx) == 0):
        return None, None, diag_U, diag_w
    if kind == "prior":
        U, w = build_prior_basis(probe, pde.prior_spec, pde.ranges, width)
    elif kind == "random":
        U, w = build_random_subspace(probe, pde.prior_spec, pde.ranges, width,
                                     seed=0)
    elif kind == "randomgauss":
        U, w = build_random_gauss_subspace(probe.shape[0], width, seed=0)
    elif kind == "data":
        U, w = build_data_subspace(splits["train"].s[data_idx][:, probe_idx].T, width)
    elif kind == "hybrid":
        Up, wp = build_prior_basis(probe, pde.prior_spec, pde.ranges, width)
        Ud, wd = build_data_subspace(splits["train"].s[data_idx][:, probe_idx].T, width)
        U, w = build_combined_subspace(Up, wp, Ud, wd, width)
    else:
        raise ValueError(f"unknown align kind {kind}")
    return U, w, diag_U, diag_w
