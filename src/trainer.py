"""PI-DeepONet 训练器：支持多阶段（数据预训练 / 物理训练）、trunk 冻结、对齐损失。"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .data import Split
from .diagnostics import alignment_loss, singular_values, subspace_metrics
from .pdes import PDEProblem
from .utils import rel_l2
from .resume import atomic_save, random_state, restore_random


@dataclass
class StageCfg:
    kind: str                       # 'physics' | 'data' | 'lbfgs'
    steps: int
    lr: float = 1e-3
    lambdas: dict = field(default_factory=dict)   # pde / ic / bc / data
    align: float = 0.0              # 对齐损失权重
    ortho: float = 0.0              # trunk 特征去相关（正交）正则权重
    causal: float = 0.0             # 因果加权强度 eps；0 = 关闭
    causal_M: int = 20              # 时间方向分块数
    lbfgs_bs: int = 0               # L-BFGS 阶段固定的输入函数个数（0=沿用 bs）
    freeze_trunk: bool = False
    trunk_lr_mult: float = 1.0
    n_col: int = 128
    n_ic: int = 64
    n_bc: int = 32
    n_pts: int = 256                # data 阶段每样本采样点数


DEFAULT_LAMBDAS = {"pde": 1.0, "ic": 1.0, "bc": 1.0, "data": 1.0}


def _to_device(obj, device, dtype=torch.float32):
    if isinstance(obj, dict):
        return {k: _to_device(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return torch.as_tensor(obj, dtype=dtype, device=device)
    return obj


class Trainer:
    def __init__(self, model: torch.nn.Module, pde: PDEProblem, splits: dict,
                 device: torch.device, seed: int = 0,
                 align_U: Optional[np.ndarray] = None,
                 align_w: Optional[np.ndarray] = None,
                 diag_U: Optional[np.ndarray] = None,
                 diag_w: Optional[np.ndarray] = None,
                 probe_coords: Optional[np.ndarray] = None,
                 n_data: int = 0, data_split: str = "train",
                 grad_clip: Optional[float] = None, logger=None,
                 causal_M: int = 20):
        self.model = model.to(device)
        self.pde = pde
        self.splits = splits
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.grad_clip = grad_clip
        self.logger = logger
        self.n_data = n_data

        self.grid_t = torch.as_tensor(pde.grid, dtype=torch.float32, device=device)
        self.probe_t = torch.as_tensor(probe_coords if probe_coords is not None
                                       else pde.grid, dtype=torch.float32, device=device)

        # 对齐目标（参与训练）
        self.align_U, self.align_w = align_U, align_w
        if align_U is not None:
            self.align_U_t = torch.as_tensor(align_U, dtype=torch.float32, device=device)
            self.align_w_t = (torch.as_tensor(align_w, dtype=torch.float32, device=device)
                              if align_w is not None else None)
        # 诊断目标（只用 val，绝不参与训练）
        self.diag_U, self.diag_w = diag_U, diag_w

        # 有监督数据的索引
        sp: Split = splits[data_split]
        n_avail = len(sp)
        if n_data > 0:
            self.data_idx = np.random.default_rng(seed + 1234).choice(
                n_avail, size=min(n_data, n_avail), replace=False)
        else:
            self.data_idx = np.zeros(0, dtype=int)

        # 因果加权状态：时间分块的 PDE 损失滑动平均
        self.causal_M = causal_M
        self.slab_loss = torch.zeros(causal_M, device=device)
        self.slab_ready = torch.zeros(causal_M, dtype=torch.bool, device=device)
        self.causal_alpha = 0.9

        self.history = {"step": [], "loss": [], "val_rel_l2": []}
        self.diag_history = {"step": [], "captured_energy": [], "eff_rank": [],
                             "angle_mean_deg": [], "cond": []}
        self.global_step = 0
        self.checkpoint_path = None
        self.checkpoint_every = 100
        self._resume_state = None

    def _save_progress(self, opt, iteration):
        if self.checkpoint_path is None:
            return
        atomic_save(dict(identity=self._checkpoint_identity, stage=self._stage_index,
                         iteration=iteration, model=self.model.state_dict(),
                         optimizer=opt.state_dict(), random=random_state(self.rng),
                         global_step=self.global_step, history=self.history,
                         diag_history=self.diag_history, initial_diag=self._initial_diag,
                         slab_loss=self.slab_loss, slab_ready=self.slab_ready),
                    self.checkpoint_path)

    # ---------------- 工具 ----------------
    def log(self, msg):
        if self.logger:
            self.logger.log(msg)
        else:
            print(msg, flush=True)

    def _align_term(self):
        if self.align_U is None:
            return None
        phi = self.model.trunk_features(self.probe_t)
        return alignment_loss(phi, self.align_U_t, self.align_w_t)

    def _ortho_term(self):
        """trunk 特征去相关正则（无目标子空间的"纯多样性"对照）。

        G = Phi^T Phi，归一化后与 I 的差距：同时惩罚列间相关与范数不均。
        整体尺度不变（除以对角均值），因此不会干扰输出幅值的学习。
        """
        phi = self.model.trunk_features(self.probe_t)      # (N, p)
        G = phi.transpose(0, 1) @ phi                      # (p, p)
        s = torch.diagonal(G).mean().clamp_min(1e-12)
        p = G.shape[0]
        Gn = G / s
        return ((Gn - torch.eye(p, device=phi.device, dtype=phi.dtype)) ** 2).mean()

    def _causal_pde(self, res: torch.Tensor, coords: torch.Tensor):
        """因果加权 PDE 损失（Wang, Shankar & Karniadakis 2022）。

        把时间轴切成 M 块，第 i 块的权重
            w_i = exp(-eps * sum_{j<i} L_j)
        即只有更早的时间块被拟合好之后，后面的时间块才获得训练权重，
        从而避免"先拟合 t 大处导致整个解塌掉"的因果性违背。
        """
        td = getattr(self.pde, "time_dim", None)
        if td is None:
            return (res ** 2).mean()
        t = coords[..., td]
        t0, T = self.pde.t_range
        tn = ((t - t0) / max(T - t0, 1e-12)).clamp(0.0, 1.0)
        idx = (tn * self.causal_M).long().clamp(0, self.causal_M - 1)   # (B,N)

        # 1) 用 detach 的逐块残差更新滑动平均
        with torch.no_grad():
            fi = idx.reshape(-1).to(torch.int64)
            r2 = (res.detach() ** 2).reshape(-1).to(torch.float32)
            M = self.causal_M
            ssum = torch.zeros(M, device=self.device).index_add(0, fi, r2)
            cnt = torch.zeros(M, device=self.device).index_add(
                0, fi, torch.ones_like(r2))
            cur = ssum / cnt.clamp_min(1.0)
            hit = cnt > 0
            prev = torch.where(self.slab_ready, self.slab_loss, cur)
            new = self.causal_alpha * prev + (1 - self.causal_alpha) * cur
            self.slab_loss = torch.where(hit, new, self.slab_loss)
            self.slab_ready = self.slab_ready | hit

        # 2) 严格"更早"的累计损失 -> 权重
        with torch.no_grad():
            cum = torch.cumsum(self.slab_loss, dim=0) - self.slab_loss
            w = torch.exp(-self.causal_eps * cum)
            w = w[idx]
            w = w / w.mean().clamp_min(1e-12)
        return (w * res ** 2).mean()

    def _optimizer(self, stage: StageCfg):
        trunk_was_trainable = any(p.requires_grad for p in self.model.trunk.parameters())
        if stage.freeze_trunk:
            for p in self.model.trunk.parameters():
                p.requires_grad_(False)
        elif not trunk_was_trainable:
            for p in self.model.trunk.parameters():
                p.requires_grad_(True)
        groups = [
            {"params": self.model.branch.parameters(), "lr": stage.lr},
            {"params": [self.model.bias], "lr": stage.lr},
        ]
        if any(p.requires_grad for p in self.model.trunk.parameters()):
            groups.append({"params": self.model.trunk.parameters(),
                           "lr": stage.lr * stage.trunk_lr_mult})
        return torch.optim.Adam(groups)

    # ---------------- 批构造 ----------------
    def _sample_train_batch(self, bs: int):
        sp = self.splits["train"]
        idx = self.rng.choice(len(sp), size=bs, replace=False)
        return idx

    def physics_terms(self, idx, stage: StageCfg):
        sp = self.splits["train"]
        coef = sp.coef[idx]
        u_b = torch.as_tensor(sp.u[idx], dtype=torch.float32, device=self.device)
        pts = self.pde.sample_points(coef, stage.n_col, stage.n_ic, stage.n_bc, self.rng)
        pts = _to_device(pts, self.device)
        evalr = self.model.evaluator(u_b)
        terms = self.pde.losses(evalr, pts)
        if stage.causal > 0.0 and "pde_pts" in terms:
            self.causal_eps = stage.causal
            self.causal_M = stage.causal_M
            if self.slab_loss.shape[0] != stage.causal_M:
                self.slab_loss = torch.zeros(stage.causal_M, device=self.device)
                self.slab_ready = torch.zeros(stage.causal_M, dtype=torch.bool,
                                              device=self.device)
            terms["pde"] = self._causal_pde(terms["pde_pts"], terms["col_coords"])
        return terms

    def data_term(self, idx, n_pts):
        sp = self.splits["train"]
        Ng = sp.s.shape[1]
        gi = self.rng.choice(Ng, size=n_pts, replace=False)
        u_b = torch.as_tensor(sp.u[idx], dtype=torch.float32, device=self.device)
        coords = torch.as_tensor(self.pde.grid[gi], dtype=torch.float32, device=self.device)
        target = torch.as_tensor(sp.s[idx][:, gi], dtype=torch.float32, device=self.device)
        pred = self.model(u_b, coords)
        return (pred - target) ** 2

    # ---------------- 评估 / 诊断 ----------------
    @torch.no_grad()
    def evaluate(self, name: str = "val", bs: int = 32) -> float:
        self.model.eval()
        sp = self.splits[name]
        preds, gts = [], []
        grid = self.grid_t
        hard = getattr(self.pde, "hard_ic", False)
        t_grid = grid[:, 0] if hard else None
        for i in range(0, len(sp), bs):
            j = min(i + bs, len(sp))
            u_b = torch.as_tensor(sp.u[i:j], dtype=torch.float32, device=self.device)
            out = self.model.predict_on_grid(u_b, grid)
            if hard:
                u0 = self.pde.ic_values(sp.coef[i:j], self.pde.grid, order=0)[0]
                out = torch.as_tensor(u0, dtype=torch.float32,
                                      device=self.device) + t_grid[None, :] * out
            preds.append(out.cpu())
            gts.append(torch.as_tensor(sp.s[i:j]))
        self.model.train()
        return rel_l2(torch.cat(preds), torch.cat(gts))

    @torch.no_grad()
    def diagnostics(self) -> dict:
        self.model.eval()
        phi = self.model.trunk_features(self.probe_t)
        Phi = phi.detach().cpu().numpy().astype(np.float64)
        self.model.train()
        m = subspace_metrics(Phi, self.diag_U, self.diag_w)
        m["sv"] = singular_values(Phi)
        return m

    # ---------------- 主循环 ----------------
    def run_stage(self, stage: StageCfg, bs: int, eval_every: int, diag_every: int):
        lam = dict(DEFAULT_LAMBDAS)
        lam.update(stage.lambdas or {})
        opt = self._optimizer(stage)
        use_data = (stage.kind == "data") or (stage.kind == "physics" and self.n_data > 0
                                              and lam.get("data", 0.0) > 0 and len(self.data_idx) > 0)
        t0 = time.time()
        start = 0
        if self._resume_state is not None and self._resume_state['stage'] == self._stage_index:
            opt.load_state_dict(self._resume_state['optimizer'])
            start = self._resume_state['iteration']
            restore_random(self._resume_state['random'], self.rng)
            self._resume_state = None
        for it in range(start, stage.steps):
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=self.device)
            terms_seen = {}
            if stage.kind == "physics":
                idx = self._sample_train_batch(bs)
                terms = self.physics_terms(idx, stage)
                for k, v in terms.items():
                    if k in lam:
                        loss = loss + lam[k] * v
                # 只在需要打印时才做 float() —— 否则每步都会强制 GPU→CPU 同步，
                # 把 GPU 的流水打散，是 GPU 利用率上不去的一个真实原因。
                if eval_every and (it + 1) % eval_every == 0:
                    terms_seen.update(
                        {k: float(v.detach()) for k, v in terms.items()
                         if torch.is_tensor(v) and v.dim() == 0})
            else:
                idx = self.data_idx if len(self.data_idx) > 0 else self._sample_train_batch(bs)
                dl = self.data_term(idx, stage.n_pts).mean()
                loss = loss + lam["data"] * dl
                terms_seen["data"] = float(dl.detach())

            # 数据项（物理阶段同时用少量监督数据）
            if stage.kind == "physics" and lam.get("data", 0.0) > 0 and len(self.data_idx) > 0:
                loss = loss + lam["data"] * self.data_term(self.data_idx, stage.n_pts).mean()

            if stage.align > 0.0:
                al = self._align_term()
                if al is not None:
                    loss = loss + stage.align * al
                    terms_seen["align"] = float(al.detach())
            if stage.ortho > 0.0:
                ot = self._ortho_term()
                loss = loss + stage.ortho * ot
                terms_seen["ortho"] = float(ot.detach())

            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], self.grad_clip)
            opt.step()
            self.global_step += 1

            if eval_every and (it + 1) % eval_every == 0:
                v = self.evaluate("val")
                self.history["step"].append(self.global_step)
                lv = float(loss.detach().cpu())
                self.history["loss"].append(lv)
                self.history["val_rel_l2"].append(v)
                det = " ".join(f"{k}={x:.3e}" for k, x in terms_seen.items())
                self.log(f"  step {self.global_step:6d} | loss {lv:.3e} | "
                         f"val relL2 {v:.4f} | {det}")
            if diag_every and (it + 1) % diag_every == 0:
                m = self.diagnostics()
                self.diag_history["step"].append(self.global_step)
                for k in ("captured_energy", "eff_rank", "angle_mean_deg", "cond"):
                    self.diag_history[k].append(float(m[k]))
            if self.checkpoint_path and ((it + 1) % self.checkpoint_every == 0 or it + 1 == stage.steps):
                self._save_progress(opt, it + 1)
        self.log(f"  [stage {stage.kind}] {stage.steps} steps in {time.time()-t0:.1f}s")

    def run_lbfgs(self, stage: StageCfg, bs: int, eval_every: int, diag_every: int):
        """L-BFGS 精调阶段：配点/监督点固定（DeepXDE 的做法），用强 Wolfe 线搜索。

        Adam 后期在病态损失面上容易抖动，L-BFGS 的二阶信息常能再压一截 opt_gap。
        """
        lam = dict(DEFAULT_LAMBDAS)
        lam.update(stage.lambdas or {})
        sp = self.splits["train"]

        idx = self._sample_train_batch(stage.lbfgs_bs or bs)
        u_b = torch.as_tensor(sp.u[idx], dtype=torch.float32, device=self.device)
        pts = _to_device(self.pde.sample_points(
            sp.coef[idx], stage.n_col, stage.n_ic, stage.n_bc, self.rng), self.device)
        gi = self.rng.choice(sp.s.shape[1], size=stage.n_pts, replace=False)
        coords_d = torch.as_tensor(self.pde.grid[gi], dtype=torch.float32,
                                   device=self.device)
        target_d = torch.as_tensor(sp.s[idx][:, gi], dtype=torch.float32,
                                   device=self.device)
        has_data = lam.get("data", 0.0) > 0 and len(self.data_idx) > 0

        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.LBFGS(params, lr=stage.lr, max_iter=20, history_size=50,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            evalr = self.model.evaluator(u_b)
            terms = self.pde.losses(evalr, pts)
            loss = torch.zeros((), device=self.device)
            for k, v in terms.items():
                if k in lam and torch.is_tensor(v) and v.dim() == 0:
                    loss = loss + lam[k] * v
            if has_data:
                idx2 = self.data_idx
                u_d = torch.as_tensor(sp.u[idx2], dtype=torch.float32, device=self.device)
                pred = self.model(u_d, coords_d)
                tgt = torch.as_tensor(sp.s[idx2][:, gi], dtype=torch.float32,
                                      device=self.device)
                loss = loss + lam["data"] * ((pred - tgt) ** 2).mean()
            if stage.align > 0.0:
                al = self._align_term()
                if al is not None:
                    loss = loss + stage.align * al
            loss.backward()
            return loss

        t0 = time.time()
        for it in range(stage.steps):
            loss = opt.step(closure)
            self.global_step += 1
            if eval_every and (it + 1) % eval_every == 0:
                v = self.evaluate("val")
                lv = float(loss.detach())
                self.history["step"].append(self.global_step)
                self.history["loss"].append(lv)
                self.history["val_rel_l2"].append(v)
                self.log(f"  [lbfgs] step {it+1:4d} | loss {lv:.3e} | val relL2 {v:.4f}")
            if diag_every and (it + 1) % diag_every == 0:
                m = self.diagnostics()
                self.diag_history["step"].append(self.global_step)
                for k in ("captured_energy", "eff_rank", "angle_mean_deg", "cond"):
                    self.diag_history[k].append(float(m[k]))
        self.log(f"  [stage lbfgs] {stage.steps} steps in {time.time()-t0:.1f}s")

    def fit(self, stages, bs: int = 16, eval_every: int = 500, diag_every: int = 500,
            checkpoint_path=None, checkpoint_every=100, resume=False, identity=None):
        if checkpoint_path and any(st.kind == 'lbfgs' for st in stages):
            raise ValueError('step-level resume currently supports Adam stages only')
        if checkpoint_every < 1:
            raise ValueError('checkpoint_every must be positive')
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_every = checkpoint_every
        self._checkpoint_identity = dict(stages=[asdict(st) for st in stages], bs=bs,
                                         eval_every=eval_every, diag_every=diag_every, user=identity)
        if resume and self.checkpoint_path and self.checkpoint_path.exists():
            state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            if state['identity'] != self._checkpoint_identity:
                raise ValueError('checkpoint identity mismatch')
            self.model.load_state_dict(state['model'])
            for key in ('global_step', 'history', 'diag_history', 'slab_loss', 'slab_ready'):
                setattr(self, key, state[key])
            self._resume_state = state
        m0 = self.diagnostics()
        if self._resume_state:
            m0 = self._resume_state['initial_diag']
        self._initial_diag = m0
        self.log(f"  init diag: captured={m0['captured_energy']:.4f} "
                 f"eff_rank={m0['eff_rank']:.2f} angle_mean={m0['angle_mean_deg']:.1f}deg")
        if not self._resume_state:
            self.diag_history["step"].append(0)
            for k in ("captured_energy", "eff_rank", "angle_mean_deg", "cond"):
                self.diag_history[k].append(float(m0[k]))
        for index, st in enumerate(stages):
            self._stage_index = index
            if self._resume_state and index < self._resume_state['stage']:
                continue
            if st.kind == "lbfgs":
                self.run_lbfgs(st, bs=bs, eval_every=eval_every, diag_every=diag_every)
            else:
                self.run_stage(st, bs=bs, eval_every=eval_every, diag_every=diag_every)
        m1 = self.diagnostics()
        tr = self.evaluate("train")
        v = self.evaluate("val")
        t = self.evaluate("test")
        self.log(f"  final diag: captured={m1['captured_energy']:.4f} "
                 f"eff_rank={m1['eff_rank']:.2f} angle_mean={m1['angle_mean_deg']:.1f}deg")
        self.log(f"  final relL2: train={tr:.4f} val={v:.4f} test={t:.4f}")
        # 关键：可达相对误差的硬下界  LB = sqrt(1 - captured_energy)
        lb = math.sqrt(max(0.0, 1.0 - m1["captured_energy"]))
        lb0 = math.sqrt(max(0.0, 1.0 - m0["captured_energy"]))
        # 三项分解（2026-09-21 新增）：
        #     test_relL2 = LB + opt_gap_train + gen_gap
        #   LB            —— 表示下界：任何 branch 网络都跨不过去的误差
        #   opt_gap_train —— train 误差 - LB：优化 / branch 容量差距
        #   gen_gap       —— test 误差 - train 误差：对新输入函数的泛化差距
        # 旧的 opt_gap = test - LB 保留（= opt_gap_train + gen_gap），用于兼容既有分析脚本。
        # ⚠️ 尺度说明：LB 在 val 解的 probe 子采样点上、按 Frobenius 范数定义；
        #    relL2 是"逐样本比值再取均值"。因此三项之和严格等于 test_relL2，
        #    但 LB 与 opt_gap_train 的归属是近似的；gen_gap 是三者里唯一
        #    完全同 metric、同点集、同模型的量，也是最可信的一项。
        return {
            "diagnostic_status": "legacy_truncated_proxy_not_certified_lower_bound; use scripts/audit_checkpoints.py",
            "gap_interpretation": "legacy cross-split metric difference; not pure optimization error",
            "val_rel_l2": v,
            "train_rel_l2": tr,
            "test_rel_l2": t,
            "lb_init": lb0,
            "lb_final": lb,
            "opt_gap": t - lb,
            "opt_gap_train": tr - lb,
            "gen_gap": t - tr,
            "final_diag": {k: float(v_) for k, v_ in m1.items() if k != "sv"},
            "init_diag": {k: float(v_) for k, v_ in m0.items() if k != "sv"},
            "final_sv": m1["sv"].tolist(),
            "history": self.history,
            "diag_history": {k: v_ for k, v_ in self.diag_history.items()},
        }
