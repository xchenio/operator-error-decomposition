"""验证「对齐 = 正则化/稳定优化」假设：测量 NTK 谱与损失面尖锐度。

三个量：

1. **Trunk NTK 谱**。trunk 特征 Phi ∈ R^{N×p}，其线性读出对应的 NTK 恰为 Phi Phi^T，
   特征值 = Phi 奇异值的平方。等价地，trunk NTK 的有效秩 = Phi 的有效秩。
   （免费，直接从特征矩阵算。）

2. **全模型输出 NTK 谱**。J = ∂u/∂θ ∈ R^{M×P}，K = J J^T ∈ R^{M×M}。
   关注：有效秩、条件数、最大特征值占比。
   结论若成立，对齐应使 NTK 谱更"平坦"（有效秩更高、条件数更低），
   而不仅是 Phi 的秩更高。

3. **损失面尖锐度**。用幂迭代估计 Hessian 最大特征值 λ_max（Hessian-vector product），
   并报告 Keskar sharpness = λ_max / (2L)。对齐若起正则作用，应显著降低 λ_max。

用法：
  python scripts/ntk_analysis.py --ckpt results/expQ/xxx.pt --out results/ntk.json
  python scripts/ntk_analysis.py --dir results/expQ          # 批量分析目录下所有 .pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataset            # noqa: E402
from src.diagnostics import effective_rank    # noqa: E402
from src.models import DeepONet               # noqa: E402
from src.pdes import make_problem             # noqa: E402
from src.trainer import _to_device           # noqa: E402
from src.utils import get_device, set_seed   # noqa: E402


def output_ntk(model, u_b, coords, max_pts: int = 512):
    """J = ∂u/∂θ (M, P)，返回 NTK = J J^T 及其谱。"""
    M = coords.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    out = model(u_b, coords)                  # (B, M)
    rows = []
    for i in range(M):
        gi = torch.autograd.grad(out[0, i], params, retain_graph=(i < M - 1),
                                 allow_unused=True)
        gi = [g if g is not None else torch.zeros_like(p)
              for g, p in zip(gi, params)]
        rows.append(torch.cat([g.reshape(-1) for g in gi]))
    J = torch.stack(rows)                     # (M, P)
    K = (J @ J.t()).detach().cpu().numpy().astype(np.float64)
    ev = np.linalg.eigvalsh(K)[::-1]
    ev = np.clip(ev, 0.0, None)
    return ev, J.detach().cpu().numpy()


def hessian_lambda_max(model, pde, splits, device, n_col=32, bs=8, iters=15, seed=0):
    """幂迭代估计物理损失的 Hessian 最大特征值。"""
    rng = np.random.default_rng(seed)
    sp = splits["train"]
    idx = rng.choice(len(sp), bs, replace=False)
    u_b = torch.as_tensor(sp.u[idx], dtype=torch.float32, device=device)
    pts = _to_device(pde.sample_points(sp.coef[idx], n_col, 32, 16, rng), device)
    params = [p for p in model.parameters() if p.requires_grad]

    def loss_fn():
        evalr = model.evaluator(u_b)
        terms = pde.losses(evalr, pts)
        return sum(v for v in terms.values()
                   if torch.is_tensor(v) and v.dim() == 0)

    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    gnorm = float(torch.sqrt(sum((g ** 2).sum() for g in grads)))

    v = [torch.randn_like(p) for p in params]
    nv = torch.sqrt(sum((x ** 2).sum() for x in v))
    v = [x / nv for x in v]
    lam = 0.0
    for _ in range(iters):
        gv = sum((g * x).sum() for g, x in zip(grads, v))
        hv = torch.autograd.grad(gv, params, retain_graph=True)
        lam = float(sum((h * x).sum() for h, x in zip(hv, v)))
        nv = torch.sqrt(sum((h ** 2).sum() for h in hv)) + 1e-12
        v = [h / nv for h in hv]
    L = float(loss.detach())
    return {"loss": L, "grad_norm": gnorm, "hess_lambda_max": lam,
            "sharpness": lam / (2.0 * max(L, 1e-12))}


def analyze_one(ckpt_path: Path, device, n_probe: int = 1024, ntk_pts: int = 256):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    pde = make_problem(ck["pde"], **ck.get("pde_kwargs", {}))
    model = DeepONet(branch_in=pde.n_sensors, trunk_in=pde.trunk_dim,
                     **ck["model_kwargs"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    splits = build_dataset(pde, cfg["n_train"], cfg["n_val"], cfg["n_test"], seed=0,
                           cache=str(ROOT / "data" /
                                     f"{ck['pde']}_default_n200_50_50_s0.npz"))
    rng = np.random.default_rng(0)
    gi = rng.choice(pde.grid.shape[0], min(n_probe, pde.grid.shape[0]), replace=False)
    probe = torch.as_tensor(pde.grid[gi], dtype=torch.float32, device=device)

    # 1) trunk NTK 谱（= Phi 奇异值平方）
    with torch.no_grad():
        Phi = model.trunk_features(probe).cpu().numpy().astype(np.float64)
    g = Phi.T @ Phi
    s = np.linalg.svd(Phi, compute_uv=False)
    trunk = {"eff_rank": effective_rank(Phi),
             "cond": float(np.linalg.cond(Phi)),
             "top1_energy_share": float((s[:1] ** 2).sum() / (s ** 2).sum())}

    # 2) 全模型输出 NTK 谱
    b = rng.choice(len(splits["train"]), 1, replace=False)
    u_b = torch.as_tensor(splits["train"].u[b], dtype=torch.float32, device=device)
    tidx = rng.choice(probe.shape[0], min(ntk_pts, probe.shape[0]), replace=False)
    coords = probe[tidx]
    ev, _ = output_ntk(model, u_b, coords, max_pts=ntk_pts)
    tot = ev.sum() + 1e-30
    p = ev / tot
    ent = -(p * np.log(p + 1e-30)).sum()
    out = {"eff_rank_entropy": float(np.exp(ent)),
           "cond": float(ev[0] / max(ev[ev > ev[0] * 1e-12][-1], 1e-30)),
           "top1_share": float(ev[0] / tot),
           "ev_ratio_top1_top10": float(ev[0] / max(ev[min(9, len(ev) - 1)], 1e-30))}

    # 3) 损失面尖锐度
    set_seed(0)
    model.train()
    hess = hessian_lambda_max(model, pde, splits, device, seed=cfg["seed"])

    # 保存谱本身，便于绘图（归一化到最大特征值 = 1）
    trunk["sv_top"] = (s[:48] / max(s[0], 1e-30)).tolist()
    out["ev_top"] = (ev[:48] / max(ev[0], 1e-30)).tolist()

    rel = None
    js = ckpt_path.with_suffix(".json")
    if js.exists():
        try:
            rel = json.load(open(js, encoding="utf-8")).get("test_rel_l2")
        except Exception:
            pass

    return {"ckpt": ckpt_path.stem, "method": cfg["method"],
            "align_kind": cfg["align_kind"], "seed": cfg["seed"],
            "rel_l2": rel,
            "trunk_ntk": trunk, "output_ntk": out, "hessian": hess}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default="results/ntk.json")
    args = ap.parse_args()

    device = get_device(True)
    pts = ([Path(args.ckpt)] if args.ckpt else
           sorted(Path(args.dir).glob("*.pt")))
    if not pts:
        print("no checkpoints found")
        return
    res = []
    for p in pts:
        print(f"[ntk] analyzing {p.name} ...", flush=True)
        try:
            r = analyze_one(p, device)
            res.append(r)
            print(f"   trunk eff_rank={r['trunk_ntk']['eff_rank']:.1f} | "
                  f"out-NTK eff_rank={r['output_ntk']['eff_rank_entropy']:.1f} "
                  f"top1={r['output_ntk']['top1_share']:.3f} | "
                  f"lambda_max={r['hessian']['hess_lambda_max']:.3e} "
                  f"sharp={r['hessian']['sharpness']:.3e}", flush=True)
        except Exception as e:
            import traceback
            print(f"   FAILED: {e}")
            traceback.print_exc()

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n[saved] {outp}")


if __name__ == "__main__":
    main()
