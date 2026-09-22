"""对比两种导数实现：
  A) 对 trunk 的 p 个基求 (p,d,d) Hessian，再与 branch 系数收缩
  B) 对"样本×配点"的标量输出 u=b·phi 直接求 (d,d) Hessian（双层 vmap）
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.func import jacfwd, jacrev, vmap
from src.models import DeepONet

torch.set_num_threads(16)
dev = torch.device("cpu")

for width in (100, 64):
    model = DeepONet(branch_in=128, trunk_in=2, width=width, branch_hidden=128,
                     branch_layers=3, trunk_hidden=128, trunk_layers=3)
    trunk = model.trunk
    B, Nc = 16, 64
    M = B * Nc
    y = torch.randn(Nc, 2)
    b = torch.randn(B, width, requires_grad=True)

    # ---- A: per-basis ----
    def A(order=2):
        phi = torch.vmap(trunk)(y)
        dphi = vmap(jacrev(trunk))(y)
        if order >= 2:
            d2phi = vmap(jacfwd(jacfwd(trunk)))(y)
        u = torch.einsum("bp,np->bn", b, phi) + model.bias
        du = torch.einsum("bp,npd->bnd", b, dphi)
        d2u = torch.einsum("bp,npde->bnde", b, d2phi) if order >= 2 else None
        return u, du, d2u

    # ---- B: per-(sample,point) scalar ----
    def f_single(yv, bv):
        return torch.dot(bv, trunk(yv))

    def B_(order=2):
        fu = vmap(vmap(f_single, in_dims=(0, None)), in_dims=(None, 0))
        u = fu(y, b) + model.bias
        if order >= 1:
            fj = vmap(vmap(jacrev(f_single), in_dims=(0, None)), in_dims=(None, 0))
            du = fj(y, b)
        else:
            du = None
        if order >= 2:
            fh = vmap(vmap(jacfwd(jacrev(f_single)), in_dims=(0, None)),
                      in_dims=(None, 0))
            d2u = fh(y, b)
        else:
            d2u = None
        return u, du, d2u

    print(f"\n=== width={width}, B={B}, Nc={Nc} ===")
    ua, dua, d2ua = A()
    ub, dub, d2ub = B_()
    print(f"  max|uA-uB|   = {(ua-ub).abs().max().item():.3e}")
    print(f"  max|duA-duB| = {(dua-dub).abs().max().item():.3e}")
    print(f"  max|d2A-d2B| = {(d2ua-d2ub).abs().max().item():.3e}")

    for name, fn in (("A per-basis", A), ("B per-scalar", B_)):
        fn()  # warmup
        t0 = time.time()
        for _ in range(5):
            u, du, d2u = fn()
            loss = (u ** 2).mean() + (du ** 2).mean() + (d2u ** 2).mean()
            loss.backward()
        print(f"  {name:12s}: {((time.time()-t0)/5)*1000:7.1f} ms/step (含反向)")
        if b.grad is not None:
            b.grad = None
