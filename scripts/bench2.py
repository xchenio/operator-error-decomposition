"""对比不同二阶导数实现的速度（含 warmup）。"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.func import jacfwd, jacrev, vmap
from src.models import DeepONet

torch.set_num_threads(16)
model = DeepONet(branch_in=101, trunk_in=2, width=100, branch_hidden=128,
                 branch_layers=3, trunk_hidden=128, trunk_layers=3)
f = model.trunk

for M in (1024, 1536):
    y = torch.randn(M, 2)
    variants = {
        "jacrev(jacrev)": lambda: vmap(jacrev(jacrev(f)))(y),
        "jacfwd(jacrev)": lambda: vmap(jacfwd(jacrev(f)))(y),
        "jacfwd(jacfwd)": lambda: vmap(jacfwd(jacfwd(f)))(y),
        "jacrev(jacfwd)": lambda: vmap(jacrev(jacfwd(f)))(y),
    }
    print(f"\n--- M={M} ---")
    for name, fn in variants.items():
        try:
            out = fn()                      # warmup
            assert out.shape == (M, 100, 2, 2), out.shape
            t0 = time.time()
            for _ in range(3):
                fn()
            dt = (time.time() - t0) / 3
            print(f"  {name:16s}: {dt*1000:8.1f} ms")
        except Exception as e:
            print(f"  {name:16s}: FAILED {type(e).__name__}: {e}")

    # 对称性与正确性交叉验证
    d2a = vmap(jacfwd(jacrev(f)))(y)
    d2b = vmap(jacfwd(jacfwd(f)))(y)
    print(f"  max|jacfwd(jacrev) - jacfwd(jacfwd)| = {(d2a-d2b).abs().max().item():.3e}")
    print(f"  symmetry err (0,1)-(1,0): {(d2a[:,:,0,1]-d2a[:,:,1,0]).abs().max().item():.3e}")
