"""Measure forward+backward FLOPs per training step for each architecture.

Purpose
-------
Section 5.9 currently matches budget by update count and wall clock only. Both are proxies:
update count ignores that the architectures do different amounts of work per update, and wall
clock depends on the machine. This measures the work directly so a FLOP-matched comparison
can be run.

Caveat recorded in the output: `FlopCounterMode` counts registered ATen ops. Fourier
transforms may not contribute a registered flop count, in which case the FNO total is a
lower bound and the FLOP-matched DeepONet budget is an upper bound. The script reports
whether any fft-related op was seen so the reader can tell which case applies.

Usage
-----
    python scripts/measure_flops.py --device cuda
    python scripts/measure_flops.py --data external_data/pdebench/darcy --family darcy
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

from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from src.fno import FNO1d  # noqa: E402
from src.fno2d import FNO2d  # noqa: E402
from src.models import DeepONet  # noqa: E402


def build(architecture, protocol, device):
    if architecture == "fno":
        if protocol.get("fno") == "fno2d":
            return FNO2d(grid_shape=tuple(protocol["grid_shape"]), width=32, modes=12, layers=4)
        return FNO1d(width=32, modes=16, layers=4)
    grid = protocol.get("grid_shape")
    trunk_in = int(protocol.get("trunk_in", 1))
    branch_in = int(protocol.get("branch_in", protocol["grid_size"]))
    _ = grid
    return DeepONet(branch_in=branch_in, trunk_in=trunk_in, width=64,
                    branch_hidden=128, trunk_hidden=128, branch_layers=3, trunk_layers=3,
                    trunk_type="fourier" if architecture == "deeponet_fourier" else "mlp")


def analytic_fft_flops(architecture, protocol, batch: int, coords_shape) -> dict:
    """Estimate the FFT work that FlopCounterMode does not register.

    Standard estimates: a complex FFT of length n costs about 5 n log2(n) flops and a real
    FFT about half that. A forward-plus-backward pass needs the transform and its inverse in
    the forward, and roughly the same work again in the backward, so the factor used below is
    4 transforms worth per spectral layer. This is an estimate, reported separately from the
    measured dense count so a reader can use either.
    """
    n = int(coords_shape[0])
    if architecture != "fno":
        return dict(estimated_fft_flops=0, detail="no spectral layer")
    layers = 4
    if protocol.get("fno") == "fno2d":
        h, w = tuple(protocol["grid_shape"])
        per_point = 2.5 * np.log2(max(h * w, 2))
        per_transform = batch * 32 * h * w * per_point
    else:
        per_point = 2.5 * np.log2(max(n, 2))
        per_transform = batch * 32 * n * per_point
    total = int(4 * layers * per_transform)
    return dict(estimated_fft_flops=total,
                detail=f"{layers} spectral layers, factor 4 for forward and backward")


def measure(model, inputs, coords, device, batch: int) -> dict:
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = inputs[:batch].to(device)
    y = torch.zeros((batch, coords.shape[0]), device=device)
    seen = set()
    with FlopCounterMode(display=False) as mode:
        pred = model(x, coords)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    total = mode.get_total_flops()
    try:
        seen = set(mode.get_flop_counts().keys())
    except Exception:  # pragma: no cover - API detail
        seen = set()
    fft_ops = sorted(k for k in seen if "fft" in str(k).lower())
    return dict(total_flops=int(total),
                flops_forward_backward=int(total),
                per_sample_flops=int(total) / float(batch),
                fft_ops_counted=fft_ops,
                n_ops_counted=len(seen))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="external_data/pdebench/fixed_t1_v1")
    ap.add_argument("--family", default="burgers_t100")
    ap.add_argument("--architectures", default="deeponet_mlp,deeponet_fourier,fno")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/flop_measurement_v1")
    args = ap.parse_args()

    data = ROOT / args.data
    protocol = json.loads((data / "protocol.json").read_text(encoding="utf-8"))
    with np.load(data / "pairs.npz") as z:
        grid = z["x"]
        inputs = torch.tensor(z["train_x"], dtype=torch.float32)
    coords = torch.tensor(grid[:, None] if grid.ndim == 1 else grid, dtype=torch.float32)
    device = torch.device(args.device)
    coords = coords.to(device)

    report = dict(family=args.family, data=str(args.data), batch=args.batch,
                  grid_size=int(protocol["grid_size"]), measurements={})
    for architecture in args.architectures.split(","):
        torch.manual_seed(0)
        model = build(architecture, protocol, device)
        params = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
        res = measure(model, inputs, coords, device, args.batch)
        res["parameters"] = int(params)
        res.update(analytic_fft_flops(architecture, protocol, args.batch, coords.shape))
        res["total_with_estimated_fft"] = int(res["total_flops"] + res["estimated_fft_flops"])
        report["measurements"][architecture] = res
        print(f"{architecture:18s} params={params:8d}  measured={res['total_flops']:,}  "
              f"est_fft={res['estimated_fft_flops']:,}  "
              f"total={res['total_with_estimated_fft']:,}", flush=True)

    base = report["measurements"].get("fno")
    if base:
        for name, res in report["measurements"].items():
            if name == "fno":
                continue
            res["fno_to_this_ratio_measured"] = float(
                base["total_flops"] / max(res["total_flops"], 1))
            res["fno_to_this_ratio_with_fft"] = float(
                base["total_with_estimated_fft"] / max(res["total_with_estimated_fft"], 1))
            res["matched_updates_for_fno_1500"] = int(
                round(1500 * base["total_with_estimated_fft"]
                      / max(res["total_with_estimated_fft"], 1)))
            print(f"  FNO/{name}: measured {res['fno_to_this_ratio_measured']:.3f}, "
                  f"with FFT {res['fno_to_this_ratio_with_fft']:.3f}, "
                  f"matched updates {res['matched_updates_for_fno_1500']}", flush=True)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.family}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("written", out / f"{args.family}.json")


if __name__ == "__main__":
    main()
