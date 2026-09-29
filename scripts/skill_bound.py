"""An exact, training-free upper bound on the fluctuation skill a trunk can support.

Statement
---------
A DeepONet with trunk features `Phi in R^{N x p}` can only output fields lying in
`span(Phi)`. Allowing a constant column and granting the branch an oracle that chooses
optimal coefficients **separately for each sample** (strictly more powerful than any real
branch, which must be a function of the input), the smallest achievable fluctuation share is

    fluct_share_min = mean_j ||(I - P) y_{c,j}||^2 / ||y_j||^2 ,   P = projector onto span([Phi, 1])

and therefore

    R^2_per_sample  <=  1 - fluct_share_min / (per-sample fluctuation energy).

This is an identity about the trunk's span, so it is a **hard upper bound**, not an
attribution. Comparing it with the achieved skill separates two failure modes:

* achieved ~ bound  -> the trunk's span is the binding constraint;
* achieved << bound -> the branch (the input-to-coefficient map) is the binding constraint.

The bound needs the trunk but no branch training, and no test labels beyond the targets the
skill is measured against.

Usage
-----
    python scripts/skill_bound.py --settings burgers_t100,advection_t100 --device cuda
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

from src.models import DeepONet  # noqa: E402
from src.public_benchmark import file_hash  # noqa: E402

RCOND = 1e-6  # matches the conditioning used by the projection audit

SETTINGS = {
    "burgers_t025": ("external_data/pdebench/fixed_t025_v1", "results/public_horizon_t025_v1",
                     "results/fluctuation_horizon_v1/t025/public_horizon_t025_v1.json"),
    "burgers_t100": ("external_data/pdebench/fixed_t1_v1", "results/public_fixed_t1_v1",
                     "results/fluctuation_v1/public_fixed_t1_v1.json"),
    "burgers_t200": ("external_data/pdebench/fixed_t200_v1", "results/public_horizon_t200_v1",
                     "results/fluctuation_horizon_v1/t200/public_horizon_t200_v1.json"),
    "advection_t100": ("external_data/pdebench/advection/fixed_t100_v1",
                       "results/advection_horizon_t100_v1",
                       "results/fluctuation_advection_v1/t100/advection_horizon_t100_v1.json"),
    "advection_t200": ("external_data/pdebench/advection/fixed_t200_v1",
                       "results/advection_horizon_t200_v1",
                       "results/fluctuation_advection_v1/t200/advection_horizon_t200_v1.json"),
}
ARCHS = {"deeponet_mlp": "mlp", "deeponet_fourier": "fourier"}


def orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = s > RCOND * max(s[0], 1e-300)
    return u[:, keep]


def bound_for_run(folder: Path, arch: str, Y: np.ndarray, coords: torch.Tensor, device):
    ck = torch.load(folder / "checkpoints" / "latest.pt", map_location=device, weights_only=False)
    model = DeepONet(branch_in=256, trunk_in=1, width=64, branch_hidden=128, trunk_hidden=128,
                     branch_layers=3, trunk_layers=3, trunk_type=ARCHS[arch]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        phi = model.trunk_features(coords).cpu().numpy().astype(np.float64)
    basis = orthonormal_basis(np.column_stack([phi, np.ones(len(phi))]))
    Yc = Y - Y.mean(axis=1, keepdims=True)
    resid = Yc - (Yc @ basis) @ basis.T
    denom = np.sum(Y ** 2, axis=1)
    fluct_share_min = float(np.mean(np.sum(resid ** 2, axis=1) / np.maximum(denom, 1e-300)))
    energy_ps = float(np.mean(np.sum(Yc ** 2, axis=1) / np.maximum(denom, 1e-300)))
    return dict(fluctuation_share_bound=fluct_share_min,
                fluctuation_energy_per_sample=energy_ps,
                skill_bound=1.0 - fluct_share_min / max(energy_ps, 1e-300),
                basis_rank=int(basis.shape[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/skill_bound_v1")
    args = ap.parse_args()
    device = torch.device(args.device)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    for name in (s for s in args.settings.split(",") if s):
        data_dir, runs_dir, diag_rel = SETTINGS[name]
        achieved = {}
        diag_path = ROOT / diag_rel
        if diag_path.exists():
            payload = json.loads(diag_path.read_text(encoding="utf-8"))
            for run in payload["runs"]:
                achieved.setdefault(run["configuration"]["architecture"], []).append(
                    run["splits"]["test"]["fluctuation_r2_per_sample"])
        data_path, runs_path = ROOT / data_dir, ROOT / runs_dir
        if not data_path.exists() or not runs_path.exists():
            print("[skip]", name)
            continue
        protocol = json.loads((data_path / "protocol.json").read_text(encoding="utf-8"))
        if file_hash(data_path / "pairs.npz") != protocol["cache_sha256"]:
            raise ValueError(f"pair cache changed: {data_path}")
        with np.load(data_path / "pairs.npz") as z:
            Y = z["test_y"].astype(np.float64)
            grid = z["x"]
        coords = torch.tensor(grid[:, None], device=device, dtype=torch.float32)
        report[name] = {}
        for arch in ARCHS:
            rows = []
            for seed in (int(s) for s in args.seeds.split(",")):
                folder = runs_path / f"{arch}_s{seed}"
                if not (folder / "checkpoints" / "latest.pt").exists():
                    continue
                rows.append(bound_for_run(folder, arch, Y, coords, device))
            if not rows:
                continue
            bounds = [r["skill_bound"] for r in rows]
            bound = float(np.mean(bounds))
            entry = dict(
                skill_bound_mean=bound,
                skill_bound_std=float(np.std(bounds, ddof=1)) if len(bounds) > 1 else 0.0,
                basis_rank_mean=float(np.mean([r["basis_rank"] for r in rows])),
                n=len(rows))
            if arch in achieved:
                got = float(np.mean(achieved[arch]))
                entry.update(achieved_mean=got,
                             branch_gap=bound - got,
                             bound_holds=bool(bound >= got))
            report[name][arch] = entry
            extra = (f" achieved={entry['achieved_mean']:+.4f} gap={entry['branch_gap']:.4f}"
                     if "achieved_mean" in entry else "")
            print(f"{name:16s} {arch:18s} bound={bound:+.4f} +- {entry['skill_bound_std']:.4f} "
                  f"(basis rank {entry['basis_rank_mean']:.1f}, n={len(rows)}){extra}", flush=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
