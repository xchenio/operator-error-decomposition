"""Mean / fluctuation decomposition recomputed from stored public-data checkpoints.

Background
----------
The manuscript reports that both DeepONet variants fail to beat an "input spatial-mean
baseline" (0.37577) while FNO reaches 0.22835 on the same protocol. That comparison does
not say *what* each model predicts. This script separates the error into a conserved-mean
part and a fluctuation part, because for periodic Burgers the target spatial mean equals
the input spatial mean (see docs/FLUCTUATION_DIAGNOSIS_V1.md), so the mean baseline is a
conservation-law predictor rather than a strawman.

For target Y and prediction P in R^{m x N}:

    fluctuation R^2 = 1 - sum||P_c - Y_c||^2 / sum||Y_c||^2        (Y_c = Y - bar(y) 1^T)

R^2 = 0 means "no better than predicting zero fluctuation". It is a skill measure against
a zero-fluctuation reference, not a causal attribution and not a decomposition of
mean relative L2; both are reported separately.

Faithfulness
------------
Every recomputed `mean_relative_l2` is compared with the value stored in the run's
`result.json`. A run whose recomputation disagrees beyond `--tolerance` is reported as a
failure rather than silently included, so R^2 numbers always come from a reconstruction
that reproduces the original metric.

Usage
-----
    python scripts/diagnose_fluctuation.py --families public_fixed_t1_v1 --device cuda
    python scripts/diagnose_fluctuation.py --device cuda          # all families
    python scripts/diagnose_fluctuation.py --limit 2 --device cpu # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fno import FNO1d                      # noqa: E402
from src.models import DeepONet, MLP           # noqa: E402
from src.public_benchmark import atomic_json, file_hash   # noqa: E402

DATA = ROOT / "external_data/pdebench/fixed_t1_v1"

# Families whose checkpoints store a whole end-to-end model.
END_TO_END = {"public_fixed_t1_v1": ("test", "val")}

# Families whose checkpoints store a branch MLP plus the frozen whitened basis.
# `mean_lift` is read from the run configuration for public_coefficients_v1 and is
# always on for the smallfit / stability / scale1000 families (they lift by construction).
BRANCH_FAMILIES = {
    "public_coefficients_v1": ("test", "val"),
    "public_smallfit_v1": ("val",),
    "public_stability_v1": ("val",),
    "public_scale1000_v1": ("val",),
}


def source_hash() -> str:
    files = [Path(__file__), ROOT / "src/models.py", ROOT / "src/fno.py"]
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def decompose(pred, truth) -> dict:
    """Separate conserved-mean skill from fluctuation skill."""
    P = np.asarray(pred, dtype=np.float64)
    Y = np.asarray(truth, dtype=np.float64)
    residual = P - Y
    denom = np.maximum(np.linalg.norm(Y, axis=1), 1e-12)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    Pc = P - P.mean(axis=1, keepdims=True)
    fluct = float(np.sum(Yc ** 2))
    return dict(
        mean_relative_l2=float(np.mean(np.linalg.norm(residual, axis=1) / denom)),
        relative_fro=float(np.linalg.norm(residual) / max(np.linalg.norm(Y), 1e-12)),
        rmse=float(np.sqrt(np.mean(residual ** 2))),
        fluctuation_r2=float(1.0 - np.sum((Pc - Yc) ** 2) / fluct) if fluct > 0 else float("nan"),
        normalized_mean_error=float(np.mean(
            np.abs(P.mean(axis=1) - Y.mean(axis=1)) / (denom / np.sqrt(Y.shape[1])))),
        fluctuation_energy_fraction=float(np.linalg.norm(Yc) / max(np.linalg.norm(Y), 1e-12)),
    )


def conservation_check(arrays: dict, split: str) -> dict:
    """Periodic Burgers conserves int u dx; quantify how closely the stored data does."""
    x = np.asarray(arrays[split + "_x"], dtype=np.float64)
    y = np.asarray(arrays[split + "_y"], dtype=np.float64)
    mx, my = x.mean(axis=1), y.mean(axis=1)
    return dict(
        split=split,
        mean_correlation=float(np.corrcoef(mx, my)[0, 1]),
        mean_abs_difference=float(np.abs(my - mx).mean()),
        mean_abs_target_mean=float(np.abs(my).mean()),
    )


def _end_to_end_prediction(cfg, ck, arrays, protocol, device, split):
    n = protocol["normalization"]
    coords = torch.tensor(arrays["x"][:, None], device=device, dtype=torch.float32)
    x = torch.tensor((arrays[split + "_x"] - n["input_mean"]) / max(n["input_std"], 1e-6), device=device)
    if cfg["architecture"] == "fno":
        model = FNO1d(width=32, modes=16, layers=4).to(device)
    else:
        model = DeepONet(branch_in=len(arrays["x"]), trunk_in=1, width=64,
                         branch_hidden=128, trunk_hidden=128, branch_layers=3,
                         trunk_layers=3,
                         trunk_type="fourier" if cfg["architecture"] == "deeponet_fourier" else "mlp").to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(v, coords) for v in x.split(64)]).cpu().numpy()
    return out * n["target_std"] + n["target_mean"]


def _branch_prediction(ck, arrays, protocol, device, split, mean_lift: bool):
    basis = ck["basis"].to(device)
    branch = MLP(len(arrays["x"]), basis.shape[1], 128, 3).to(device)
    branch.load_state_dict(ck["model"])
    branch.eval()
    n = protocol["normalization"]
    scale = max(n["target_std"], 1e-6)
    x = torch.tensor((arrays[split + "_x"] - n["input_mean"]) / max(n["input_std"], 1e-6), device=device)
    if mean_lift:
        offsets = arrays[split + "_x"].mean(axis=1, keepdims=True)
    else:
        offsets = np.full((len(arrays[split + "_x"]), 1), n["target_mean"], dtype="float32")
    with torch.no_grad():
        out = torch.cat([branch(v) @ basis.T for v in x.split(64)]).cpu().numpy()
    return out * scale + offsets


def run_family(family: str, arrays: dict, protocol: dict, device, out_dir: Path,
               tolerance: float, limit: int | None):
    splits = END_TO_END[family] if family in END_TO_END else BRANCH_FAMILIES[family]
    base = ROOT / "results" / family
    runs = sorted(p for p in base.iterdir() if (p / "checkpoints" / "latest.pt").exists())
    if limit is not None:
        runs = runs[:limit]

    records = []
    worst = 0.0
    failures = []
    for folder in runs:
        ck = torch.load(folder / "checkpoints" / "latest.pt", map_location=device, weights_only=False)
        cfg = ck["identity"]["configuration"]
        result_path = folder / "result.json"
        stored = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        per_split = {}
        for split in splits:
            if family in END_TO_END:
                pred = _end_to_end_prediction(cfg, ck, arrays, protocol, device, split)
            else:
                mean_lift = bool(cfg.get("mean_lift", True))
                pred = _branch_prediction(ck, arrays, protocol, device, split, mean_lift)
            score = decompose(pred, arrays[split + "_y"])
            if stored is not None:
                ref = stored["metrics"][split]["mean_relative_l2"]
                diff = abs(score["mean_relative_l2"] - ref)
                worst = max(worst, diff)
                score["stored_mean_relative_l2"] = float(ref)
                score["metric_abs_difference"] = float(diff)
                if diff > tolerance:
                    failures.append(dict(run=folder.name, split=split, difference=diff))
            per_split[split] = score
        records.append(dict(run=folder.name, configuration={k: cfg[k] for k in sorted(cfg)},
                            splits=per_split))

    payload = dict(
        family=family,
        splits=list(splits),
        source_sha256=source_hash(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        protocol_cache_sha256=protocol["cache_sha256"],
        tolerance=tolerance,
        runs=records,
        max_abs_metric_difference=worst,
        faithfulness_failures=failures,
        splits_note="recomputed from stored checkpoints; train split omitted for branch "
                    "families that subsample training rows",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(out_dir / f"{family}.json", payload)
    for split in splits:
        vals = [r["splits"][split]["fluctuation_r2"] for r in records]
        rels = [r["splits"][split]["mean_relative_l2"] for r in records]
        if vals:
            print(f"{family} [{split}] n={len(vals)} relL2={np.mean(rels):.5f}+-{np.std(rels):.5f} "
                  f"R2={np.mean(vals):+.4f}+-{np.std(vals):.4f} "
                  f"max|metric diff|={worst:.2e}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(list(END_TO_END) + list(BRANCH_FAMILIES)))
    ap.add_argument("--out", default="results/fluctuation_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tolerance", type=float, default=2e-5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    protocol = json.loads((DATA / "protocol.json").read_text(encoding="utf-8"))
    if file_hash(DATA / "pairs.npz") != protocol["cache_sha256"]:
        raise ValueError("pair cache changed; refusing to diagnose against a moved dataset")
    with np.load(DATA / "pairs.npz") as z:
        arrays = {k: z[k].copy() for k in z.files}

    device = torch.device(args.device)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle = {}
    for split in ("val", "test"):
        mean_field = np.repeat(arrays[split + "_x"].mean(axis=1)[:, None],
                               arrays[split + "_y"].shape[1], axis=1)
        oracle[split] = decompose(mean_field, arrays[split + "_y"])
    conservation = [conservation_check(arrays, s) for s in ("train", "val", "test")]
    atomic_json(out_dir / "reference.json", dict(
        oracle_input_mean=oracle,
        conservation=conservation,
        note="oracle predicts the exact input spatial mean and zero fluctuation",
    ))
    for c in conservation:
        print(f"conservation [{c['split']}] corr={c['mean_correlation']:.6f} "
              f"|dmean|={c['mean_abs_difference']:.3e} vs |mean|={c['mean_abs_target_mean']:.3e}", flush=True)
    for split in oracle:
        print(f"oracle input mean [{split}] relL2={oracle[split]['mean_relative_l2']:.5f} "
              f"R2={oracle[split]['fluctuation_r2']:+.4f}", flush=True)

    summaries = {}
    for family in args.families.split(","):
        if family in ("reference", ""):
            continue
        summaries[family] = run_family(family, arrays, protocol, device, out_dir,
                                       args.tolerance, args.limit)
    atomic_json(out_dir / "summary.json", dict(
        families={k: dict(n_runs=len(v["runs"]),
                          max_abs_metric_difference=v["max_abs_metric_difference"],
                          faithfulness_failures=v["faithfulness_failures"])
                  for k, v in summaries.items()},
        oracle_input_mean=oracle,
        conservation=conservation,
    ))


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
