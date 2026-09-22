"""Resumable FNO control for an HDF5 PDEBench 1D Burgers file.

It deliberately uses supervised initial-to-final-time prediction: this is a cross-architecture
control and a public-data validation path, not a claim that FNO optimizes this project's
physics-informed residual. Use `--inspect-only` before selecting an HDF5 field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fno import FNO1d
from src.utils import rel_l2, set_seed


def _datasets(h5: h5py.File) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            found[name] = list(obj.shape)
    h5.visititems(visit)
    return found


def _pick_field(h5: h5py.File, field: str) -> h5py.Dataset:
    if field != "auto":
        return h5[field]
    candidates = [(name, shape) for name, shape in _datasets(h5).items()
                  if len(shape) in (3, 4) and shape[0] > 1 and shape[1] > 1]
    if len(candidates) != 1:
        raise ValueError("could not infer solution field; rerun --inspect-only then pass --field. "
                         f"candidates={candidates}")
    return h5[candidates[0][0]]


def _as_trajectory(array: np.ndarray) -> np.ndarray:
    """Normalize common PDEBench 1D layouts to `(sample, time, space)`."""
    x = np.asarray(array, dtype=np.float32)
    if x.ndim == 4:
        if x.shape[-1] != 1:
            raise ValueError(f"expected scalar PDE field, got shape {x.shape}")
        x = x[..., 0]
    if x.ndim != 3:
        raise ValueError(f"expected 3D/4D solution tensor, got {x.shape}")
    return x


def _atomic_torch_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _file_sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash multi-GB public files without loading them into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk_bytes):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--field", default="auto")
    ap.add_argument("--inspect-only", action="store_true")
    ap.add_argument("--out", default="results/pdebench_burgers_fno_smoke/fno_seed0.json")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=128)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    data_path = Path(args.data)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(f"PDEBench file not found: {data_path}. See docs/PDEBENCH_PROTOCOL.md")
    with h5py.File(data_path, "r") as h5:
        schema = _datasets(h5)
        if args.inspect_only:
            target = ROOT / "results/pdebench_burgers_fno_smoke/dataset_schema.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"data": str(data_path), "datasets": schema}, indent=2),
                              encoding="utf-8")
            print(json.dumps(schema, indent=2))
            return
        ds = _pick_field(h5, args.field)
        n = min(args.max_samples, ds.shape[0])
        data = _as_trajectory(ds[:n])

    if data.shape[1] < 2:
        raise ValueError(f"need time dimension >= 2, got {data.shape}")
    split = max(1, int(0.8 * len(data)))
    train, test = data[:split], data[split:]
    if len(test) == 0:
        raise ValueError("need at least two samples for train/test split")
    mean, std = float(train.mean()), float(train.std() + 1e-6)
    train_x, train_y = (train[:, 0] - mean) / std, (train[:, -1] - mean) / std
    test_x, test_y = (test[:, 0] - mean) / std, (test[:, -1] - mean) / std

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model = FNO1d(args.width, args.modes, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = out.with_suffix(".pt")
    start = 0
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        start = int(state["step"])
        print(f"[resume] step={start}")

    tx = torch.as_tensor(train_x, device=device); ty = torch.as_tensor(train_y, device=device)
    vx = torch.as_tensor(test_x, device=device); vy = torch.as_tensor(test_y, device=device)
    rng = np.random.default_rng(args.seed + 1)
    t0 = time.time()
    for step in range(start, args.steps):
        idx = rng.choice(len(tx), size=min(args.batch_size, len(tx)), replace=False)
        pred = model(tx[idx])
        loss = ((pred - ty[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            _atomic_torch_save({"step": step + 1, "model": model.state_dict(),
                                "optimizer": opt.state_dict(), "args": vars(args)}, checkpoint)
            print(f"step={step + 1} loss={float(loss.detach()):.4e}", flush=True)

    with torch.no_grad():
        test_rel = rel_l2(model(vx) * std + mean, vy * std + mean)
    out.write_text(json.dumps({"architecture": "FNO1d", "dataset": str(data_path),
        "field": args.field, "samples": int(n), "shape": list(data.shape), "seed": args.seed,
        "steps": args.steps, "test_rel_l2": test_rel, "wall_time": time.time() - t0,
        "data_sha256": _file_sha256(data_path)}, indent=2), encoding="utf-8")
    print(f"[done] test relL2={test_rel:.4f}; saved {out}")


if __name__ == "__main__":
    main()
