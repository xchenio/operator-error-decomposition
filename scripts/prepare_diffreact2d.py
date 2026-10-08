"""Prepare PDEBench 2D reaction-diffusion caches, a two-component 2D system.

Layout and the two things that had to be checked first
------------------------------------------------------
The file stores one HDF5 group per sample (`"0000"`..`"0999"`), each with
`data (101, 128, 128, 2)` and `grid/t (101,)`, `grid/x (128,)`, `grid/y (128,)`. There are
**1000 samples**, not 10000, so the split sizes differ from every other family.

Two properties were measured before writing any code, because the 1D diffusion-sorption file
had already shown that a stored initial layer can be unusable:

1. **The stored initial layer is broadband noise.** Only 4.9% of its spectral energy lies
   inside r < 0.25 Nyquist (19% inside r < 0.5). Subsampling such a field discards the
   fine-scale detail that determines the later pattern, so the initial layer is **not** used.
2. **The field smooths very fast.** Energy inside r < 0.25 Nyquist rises to 70.6% by layer 5
   and 96.8% by layer 20. The input is therefore taken at **layer 20** (t = 1.0), where
   subsampling is safe, and all targets are strictly later. This follows the same convention
   as the 1D diffusion-sorption family.

Input is both components at layer 20; each cache predicts **one** component at one later
layer, which keeps every downstream script unchanged (one output field, trunk over (x, y)).

Usage
-----
    python scripts/prepare_diffreact2d.py
    python scripts/prepare_diffreact2d.py --time-indices 40,100 --stride 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.public_benchmark import (  # noqa: E402  reused, not modified
    atomic_json, file_hash, split_indices)

SOURCE = ROOT / "external_data/pdebench/darus/2D_diff-react_NA_NA.h5"
EXPECTED_MD5 = "b8d0b86064193195ddc30c33be5dc949"
OUT_ROOT = ROOT / "external_data/pdebench/diffreact2d"
INPUT_INDEX = 20          # measured: smooth enough to subsample
N_COMPONENTS = 2


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--input-index", type=int, default=INPUT_INDEX)
    ap.add_argument("--time-indices", default="40,70,100")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--sizes", default="600,200,200", help="only 1000 samples exist")
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    source, out_root = Path(args.source), Path(args.out)
    sizes = tuple(int(s) for s in args.sizes.split(","))
    targets = [int(t) for t in args.time_indices.split(",") if t]
    digest = md5_of(source)
    if digest != EXPECTED_MD5:
        raise ValueError(f"MD5 mismatch for {source.name}: {digest}")

    with h5py.File(source, "r") as h:
        names = sorted(h.keys())
        if len(names) != 1000:
            raise ValueError(f"expected 1000 samples, found {len(names)}")
        if sum(sizes) > len(names):
            raise ValueError("split sizes exceed the 1000 available samples")
        first = h[names[0]]
        n_time = first["data"].shape[0]
        if any(not (args.input_index < t < n_time) for t in targets):
            raise ValueError("every target index must be strictly after the input index")
        ts = first["grid/t"][:]
        xs, ys = first["grid/x"][:], first["grid/y"][:]
        if 128 % args.stride:
            raise ValueError("stride must divide 128")
        ids = split_indices(len(names), sizes, args.seed)
        layers = [args.input_index] + targets
        per_split = {}
        for name, index in ids.items():
            arr = np.empty((len(index), len(layers), N_COMPONENTS,
                            128 // args.stride, 128 // args.stride), dtype=np.float32)
            for row, sample in enumerate(index):
                block = h[names[int(sample)]]["data"][layers][:, ::args.stride, ::args.stride, :]
                arr[row] = np.moveaxis(block, -1, 1)      # (layers, H, W, C) -> (layers, C, H, W)
            per_split[name] = arr

    side = 128 // args.stride
    gy, gx = np.meshgrid(ys[::args.stride], xs[::args.stride], indexing="ij")
    grid = np.stack([gy.reshape(-1), gx.reshape(-1)], axis=1).astype(np.float32)
    n_points = side * side

    for comp in range(N_COMPONENTS):
        for column, ti in enumerate(targets):
            folder = out_root / f"c{comp}_in{args.input_index:03d}_t{ti:03d}_v1"
            folder.mkdir(parents=True, exist_ok=True)
            arrays = {"x": grid}
            for name, arr in per_split.items():
                arrays[name + "_x"] = np.ascontiguousarray(
                    arr[:, 0].reshape(len(arr), N_COMPONENTS * n_points))
                arrays[name + "_y"] = np.ascontiguousarray(
                    arr[:, column + 1, comp].reshape(len(arr), n_points))
            for key, value in arrays.items():
                if not np.isfinite(value).all():
                    raise ValueError(f"nonfinite values in {folder.name}/{key}")
            cache = folder / "pairs.npz"
            tmp = cache.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                np.savez_compressed(fh, **arrays)
            tmp.replace(cache)
            info = dict(protocol_version=1,
                        task="supervised two-component initial state to one component later",
                        equation="2D reaction-diffusion; system of two coupled fields",
                        source=str(source.resolve()), source_md5=digest, source_name=source.name,
                        output_component=comp, input_components=list(range(N_COMPONENTS)),
                        n_samples_available=len(names), split_seed=args.seed,
                        sample_ids={k: v.tolist() for k, v in ids.items()},
                        input_index=args.input_index, input_time=float(ts[args.input_index]),
                        time_index=ti, target_time=float(ts[ti]),
                        stored_time_coordinate_count=len(ts), solution_time_count=n_time,
                        time_step=float(ts[1] - ts[0]),
                        spatial_stride=args.stride, grid_shape=[side, side], grid_size=n_points,
                        branch_in=N_COMPONENTS * n_points, trunk_in=2,
                        spatial_policy="point subsampling, no filtering; same grid for every model",
                        physics_note="the stored t=0 layer is broadband noise (4.9% of spectral "
                                     "energy inside r < 0.25 Nyquist), so it is not used; the "
                                     "input is layer 20, where 96.8% of the energy is smooth",
                        official_split=False,
                        hash_note="MD5 recorded and checked against the DaRUS published value")
            info["cache_sha256"] = file_hash(cache)
            info["normalization"] = dict(input_mean=float(arrays["train_x"].mean()),
                                         input_std=float(arrays["train_x"].std()),
                                         target_mean=float(arrays["train_y"].mean()),
                                         target_std=float(arrays["train_y"].std()))
            atomic_json(folder / "protocol.json", info)
            print("prepared", folder.name, "target_time", info["target_time"],
                  "grid", info["grid_shape"], flush=True)


if __name__ == "__main__":
    main()
