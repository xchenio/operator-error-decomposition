"""Prepare the steady 2D Darcy control from PDEBench.

Task
----
`nu (10000,128,128)` is the coefficient field and `tensor (10000,1,128,128)` the solution.
This is a **steady elliptic** operator: coefficient -> solution. There is no time axis, so
there is no forecast horizon and no conservation law linking the input mean to the target
mean. What the fluctuation decomposition still measures is whether a model predicts the
deviation of the target from its own per-sample mean better than not predicting it, which is
exactly the quantity the threshold rule is about.

Layout
------
`x` is stored as `(N, 2)` coordinate pairs rather than the 1D `(N,)`, so 2D runners and the
diagnostic can distinguish the two. All 1D caches keep the 1D form and are unaffected.
Protocol keys match `prepare()` (`cache_sha256`, `normalization`) plus explicit
`grid_shape`, `branch_in` and `trunk_in`, so downstream code can be driven from the cache
instead of hard-coded sizes.

Usage
-----
    python scripts/prepare_darcy.py
    python scripts/prepare_darcy.py --stride 4
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

# published MD5s from the DaRUS deposit, keyed by filename, so beta variants can be
# verified without loosening the check
EXPECTED_MD5 = {
    "2D_DarcyFlow_beta0.01_Train.hdf5": "d05c287d4c0b7d3178b0097084238251",
    "2D_DarcyFlow_beta0.1_Train.hdf5": "294f9a03a4aa16b0e386469ca8b471be",
    "2D_DarcyFlow_beta1.0_Train.hdf5": "81694ed31306ff2e5f6b76349b0b4389",
    "2D_DarcyFlow_beta10.0_Train.hdf5": "a7f23cf8011fc211b180828af39b7d1a",
    "2D_DarcyFlow_beta100.0_Train.hdf5": "7c09f3b1bb097737d3fd52ffb0c6f1c8",
}
SOURCE = ROOT / "external_data/pdebench/darus/2D_DarcyFlow_beta1.0_Train.hdf5"
OUT = ROOT / "external_data/pdebench/darcy"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--stride", type=int, default=2,
                    help="2 gives 64x64 = 4096 points; 4 gives 32x32 = 1024")
    ap.add_argument("--sizes", default="1000,200,200")
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    source, out = Path(args.source), Path(args.out)
    sizes = tuple(int(s) for s in args.sizes.split(","))
    digest = md5_of(source)
    expected = EXPECTED_MD5.get(source.name)
    if expected is None:
        raise ValueError(f"unknown Darcy file {source.name}; add its published MD5 first")
    if digest != expected:
        raise ValueError(f"MD5 mismatch for {source.name}: {digest}")

    with h5py.File(source, "r") as h:
        nu, sol = h["nu"], h["tensor"]
        if nu.shape != (10000, 128, 128):
            raise ValueError(f"unexpected coefficient shape {nu.shape}")
        if sol.shape != (10000, 1, 128, 128):
            raise ValueError(f"unexpected solution shape {sol.shape}")
        if 128 % args.stride:
            raise ValueError("stride must divide 128")
        xs, ys = h["x-coordinate"][:], h["y-coordinate"][:]
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in h.attrs.items()}
        ids = split_indices(len(nu), sizes, args.seed)
        arrays_points = {}
        for name, index in ids.items():
            idx = np.asarray(index, dtype=np.int64)
            arrays_points[name + "_x"] = np.stack([nu[int(i), ::args.stride, ::args.stride]
                                                   for i in idx]).astype(np.float32)
            arrays_points[name + "_y"] = np.stack([sol[int(i), 0, ::args.stride, ::args.stride]
                                                   for i in idx]).astype(np.float32)

    side = 128 // args.stride
    for key, value in list(arrays_points.items()):
        arrays_points[key] = np.ascontiguousarray(value.reshape(len(value), side * side))
    gy, gx = np.meshgrid(ys[::args.stride], xs[::args.stride], indexing="ij")
    arrays_points["x"] = np.stack([gy.reshape(-1), gx.reshape(-1)], axis=1).astype(np.float32)

    for key, value in arrays_points.items():
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite values in {key}")

    out.mkdir(parents=True, exist_ok=True)
    cache = out / "pairs.npz"
    tmp = cache.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays_points)
    tmp.replace(cache)

    n_points = side * side
    info = dict(protocol_version=1,
                task="steady coefficient field to solution field; no time axis",
                operator_class="elliptic, steady",
                equation="2D Darcy flow, beta from file attrs",
                source=str(source.resolve()), source_md5=digest, source_name=source.name,
                split_seed=args.seed, sample_ids={k: v.tolist() for k, v in ids.items()},
                grid_shape=[side, side], grid_size=n_points,
                branch_in=n_points, trunk_in=2, fno="fno2d",
                spatial_stride=args.stride,
                spatial_policy="point subsampling, no filtering; same grid for every model",
                file_attrs=attrs,
                physics_note="no conservation law links the input mean to the target mean, so "
                             "the input-spatial-mean predictor is not a conservation-law "
                             "predictor here; only the zero-fluctuation reference applies",
                official_split=False,
                hash_note="MD5 recorded and checked against the DaRUS published value")
    info["cache_sha256"] = file_hash(cache)
    info["normalization"] = dict(input_mean=float(arrays_points["train_x"].mean()),
                                 input_std=float(arrays_points["train_x"].std()),
                                 target_mean=float(arrays_points["train_y"].mean()),
                                 target_std=float(arrays_points["train_y"].std()))
    atomic_json(out / "protocol.json", info)
    print("prepared", out.name, "grid", info["grid_shape"], "points", n_points,
          "attrs", info["file_attrs"], "cache", info["cache_sha256"][:16])


if __name__ == "__main__":
    main()
