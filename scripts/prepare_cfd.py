"""Prepare PDEBench 1D CFD caches, one per output component.

Layout
------
The file stores three fields, `density`, `Vx` and `pressure`, each `(10000, 101, 1024)`, plus
`t-coordinate (102,)` and `x-coordinate (1024,)`, with attributes `eta` and `zeta`. There is
no single `tensor` dataset, so `src.public_benchmark.prepare()` cannot read it, and it also
pins the Burgers digest.

Design decision, recorded rather than assumed
--------------------------------------------
This is the first **system** in the study, so a choice has to be made explicit. The branch
receives all three components at the initial time (a 3-channel 1D field, 768 values after
subsampling) and predicts **one** component at the target time. One cache is written per
output component.

Doing it this way keeps every downstream script unchanged: each cache is an ordinary
single-output task with `branch_in = 768` and `trunk_in = 1`, so the existing runner and the
existing diagnostic apply without modification. The alternative — one cache with a
three-channel output and a `(x, channel)` trunk — would have required new coordinate handling
and would have broken the FNO control, which expects a field and a coordinate of equal length.

The cost of this choice is that FNO cannot be run here: `FNO1d` lifts a value channel plus a
coordinate channel and requires the input and the coordinate to have the same length, which a
3-channel input violates. Rather than edit `src/fno.py`, which is hashed into every existing
run identity, **the FNO control is deferred for this family** and only the two DeepONet
trunks are reported.

Usage
-----
    python scripts/prepare_cfd.py
    python scripts/prepare_cfd.py --time-index 50
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

SOURCE = ROOT / "external_data/pdebench/darus/1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5"
EXPECTED_MD5 = "2fdd00138f1d7abc794fb953021a9f43"
OUT_ROOT = ROOT / "external_data/pdebench/cfd"
FIELDS = ("density", "Vx", "pressure")


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
    ap.add_argument("--time-index", type=int, default=100,
                    help="target layer; there are 101 stored layers")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--sizes", default="1000,200,200")
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    source, out_root = Path(args.source), Path(args.out)
    sizes = tuple(int(s) for s in args.sizes.split(","))
    digest = md5_of(source)
    if digest != EXPECTED_MD5:
        raise ValueError(f"MD5 mismatch for {source.name}: {digest}")

    with h5py.File(source, "r") as h:
        shapes = {f: h[f].shape for f in FIELDS}
        if any(s != (10000, 101, 1024) for s in shapes.values()):
            raise ValueError(f"unexpected field shapes {shapes}")
        if not 0 < args.time_index < 101:
            raise ValueError("time index out of range")
        if 1024 % args.stride:
            raise ValueError("stride must divide 1024")
        ts, xs = h["t-coordinate"][:], h["x-coordinate"][:]
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in h.attrs.items()}
        ids = split_indices(10000, sizes, args.seed)
        blocks = {}
        for name, index in ids.items():
            idx = np.asarray(index, dtype=np.int64)
            channels = []
            for field in FIELDS:
                ds = h[field]
                channels.append(np.stack([ds[int(i), 0, ::args.stride] for i in idx])
                                .astype(np.float32))
                blocks.setdefault(field, {})[name] = np.stack(
                    [ds[int(i), args.time_index, ::args.stride] for i in idx]).astype(np.float32)
            blocks.setdefault("__input__", {})[name] = np.concatenate(channels, axis=1)

    grid = xs[::args.stride]
    n_points = len(grid)
    for field in FIELDS:
        folder = out_root / f"{field}_t{args.time_index:03d}_v1"
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {"x": grid}
        for name in ("train", "val", "test"):
            arrays[name + "_x"] = np.ascontiguousarray(blocks["__input__"][name])
            arrays[name + "_y"] = np.ascontiguousarray(blocks[field][name])
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite values in {field}/{key}")
        cache = folder / "pairs.npz"
        tmp = cache.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(cache)
        info = dict(protocol_version=1,
                    task="supervised three-channel initial state to one component at a later time",
                    equation="1D compressible Navier-Stokes; system of three fields",
                    source=str(source.resolve()), source_md5=digest, source_name=source.name,
                    output_component=field, input_components=list(FIELDS),
                    split_seed=args.seed, sample_ids={k: v.tolist() for k, v in ids.items()},
                    time_index=args.time_index, target_time=float(ts[args.time_index]),
                    stored_time_coordinate_count=len(ts), solution_time_count=101,
                    time_step=float(ts[1] - ts[0]),
                    spatial_stride=args.stride, grid_size=n_points,
                    branch_in=3 * n_points, trunk_in=1,
                    spatial_policy="point subsampling, no filtering; same grid for every model",
                    file_attrs=attrs,
                    physics_note="mass and momentum are conserved on a periodic domain; the "
                                 "spatial mean of density and of density*Vx are conserved, but "
                                 "the spatial mean of Vx or pressure is not, so the "
                                 "input-spatial-mean predictor is a conservation-law predictor "
                                 "for some components and not others",
                    fno_note="no FNO control: FNO1d requires the input and the coordinate to "
                             "have equal length, which a three-channel input violates",
                    official_split=False,
                    hash_note="MD5 recorded and checked against the DaRUS published value")
        info["cache_sha256"] = file_hash(cache)
        info["normalization"] = dict(input_mean=float(arrays["train_x"].mean()),
                                     input_std=float(arrays["train_x"].std()),
                                     target_mean=float(arrays["train_y"].mean()),
                                     target_std=float(arrays["train_y"].std()))
        atomic_json(folder / "protocol.json", info)
        print("prepared", folder.name, "target_time", info["target_time"], "grid", n_points,
              "branch_in", info["branch_in"], flush=True)


if __name__ == "__main__":
    main()
