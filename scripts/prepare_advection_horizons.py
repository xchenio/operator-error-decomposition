"""Prepare PDEBench 1D Advection caches at several forecast times.

Why a separate script rather than reusing `src.public_benchmark.prepare`
------------------------------------------------------------------------
`prepare()` pins the publisher SHA-256 of the **Burgers** file and reads the `Nu` HDF5
attribute; neither applies to the advection file. Editing `src/public_benchmark.py` would
change `source_hash()` inside every run identity and invalidate code-fingerprint comparison
against the finished Burgers families, so this script imports and reuses the existing
helpers unchanged and only omits the Burgers-specific checks, recording the advection
file's own hash in the protocol instead.

The output layout (pairs.npz keys, protocol.json keys including `cache_sha256` and
`normalization`) is deliberately identical to `prepare()`, so
`scripts/run_public_benchmark.py --data-dir` and
`scripts/diagnose_fluctuation.py --data` work without modification.

Usage
-----
    python scripts/prepare_advection_horizons.py
    python scripts/prepare_advection_horizons.py --time-indices 25,50,100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.public_benchmark import (  # noqa: E402  reused, not modified
    atomic_json, extract_pair, file_hash, split_indices)

DEFAULT_SOURCE = ROOT / "external_data/pdebench/advection/1D_Advection_Sols_beta0.4.hdf5"
DEFAULT_OUT = ROOT / "external_data/pdebench/advection"


def prepare_one(source: Path, folder: Path, time_index: int, stride: int,
                sizes=(1000, 200, 200), seed: int = 20260922) -> dict:
    digest = file_hash(source)
    with h5py.File(source, "r") as f:
        data = f["tensor"]
        if data.shape != (10000, 201, 1024) or not 0 < time_index < data.shape[1]:
            raise ValueError("unexpected tensor layout/time index")
        ts, xs = f["t-coordinate"][:], f["x-coordinate"][:]
        if len(ts) not in (data.shape[1], data.shape[1] + 1) or len(xs) != data.shape[2]:
            raise ValueError("unexpected coordinate lengths")
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
        ids = split_indices(len(data), sizes, seed)
        arrays = {"x": xs[::stride]}
        for name, index in ids.items():
            arrays[name + "_x"], arrays[name + "_y"] = extract_pair(data, index, time_index, stride)

    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / "pairs.npz"
    tmp = cache.with_suffix(".tmp")
    with tmp.open("wb") as out:
        np.savez_compressed(out, **arrays)
    tmp.replace(cache)

    info = dict(protocol_version=1,
                task="supervised initial field to one fixed future time",
                equation="1D linear advection, periodic, beta from file attrs",
                source=str(source.resolve()), source_sha256=digest,
                source_shape=[10000, 201, 1024], source_dtype="float32", field="tensor",
                split_seed=seed, sample_ids={k: v.tolist() for k, v in ids.items()},
                time_index=time_index, target_time=float(ts[time_index]),
                stored_time_coordinate_count=len(ts), solution_time_count=201,
                unused_trailing_time_coordinates=ts[201:].tolist(),
                spatial_stride=stride, grid_size=len(arrays["x"]),
                spatial_policy="point subsampling, no filtering; same grid for every model",
                file_attrs=attrs,
                physics_note="linear transport; spatial mean is conserved and, unlike "
                             "Burgers, fluctuation energy is expected to be horizon-independent",
                official_split=False,
                hash_note="file hash is recorded, not pinned; unlike prepare() this script "
                          "does not compare against a hard-coded publisher digest")
    info["cache_sha256"] = file_hash(cache)
    info["normalization"] = dict(input_mean=float(arrays["train_x"].mean()),
                                 input_std=float(arrays["train_x"].std()),
                                 target_mean=float(arrays["train_y"].mean()),
                                 target_std=float(arrays["train_y"].std()))
    atomic_json(folder / "protocol.json", info)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--time-indices", default="10,25,50,75,100,150,200")
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    source = Path(args.source)
    out_root = Path(args.out)
    for time_index in (int(t) for t in args.time_indices.split(",") if t):
        folder = out_root / f"fixed_t{time_index:03d}_v1"
        if (folder / "protocol.json").exists():
            print("[skip] already prepared", folder.name, flush=True)
            continue
        info = prepare_one(source, folder, time_index, args.stride)
        print("prepared", folder.name, "target_time", info["target_time"],
              "grid", info["grid_size"], "attrs", info["file_attrs"], flush=True)


if __name__ == "__main__":
    main()
