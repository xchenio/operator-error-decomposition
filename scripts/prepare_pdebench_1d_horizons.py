"""Prepare horizons for any PDEBench 1D file that uses the `(10000, 201, 1024)` tensor layout.

Why this exists
---------------
`src.public_benchmark.prepare()` pins the SHA-256 of the **Burgers Nu0.001** file, so it
refuses every other file even when the layout is identical. Editing that module would change
`source_hash()` inside every run identity and invalidate code-fingerprint comparison with
the finished families, so the helpers are imported and reused unchanged and this script only
skips the pinned-digest check, recording the file's own hash instead.

`scripts/prepare_advection_horizons.py` does the same for the advection file and stays as the
committed record of that run; this is the generic form used for new files so later additions
do not need another copy.

The output layout (pairs.npz keys, protocol.json keys including `cache_sha256` and
`normalization`) is identical to `prepare()`, so
`scripts/run_public_benchmark.py --data-dir` and `scripts/diagnose_fluctuation.py --data`
work unmodified.

Usage
-----
    python scripts/prepare_pdebench_1d_horizons.py \
        --source external_data/pdebench/darus/1D_Burgers_Sols_Nu0.01.hdf5 \
        --out external_data/pdebench/burgers_nu001 \
        --equation "1D Burgers, Nu=0.01"
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
    atomic_json, extract_pair, file_hash, split_indices)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def prepare_one(source: Path, folder: Path, time_index: int, stride: int, equation: str,
                sizes=(1000, 200, 200), seed: int = 20260922) -> dict:
    with h5py.File(source, "r") as f:
        data = f["tensor"]
        if data.shape != (10000, 201, 1024) or not 0 < time_index < data.shape[1]:
            raise ValueError("unexpected tensor layout/time index")
        ts, xs = f["t-coordinate"][:], f["x-coordinate"][:]
        if len(ts) not in (data.shape[1], data.shape[1] + 1) or len(xs) != data.shape[2]:
            raise ValueError("unexpected coordinate lengths")
        if stride <= 0 or data.shape[2] % stride:
            raise ValueError("stride must divide spatial resolution")
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
                equation=equation,
                source=str(source.resolve()), source_md5=md5_of(source),
                source_shape=[10000, 201, 1024], source_dtype="float32", field="tensor",
                split_seed=seed, sample_ids={k: v.tolist() for k, v in ids.items()},
                time_index=time_index, target_time=float(ts[time_index]),
                stored_time_coordinate_count=len(ts), solution_time_count=201,
                unused_trailing_time_coordinates=ts[201:].tolist(),
                spatial_stride=stride, grid_size=len(arrays["x"]),
                spatial_policy="point subsampling, no filtering; same grid for every model",
                file_attrs=attrs,
                official_split=False,
                hash_note="file MD5 recorded, not pinned; prepare() is not used because it "
                          "compares against the hard-coded Burgers Nu0.001 digest")
    info["cache_sha256"] = file_hash(cache)
    info["normalization"] = dict(input_mean=float(arrays["train_x"].mean()),
                                 input_std=float(arrays["train_x"].std()),
                                 target_mean=float(arrays["train_y"].mean()),
                                 target_std=float(arrays["train_y"].std()))
    atomic_json(folder / "protocol.json", info)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--equation", required=True)
    ap.add_argument("--time-indices", default="10,25,50,75,100,150,200")
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    source, out_root = Path(args.source), Path(args.out)
    for time_index in (int(t) for t in args.time_indices.split(",") if t):
        folder = out_root / f"fixed_t{time_index:03d}_v1"
        if (folder / "protocol.json").exists():
            print("[skip] already prepared", folder.name, flush=True)
            continue
        info = prepare_one(source, folder, time_index, args.stride, args.equation)
        print("prepared", folder.name, "target_time", info["target_time"],
              "grid", info["grid_size"], "attrs", info["file_attrs"], flush=True)


if __name__ == "__main__":
    main()
