"""Prepare PDEBench 1D diffusion-sorption caches at several forecast times.

Layout differences from the Burgers and advection files
-------------------------------------------------------
This file stores **one HDF5 group per sample** (`"0000"` ... `"9999"`), each containing
`data (101, 1024, 1)`, `grid/t (101,)` and `grid/x (1024,)`. There is no single `tensor`
dataset, so `src.public_benchmark.prepare()` cannot read it, and it also pins the Burgers
SHA-256. As with `prepare_advection_horizons.py`, the helpers are imported and reused
unchanged so that `source_hash()` in every run identity stays untouched.

Every sample is read **once** and all requested horizons are extracted from that single
read, because the file is organised by sample rather than by time.

Physics: parabolic diffusion-sorption on `x in [0,1]`, `t in [0, 500]` with `dt = 5` and
101 stored layers, values in `[0,1]`. Unlike Burgers there are no shocks, and unlike
advection the fluctuation is not merely transported, so this is the setting that separates
"nonlinear steepening" from "dissipation" as the driver of the skill horizon.

**Documented data caveat (checked, not assumed):** the stored initial layer is spatially
constant for every sample (`std < 4e-8`; only the constant value varies between samples,
e.g. 0.10976, 0.08340, 0.13929). The true `t=0` state therefore carries a single degree of
freedom, so "constant -> field" is a degenerate operator-learning task and the
fluctuation decomposition would be measuring generation from a scalar rather than the
transport of structure. For that reason the default input layer is **not** 0 but a later
layer that already has spatial structure (`--input-index 20`, i.e. `t = 100`), and the
horizons are taken strictly after it. This is recorded in every protocol file.

Usage
-----
    python scripts/prepare_diffsorp_horizons.py
    python scripts/prepare_diffsorp_horizons.py --time-indices 10,40,100
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

SOURCE = ROOT / "external_data/pdebench/darus/1D_diff-sorp_NA_NA.h5"
EXPECTED_MD5 = "9d466d1213065619d087319e16d9a938"
OUT_ROOT = ROOT / "external_data/pdebench/diffsorp"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--time-indices", default="25,40,55,70,85,100",
                    help="target layer indices, all strictly after --input-index; t = index * 5")
    ap.add_argument("--input-index", type=int, default=20,
                    help="layer used as the input field; layer 0 is spatially constant")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--sizes", default="1000,200,200")
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    source = Path(args.source)
    sizes = tuple(int(s) for s in args.sizes.split(","))
    time_indices = [int(t) for t in args.time_indices.split(",") if t]
    out_root = Path(args.out)

    digest = md5_of(source)
    if digest != EXPECTED_MD5:
        raise ValueError(f"MD5 mismatch for {source.name}: {digest}")

    with h5py.File(source, "r") as h:
        names = sorted(h.keys())
        total = len(names)
        first = h[names[0]]
        t_coord = first["grid/t"][:]
        x_coord = first["grid/x"][:]
        n_time, n_space = first["data"].shape[0], first["data"].shape[1]
        if any(not (0 <= ti < n_time) for ti in time_indices):
            raise ValueError("time index out of range")
        if not 0 <= args.input_index < n_time:
            raise ValueError("input index out of range")
        if min(time_indices) <= args.input_index:
            raise ValueError("every target index must be strictly after the input index")
        input_std = float(h[names[0]]["data"][args.input_index, :, 0].std())
        layer0_std = float(h[names[0]]["data"][0, :, 0].std())
        layers = [args.input_index] + time_indices
        if n_space % args.stride:
            raise ValueError("stride must divide spatial resolution")

        ids = split_indices(total, sizes, args.seed)
        # read every selected sample once, keeping all requested layers
        per_split = {}
        for name, index in ids.items():
            arr = np.empty((len(index), len(layers), n_space // args.stride),
                           dtype=np.float32)
            for row, sample in enumerate(index):
                arr[row] = h[names[int(sample)]]["data"][:, ::args.stride, 0][layers]
            per_split[name] = arr
        grid = x_coord[::args.stride]

    checks = []
    for name, arr in per_split.items():
        flat = arr.reshape(-1)
        finite = bool(np.isfinite(flat).all())
        checks.append((name, arr.shape, finite, float(flat.min()), float(flat.max())))
        if not finite:
            raise ValueError(f"nonfinite values in {name}")

    for column, ti in enumerate(time_indices):
        folder = out_root / f"in{args.input_index:03d}_t{ti:03d}_v1"
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {"x": grid}
        for name, arr in per_split.items():
            arrays[name + "_x"] = np.ascontiguousarray(arr[:, 0, :])          # input layer
            arrays[name + "_y"] = np.ascontiguousarray(arr[:, column + 1, :])  # target layer
        cache = folder / "pairs.npz"
        tmp = cache.with_suffix(".tmp")
        with tmp.open("wb") as out:
            np.savez_compressed(out, **arrays)
        tmp.replace(cache)
        info = dict(
            protocol_version=1,
            task="supervised initial field to one fixed future time",
            equation="1D diffusion-sorption; parabolic, diffusion-dominated, no shocks",
            source=str(source.resolve()), source_md5=digest, source_name=source.name,
            layout="one HDF5 group per sample: data (101,1024,1), grid/t, grid/x",
            sample_count=total, split_seed=args.seed,
            sample_ids={k: v.tolist() for k, v in ids.items()},
            time_index=ti, target_time=float(t_coord[ti]),
            input_index=args.input_index, input_time=float(t_coord[args.input_index]),
            stored_time_coordinate_count=len(t_coord), solution_time_count=n_time,
            layer0_spatial_std=layer0_std, input_layer_spatial_std=input_std,
            degenerate_initial_state=bool(layer0_std < 1e-6),
            time_step=float(t_coord[1] - t_coord[0]),
            spatial_stride=args.stride, grid_size=len(grid),
            spatial_policy="point subsampling, no filtering; same grid for every model",
            value_checks=[dict(split=c[0], shape=list(c[1]), finite=c[2],
                               min=c[3], max=c[4]) for c in checks],
            physics_note="dissipative: fluctuation energy is expected to decay with horizon, "
                         "unlike pure transport and unlike a conserved-mean Burgers field",
            official_split=False,
            hash_note="MD5 recorded and checked against the DaRUS published value; "
                      "prepare() is not used because it pins the Burgers SHA-256 and "
                      "expects a single tensor dataset")
        info["cache_sha256"] = file_hash(cache)
        info["normalization"] = dict(input_mean=float(arrays["train_x"].mean()),
                                     input_std=float(arrays["train_x"].std()),
                                     target_mean=float(arrays["train_y"].mean()),
                                     target_std=float(arrays["train_y"].std()))
        atomic_json(folder / "protocol.json", info)
        print("prepared", folder.name, "target_time", info["target_time"],
              "grid", info["grid_size"], "cache", info["cache_sha256"][:16], flush=True)


if __name__ == "__main__":
    main()
