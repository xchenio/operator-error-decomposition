"""Explicit, reproducible initial-to-fixed-time public-data benchmark protocol."""
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

EXPECTED_SHA256 = 'afeeed1c40ce01d2ba5e1702f4f66b6bf95c65943eb0e4847691e9a10d1cb50d'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def split_indices(total, sizes, seed):
    if any(n <= 0 for n in sizes) or sum(sizes) > total:
        raise ValueError('invalid split sizes')
    order = np.random.default_rng(seed).permutation(total)[:sum(sizes)]
    return dict(zip(('train', 'val', 'test'), np.split(order, np.cumsum(sizes)[:-1])))


def extract_pair(dataset, indices, time_index, stride):
    # Scalar sample selection avoids ambiguous advanced indexing of HDF5 axes.
    inputs = np.stack([dataset[int(i), 0, ::stride] for i in indices]).astype('float32')
    targets = np.stack([dataset[int(i), time_index, ::stride] for i in indices]).astype('float32')
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError('nonfinite public data')
    return inputs, targets


def prepare(source, folder, sizes=(1000, 200, 200), seed=20260922, time_index=100, stride=4):
    source, folder = Path(source), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    digest = file_hash(source)
    if digest != EXPECTED_SHA256:
        raise ValueError('public file checksum differs from pinned publisher SHA-256')
    with h5py.File(source, 'r') as f:
        data = f['tensor']
        if data.shape != (10000, 201, 1024) or not 0 < time_index < data.shape[1]:
            raise ValueError('unexpected tensor layout/time index')
        if stride <= 0 or data.shape[2] % stride:
            raise ValueError('stride must divide spatial resolution')
        ts, xs = f['t-coordinate'][:], f['x-coordinate'][:]
        if len(ts) not in (data.shape[1], data.shape[1]+1) or len(xs) != data.shape[2]:
            raise ValueError('unexpected coordinate lengths')
        if not np.all(np.diff(ts) > 0) or not np.allclose(np.diff(xs), np.diff(xs)[0]):
            raise ValueError('invalid coordinate grid')
        ids = split_indices(len(data), sizes, seed)
        arrays = {'x': xs[::stride]}
        for name, index in ids.items():
            arrays[name+'_x'], arrays[name+'_y'] = extract_pair(data,index,time_index,stride)
        info = dict(protocol_version=1, task='supervised initial field to one fixed future time',
                    source=str(source.resolve()), source_sha256=digest, source_shape=list(data.shape),
                    source_dtype=str(data.dtype), field='tensor', split_seed=seed,
                    sample_ids={k:v.tolist() for k,v in ids.items()},
                    time_index=time_index, target_time=float(ts[time_index]),
                    stored_time_coordinate_count=len(ts), solution_time_count=data.shape[1],
                    unused_trailing_time_coordinates=ts[data.shape[1]:].tolist(),
                    spatial_stride=stride, grid_size=len(arrays['x']),
                    spatial_policy='point subsampling, no filtering; same grid for every model',
                    file_Nu=float(f.attrs['Nu']),
                    physics_note='supervised only; upstream generator diffusion uses epsilon/pi',
                    official_split=False)
    cache = folder/'pairs.npz'
    tmp = cache.with_suffix('.tmp')
    with tmp.open('wb') as out:
        np.savez_compressed(out, **arrays)
    tmp.replace(cache)
    info['cache_sha256'] = file_hash(cache)
    info['normalization'] = dict(input_mean=float(arrays['train_x'].mean()),
                                 input_std=float(arrays['train_x'].std()),
                                 target_mean=float(arrays['train_y'].mean()),
                                 target_std=float(arrays['train_y'].std()))
    atomic_json(folder/'protocol.json', info)
    return info
