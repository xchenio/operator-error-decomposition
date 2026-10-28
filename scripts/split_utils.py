"""Drawing reproducible data splits, optionally disjoint from an earlier one.

Why this exists
---------------
Every family in this study uses the same split (seed 20260922, sizes 1000/200/200) so that
trajectory ids match across families. That single split is also the study's most attackable
limitation, since its test set was exposed during exploratory development
(`paper/main.md` section 6).

The fix is a second split on the same files. But two independently seeded draws of 1400 from
10000 trajectories overlap on about 196 of them, which is not an independent sample. So the
second split is drawn from the **complement** of the first, which makes it share no trajectory
at all. See `docs/REPLICATION_SPLIT2_V1.md`, which was committed before any run on split 2.

Determinism
-----------
`draw_split(total, sizes, seed, exclude_seed)` depends only on its four arguments and on
`numpy.random.default_rng`, whose bit stream is guaranteed stable across NumPy versions by the
NumPy stream-compatibility policy. Re-running it must always give the same ids, and
`tests/test_replication_split.py` pins that.

This module is deliberately *not* in `src/`: nothing here is hashed into a run identity, so
these helpers can be changed without invalidating any recorded run.
"""
from __future__ import annotations

import numpy as np


def draw_split(total: int, sizes=(1000, 200, 200), seed: int = 20260922,
               exclude_seed: int | None = None) -> dict[str, np.ndarray]:
    """Return `{'train', 'val', 'test'}` index arrays.

    With `exclude_seed=None` this reproduces the original single-split behaviour exactly: a
    permutation of `range(total)` truncated to `sum(sizes)`. Otherwise the sample is taken from
    the trajectories **not** used by the excluded split, so the two are disjoint by
    construction.

    Raises ValueError if the sizes do not fit, including when the exclusion leaves too few
    trajectories — which would otherwise silently produce a smaller split.
    """
    if any(n <= 0 for n in sizes):
        raise ValueError('invalid split sizes')
    if sum(sizes) > total:
        raise ValueError('split sizes exceed the number of available trajectories')

    if exclude_seed is None:
        order = np.random.default_rng(seed).permutation(total)[:sum(sizes)]
    else:
        # Imported here to keep this module importable without the src package on the path.
        from src.public_benchmark import split_indices
        excluded = np.concatenate(
            list(split_indices(total, sizes, exclude_seed).values()))
        pool = np.setdiff1d(np.arange(total), excluded)
        if len(pool) < sum(sizes):
            raise ValueError(
                f'excluding split seed {exclude_seed} leaves {len(pool)} trajectories, '
                f'which is fewer than the requested {sum(sizes)}')
        order = np.random.default_rng(seed).permutation(pool)[:sum(sizes)]

    return dict(zip(('train', 'val', 'test'), np.split(order, np.cumsum(sizes)[:-1])))


def split_fields(ids: dict[str, np.ndarray], seed: int, sizes=(1000, 200, 200),
                 exclude_seed: int | None = None) -> dict:
    """The protocol.json entries that record how the split was drawn."""
    return dict(split_seed=seed,
                split_sizes=list(sizes),
                excluded_seed=exclude_seed,
                disjoint_from_split1=exclude_seed is not None,
                sample_ids={k: v.tolist() for k, v in ids.items()})
