"""Contracts that prevent silent sample-pairing errors in operator-learning code."""
from __future__ import annotations

import numpy as np


def assert_paired_values(name: str, values: np.ndarray, batch: int, points: int) -> None:
    """Require one target value per `(sample, query-point)` pair.

    This deliberately rejects `(B, B, N)`, the silent-broadcasting shape that motivated this
    project. Keep this assertion at the sampler boundary, before conversion to torch tensors.
    """
    got = tuple(np.asarray(values).shape)
    expected = (int(batch), int(points))
    if got != expected:
        raise AssertionError(f"{name}: expected paired shape {expected}, got {got}")


def assert_matches_loop(name: str, batched: np.ndarray, loop_values: np.ndarray,
                        atol: float = 1e-6, rtol: float = 1e-6) -> None:
    """Check a vectorized paired evaluator against a one-sample-at-a-time reference."""
    if not np.allclose(batched, loop_values, atol=atol, rtol=rtol):
        err = float(np.max(np.abs(np.asarray(batched) - np.asarray(loop_values))))
        raise AssertionError(f"{name}: paired evaluator differs from loop reference; max err={err:.3e}")


def paired_loop_reference(fn, coef: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Evaluate `fn` once per sample on its own coordinates; independent batching oracle."""
    rows = [np.asarray(fn(coef[i:i + 1], coords[i])).reshape(-1) for i in range(len(coef))]
    return np.stack(rows, axis=0)
