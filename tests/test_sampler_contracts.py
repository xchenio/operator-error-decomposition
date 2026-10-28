"""Fast regression tests for every sampler's per-sample pairing contract.

Run with: `python tests/test_sampler_contracts.py` (from the repository root, with the
`pytorch` conda environment active).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.health_checks import assert_matches_loop, assert_paired_values, paired_loop_reference
from src.pdes import make_problem


def _check_scalar_input(problem_name: str) -> None:
    pde = make_problem(problem_name)
    rng = np.random.default_rng(7)
    batch, points = 3, 11
    coef = pde.sample_coeffs(batch, rng)
    coords = rng.uniform(0.0, 1.0, (batch, points))
    got = pde.input_at(coef, coords)
    ref = paired_loop_reference(pde.input_at, coef, coords)
    assert_paired_values(f"{problem_name}.input_at", got, batch, points)
    assert_matches_loop(f"{problem_name}.input_at", got, ref)


def _check_darcy() -> None:
    pde = make_problem("darcy")
    rng = np.random.default_rng(8)
    batch, points = 3, 9
    coef = pde.sample_coeffs(batch, rng)
    coords = rng.uniform(0.0, 1.0, (batch, points, 2))
    got = pde.input_and_grad(coef, coords)
    ref = tuple(np.stack([pde.input_and_grad(coef[i:i + 1], coords[i])[j].reshape(-1)
                          for i in range(batch)], axis=0) for j in range(3))
    for label, value, expected in zip(("a", "ax", "ay"), got, ref):
        assert_paired_values(f"darcy.{label}", value, batch, points)
        assert_matches_loop(f"darcy.{label}", value, expected)


def main() -> None:
    for name in ("burgers", "dr"):
        _check_scalar_input(name)
    _check_darcy()
    print("[ok] paired sampler contracts: burgers, dr, darcy")


if __name__ == "__main__":
    main()
