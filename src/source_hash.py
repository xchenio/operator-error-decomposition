"""One definition of the run-identity source hash, with every historic value preserved.

Background
----------
Four scripts each carried their own copy of `source_hash()` with a different file list, and
one composed two of them by hand. The duplication is a maintenance hazard: a new file added
to one list is invisible to the others, and the four values are not comparable.

Why the lists are mirrored here rather than imported by the scripts
------------------------------------------------------------------
Each list is part of the **identity** of every run recorded under it: `train_one` refuses to
resume or overwrite a run whose identity changed, and roughly a thousand runs are already
recorded. Crucially, **each list begins with the entry script itself**, so simply replacing a
local `source_hash()` with an import from here would change that script's bytes and therefore
change its own hash — invalidating every run of that family in one step. That is the exact
failure this repository guards against elsewhere (two files four days apart, no record of
which code produced which result).

The four `source_hash()` definitions therefore stay where they are. This module records the
authoritative lists in one place and `tests/test_source_hash.py` **pins each live value**
against `results/source_hash_baseline.json`, which was recorded before this module existed.
Any edit to a listed file, including a line-ending change, now fails that test instead of
silently shifting a run identity.

If a future change genuinely needs to move a list, do it deliberately and accept a new result
directory per family: the point is that the decision is recorded, not that it never happens.

Hashing scheme (unchanged)
--------------------------
For each file in order: update with the file's *name*, then with its *bytes*. Including the
name and not the full path keeps the value independent of where the repository is checked
out.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# FROZEN. The first entry is the entry script; the remainder are source modules.
FROZEN_FILE_LISTS: dict[str, list[str]] = {
    "public_benchmark": [
        "scripts/run_public_benchmark.py",
        "src/fno.py",
        "src/models.py",
        "src/resume.py",
        "src/public_benchmark.py",
        "src/projection_audit.py",
        "src/utils.py",
        "src/derivatives.py",
    ],
    "physics_public": [
        "scripts/run_physics_public.py",
        "src/models.py",
        "src/derivatives.py",
        "src/resume.py",
        "src/public_benchmark.py",
        "src/utils.py",
    ],
    "darcy": [
        "scripts/run_darcy_benchmark.py",
        "src/fno2d.py",
        "src/models.py",
        "src/fno1d_multi.py",
        "src/fno2d_multi.py",
        "src/resume.py",
        "src/public_benchmark.py",
        "src/utils.py",
        "src/derivatives.py",
    ],
    "diagnose_fluctuation": [
        "scripts/diagnose_fluctuation.py",
        "src/models.py",
        "src/fno.py",
    ],
}


def source_hash(kind: str) -> str:
    """Return the frozen source hash for `kind`.

    Raises KeyError for an unknown kind rather than silently returning something else, so a
    typo cannot produce a run whose identity happens to look plausible.
    """
    if kind not in FROZEN_FILE_LISTS:
        raise KeyError(f"unknown source_hash kind {kind!r}; "
                       f"known: {sorted(FROZEN_FILE_LISTS)}")
    h = hashlib.sha256()
    for relative in FROZEN_FILE_LISTS[kind]:
        path = ROOT / relative
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def source_files(kind: str) -> list[Path]:
    """The frozen file list for `kind`, resolved. Useful for error messages and tests."""
    return [ROOT / relative for relative in FROZEN_FILE_LISTS[kind]]
