"""Preflight for the disjoint-split replication: split 2 must share no trajectory with split 1.

This is a hard gate, not a note. `docs/REPLICATION_SPLIT2_V1.md` commits to a second split that
is disjoint from the first, so if this check fails the replication has not been set up
correctly and no training should start.

Draw                                                                            expected result
----------------------------------------------------------------------------------------------
two independently seeded draws of 1400 from 10000 trajectories                  ~196 shared ids
seed 20261028 drawn from the complement of seed 20260922 (what we use)          0 shared ids

Usage
-----
    python scripts/verify_split_disjointness.py --seed-a 20260922 --seed-b 20261028 \
        --sizes 1000,200,200 --total 10000

    # also check what is actually recorded in prepared protocol.json files
    python scripts/verify_split_disjointness.py --seed-a 20260922 --seed-b 20261028 \
        --protocols external_data/pdebench/rs2/burgers_nu0001/fixed_t100_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.split_utils import draw_split  # noqa: E402


def flatten(ids) -> set[int]:
    return {int(i) for split in ids for i in split}


def check(seed_a: int, seed_b: int, sizes: tuple[int, ...], total: int) -> bool:
    """Return True if seed_b, drawn disjointly from seed_a, shares no id with it."""
    a = draw_split(total, sizes, seed_a)
    b = draw_split(total, sizes, seed_b, exclude_seed=seed_a)
    a_ids, b_ids = flatten(a.values()), flatten(b.values())

    print(f"total trajectories   {total}")
    print(f"sizes                {sizes} (sum {sum(sizes)})")
    print(f"split A seed         {seed_a}  -> {len(a_ids)} ids")
    print(f"split B seed         {seed_b}  -> {len(b_ids)} ids, drawn outside A")

    ok = True
    shared = a_ids & b_ids
    print(f"shared overall       {len(shared)}")
    if shared:
        ok = False
        print("FAIL: the splits overlap. Expected an empty intersection.")
    else:
        print("OK:   disjoint overall")

    print()
    print(f"{'split':6s} {'n':>5s} {'shared':>7s}")
    for name in ("train", "val", "test"):
        s = set(int(i) for i in a[name]) & set(int(i) for i in b[name])
        print(f"{name:6s} {len(b[name]):5d} {len(s):7d}")
        if s:
            ok = False
    if ok:
        print("OK:   disjoint per split too")
    else:
        print("FAIL: at least one split-level intersection is non-empty")

    # Every id in B must be absent from A by construction; also confirm B never reuses ids
    # within itself (it is a permutation of the pool, so this must hold).
    all_b = [int(i) for split in b.values() for i in split]
    if len(set(all_b)) != len(all_b):
        print("FAIL: split B repeats ids internally")
        ok = False
    return ok


def check_protocols(paths: list[Path], seed_a: int, sizes: tuple[int, ...], total: int) -> bool:
    """Check the ids actually written into prepared protocol.json files."""
    a_ids = flatten(draw_split(total, sizes, seed_a).values())
    ok = True
    for p in paths:
        protocol = p / "protocol.json"
        if not protocol.exists():
            print(f"[skip] no protocol.json under {p}")
            continue
        info = json.loads(protocol.read_text(encoding="utf-8"))
        written = flatten(info.get("sample_ids", {}).values())
        shared = a_ids & written
        seed = info.get("split_seed")
        excl = info.get("excluded_seed")
        disjoint = info.get("disjoint_from_split1", False)
        print(f"\n{p}")
        print(f"  split_seed={seed} excluded_seed={excl} disjoint_from_split1={disjoint}")
        print(f"  ids={len(written)}  shared with split 1={len(shared)}")
        if not disjoint or excl is None:
            print("  FAIL: not declared disjoint")
            ok = False
        if shared:
            print("  FAIL: overlaps split 1")
            ok = False
        if not shared and disjoint:
            print("  OK")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-a", type=int, default=20260922)
    ap.add_argument("--seed-b", type=int, default=20261028)
    ap.add_argument("--sizes", default="1000,200,200")
    ap.add_argument("--total", type=int, default=10000)
    ap.add_argument("--protocols", nargs="*", default=[])
    args = ap.parse_args()

    sizes = tuple(int(s) for s in args.sizes.split(",") if s)
    ok = check(args.seed_a, args.seed_b, sizes, args.total)

    if args.protocols:
        print("\n--- prepared caches ---")
        ok = check_protocols([Path(p) for p in args.protocols], args.seed_a, sizes,
                             args.total) and ok

    print()
    print("PREFLIGHT PASSED" if ok else "PREFLIGHT FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
