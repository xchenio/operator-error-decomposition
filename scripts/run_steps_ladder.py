"""Drive the E0 update-budget ladder with retries around Windows atomic-replace failures.

Why this exists
---------------
`src/public_benchmark.atomic_json` uses `Path.replace`, which on this machine
occasionally raises `PermissionError: [WinError 5]` when the destination file is
momentarily locked. The training scripts themselves are resumable and skip completed
runs, so the safe response is to re-invoke them rather than to weaken the atomic write
or to change any source file that participates in a run identity. This driver only
re-launches an existing, unchanged script and stops when every run of a budget is
complete.

Changing `src/public_benchmark.py` would also change `source_sha256` inside every run
identity and invalidate comparisons with already-finished families, so it is avoided.

Usage
-----
    python scripts/run_steps_ladder.py --budgets 20000,50000 --device cuda
    python scripts/run_steps_ladder.py --budgets 20000 --seeds 0,1 --device cuda
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def completed(out_dir: Path) -> int:
    return len(list((ROOT / out_dir).glob("*/result.json")))


def run_budget(budget: int, device: str, seeds: str, expected: int, attempts: int) -> dict:
    out = f"results/public_steps{budget // 1000}k_v1"
    out_dir = ROOT / out
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "results"
    history = []
    for attempt in range(1, attempts + 1):
        done = completed(out_dir)
        if done >= expected:
            break
        log = log_dir / f"steps{budget // 1000}k_try{attempt}.log"
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                [sys.executable, "scripts/run_public_benchmark.py",
                 "--steps", str(budget), "--out", out, "--device", device, "--seeds", seeds],
                cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        done = completed(out_dir)
        history.append(dict(attempt=attempt, exit_code=proc.returncode, completed_after=done))
        print(f"budget={budget} attempt={attempt} exit={proc.returncode} completed={done}/{expected}",
              flush=True)
        if proc.returncode == 0 and done >= expected:
            break
        time.sleep(2)
    return dict(budget=budget, out=out, completed=completed(out_dir), expected=expected,
                attempts=history)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="20000,50000")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--expected", type=int, default=15,
                    help="runs per budget: architectures x seeds")
    ap.add_argument("--attempts", type=int, default=40)
    args = ap.parse_args()

    report = []
    for budget in (int(b) for b in args.budgets.split(",") if b):
        report.append(run_budget(budget, args.device, args.seeds, args.expected, args.attempts))
    summary = ROOT / "results/steps_ladder_v1" / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(dict(report=report), indent=2), encoding="utf-8")
    for r in report:
        print(f"[done] {r['out']}: {r['completed']}/{r['expected']}")


if __name__ == "__main__":
    main()
