"""Put a number on the physics-informed arm's unmatched work per update.

The gap this closes
-------------------
Section 5.11 compares physics-informed training with supervised training and states that the
arm "uses a different batch and collocation size, so its work per update is not matched to the
supervised runs". True, but unquantified — which lets a reader supply their own guess in either
direction. This script measures the ratio instead.

What is measured
----------------
Wall-clock seconds per training update for each arm, taken from the `train_seconds` and final
`step` recorded in every stored `result.json`. These are the real training runs, timed by the
training loop itself with `torch.cuda.synchronize` around the update, so this is not a
micro-benchmark and nothing is re-implemented: whatever the update cost, this is what it cost.

  seconds_per_update = train_seconds / final_step
  ratio              = physics_informed / supervised

Wall-clock is machine dependent, so the ratio is reported together with the absolute numbers and
the device they were taken on; section 5.9 already uses wall-clock per update as one of three
yardsticks for the same reason.

Usage
-----
    python scripts/measure_pi_update_cost.py --out results/pi_update_cost_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# (label, physics-informed family, supervised family)
PAIRS = [
    ("Burgers Nu=0.01, t=0.50", "physics_nu001_t050_v1", "burgers_nu001_t050_v1"),
    ("Burgers Nu=0.01, t=1.00", "physics_nu001_t100_v1", "burgers_nu001_t100_v1"),
    ("linear advection, t=1.00", "physics_advection_t100_v1", "advection_horizon_t100_v1"),
]

ARCHS = ("deeponet_mlp", "deeponet_fourier", "fno")


def per_update_seconds(folder: Path) -> dict:
    """Mean seconds per update, per architecture, over the seeds present."""
    out = {}
    for result_path in sorted(folder.glob("*/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        arch = (payload.get("identity") or {}).get("configuration", {}).get("architecture")
        history = payload.get("history") or []
        final_step = history[-1].get("step") if history else None
        if arch is None or not final_step:
            continue
        out.setdefault(arch, []).append(payload["train_seconds"] / float(final_step))
    return {arch: float(np.mean(vals)) for arch, vals in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pi_update_cost_v1")
    args = ap.parse_args()

    rows = []
    for label, pi_family, sup_family in PAIRS:
        pi = per_update_seconds(ROOT / "results" / pi_family)
        sup = per_update_seconds(ROOT / "results" / sup_family)
        for arch in ARCHS:
            if arch not in pi or arch not in sup:
                continue
            rows.append(dict(setting=label, architecture=arch,
                             seconds_per_update_pi=pi[arch],
                             seconds_per_update_supervised=sup[arch],
                             ratio=pi[arch] / sup[arch]))
            print(f"{label:26s} {arch:17s} "
                  f"PI {pi[arch] * 1e3:7.3f} ms  sup {sup[arch] * 1e3:7.3f} ms  "
                  f"ratio {pi[arch] / sup[arch]:5.2f}x", flush=True)

    ratios = [r["ratio"] for r in rows]
    report = dict(
        purpose="Quantify the unmatched work per update in the physics-informed arm "
                "(manuscript section 5.11), so the caveat carries a measured factor instead of "
                "an unquantified one.",
        method="Wall-clock seconds per training update read from the train_seconds and final "
               "step recorded in each stored result.json; mean over the five seeds.",
        device="CUDA (NVIDIA GeForce RTX 5060 Laptop GPU)",
        unit="seconds per optimizer step, i.e. one forward, loss, backward and step",
        rows=rows,
        ratio_min=float(np.min(ratios)),
        ratio_max=float(np.max(ratios)),
        ratio_mean=float(np.mean(ratios)),
    )
    print(f"\nratio over {len(ratios)} comparisons: "
          f"min {np.min(ratios):.2f}x, mean {np.mean(ratios):.2f}x, max {np.max(ratios):.2f}x",
          flush=True)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("written", out / "report.json")


if __name__ == "__main__":
    main()
