"""Fit the fluctuation-energy thresholds on split 2 and score the held-out family.

Why a separate script
---------------------
`scripts/predict_skill_horizon.py` hardcodes the split-1 cache and diagnostic paths, so the
command written in section 7 of `docs/REPLICATION_SPLIT2_V1.md` cannot be run as-is against
split 2. Editing that script to take roots would risk changing behaviour under its defaults, and
its output is part of the record for section 5.8. Rather than risk either, this script reuses its two
statistics verbatim -- `a_priori()`, computed from training arrays only, and `measured()` from the
recomputed diagnostic -- and reimplements only the interpolation, which is three lines.

The deviation from the preregistered command list is recorded in section 9 of that document.

What it does
------------
1. Recompute the a-priori fluctuation-energy fraction for every split-2 family.
2. Fit one threshold per architecture by linear interpolation, using **only** the split-2
   fitting set: Burgers Nu = 0.001 (seven horizons) and linear advection (seven horizons).
3. Apply those thresholds, unchanged, to the split-2 held-out family, Burgers Nu = 0.01
   (seven horizons). An architecture whose measured skill never crosses zero on the fitting set
   has **no** fitted threshold and therefore contributes no prediction; this is reported rather
   than worked around.
4. Write `experiments/prediction_ledger_rs2.json` in the same schema as the split-1 ledger, so
   `scripts/analyze_prediction_ledger.py --ledger ...` stratifies it unchanged.

Usage
-----
    python scripts/fit_rs2_thresholds.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_prediction_ledger import (  # noqa: E402
    baseline_always_positive, group, pct, wilson)
from scripts.predict_skill_horizon import a_priori, measured  # noqa: E402

TAGS = ("010", "025", "050", "075", "100", "150", "200")
ARCHS = ("deeponet_mlp", "deeponet_fourier", "fno")

DATA_ROOT = "external_data/pdebench/rs2"
DIAG_ROOT = "results/rs2_fluctuation"

# The Burgers Nu = 0.001 family is the one whose measured skill crosses zero.
FITTING = [("burgers_nu0001", t) for t in TAGS] + [("advection", t) for t in TAGS]
HELD_OUT = [("burgers_nu001", t) for t in TAGS]

SPLIT1_THRESHOLDS = {"deeponet_mlp": 0.2477, "deeponet_fourier": 0.2728, "fno": 0.1502}

SATURATED = 0.9
MARGIN = 0.15
Z = 2.0


def rows_for(specs):
    rows = []
    for pde, tag in specs:
        data = ROOT / DATA_ROOT / pde / f"fixed_t{tag}_v1"
        diag = ROOT / DIAG_ROOT / pde / f"t{tag}" / f"rs2_{pde}_t{tag}_v1.json"
        if not (data / "protocol.json").exists() or not diag.exists():
            print(f"[skip] missing {pde} t={tag}")
            continue
        ap = a_priori(data)
        protocol = json.loads((data / "protocol.json").read_text(encoding="utf-8"))
        m = measured(diag)
        rows.append(dict(pde=pde, tag=tag, target_time=float(protocol["target_time"]),
                         fluctE=ap["fluctuation_energy_fraction"], diag=str(diag),
                         **{arch: m[arch] for arch in ARCHS}))
    return rows


def per_seed(diag, arch):
    payload = json.loads(Path(diag).read_text(encoding="utf-8"))
    return [run["splits"]["test"]["fluctuation_r2_per_sample"]
            for run in payload["runs"]
            if run["configuration"]["architecture"] == arch]


def fit_threshold(rows, arch):
    """fluctE at which measured skill crosses zero, or None if it never does."""
    ordered = sorted(rows, key=lambda r: -r["fluctE"])
    for a, b in zip(ordered, ordered[1:]):
        if a[arch] > 0 >= b[arch]:
            w = a[arch] / (a[arch] - b[arch])
            return float(a["fluctE"] * (1 - w) + b["fluctE"] * w)
    return None


def build(specs, thresholds, family_label):
    out = []
    for r in specs:
        for arch in ARCHS:
            if thresholds[arch] is None:
                continue
            values = per_seed(r["diag"], arch)
            mean_value = float(np.mean(values))
            out.append(dict(
                family=family_label,
                setting=f"t={r['target_time']:.2f}",
                architecture=arch,
                fluctuation_energy=r["fluctE"],
                predicted="+" if r["fluctE"] >= thresholds[arch] else "-",
                measured=mean_value,
                sd=float(np.std(values, ddof=1)),
                measured_sign="+" if mean_value > 0 else "-",
            ))
    return out


def derive(row, thresholds, saturated=SATURATED, margin=MARGIN, z=Z):
    thr = thresholds[row["architecture"]]
    row["correct"] = row["predicted"] == row["measured_sign"]
    row["saturated"] = row["measured"] >= saturated
    row["informative"] = not row["saturated"]
    row["boundary"] = abs(row["fluctuation_energy"] - thr) / thr <= margin
    row["resolvable"] = abs(row["measured"]) > z * row["sd"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rs2_skill_prediction_v1")
    args = ap.parse_args()

    fitting = rows_for(FITTING)
    held_out = rows_for(HELD_OUT)

    print("Refitting set on split 2 (Burgers Nu=0.001, then linear advection):")
    for r in fitting:
        print(f"  {r['pde']:14s} t={r['target_time']:<5.2f} fluctE={r['fluctE']:.4f}"
              f"  mlp={r['deeponet_mlp']:+.4f}  fourier={r['deeponet_fourier']:+.4f}"
              f"  fno={r['fno']:+.4f}")

    thresholds = {arch: fit_threshold(fitting, arch) for arch in ARCHS}
    print("\nRefitted thresholds, fitting on split 2 only:")
    for arch in ARCHS:
        t = thresholds[arch]
        drift = "" if t is None else (
            f"   (split 1 {SPLIT1_THRESHOLDS[arch]}, drift "
            f"{abs(t - SPLIT1_THRESHOLDS[arch]) / SPLIT1_THRESHOLDS[arch] * 100:.1f}%)")
        print(f"  {arch:17s} {'-' if t is None else format(t, '.4f')}{drift}")

    predictions = build(held_out, thresholds, "rs2_burgers_nu001")
    for p in predictions:
        derive(p, thresholds)

    ledger = dict(
        purpose="Preregistered held-out sign predictions on the second, disjoint split "
                "(seed 20261028, sharing no trajectory with split 1). Thresholds refitted on the "
                "split-2 fitting set only, then applied unchanged to the held-out family. "
                "See docs/REPLICATION_SPLIT2_V1.md.",
        rule="Predicted sign is positive iff the a-priori fluctuation-energy fraction of the "
             "setting is at least the architecture's threshold.",
        thresholds={arch: thresholds[arch] for arch in ARCHS},
        threshold_provenance="Interpolated from split 2 only: Burgers Nu=0.001 (seven horizons) plus "
                     "linear advection (seven horizons). Never refitted on the held-out family.",
        sources={"rs2_burgers_nu0001": {"doc": "docs/REPLICATION_SPLIT2_V1.md",
                                         "split_seed": 20261028, "role": "fitting"},
                 "rs2_advection": {"doc": "docs/REPLICATION_SPLIT2_V1.md",
                                   "split_seed": 20261028, "role": "fitting"},
                 "rs2_burgers_nu001": {"doc": "docs/REPLICATION_SPLIT2_V1.md",
                                       "split_seed": 20261028, "role": "held out"}},
        notes="Measured values are per-sample fluctuation R^2 with the seed standard deviation "
              "over five seeds, recomputed from stored checkpoints.",
        predictions=predictions,
    )
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    (ROOT / "experiments/prediction_ledger_rs2.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8")

    print(f"\nHeld-out predictions on split 2 ({len(predictions)}):")
    print(f"{'t':>6s} {'fluctE':>8s} {'arch':17s} {'pred':>5s} {'measured':>18s} "
          f"{'sat':>4s} {'bnd':>4s} {'res':>4s}")
    for p in predictions:
        print(f"{p['setting']:>6s} {p['fluctuation_energy']:8.4f} {p['architecture']:17s} "
              f"{p['predicted']:>5s} {p['measured']:+8.4f} +- {p['sd']:.4f}  "
              f"{'y' if p['saturated'] else '.':>3s}  "
              f"{'y' if p['boundary'] else '.':>3s}  "
              f"{'y' if p['resolvable'] else '.':>3s}")

    print("\nStratified (same cuts as manuscript section 5.16):")
    print(f"{'stratum':38s} {'n':>3s} {'correct':>7s} {'accuracy':>9s}  95% CI")
    cuts = [
        ("all", lambda p: True),
        ("saturated (R^2 >= 0.9)", lambda p: p["saturated"]),
        ("informative (not saturated)", lambda p: p["informative"]),
        ("boundary (within 15% of threshold)", lambda p: p["boundary"]),
        ("resolvable (|R^2| > 2 sd)", lambda p: p["resolvable"]),
        ("not resolvable", lambda p: not p["resolvable"]),
        ("informative AND resolvable", lambda p: p["informative"] and p["resolvable"]),
        ("informative AND boundary", lambda p: p["informative"] and p["boundary"]),
    ]
    for label, pred in cuts:
        g = group(predictions, pred, label)
        print(f"{g['label']:38s} {g['n']:3d} {g['correct']:7d} "
              f"{g['accuracy'] * 100:8.1f}%  {pct(*g['ci'])}")

    base = baseline_always_positive(predictions, lambda p: True, "all held-out")
    print(f"\nAlways-positive baseline: {base['correct']}/{base['n']}  "
          f"({base['accuracy'] * 100:.1f}%, 95% Wilson CI {pct(*base['ci'])})")


if __name__ == "__main__":
    main()
