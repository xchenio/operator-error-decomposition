"""Stratify the preregistered sign predictions instead of reporting one headline count.

Background
----------
The manuscript reports that the fluctuation-energy threshold rule got 52 of 55 preregistered
sign predictions correct. That single number is honest but easy to attack, because the 55 are
not equally informative:

* 25 of them are on settings where every architecture scores |R^2| >= 0.9, so the sign is
  not in doubt for any method and the prediction cannot discriminate;
* some sit far from the threshold, where the rule only has to get the sign of an obviously
  easy case right;
* some measured values are within their own seed standard deviation of zero, so their sign is
  not statistically resolvable in the first place.

This script recomputes the count inside each stratum and compares it with trivial baselines,
so the strength of the rule can be read at the level it is actually claimed.

Strata
------
* `saturated`   : measured >= --saturated (default 0.9). The architecture is essentially
                  perfect, so the sign is not in doubt and the setting cannot discriminate
                  between methods. The test is signed on purpose: a large *negative* R^2 is a
                  decisive failure and is informative, not saturated.
* `boundary`    : |fluctuation_energy - threshold| / threshold <= --margin (default 0.15).
                  The rule is being tested where it could go either way.
* `resolvable`  : |measured| > --z * sd (default 2). The measured sign is separated from zero
                  by more than the seed spread, so the verdict is not a coin flip.

Baselines
---------
* `always positive` : predict '+' everywhere. The rule predicted '+' in 44 of 55 cases, so
  this is the baseline a skeptic will reach for.
* Wilson score intervals are reported for every proportion with n >= 1.

Usage
-----
    python scripts/analyze_prediction_ledger.py
    python scripts/analyze_prediction_ledger.py --ledger experiments/prediction_ledger.json \
        --out results/prediction_ledger_v1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(successes: int, total: int, z: float = 1.96):
    """Wilson score interval; behaves sensibly at 0 and 1 unlike the normal approximation."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(lo: float, hi: float) -> str:
    return f"[{lo*100:.1f}, {hi*100:.1f}]"


def load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    thresholds = data["thresholds"]
    rows = []
    for p in data["predictions"]:
        thr = thresholds[p["architecture"]]
        # Recompute the prediction from the rule rather than trusting the recorded string.
        rule_sign = "+" if p["fluctuation_energy"] >= thr else "-"
        measured_sign = "+" if p["measured"] > 0 else "-"
        rows.append(dict(
            family=p["family"], setting=p["setting"], architecture=p["architecture"],
            fluctE=p["fluctuation_energy"], threshold=thr,
            predicted=p["predicted"], predicted_by_rule=rule_sign,
            measured=p["measured"], sd=p["sd"], measured_sign=measured_sign,
            correct=p["predicted"] == measured_sign,
            margin=abs(p["fluctuation_energy"] - thr) / thr,
        ))
    return data, thresholds, rows


def strata(rows, saturated, margin, z):
    for r in rows:
        r["saturated"] = r["measured"] >= saturated
        r["boundary"] = r["margin"] <= margin
        r["resolvable"] = abs(r["measured"]) > z * r["sd"]
        r["informative"] = not r["saturated"]


def group(rows, predicate, label):
    sel = [r for r in rows if predicate(r)]
    ok = sum(r["correct"] for r in sel)
    lo, hi = wilson(ok, len(sel))
    return dict(label=label, n=len(sel), correct=ok, accuracy=ok / len(sel) if sel else float("nan"),
                ci=[lo, hi])


def baseline_always_positive(rows, predicate, label):
    sel = [r for r in rows if predicate(r)]
    ok = sum(1 for r in sel if r["measured_sign"] == "+")
    lo, hi = wilson(ok, len(sel))
    return dict(label=label, n=len(sel), correct=ok,
                accuracy=ok / len(sel) if sel else float("nan"), ci=[lo, hi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="experiments/prediction_ledger.json")
    ap.add_argument("--out", default="results/prediction_ledger_v1")
    ap.add_argument("--saturated", type=float, default=0.9,
                    help="|R^2| at or above this counts as a saturated, non-discriminating setting")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="relative distance from the threshold that counts as a boundary test")
    ap.add_argument("--z", type=float, default=2.0,
                    help="measured |R^2| must exceed z * seed sd to count as resolvable")
    args = ap.parse_args()

    data, thresholds, rows = load(ROOT / args.ledger)
    strata(rows, args.saturated, args.margin, args.z)

    # Self-check: every recorded prediction must be reproducible from the stated rule.
    mismatched = [r for r in rows if r["predicted"] != r["predicted_by_rule"]]
    if mismatched:
        for r in mismatched:
            print(f"[FAIL] {r['family']} {r['setting']} {r['architecture']}: "
                  f"recorded {r['predicted']} but the rule gives {r['predicted_by_rule']}")
        raise SystemExit("recorded predictions do not follow the stated rule")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preregistered predictions: {len(rows)}")
    print(f"Thresholds (never refitted): "
          + ", ".join(f"{k}={v}" for k, v in thresholds.items()))
    print(f"Strata: saturated |R^2| >= {args.saturated}; "
          f"boundary margin <= {args.margin}; resolvable |R^2| > {args.z} * sd\n")

    table = [
        group(rows, lambda r: True, "all"),
        group(rows, lambda r: r["saturated"], "saturated (cannot discriminate)"),
        group(rows, lambda r: r["informative"], "informative (not saturated)"),
        group(rows, lambda r: r["boundary"], "boundary (near threshold)"),
        group(rows, lambda r: r["resolvable"], "resolvable (|R^2| > z*sd)"),
        group(rows, lambda r: not r["resolvable"], "not resolvable"),
        group(rows, lambda r: r["informative"] and r["resolvable"], "informative AND resolvable"),
        group(rows, lambda r: r["informative"] and r["boundary"], "informative AND boundary"),
    ]
    print(f"{'stratum':34s} {'n':>3s} {'correct':>7s} {'accuracy':>9s}  95% Wilson CI")
    for g in table:
        print(f"{g['label']:34s} {g['n']:3d} {g['correct']:7d} "
              f"{g['accuracy']*100:8.1f}%  {pct(*g['ci'])}")

    print("\nBaseline: always predict '+'")
    print(f"{'stratum':34s} {'n':>3s} {'correct':>7s} {'accuracy':>9s}  95% Wilson CI")
    baselines = [
        baseline_always_positive(rows, lambda r: True, "all"),
        baseline_always_positive(rows, lambda r: r["informative"], "informative (not saturated)"),
        baseline_always_positive(rows, lambda r: r["informative"] and r["resolvable"],
                                 "informative AND resolvable"),
    ]
    for g in baselines:
        print(f"{g['label']:34s} {g['n']:3d} {g['correct']:7d} "
              f"{g['accuracy']*100:8.1f}%  {pct(*g['ci'])}")

    print("\nPer family")
    print(f"{'family':16s} {'n':>3s} {'correct':>7s} {'saturated':>10s} {'boundary':>9s}")
    for fam in dict.fromkeys(r["family"] for r in rows):
        sel = [r for r in rows if r["family"] == fam]
        print(f"{fam:16s} {len(sel):3d} {sum(r['correct'] for r in sel):7d} "
              f"{sum(r['saturated'] for r in sel):10d} {sum(r['boundary'] for r in sel):9d}")

    misses = [r for r in rows if not r["correct"]]
    print("\nMisses")
    for r in misses:
        ratio = abs(r["measured"]) / r["sd"] if r["sd"] else float("inf")
        print(f"  {r['family']:14s} {r['setting']:12s} {r['architecture']:18s} "
              f"predicted {r['predicted']}  measured {r['measured']:+.4f} +- {r['sd']:.4f} "
              f"(|R^2|/sd = {ratio:.2f})")

    report = dict(
        ledger=args.ledger,
        settings=dict(saturated=args.saturated, margin=args.margin, z=args.z),
        thresholds=thresholds,
        n_predictions=len(rows),
        n_correct=sum(r["correct"] for r in rows),
        strata=table,
        baselines=baselines,
        by_family={fam: dict(n=len([r for r in rows if r["family"] == fam]),
                             correct=sum(r["correct"] for r in rows if r["family"] == fam),
                             saturated=sum(r["saturated"] for r in rows if r["family"] == fam),
                             boundary=sum(r["boundary"] for r in rows if r["family"] == fam))
                   for fam in dict.fromkeys(r["family"] for r in rows)},
        misses=[{k: r[k] for k in ("family", "setting", "architecture", "predicted",
                                   "measured", "sd", "resolvable", "boundary")}
                for r in misses],
        rows=rows,
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}/report.json")


if __name__ == "__main__":
    main()
