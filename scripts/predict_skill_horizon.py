"""A-priori predictors of fluctuation skill, computed from training targets only.

Motivation
----------
The skill horizon (section 5.5) is currently measured *after* training. If a quantity
computed from the training arrays alone predicts it, the diagnostic becomes prospective
instead of retrospective, which is the step from "a case study" to "a diagnostic method".

Every statistic below is computed from the **training split only**, before any model exists,
and never touches test targets.

Statistics
----------
- `fluctuation_energy_fraction`: ||Y_c|| / ||Y||.
- `target_modes_99`: POD modes of the target fluctuation needed for 99% energy.
- `target_effective_rank`: exponential of the spectral entropy of the target fluctuation.
- `spectral_decay_exponent`: slope of log singular values over the leading decade;
  steeper means energy concentrated in fewer modes.
- `input_target_subspace_distance`: `1 - captured`, where `captured` is the energy-weighted
  fraction of the target fluctuation subspace captured by the leading input fluctuation
  modes. For pure transport the ensemble subspace should barely move, so this should stay
  near zero; for nonlinear steepening it should grow. This is the main hypothesis.

Usage
-----
    python scripts/predict_skill_horizon.py --out results/skill_prediction_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.public_benchmark import file_hash  # noqa: E402  reused, not modified

# settings: (label, pde, time tag, data dir, diagnostic json with measured skill)
# the original Burgers cache for t=1.0 is named fixed_t1_v1, not fixed_t100_v1
BURGERS_DATA = {"010": "fixed_t010_v1", "025": "fixed_t025_v1", "050": "fixed_t050_v1",
                "075": "fixed_t075_v1", "100": "fixed_t1_v1",
                "150": "fixed_t150_v1", "200": "fixed_t200_v1"}
BURGERS = [("burgers", t, f"external_data/pdebench/{BURGERS_DATA[t]}",
            (f"results/fluctuation_v1/public_fixed_t1_v1.json" if t == "100"
             else f"results/fluctuation_horizon_v1/t{t}/public_horizon_t{t}_v1.json"))
           for t in ("010", "025", "050", "075", "100", "150", "200")]
ADVECTION = [("advection", t, f"external_data/pdebench/advection/fixed_t{t}_v1",
              f"results/fluctuation_advection_v1/t{t}/advection_horizon_t{t}_v1.json")
             for t in ("010", "025", "050", "075", "100", "150", "200")]

ALL = BURGERS + ADVECTION
TARGET_TIME = {"010": 0.10, "025": 0.25, "050": 0.50, "075": 0.75,
               "100": 1.00, "150": 1.50, "200": 2.00}
MODES = 64  # fixed subspace dimension for the input/target overlap


def _modes_for_energy(singular_values: np.ndarray, fraction: float) -> int:
    energy = np.cumsum(singular_values ** 2)
    total = energy[-1]
    return int(np.searchsorted(energy, fraction * total) + 1)


def _effective_rank(singular_values: np.ndarray) -> float:
    p = singular_values ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def _decay_exponent(singular_values: np.ndarray) -> float:
    """Slope of log sigma over its leading decade of magnitude, or zero if degenerate."""
    s = singular_values[singular_values > 0]
    if len(s) < 4:
        return 0.0
    tail = s[len(s) // 2]
    keep = s > tail
    if keep.sum() < 3:
        return 0.0
    y = np.log(s[keep])
    x = np.arange(keep.sum(), dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def a_priori(data_dir: Path) -> dict:
    protocol = json.loads((data_dir / "protocol.json").read_text(encoding="utf-8"))
    if file_hash(data_dir / "pairs.npz") != protocol["cache_sha256"]:
        raise ValueError(f"pair cache changed: {data_dir}")
    with np.load(data_dir / "pairs.npz") as z:
        X = z["train_x"].astype(np.float64)
        Y = z["train_y"].astype(np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)

    sx = np.linalg.svd(Xc, full_matrices=False)[1]
    Uy, sy, Vy = np.linalg.svd(Yc, full_matrices=False)
    Vy = Vy.T  # columns are modes

    k = min(MODES, Xc.shape[1], Vy.shape[1])
    # energy-weighted captured fraction of the target subspace by the leading input modes
    B = np.linalg.svd(Xc, full_matrices=False)[2][:k].T  # (N, k) orthonormal input modes
    proj = B.T @ Vy                       # (k, N)
    per_mode = np.sum(proj ** 2, axis=0)  # squared norm of each target mode's projection
    weights = sy ** 2
    captured = float(np.sum(weights * per_mode) / np.sum(weights))

    return dict(
        cache_sha256=protocol["cache_sha256"],
        fluctuation_energy_fraction=float(np.linalg.norm(Yc) / np.linalg.norm(Y)),
        target_modes_99=_modes_for_energy(sy, 0.99),
        target_modes_999=_modes_for_energy(sy, 0.999),
        target_effective_rank=_effective_rank(sy),
        spectral_decay_exponent=_decay_exponent(sy),
        input_target_subspace_distance=float(1.0 - captured),
        subspace_modes_used=int(k),
    )


def measured(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for run in payload["runs"]:
        key = run["configuration"]["architecture"]
        out.setdefault(key, []).append(run["splits"]["test"]["fluctuation_r2_per_sample"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if len(a) < 2 or np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/skill_prediction_v1")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pde, tag, data_dir, diag in ALL:
        data_path, diag_path = ROOT / data_dir, ROOT / diag
        if not data_path.exists() or not diag_path.exists():
            print(f"[skip] missing {pde} t={tag}", flush=True)
            continue
        row = dict(pde=pde, time_tag=tag, target_time=TARGET_TIME[tag],
                   **a_priori(data_path), **measured(diag_path))
        rows.append(row)
        print(f"{pde:9s} t={row['target_time']:.2f} "
              f"fluctE={row['fluctuation_energy_fraction']:.4f} "
              f"modes99={row['target_modes_99']:3d} "
              f"effrank={row['target_effective_rank']:6.2f} "
              f"subspace_dist={row['input_target_subspace_distance']:.4f} "
              f"| R2ps mlp={row['deeponet_mlp']:+.4f} fourier={row['deeponet_fourier']:+.4f} "
              f"fno={row['fno']:+.4f}", flush=True)

    stats = ["fluctuation_energy_fraction", "target_modes_99", "target_effective_rank",
             "spectral_decay_exponent", "input_target_subspace_distance"]
    report = {"rows": rows, "spearman": {}}
    for arch in ("deeponet_mlp", "deeponet_fourier", "fno"):
        skill = np.array([r[arch] for r in rows], dtype=np.float64)
        report["spearman"][arch] = {s: spearman(np.array([r[s] for r in rows]), skill)
                                    for s in stats}
    # pooled over both PDEs, and within Burgers only, where the decay actually happens
    report["spearman_burgers_only"] = {}
    burg = [r for r in rows if r["pde"] == "burgers"]
    for arch in ("deeponet_mlp", "deeponet_fourier", "fno"):
        skill = np.array([r[arch] for r in burg], dtype=np.float64)
        report["spearman_burgers_only"][arch] = {
            s: spearman(np.array([r[s] for r in burg]), skill) for s in stats}

    # Critical fluctuation-energy threshold per architecture: interpolate the value of
    # fluctuation_energy_fraction at which measured skill crosses zero. Purely descriptive;
    # with two PDEs it is a hypothesis to be tested on a third, not an established law.
    report["critical_fluctuation_energy"] = {}
    for arch in ("deeponet_mlp", "deeponet_fourier", "fno"):
        ordered = sorted(rows, key=lambda r: -r["fluctuation_energy_fraction"])
        threshold = None
        for a, b in zip(ordered, ordered[1:]):
            if a[arch] > 0 >= b[arch]:
                w = a[arch] / (a[arch] - b[arch])
                threshold = float(a["fluctuation_energy_fraction"] * (1 - w)
                                  + b["fluctuation_energy_fraction"] * w)
                break
        report["critical_fluctuation_energy"][arch] = threshold
        if threshold is not None:
            # check the rule against every setting of both PDEs
            verdicts = []
            for r in rows:
                predicted = r["fluctuation_energy_fraction"] >= threshold
                actual = r[arch] > 0
                verdicts.append(predicted == actual)
            report["critical_fluctuation_energy"][arch + "_rule_accuracy"] = (
                f"{sum(verdicts)}/{len(verdicts)}")
            report["critical_fluctuation_energy"][arch + "_rule_failures"] = [
                f"{r['pde']} t={r['target_time']:.2f}" for r, ok in zip(rows, verdicts) if not ok]

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nCritical fluctuation-energy threshold (interpolated zero crossing):")
    for arch in ("deeponet_mlp", "deeponet_fourier", "fno"):
        info = report["critical_fluctuation_energy"]
        print(f"  {arch}: {info[arch]}  rule accuracy {info.get(arch + '_rule_accuracy')} "
              f"failures={info.get(arch + '_rule_failures')}")
    print("\nSpearman correlation of a-priori statistic with measured per-sample R^2 "
          "(pooled over both PDEs, n=%d):" % len(rows))
    for arch in ("deeponet_mlp", "deeponet_fourier", "fno"):
        print(" ", arch, {k: round(v, 3) for k, v in report["spearman"][arch].items()})
    print("Burgers only, n=%d:" % len(burg))
    for arch in ("deeponet_mlp", "deeponet_fourier", "fno"):
        print(" ", arch, {k: round(v, 3)
                          for k, v in report["spearman_burgers_only"][arch].items()})


if __name__ == "__main__":
    main()
