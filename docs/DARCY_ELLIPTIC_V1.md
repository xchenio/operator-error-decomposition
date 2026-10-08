# Steady elliptic operator: 2D Darcy — predeclared protocol

Date: 2026-09-30. Committed **before** any model is trained on this data.

## 1. Why this operator

The three families used so far are all 1D scalar **evolution** problems: an initial field is
mapped to a later field. That leaves open whether anything in this study is specific to
evolution problems. PDEBench 2D Darcy is different in three independent ways:

* **steady** — there is no time axis, so there is no forecast horizon;
* **elliptic** — coefficient field to solution field, not state to later state;
* **2D** — a 64x64 grid rather than a 256-point line.

## 2. Data and task

`2D_DarcyFlow_beta1.0_Train.hdf5`, MD5 `81694ed31306ff2e5f6b76349b0b4389`, verified.
`nu (10000,128,128)` is the coefficient, `tensor (10000,1,128,128)` the solution.
Subsampled with stride 2 to 64x64 = 4096 points. Same split seed 20260922 and sizes
1000/200/200, so trajectory ids match every other family.

## 3. What changes in the analysis, stated in advance

**There is no conservation law here.** On periodic Burgers and on linear advection the
spatial mean is conserved, which is what made the input-spatial-mean predictor a
*conservation-law* predictor and gave the zero reference a physical meaning. For Darcy no
such law links the input mean to the target mean, so:

* the input-spatial-mean predictor is **not** a conservation-law predictor and is reported
  only as a trivial reference;
* the zero-fluctuation reference itself is unaffected: per-sample fluctuation R^2 = 0 still
  means "no better than predicting each target's own mean", which is well defined for any
  field.

The exact orthogonal split of squared error (mean share plus fluctuation share) also still
holds, since it is an identity about orthogonality and not about physics.

## 4. Models

Two new pieces, both added as **new files** so that no existing run fingerprint changes:
`src/fno2d.py` (a 2D FNO) and `scripts/run_darcy_benchmark.py`. `src/fno.py` is hashed by
`source_hash()` in every existing run identity and was deliberately not edited.

DeepONet: branch_in 4096, trunk_in 2, width 64, three layers of width 128 for both nets,
MLP or fixed random Fourier trunk. FNO2d: width 32, 12 modes, 4 layers. Same budget as
every other family: Adam 1e-3, batch 32, **1500 updates**, seeds 0-4, three architectures.

## 5. A-priori statistic and the predeclared prediction

Computed from the training targets only, before any model exists:

| quantity | value |
|---|---:|
| fluctuation energy fraction | **0.4993** |
| POD modes for 99% energy | 11 |
| input spatial std | 0.4500 |
| target spatial std | 0.1371 |

Thresholds, unchanged from the fit on Burgers Nu0.001 plus advection: 0.2477 (MLP DeepONet),
0.2728 (Fourier DeepONet), 0.1502 (FNO). Since 0.4993 exceeds all three:

**Predicted: positive per-sample fluctuation skill for all three architectures.**

## 6. Falsification criteria, fixed in advance

1. Any architecture scoring at or below zero falsifies the rule for that architecture on
   this operator class.
2. The thresholds are applied unchanged and will **not** be refitted on this data. A failure
   is reported as a failure.

## 7. Known limitation, stated in advance

As with diffusion-sorption, this prediction is **far from the thresholds**, so it is a
non-boundary test: it can falsify the rule but a confirmation is weak. The boundary was
tested by the viscosity ladder (`VISCOSITY_NU001_V1.md`). There is also no horizon here, so
the skill-horizon part of the study is not applicable and nothing about it is claimed.

## 8. Commands

```bash
python scripts/prepare_darcy.py
python scripts/run_darcy_benchmark.py --steps 1500 --device cuda
python scripts/diagnose_fluctuation.py \
    --data external_data/pdebench/darcy --families darcy_v1=endtoend:test,val \
    --out results/fluctuation_darcy_v1 --device cuda
```

## 9. Result (measured after the protocol was committed)

Per-sample fluctuation R^2 on the test split, 5 seeds, 1500 updates:

| Model | test relL2 | **R^2 per sample** | R^2 energy-weighted | predicted |
|---|---:|---:|---:|---|
| MLP DeepONet | 0.26353 | **+0.7614 ± 0.0160** | +0.8081 | + |
| Fourier DeepONet | 0.16450 | **+0.9153 ± 0.0079** | +0.9333 | + |
| FNO | 0.08330 | **+0.9790 ± 0.0009** | +0.9731 | + |

**3 of 3 predictions correct.** The threshold rule, fitted on 1D evolution problems and
never refitted, predicts the sign of fluctuation skill on a steady elliptic 2D operator.

The measured target fluctuation energy is 0.4964, close to the a-priori 0.4993 (the small
gap is the train/test split).

### The predeclared caveat is confirmed by the data

The input-output mean correlation is **-0.873** on the test split (and -0.872 / -0.885 on
train / validation), so there is no conservation law, exactly as stated in advance. The
input-spatial-mean predictor consequently gives a test relative L2 of **3.6843** — it is
worse than useless here — whereas on Burgers the same construction was a genuine conservation-
law predictor at 0.37577. This is the clearest demonstration in the study that the mean
baseline's meaning comes from the physics of the operator and not from the metric.

Ordering is consistent with the other families: FNO > Fourier trunk > MLP trunk.

### Notes

* Recomputation reproduces stored metrics to 1.8e-08 here rather than exactly, because the
  2D grid is stored as float32 coordinate pairs; it remains far below the faithfulness
  tolerance and no run was flagged.
* As predeclared, this is a non-boundary test: it can falsify the rule and did not, so it is
  weak support. The boundary was tested by the viscosity ladder. Nothing about a skill
  horizon is claimed here because a steady problem has no horizon.
