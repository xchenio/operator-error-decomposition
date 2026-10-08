# Darcy beta ladder: searching for the threshold boundary on a steady operator

Date: 2026-10-08. Committed **before** any run on these files.

## 1. What was being looked for

`REMAINING_GAPS_V1.md` identified the untested item: the threshold rule has three
out-of-sample confirmations, but only the viscosity ladder probes its boundary, and Darcy at
`beta = 1.0` sits far above every threshold. The hypothesis was that `beta`, which controls
the roughness of the coefficient field, would move the target fluctuation energy down far
enough to cross a threshold on a **steady, non-evolution** operator.

## 2. The hypothesis fails, found from a-priori statistics before training

Computed from training targets only:

| beta | fluctuation energy | modes for 99% | target std |
|---:|---:|---:|---:|
| 0.01 | **0.5638** | 5 | 0.0307 |
| 1.0 (earlier) | 0.4964 | 11 | 0.1371 |
| 10.0 | **0.4918** | 12 | 1.1850 |

The fluctuation energy barely moves across a 1000-fold range of `beta`, and it is *higher* at
the smoothest setting, not lower. Thresholds are 0.2477 (MLP DeepONet), 0.2728 (Fourier) and
0.1502 (FNO).

**Conclusion stated before the runs: this family does not contain a boundary setting.** Every
Darcy variant available lies far above every threshold, so:

* the prediction is **positive skill for all three architectures at both betas**;
* the runs will **not** test the rule's boundary, and must not be presented as if they do;
* the boundary on a steady operator remains **untestable with the data in this deposit**.

This is a negative result about the data, recorded rather than worked around by refitting.

## 3. Why run them at all

Because a 1000-fold change in coefficient roughness that leaves the prediction unchanged is
itself informative, and because it tests the rule across the whole Darcy family rather than at
one point. The expectation is low: this is a weak confirmation, like diffusion-sorption.

## 4. Protocol

Identical to `DARCY_ELLIPTIC_V1.md`: stride 2 to 64x64, split seed 20260922 with sizes
1000/200/200, Adam 1e-3, batch 32, **1500 updates**, seeds 0-4, three architectures. Thresholds
unchanged and not refitted.

## 5. Falsification

Any architecture at or below zero falsifies the rule for that architecture. Note that because
both settings are far above the thresholds, a *confirmation* here adds little; a falsification
would be decisive.

## 6. Commands

```bash
python scripts/prepare_darcy.py --source external_data/pdebench/darus/2D_DarcyFlow_beta0.01_Train.hdf5 \
    --out external_data/pdebench/darcy_beta0.01
python scripts/prepare_darcy.py --source external_data/pdebench/darus/2D_DarcyFlow_beta10.0_Train.hdf5 \
    --out external_data/pdebench/darcy_beta10.0
python scripts/run_darcy_benchmark.py --data-dir external_data/pdebench/darcy_beta0.01 \
    --steps 1500 --out results/darcy_beta001_v1 --device cuda
python scripts/run_darcy_benchmark.py --data-dir external_data/pdebench/darcy_beta10.0 \
    --steps 1500 --out results/darcy_beta10_v1 --device cuda
```

## 7. Result (measured after the protocol was committed)

### beta = 10.0: prediction confirmed

| Model | test relL2 | **R^2 per sample** | predicted |
|---|---:|---:|---|
| MLP DeepONet | 0.23579 | **+0.7780 ± 0.0052** | + |
| Fourier DeepONet | 0.13150 | **+0.9387 ± 0.0046** | + |
| FNO | 0.06309 | **+0.9874 ± 0.0007** | + |

3 of 3. Validation curves converge monotonically (for example Fourier seed 0:
0.170 -> 0.112 over the run). This is a weak, non-boundary confirmation, as predeclared.

### beta = 0.01: the prediction could not be evaluated, and not because it was false

Every architecture failed outright: relL2 2.83 (MLP), 3.80 (Fourier), 0.70 (FNO), with
R^2 per sample of -28.7, -86.8 and -1.86. Two independent reasons were checked before
concluding anything.

**Training did not converge.** Validation curves are noisy and non-monotone, and some seeds
end worse than they started (Fourier seed 1: 4.130, 3.753, 4.391, 2.574, 3.286, **4.866**).
Compare beta = 10.0, which converges cleanly. So this is not a converged-but-poor result.

**The per-sample metric is ill-conditioned here.** The decomposition divides by each
sample's own norm, so target norms spanning orders of magnitude make the per-sample
quantities uninterpretable. Measured on the test split:

| family | min \|\|y\|\| | max / min | CV of \|\|y\|\| |
|---|---:|---:|---:|
| **darcy beta 0.01** | 3.4e-02 | **343** | **2.43** |
| darcy beta 1.0 | 3.4e+00 | 10.7 | 0.54 |
| darcy beta 10.0 | 3.4e+01 | 7.9 | 0.47 |
| Burgers Nu0.001 | 5.4e-01 | 59.5 | 0.65 |
| Burgers Nu0.01 | 3.5e-01 | 90.6 | 0.66 |
| advection | 2.3e+00 | 14.2 | 0.45 |
| diffusion-sorption | 4.8e+00 | 1.3 | 0.06 |

Every family reported in the manuscript has a coefficient of variation at most 0.66. beta
0.01 has 2.43 and a 343-fold spread, so its per-sample R^2 is dominated by a handful of
near-zero-norm samples and carries no interpretation.

**Therefore beta = 0.01 is excluded, and it is neither a confirmation nor a falsification of
the rule.** Reporting it as a falsification would be wrong: the number is an artifact of an
inapplicable metric plus a non-converged run, not a measurement of skill.

### Scope condition this establishes

The per-sample measures require target norms of comparable scale. Concretely, all reported
families have CV <= 0.66 and a spread below 100-fold; beta 0.01 has CV 2.43 and is excluded.
This is now a stated scope condition of the decomposition rather than an implicit assumption.

### Net effect on the original gap

The gap was the threshold boundary on a steady, non-evolution operator. Both new beta
variants sit far above the thresholds, one of them is unusable, and no Darcy setting in this
deposit approaches a threshold. **The boundary on a steady operator is untestable with the
available data**, and this is reported as a data limitation instead of being manufactured by
refitting or by redefining the metric.
