> Compatibility filename retained from an earlier venue-specific draft. This file does not assert submission, anonymity compliance, a verified 2026 call, or four-page length.

# DeepONet Learns the Conserved Mean and Loses the Fluctuation

**Short research draft — revised 2026-09-29. Not submitted; no venue acceptance or page-limit compliance implied.**

## Abstract

On periodic Burgers the spatial mean is conserved, so a model can look competitive on
relative L2 while predicting nothing beyond a conserved scalar. We separate prediction
error into a conserved-mean part and a fluctuation part. On PDEBench 1D Burgers (Nu0.001),
t=0 to t=1.0, predicting the exact input mean and zero fluctuation already gives 0.37577
test mean relative L2. Against that zero-fluctuation reference, MLP DeepONet scores
-0.0042 +- 0.0072 and Fourier DeepONet -0.0934 +- 0.0886 in per-sample fluctuation R^2,
while FNO scores +0.6503 +- 0.0222 on identical data, split and a 1500-update budget.
Across 175 reconstructed runs no DeepONet configuration scores above zero. Raising the
budget to 20000 updates does not help: per-sample R^2 moves -0.0042 -> -0.0640 -> -0.2470
and total relative L2 is flat (0.39357 / 0.39729 / 0.39311). The deficit is
horizon-dependent: DeepONet scores +0.1695 at t=0.25 and +0.1238 at t=0.50 before decaying
to -0.0042 at t=1.00, while FNO falls only from +0.9474 to +0.6503. We claim a diagnostic
decomposition and a reproducible negative result, not a new architecture or optimizer.

## 1. Scope and prior work

Two-step DeepONet training and orthonormalization already exist (Lee and Shin,
https://arxiv.org/abs/2309.01020). Learned PI-DeepONet basis diagnostics are studied by
Williams et al. (https://arxiv.org/abs/2411.18459); our horizon results qualify that line,
since the Fourier trunk helps at t=0.25 and stops helping at t=1.00. Lu et al. compare
DeepONet and FNO with POD-DeepONet extensions (https://arxiv.org/abs/2111.05512). This
study is supervised, not physics-informed.

## 2. Decomposition

Because the constant vector and the zero-mean residual are orthogonal, each sample's
squared error splits exactly into a mean share and a fluctuation share. Both are reported.
Fluctuation skill is measured against a zero-fluctuation reference and is invariant to the
mean, so the exact-mean predictor and the all-zero predictor both score 0. Two weightings
are reported: an energy-weighted R^2 dominated by high-energy samples, and a per-sample R^2
matching the manuscript's per-sample metric. They can disagree in sign; the per-sample one
decides the claims.

Conservation is verified on the data: corr(mean(x), mean(y)) = 1.000000 on all three
splits, with absolute mean differences of 2.8e-05 to 3.2e-05 against mean magnitudes near
0.5.

## 3. Findings

| Method | Test relL2 | R^2 (energy) | R^2 (per sample) |
|---|---:|---:|---:|
| oracle input mean | 0.37577 | 0.0000 | 0.0000 |
| MLP DeepONet | 0.39357 | -0.0056 +- 0.0094 | -0.0042 +- 0.0072 |
| Fourier DeepONet | 0.41231 | -0.0124 +- 0.0356 | -0.0934 +- 0.0886 |
| FNO | 0.22835 | +0.6282 +- 0.0187 | +0.6503 +- 0.0222 |

Across the original four families plus 5k and 20k budgets, every DeepONet arm scores at or
below zero; conditioning, whitening, mean lifting, zero initialization, schedule and batch
size all fail to move it. More updates make the per-sample measure worse while total
relative L2 stays flat, so the near-zero skill is not undertraining.

At matched 1500 updates, per-sample R^2 by forecast horizon is +0.1695 / +0.1238 / -0.0042
for MLP DeepONet, +0.4248 / +0.2510 / -0.0934 for Fourier DeepONet and +0.9474 / +0.8369 /
+0.6503 for FNO at t = 0.25 / 0.50 / 1.00. Parameter counts do not explain the gap
(Fourier DeepONet 131,969 versus FNO 136,481).

## 4. Limits and conclusion

One PDE, one viscosity, one split and one solver; the horizon ladder is a sensitivity check
on that PDE, not a second PDE. Compute is not matched between FNO and DeepONet, and the
test split was exposed during exploratory development. Five seeds share one split, so seed
standard deviations do not measure task variability.

Within these limits, the dissociation reported in earlier drafts of this project now has a
measurement: the error of both DeepONet variants is dominated by a component they do not
predict, namely the deviation from the conserved spatial mean, and the component they do
predict improves with fitting while the fluctuation does not.

See [the current full manuscript](main.md) for complete methods, references, tables,
provenance and withdrawals, and
[the fluctuation protocol](../docs/FLUCTUATION_DIAGNOSIS_V1.md) for per-family artifacts.
