> Compatibility filename retained from an earlier venue-specific draft. This file does not
> assert submission, anonymity compliance, a verified 2026 call, or four-page length.

# Conserved Means and Lost Fluctuations: A Mean/Fluctuation Diagnosis of Operator Learning Across Six PDEBench Operator Families

**Short research draft — revised 2026-10-28. Not submitted; no venue acceptance or page-limit
compliance implied.**

## Abstract

On periodic Burgers the spatial mean is conserved, so a model can look competitive on
relative L2 while predicting nothing beyond a conserved scalar. We separate prediction error
into a conserved-mean part and a fluctuation part, and measure fluctuation skill against a
zero-fluctuation reference with two sample weightings. On PDEBench 1D Burgers (Nu0.001),
t=0 to t=1.0, predicting the exact input mean and zero fluctuation already gives 0.37577 test
mean relative L2. Against that reference, per-sample fluctuation R^2 is -0.0042 ± 0.0072
(MLP DeepONet), -0.0934 ± 0.0886 (Fourier DeepONet) and +0.6503 ± 0.0222 (FNO) on identical
data, split and a 1500-update budget.

The failure is a matter of degree, not kind. Over seven forecast horizons every architecture
loses fluctuation skill, and FNO also falls below zero at t=2.00. What differs is where each
crosses zero, giving an interpolated **skill horizon** of about 0.86 (Fourier DeepONet), 0.98
(MLP DeepONet) and 1.94 (FNO). An exact upper bound depending only on the trunk span shows the
constraint is the branch, not the basis. A fluctuation-energy threshold fitted on two scalar
families predicts the sign of skill on four held-out families — 52 of 55 preregistered
predictions, thresholds never refitted — but fails on one compressible system, which is
reported as a boundary rather than defended. Stratified, 25 of the 55 are on settings where the
architecture is essentially perfect; the rule is 27 of 30 on the remaining discriminating
predictions against 16 of 30 for an always-positive baseline. A physics-informed arm reaches at
most 0.38 of supervised fluctuation skill. We claim a diagnostic decomposition and a reproducible
negative result, not a new architecture or optimizer.

## 1. Scope and prior work

The architectures are DeepONet [10] and the Fourier neural operator [11]; neither is ours.
Lee and Shin introduce two-step training with orthonormalization [1]. Williams et al. examine
learned PI-DeepONet bases [2]; our horizon results qualify that line, since the Fourier trunk
helps at t=0.25 and stops helping at t=1.00. Lu et al. compare DeepONet and FNO including
POD-DeepONet [3]. Khodakarami et al. report that spectral bias in operator learning is
architecture dependent [12]. This study is supervised throughout except for one
physics-informed arm.

## 2. Decomposition

Because the constant vector and the zero-mean residual are orthogonal, each sample's squared
error splits exactly into a mean share and a fluctuation share (`squared_error_closure` is
asserted to be 0 by test). Fluctuation skill is measured against a zero-fluctuation
reference and is invariant to the mean, so the exact-mean and all-zero predictors both score
0. Two weightings are reported: an energy-weighted R^2 dominated by high-energy samples and a
per-sample R^2 matching the manuscript metric. They can disagree in sign; the per-sample one
decides the claims. At 20000 updates MLP DeepONet reads +0.2126 energy-weighted but -0.2470
per sample, so reporting only the former would be a false positive.

Conservation is verified on the data: corr(mean(x), mean(y)) = 1.000000 on all splits.

## 3. Findings

| Method | Test relL2 | R^2 (energy) | R^2 (per sample) |
|---|---:|---:|---:|
| oracle input mean | 0.37577 | 0.0000 | 0.0000 |
| MLP DeepONet | 0.39357 | -0.0056 ± 0.0094 | -0.0042 ± 0.0072 |
| Fourier DeepONet | 0.41231 | -0.0124 ± 0.0356 | -0.0934 ± 0.0886 |
| FNO | 0.22835 | +0.6282 ± 0.0187 | +0.6503 ± 0.0222 |

**Not undertraining.** Across 1500/5000/20000/50000 updates the total relative L2 of MLP
DeepONet never improves (0.39357/0.39729/0.39311/0.42075) while per-sample skill degrades
monotonically (-0.0042/-0.0640/-0.2470/-0.4157). FNO stays at +0.65 to +0.68.

**Skill horizon.** FNO's skill horizon is about 2.0x MLP DeepONet's. Parameter counts do not
explain the gap (131,969 versus 136,481), and neither does a wall-clock-matched control
(DeepONet at 3500 updates, 14 s, scores -0.0057 against FNO's +0.6503 at 1500 updates, 14 s).

**Six settings.** Linear advection, Burgers at two viscosities, diffusion-sorption, steady 2D
Darcy, a 1D three-field compressible system and a 2D two-field system. A fluctuation-energy
threshold fitted on two scalar families predicts the sign of skill on four held-out families:
52 of 55 preregistered predictions correct. Stratified: 25 of the 55 are on settings where the
architecture scores R^2 >= 0.9 and the sign is not in doubt; the rule is 27 of 30 on the
remaining discriminating predictions, against 16 of 30 for always predicting positive. It fails
on 1D compressible CFD (3 of 6). Two of those three misses are inside |R^2| < 0.03; the
substantial one, Vx, is the only setting in the study where the mean share dominates the error
and the spatial mean is not conserved.

**Where the constraint is.** An exact upper bound from the trunk span shows the Fourier
trunk's bound is +0.96 to +1.00 on both PDEs at every horizon, including t=2.00 where the
achieved value is -0.3774, so the branch is the binding constraint.

## 4. Limits and conclusion

All six settings come from PDEBench [4]: paper 2022, data deposit `doi:10.18419/darus-2986`
version 8 released 2024-02-13, files downloaded 2026-09 to 2026-10 and byte-pinned. The
claim is a diagnostic decomposition, not state-of-the-art accuracy. Compute is not
FLOP-matched and the FFT portion of the FLOP estimate is analytic. The test split was exposed
during exploratory development. Five seeds share one split, so seed deviations do not measure
task variability. The physics-informed arm covers two operators and three horizons.

Within these limits, the dissociation reported in earlier drafts of this project now has a
measurement and a boundary: the error of both DeepONet variants is dominated by a component
they do not predict, namely the deviation from the conserved spatial mean; the deficit is
horizon-dependent and shared by FNO; and the binding constraint is the branch rather than the
trunk basis.

See [the current full manuscript](main.md) for complete methods, references, tables,
provenance and withdrawals, and
[the fluctuation protocol](../docs/FLUCTUATION_DIAGNOSIS_V1.md) for per-family artifacts.
