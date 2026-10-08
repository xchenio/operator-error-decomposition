# Conserved Means and Lost Fluctuations: A Mean/Fluctuation Diagnosis of Operator Learning Across Six PDEBench Settings

**Short research draft — revised 2026-10-08. Not submitted; no venue acceptance or page-limit
compliance implied. Superseded in scope by [main.md](main.md); kept as a one-page summary.**

> This file previously carried the title *Better Projection and Stable Fitting Do Not
> Guarantee Generalization in DeepONet* and the 2026-09-24 evidence base (155 runs, one PDE,
> fitting-side interventions). It was rewritten on 2026-10-08 to match the current
> manuscript. The old text is archived at
> `archive/pre_fluctuation_20260929/` for reference.

## Abstract

On periodic Burgers the spatial mean is conserved, so a model can look competitive on
relative L2 while predicting nothing beyond a conserved scalar. Separating prediction error
into a conserved-mean part and a fluctuation part shows that on PDEBench 1D Burgers (Nu0.001),
t=0 to t=1.0, the exact input mean with zero fluctuation already gives 0.37577 test mean
relative L2. Against that zero-fluctuation reference, per-sample fluctuation R^2 is
-0.0042 ± 0.0072 (MLP DeepONet), -0.0934 ± 0.0886 (Fourier DeepONet) and +0.6503 ± 0.0222
(FNO) on identical data, split and budget.

The deficit is horizon-dependent and shared: over seven horizons every architecture loses
fluctuation skill and FNO also falls below zero at t=2.00, giving interpolated skill horizons
of about 0.86, 0.98 and 1.94. Across six PDEBench settings a fluctuation-energy threshold
fitted on two scalar families predicts the sign of skill on four held-out families — 52 of 55
preregistered predictions, thresholds never refitted — and fails on one compressible system.
An exact trunk-span bound shows the constraint is the branch, not the basis. A
physics-informed arm reaches at most 0.38 of supervised fluctuation skill.

## 1. What is measured

The constant vector and the zero-mean residual are orthogonal, so each sample's squared error
splits exactly into a mean share and a fluctuation share (closure asserted to be 0 by test).
Fluctuation skill is measured against a zero-fluctuation reference and is invariant to the
mean. Two sample weightings are reported because they can disagree in sign: at 20000 updates
MLP DeepONet reads +0.2126 energy-weighted but -0.2470 per sample.

Conservation is verified on the data: corr(mean(x), mean(y)) = 1.000000 on all splits, so the
input-mean baseline is a **conservation-law predictor** rather than a strawman.

## 2. Main numbers

| Method | Test relL2 | R^2 (energy) | R^2 (per sample) |
|---|---:|---:|---:|
| oracle input mean | 0.37577 | 0.0000 | 0.0000 |
| MLP DeepONet | 0.39357 | -0.0056 ± 0.0094 | -0.0042 ± 0.0072 |
| Fourier DeepONet | 0.41231 | -0.0124 ± 0.0356 | -0.0934 ± 0.0886 |
| FNO | 0.22835 | +0.6282 ± 0.0187 | +0.6503 ± 0.0222 |

- **Budget is not the cause.** 1500/5000/20000/50000 updates: total relL2 0.39357/0.39729/
  0.39311/0.42075, per-sample skill -0.0042/-0.0640/-0.2470/-0.4157.
- **Capacity and compute are not the cause.** Parameters within 18%; FNO costs about 2.2x
  more wall-clock per update; a wall-clock-matched DeepONet (3500 updates, 14 s) scores
  -0.0057 against FNO's +0.6503 (1500 updates, 14 s).
- **The branch is the constraint.** The Fourier trunk's span supports +0.96 to +1.00
  everywhere, including where the achieved value is -0.3774.

## 3. Limits

All six settings come from PDEBench: paper 2022, data deposit `doi:10.18419/darus-2986`
version 8 released 2024-02-13, files downloaded 2026-09 to 2026-10 and byte-pinned. The
claim is a diagnostic decomposition, not state-of-the-art accuracy. Compute is not FLOP-matched and
the FFT portion of the estimate is analytic. The test split was exposed during exploratory
development, so these numbers are descriptive. Five seeds share one split. The
physics-informed arm covers two operators and three horizons. The threshold rule is valid for
the scalar families and the 2D system tested; it fails on 1D compressible CFD, and why is not
established.

## 4. Claim boundary

We claim a diagnostic decomposition with an exact squared-error split, a reproducible
negative result across six settings, and a quantitative skill horizon. We do **not** claim
that DeepONet cannot learn fluctuations in general (it scores +0.22 to +0.64 at t=0.10), a
causal mechanism, FNO's universal superiority (it is below zero at t=2.00), or transfer to
foundation-model-scale operator training.

See [the current full manuscript](main.md) for methods, references, provenance and
withdrawals, and [the 4-page draft](ml4ps2026_4page.md) for a slightly longer version.
