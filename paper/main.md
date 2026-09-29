# DeepONet Learns the Conserved Mean and Loses the Fluctuation: A Mean/Fluctuation Diagnosis on PDEBench Burgers

**Evidence-audited research draft v3 — 2026-09-29. Not a submitted manuscript.**

**Latest 2026-09-29 follow-up:** [mean/fluctuation decomposition](../docs/FLUCTUATION_DIAGNOSIS_V1.md).
The target spatial mean is a conserved quantity, so the previously reported "input-mean
baseline" is a conservation-law predictor rather than a strawman. Measuring fluctuation
skill separately shows that every DeepONet configuration in this repository scores at or
below the zero-fluctuation reference at forecast time 1.0, at every update budget tested,
while FNO explains about 65% of the per-sample fluctuation variance. The deficit is
horizon-dependent, not absolute: DeepONet has positive fluctuation skill at t=0.25 and
t=0.50 that decays to zero by t=1.00.

**Earlier 2026-09-29:** the [modal-shrinkage line](../docs/UNCERTAINTY_MODAL_V1.md) was
closed after its confirmation failed 0 of 12 comparisons against strong controls; those
experiments are recorded as a negative result and are not part of the current argument.

**2026-09-24 audit:** see [the historical archive](archive/pre_audit_20260924/main.md)
and [the pre-rewrite version](archive/pre_fluctuation_20260929/main.md). The study is
supervised operator learning, not a physics-informed training result.

## Abstract

On periodic Burgers the spatial mean of the solution is conserved, so a model can appear
competitive on relative L2 while predicting nothing beyond a conserved scalar. We separate
prediction error into a conserved-mean part and a fluctuation part and measure skill on
each. On PDEBench 1D Burgers (Nu0.001), forecasting t=0 to t=1.0, the conserved mean
carries 97.4% of the target energy and the fluctuation carries 23.7% of the per-sample
squared norm; the exact input mean alone already achieves 0.37577 mean relative L2.

Against a zero-fluctuation reference, MLP DeepONet scores -0.0042 +- 0.0072 and Fourier
DeepONet -0.0934 +- 0.0886 in per-sample fluctuation R^2, while FNO scores +0.6503 +-
0.0222 on identical data, split and update budget. Across 175 reconstructed runs spanning
trunk architecture, whitening, mean lifting, zero initialization, learning-rate schedule,
batch size and training-subset size, no DeepONet configuration scores above zero. Raising
the budget from 1500 to 20000 updates does not help: per-sample fluctuation R^2 moves
-0.0042 -> -0.0640 -> -0.2470 for MLP DeepONet and total relative L2 is unchanged
(0.39357 / 0.39729 / 0.39311), whereas FNO stays at +0.65 to +0.68 and 0.222 to 0.228.
The deficit is horizon-dependent: DeepONet's per-sample fluctuation R^2 is +0.1695 at
t=0.25 and +0.1238 at t=0.50, decaying to -0.0042 at t=1.00, while FNO falls only from
+0.9474 to +0.6503. Parameter counts do not explain the gap (Fourier DeepONet 131,969
versus FNO 136,481).

We claim a diagnostic decomposition and a reproducible negative result, not a new
architecture or optimizer. One PDE, one split and one viscosity are studied; a
compute-matched FNO control and a second public PDE remain open.

## 1. Research question and contribution

Prior drafts of this study reported a dissociation: better offline basis quality and more
stable training fit did not produce better validation error. That statement is descriptive.
This version asks what the models actually predict, and answers it with a decomposition
that turns the dissociation into a measurement.

Contribution candidates:

1. A **mean/fluctuation decomposition** with a defined zero-fluctuation reference, an
   exact orthogonal split of squared error, and two skill measures that differ in how they
   weight samples. It identifies the widely used "input spatial-mean baseline" as a
   conservation-law predictor.
2. A **negative result with a mechanism**: on this protocol DeepONet reproduces a conserved
   quantity and has no per-sample fluctuation skill, and the skill it does have at short
   forecast horizons decays to zero as the horizon grows.
3. A **budget control**: the near-zero skill is not undertraining, and additional updates
   make the per-sample measure worse while leaving total relative L2 flat.
4. A **measurement warning**: an energy-weighted skill measure can report improvement where
   the per-sample measure reports degradation. Both are reported throughout.

The linear-algebra identities below are standard, not new approximation theory. Neither
orthonormalization, zero-residual initialization, cosine decay, nor POD truncation is
claimed as a new method.

## 2. Related work and overlap

Lee and Shin introduce two-step DeepONet training with trunk-first learning followed by
branch learning and orthonormalization [1]. Fixed-trunk training and basis
orthogonalization are therefore comparison points, not original contributions.

Williams et al. examine learned PI-DeepONet bases, singular-value/coefficient behavior and
transfer learning [2]. Our results qualify rather than contradict that line: trunk features
matter at short forecast horizons (the Fourier trunk scores +0.4248 against +0.1695 for the
MLP trunk at t=0.25) and stop mattering at t=1.00 (-0.0934 against -0.0042). Any claim
about trunk basis quality should state the forecast horizon.

Lu et al. compare DeepONet and FNO with practical extensions, including POD-DeepONet,
across multiple benchmarks [3]. Their result that architecture choice dominates on these
tasks is consistent with ours; our decomposition locates the difference in the fluctuation
component rather than in total error. Demo et al. learn a residual correction on a POD
basis [8], and Koike et al. study mode selection for sparse networks [9]; a modal-shrinkage
line in this repository was closed after failing its predeclared confirmation and is not
part of the current argument [see UNCERTAINTY_MODAL_V1.md](../docs/UNCERTAINTY_MODAL_V1.md).

PDEBench supplies the public data [4]. Zero-initialized residual mechanisms precede this
study, for example ReZero [5]; our zero final readout around an input-mean baseline is not
identical to ReZero's residual gates and no dynamical-isometry claim is made. Cosine
scheduling [6] and large-batch generalization studies [7] are established; we do not present
convergence-versus-generalization as a new general discovery, and no sharpness mechanism is
claimed.

## 3. Mean/fluctuation decomposition

Let `Y` and `P` in `R^{m x N}` be target and predicted fields for `m` trajectories on the
same `N`-point grid. Write the per-sample spatial mean as `bar(y)_j` and the fluctuation as
`Y_c = Y - bar(y) 1^T`, and similarly `P_c`.

### 3.1 An exact split of squared error

The constant vector and the zero-mean residual are orthogonal, so for every sample

\[
  \|p_j - y_j\|^2 = N\,(\bar p_j - \bar y_j)^2 + \|p_{c,j} - y_{c,j}\|^2 .
\]

Dividing by `||y_j||^2` and averaging gives an identity, not an attribution:

\[
  \underbrace{\mathbb{E}_j \frac{N(\bar p_j-\bar y_j)^2}{\|y_j\|^2}}_{\text{mean share}}
  +
  \underbrace{\mathbb{E}_j \frac{\|p_{c,j}-y_{c,j}\|^2}{\|y_j\|^2}}_{\text{fluctuation share}}
  = \mathbb{E}_j\,\mathrm{relL2}_j^2 .
\]

`scripts/diagnose_fluctuation.py` reports both shares and a closure term that the test
suite asserts to be zero to machine precision.

### 3.2 Two skill measures

| Measure | Definition | Dominated by |
|---|---|---|
| `fluctuation_r2` | `1 - sum_j \|\|p_c-y_c\|\|^2 / sum_j \|\|y_c\|\|^2` | high-energy samples |
| `fluctuation_r2_per_sample` | `1 - (fluctuation share) / (per-sample fluctuation energy)` | every sample equally |

Both are referenced to a **zero-fluctuation** predictor: any prediction whose fluctuation
is zero scores 0, including the exact-mean predictor and the all-zero predictor. `R^2` is
therefore invariant to the mean and measures fluctuation skill only; it is never a measure
of total accuracy. The manuscript's headline metric is a per-sample mean, so the per-sample
measure decides the claims, and the energy-weighted one is reported alongside because the
two can disagree in sign (section 5.4).

### 3.3 The mean baseline is a conservation-law predictor

For periodic Burgers, `d/dt int u dx = -[u^2/2] + nu[u_x] = 0`, so the spatial mean of the
target equals that of the input. Measured on the stored data:

| Split | corr(mean(x), mean(y)) | mean abs difference | mean abs target mean |
|---|---:|---:|---:|
| train | 1.000000 | 3.179e-05 | 4.959e-01 |
| validation | 1.000000 | 2.816e-05 | 5.097e-01 |
| test | 1.000000 | 2.977e-05 | 5.012e-01 |

Predicting the exact input mean and zero fluctuation therefore yields 0.37577 test mean
relative L2, with a fluctuation share of 0.22002 and a mean share of 0. The reported
"input spatial-mean baseline" is not a weak strawman: it is the best possible
conservation-law predictor, and matching it is a real, nontrivial requirement.

## 4. Data, controls and sequential study design

We use PDEBench's 1D Burgers Nu0.001 file, verified by SHA-256, with 10000 trajectories,
201 stored time layers and 1024 cell-centered spatial points. The primary task maps t=0 to
t=1.0 (index 100), with stride-four point sampling to 256 points and no anti-alias filter.
The generator uses epsilon/pi as the diffusion coefficient.

The initial split is 1000 training, 200 validation and 200 test trajectories, sampled with
split seed 20260922. Input/target scalar normalization uses training arrays only. DeepONet
uses 64 trunk features and three-layer, width-128 branch/trunk MLPs; the alternative trunk
uses fixed random Fourier input features. The FNO reference has four blocks, width 32 and
16 spectral modes. All training seeds are 0-4.

| Family | Run records | Main intervention | Evaluation status |
|---|---:|---|---|
| public_fixed_t1_v1 | 15 | MLP/Fourier DeepONet and FNO, 1500 updates | Train/validation/test; exploratory subset |
| public_coefficients_v1 | 40 | Conditioning x mean offset, 3000 branch updates | Same exploratory split |
| public_smallfit_v1 | 60 | n=32/128/1000 x random/zero readout | Train/validation only |
| public_stability_v1 | 40 | Constant/cosine LR x batch 32/128, n=128 | Train/validation only |
| public_steps5k_v1 | 15 | Same three architectures, 5000 updates | Train/validation/test |
| public_steps20k_v1 | 15 | Same three architectures, 20000 updates | Train/validation/test |
| public_horizon_t025_v1 | 15 | Target time 0.25, 1500 updates | Train/validation/test |
| public_horizon_t050_v1 | 15 | Target time 0.50, 1500 updates | Train/validation/test |

Horizon variants are prepared by calling `src.public_benchmark.prepare` with
`time_index` 25 and 50 and the **same split seed and sizes**, so the trajectories are
identical and only the target time changes. This is a sensitivity check over forecast
horizon, **not a second PDE**.

These families were designed sequentially after inspecting preceding results. Protocols
were fixed before their respective launches, not preregistered as one independent
confirmatory study. Ten stability controls reproduce earlier results exactly and are not
independent replicates. The test split was exposed during early exploratory development.

Every number in sections 5.1-5.5 was recomputed from stored checkpoints by
`scripts/diagnose_fluctuation.py`, which compares each recomputed `mean_relative_l2`
against the value stored in that run's `result.json`. All recomputations reproduce their
stored metrics exactly (`max_abs_metric_difference = 0.00e+00` in every family), so the
decomposition is applied to a faithful reconstruction rather than an approximation.

## 5. Results

### 5.1 Representability is not operator accuracy

| Initial method | Test mean relative L2 (mean ± seed SD) |
|---|---:|
| Input spatial-mean baseline | 0.37577 |
| MLP DeepONet | 0.39357 ± 0.01014 |
| Fourier DeepONet | 0.41231 ± 0.01344 |
| FNO | 0.22835 ± 0.00680 |

Both initial DeepONets fail to beat the conservation-law baseline while FNO beats it by a
wide margin. Equal update counts do not match parameter counts or compute (section 5.6).

### 5.2 Fluctuation skill at t = 1.0

| Method | Test relL2 | R^2 (energy) | **R^2 (per sample)** | mean share | fluctuation share |
|---|---:|---:|---:|---:|---:|
| oracle input mean | 0.37577 | 0.0000 | **0.0000** | 0.0000 | 0.2200 |
| MLP DeepONet | 0.39357 | -0.0056 ± 0.0094 | **-0.0042 ± 0.0072** | 0.0180 | 0.2209 |
| Fourier DeepONet | 0.41231 | -0.0124 ± 0.0356 | **-0.0934 ± 0.0886** | 0.0340 | 0.2406 |
| FNO | 0.22835 | +0.6282 ± 0.0187 | **+0.6503 ± 0.0222** | 0.0023 | 0.0769 |

MLP DeepONet's fluctuation share (0.2209) is indistinguishable from the zero-fluctuation
reference (0.2200): at 1500 updates it predicts the fluctuation no better than not
predicting it at all, and its remaining error over the oracle comes from the mean
component. FNO reduces the fluctuation share by 65% and its mean share by a further order
of magnitude.

### 5.3 No intervention produces positive fluctuation skill

`scripts/diagnose_fluctuation.py` recomputed all 175 stored runs of the original four
families plus the 5k and 20k families. Per-sample fluctuation R^2 by family (mean ± SD
over the runs in that family):

| Family | Split | MLP DeepONet | Fourier DeepONet |
|---|---|---:|---:|
| public_fixed_t1_v1 (1500) | test | -0.0042 ± 0.0072 | -0.0934 ± 0.0886 |
| public_coefficients_v1 (3000) | test | -0.1286 ± 0.1054 | -0.3030 ± 0.1558 |
| public_stability_v1 (3000) | validation | -0.6438 ± 0.2273 | -1.4062 ± 0.1736 |
| public_scale1000_v1 (3000) | validation | -0.0354 ± 0.0556 | -0.3181 ± 0.0895 |
| public_smallfit_v1 (3000) | validation | -2.2911 ± 4.0722 | -5.5548 ± 8.7244 |

Not one configuration, across trunk architecture, whitening, mean lifting, zero
initialization, schedule, batch size and training-subset size, scores above the
zero-fluctuation reference. Interventions that improve fitting score worse than the
end-to-end baselines, which is the dissociation reported in earlier drafts, now measured on
the component that causes it.

### 5.4 More updates do not help, and hurt the per-sample measure

| Budget | Method | Test relL2 | R^2 (energy) | **R^2 (per sample)** |
|---|---|---:|---:|---:|
| 1500 | MLP DeepONet | 0.39357 | -0.0056 ± 0.0094 | **-0.0042 ± 0.0072** |
| 1500 | Fourier DeepONet | 0.41231 | -0.0124 ± 0.0356 | **-0.0934 ± 0.0886** |
| 1500 | FNO | 0.22835 | +0.6282 ± 0.0187 | **+0.6503 ± 0.0222** |
| 5000 | MLP DeepONet | 0.39729 | +0.0165 ± 0.0159 | **-0.0640 ± 0.0503** |
| 5000 | Fourier DeepONet | 0.40250 | +0.0175 ± 0.0450 | **-0.1272 ± 0.0787** |
| 5000 | FNO | 0.21619 | +0.5864 ± 0.0166 | **+0.6830 ± 0.0325** |
| 20000 | MLP DeepONet | 0.39311 | +0.2126 ± 0.0557 | **-0.2470 ± 0.1304** |
| 20000 | Fourier DeepONet | 0.47689 | -0.1623 ± 0.0609 | **-0.8037 ± 0.2393** |
| 20000 | FNO | 0.22243 | +0.5538 ± 0.0139 | **+0.6660 ± 0.0170** |

Two readings, and the second is the one that matters:

1. **Undertraining is not the explanation.** Total relative L2 is flat across a 13x budget
   range: 0.39357 / 0.39729 / 0.39311 for MLP DeepONet and 0.22835 / 0.21619 / 0.22243 for
   FNO. The ranking never changes.
2. **The energy-weighted measure would have produced a false positive.** At 20000 updates
   MLP DeepONet reads +0.2126 energy-weighted but -0.2470 per sample: improvement is
   concentrated in high-energy samples while typical samples get worse. Since the
   manuscript's metric is a per-sample mean, the per-sample measure decides the claim and
   the energy-weighted one is reported only alongside it.

The normalized mean error also grows with budget for MLP DeepONet (0.081 -> 0.076 ->
0.111) while staying at 0.024-0.034 for FNO, so extra updates trade conserved-mean accuracy
for fluctuation fitting with no net gain.

### 5.5 The deficit is horizon-dependent, not absolute

Same 1500-update budget, same trajectories, only the target time changes.

| Horizon | fluctuation energy | MLP DeepONet | Fourier DeepONet | FNO |
|---|---:|---:|---:|---:|
| t = 0.25 | 0.4785 | **+0.1695 ± 0.0154** | **+0.4248 ± 0.0168** | +0.9474 ± 0.0058 |
| t = 0.50 | 0.3580 | **+0.1238 ± 0.0137** | **+0.2510 ± 0.0389** | +0.8369 ± 0.0075 |
| t = 1.00 | 0.2367 | **-0.0042 ± 0.0072** | **-0.0934 ± 0.0886** | +0.6503 ± 0.0222 |

(per-sample R^2; the energy-weighted measure gives the same ordering: +0.2454 / +0.1387 /
-0.0056 for MLP, +0.4915 / +0.3213 / -0.0124 for Fourier, +0.9327 / +0.8149 / +0.6282 for
FNO.)

DeepONet does have fluctuation skill at short horizons, and loses all of it by t=1.00,
while FNO retains 69% of its short-horizon skill. The fluctuation also carries less energy
at longer horizons (0.4785 -> 0.3580 -> 0.2367) even though models do worse on it, so the
difficulty comes from the transport- and shock-dominated character of the long-horizon
problem rather than from the amount of fluctuation. The Fourier trunk helps substantially at
t=0.25 (+0.4248 against +0.1695) and stops helping at t=1.00.

### 5.6 Compute and parameter budget

Recorded optimizer time (excluding evaluation and checkpointing) and parameter counts:

| Model | parameters | ms per update | total s at 1500 | total s at 5000 | total s at 20000 |
|---|---:|---:|---:|---:|---:|
| MLP DeepONet | 115,713 | 3.97-4.25 | 6 | 20 | 85 |
| Fourier DeepONet | 131,969 | 3.67-4.13 | 6 | 18 | 83 |
| FNO | 136,481 | 9.02-9.32 | 14 | 46 | 180 |

FNO costs about 2.2x more wall-clock per update, so an equal-update comparison gives FNO
more compute, not less. Parameter counts are within 18% (Fourier DeepONet has 97% of FNO's
parameters). A compute-favoured comparison is therefore available and goes the other way:
MLP DeepONet trained for 20000 updates (85 s) scores -0.2470 per-sample fluctuation R^2 and
0.39311 relative L2, while FNO trained for 1500 updates (14 s) scores +0.6503 and 0.22835.
DeepONet receives roughly 6x the compute and remains worse by about 0.9 in R^2.

This does not establish a FLOP-matched result: the two architectures use different
operations, and no FLOP-level accounting was performed. It does make clear that the gap is
not explained by DeepONet having less compute or fewer parameters.

### 5.7 Fitting interventions stabilize training without improving validation

| Trunk | LR | Batch | Final train excess | Stable seeds | Validation relative L2 (mean ± SD) |
|---|---|---:|---:|---:|---:|
| MLP | constant | 32 | 0.034534 | 0/5 | 0.44657 ± 0.00981 |
| MLP | constant | 128 | 0.004113 | 0/5 | 0.50846 ± 0.02313 |
| MLP | cosine | 32 | 0.001117 | 5/5 | 0.45708 ± 0.00689 |
| MLP | cosine | 128 | 0.000061 | 5/5 | 0.47377 ± 0.01201 |
| Fourier | constant | 32 | 0.030453 | 0/5 | 0.55232 ± 0.00913 |
| Fourier | constant | 128 | 0.008727 | 1/5 | 0.56900 ± 0.01195 |
| Fourier | cosine | 32 | 0.005602 | 5/5 | 0.58755 ± 0.01125 |
| Fourier | cosine | 128 | 0.000809 | 5/5 | 0.59345 ± 0.01332 |

Cosine decay yields stable final-tail fitting in 20/20 runs; constant learning rates do so
in 0/10 with batch 32 and 1/10 with batch 128. At batch 32 decay nevertheless worsens
validation for every paired seed. Section 5.3 gives the reason these runs can be made to
fit without improving: fitting is measured on the full field, whose error is dominated by
the conserved mean, while the fluctuation component remains unexplained.

![Training fit and validation behavior](../results/public_stability_v1/stability_curves.png)

Full-batch trials process four times as many examples at equal updates. Recorded update
times exclude evaluation and checkpointing; equal update counts are not equal compute.

### 5.8 Zero initialization helps, but fitting and generalization remain distinct

Zero initialization improves final validation relative L2 in 29/30 paired comparisons; all
six group means and seed standard deviations improve. The exception is Fourier at n=1000,
seed 2. Shared inputs and related seeds do not constitute 30 independent datasets, so no
aggregate significance claim is made. The best zero-init validation group, MLP/n=1000 at
0.38981, still trails the validation conservation-law baseline of 0.374509.

## 6. Interpretation and limitations

The supported finding is a **localization**: on this protocol the error of both DeepONet
variants is dominated by a component they do not predict at all, and the component is
identified as the deviation from the conserved spatial mean. FNO predicts it. The
difference is architectural in the sense that it tracks the forecast horizon and is not
removed by 13x more updates, more basis conditioning, better initialization, or a different
schedule.

We do **not** claim that DeepONet cannot learn fluctuations in general: at t=0.25 it scores
+0.17 to +0.42. We do not claim a causal mechanism, and we do not claim FNO is universally
superior.

Limitations:

- One PDE, one viscosity, one split and one solver. The horizon ladder is a sensitivity
  check on that PDE, not a second PDE; the cross-PDE gap identified in
  [the publication assessment](../docs/PUBLICATION_ASSESSMENT.md) is not closed.
- **Compute is not FLOP-matched.** Parameter counts are within 18% and FNO costs about
  2.2x more wall-clock per update, so the gap is not explained by DeepONet having less of
  either (section 5.6). No FLOP-level accounting or FLOP-matched control was run, and the
  two architectures perform different operations per update.
- The test split was exposed during exploratory development and validation guided later
  study choices; these numbers are descriptive and no fresh holdout was used.
- Five seeds share one data split, so seed standard deviations do not measure task
  variability.
- Numerical-rank sensitivity of the projection oracle (rcond 1e-12 versus 1e-6) is reported
  in the retained diagnostic section and affects how basis quality is read.

## 7. Historical correctness audit and withdrawn claims

The project previously found shared-grid helpers called on per-sample coordinates,
producing `(B,B,N)` instead of `(B,N)`. Current sampler tests compare batched evaluation
against an independent per-sample oracle. This is an implementation lesson, not evidence
that other published systems contain the same bug.

We withdraw the old residual assertion as a mathematical test: small value error does not
ensure small derivative error or PDE residual. Earlier claims that test relative L2 minus a
truncated captured-energy proxy is pure optimization error, that alignment has a
problem-independent variance floor, that output-NTK/sharpness correlations establish a
causal mechanism, and that a modal shrinkage rule beats strong controls are not maintained.
The modal-shrinkage line failed its predeclared confirmation (0 of 12 comparisons against
strong controls) and is closed. Historical unknown source fingerprints remain unknown;
historical p-values are not reused as confirmatory evidence. Full records remain archived,
and the endpoint of a failed study was not replaced by a secondary metric.

## 8. Reproducibility and next confirmation

Versioned result directories contain manifests, events, final metrics and summaries.
Dataset SHA-256, exact row identities, code hashes and environment records are retained.
Large downloaded data and resumable checkpoints remain local rather than in Git. A
redistributable release and author/affiliation details are not yet prepared.

Recomputation is enforced rather than assumed: `scripts/diagnose_fluctuation.py` refuses to
proceed if the pair cache hash changed, and reports any run whose recomputed metric differs
from its stored value beyond a tolerance. The test suite asserts the exact squared-error
closure, the invariance of the skill measure to the mean, and that stored metrics are
reproduced.

Next experiments, in priority order: (i) a compute- or FLOP-matched FNO control; (ii) a
second public PDE with a different operator class, prepared under a locked protocol with a
reserved holdout; (iii) applying the same decomposition to a physics-informed training
objective, which this study does not cover.

## References

Primary sources checked 2026-09-24 unless noted; versions are given without guessing later
journal metadata.

1. Sanghyun Lee and Yeonjong Shin. *On the training and generalization of deep operator
   networks*. 2023. https://arxiv.org/abs/2309.01020
2. Emily Williams, Amanda Howard, Brek Meuris and Panos Stinis. *What do physics-informed
   DeepONets learn? Understanding and improving training for scientific computing
   applications*. 2024. https://arxiv.org/abs/2411.18459
3. Lu Lu et al. *A comprehensive and fair comparison of two neural operators (with
   practical extensions) based on FAIR data*. 2021 preprint; journal DOI
   https://doi.org/10.1016/j.cma.2022.114778 ; https://arxiv.org/abs/2111.05512
4. Makoto Takamoto et al. *PDEBench: An Extensive Benchmark for Scientific Machine
   Learning*. NeurIPS Datasets and Benchmarks, 2022. https://arxiv.org/abs/2210.07182
5. Thomas Bachlechner et al. *ReZero is all you need: fast convergence at large depth*.
   UAI / PMLR 161, 2021. https://proceedings.mlr.press/v161/bachlechner21a.html
6. Ilya Loshchilov and Frank Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts*.
   ICLR 2017. https://arxiv.org/abs/1608.03983
7. Nitish Shirish Keskar et al. *On Large-Batch Training for Deep Learning:
   Generalization Gap and Sharp Minima*. ICLR 2017. https://arxiv.org/abs/1609.04836
8. Niccolo Demo, Marco Tezzele and Gianluigi Rozza. *A DeepONet multi-fidelity approach for
   residual learning in reduced order modeling*. 2023. https://arxiv.org/abs/2302.12682
   (checked 2026-09-28)
9. Koike et al. *POD mode selection for sparse neural networks*. 2026.
   https://arxiv.org/abs/2605.27756 (checked 2026-09-28; targets and implementation differ
   from the closed modal-shrinkage line and do not establish priority for it)
