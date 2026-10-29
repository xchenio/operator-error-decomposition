# Mean / fluctuation decomposition of the public fixed-t1 protocol

2026-09-29. Motivation: the manuscript reports that both DeepONet variants fail to beat an
"input spatial-mean baseline" (0.37577) while FNO reaches 0.22835. This document asks a
narrower question than "which model is better": **what part of the target field does each
model actually predict?** The answer changes the baseline from an embarrassment into the
central finding, so it is recorded before any rewrite of the manuscript.

## Definitions

For target fields `Y` and predictions `P` in `R^{m x N}` (m trajectories, N grid points),
write the per-sample spatial mean as `bar(y)_j = mean_i Y_ji` and the fluctuation as
`Y_c = Y - bar(y) 1^T` (similarly `P_c`).

| Quantity | Definition | Reading |
|---|---|---|
| `mean_relative_l2` | `mean_j \|\|p_j - y_j\|\|_2 / \|\|y_j\|\| _2` | already reported everywhere |
| `fluctuation R^2` | `1 - sum\|\|P_c - Y_c\|\|^2 / sum\|\|Y_c\|\|^2` | skill on the non-mean part; 0 = no better than predicting zero fluctuation |
| normalized mean error | `mean_j \|bar(p)_j - bar(y)_j\| / (\|\|y_j\|\|_2 / sqrt(N))` | skill on the conserved mean |

`R^2` is a skill measure relative to a **zero-fluctuation** reference. It is not a causal
attribution and not a decomposition of `mean_relative_l2`; the two are reported separately.

### Two skill measures, because they weight samples differently

| Measure | Definition | Dominated by |
|---|---|---|
| `fluctuation_r2` | `1 - sum_j \|\|p_c-y_c\|\|^2 / sum_j \|\|y_c\|\|^2` (energy-weighted) | high-energy samples |
| `fluctuation_r2_per_sample` | `1 - (fluctuation squared share) / (per-sample fluctuation energy)` | every sample equally |

The manuscript's headline metric is a **per-sample mean** of relative L2, so
`fluctuation_r2_per_sample` is the measure that corresponds to it. The two can disagree in
magnitude and even in sign; both are reported everywhere. By construction the per-sample
version satisfies the exact identity
`fluctuation share = (1 - r2_per_sample) * per-sample fluctuation energy`, whereas the
energy-weighted version does not (a product of means is not a mean of products).

An exact orthogonal split of the squared error is also reported: since the constant vector
and the zero-mean residual are orthogonal,
`||p-y||^2 = N (p_bar - y_bar)^2 + ||p_c - y_c||^2`, giving `squared_error_mean_share +
squared_error_fluctuation_share = mean_j relative_L2_j^2` to machine precision
(`squared_error_closure` is asserted to be 0 in the tests). This split is an identity, not
an attribution of cause.

## Why the mean baseline is not arbitrary

Verified directly on `external_data/pdebench/fixed_t1_v1/pairs.npz` (train split, zero
training):

```
corr(mean(x), mean(y))          = 1.000000
mean |bar(y) - bar(x)|          = 3.18e-05     vs   mean |bar(y)| = 4.96e-01
||Y_c|| / ||Y||                 = 0.2442
```

Periodic Burgers conserves `int u dx`, so the input spatial mean **is** the target spatial
mean to solver accuracy. The previously reported "input spatial-mean baseline" is therefore
a **conservation-law predictor with zero fluctuation content**, not a strawman. The
fluctuation carries only 24.4% of the target energy in the Frobenius sense, so a model can
sit near 0.375 relative L2 purely by matching the conserved mean.

## Results (recomputed from stored checkpoints, no training)

Public family `results/public_fixed_t1_v1`, 1500 updates, 5 seeds, test split. Predictions
were regenerated from `checkpoints/latest.pt` with the training-time normalization
(`input_mean/input_std/target_mean/target_std` from `protocol.json`).

| Predictor | mean relative L2 | **fluctuation R^2** | normalized mean error |
|---|---:|---:|---:|
| oracle input mean (zero fluctuation) | 0.37577 | 0.0000 | 0.0001 |
| MLP DeepONet | 0.39357 +- 0.00907 | **-0.0056 +- 0.0084** | 0.053-0.114 |
| Fourier DeepONet | 0.41231 +- 0.01202 | **-0.0124 +- 0.0319** | 0.085-0.159 |
| FNO (width 32, 16 modes, 4 layers) | 0.22835 +- 0.00608 | **+0.6282 +- 0.0167** | 0.023-0.060 |

Reproduction (read-only, ~10 s on CUDA):

```text
# conservation check
python -c "import numpy as np; z=np.load('external_data/pdebench/fixed_t1_v1/pairs.npz'); \
x,y=z['train_x'],z['train_y']; print(np.corrcoef(x.mean(1),y.mean(1))[0,1], \
np.abs(y.mean(1)-x.mean(1)).mean(), np.linalg.norm(y-y.mean(1,keepdims=True))/np.linalg.norm(y))"

# per-architecture R^2 from stored checkpoints: see scripts/diagnose_fluctuation.py
```

## Interpretation (hypothesis, not yet a confirmed mechanism)

DeepONet on this protocol has **zero or slightly negative fluctuation skill**: its predicted
fluctuation is no better than predicting none, and it does not even reproduce the conserved
mean as accurately as the exact input mean does. FNO, on identical data, split, update
budget and evaluation, explains 63% of the fluctuation variance.

If this survives the step-budget control below, it gives a single explanation for every
negative result already in the manuscript:

| Existing observation | Explanation under this hypothesis |
|---|---|
| Fourier trunk has a better projection oracle but worse predictions | the binding constraint is not the trunk basis |
| zero-init / cosine decay stabilize fitting without improving validation | what stabilizes is the mean fit; the fluctuation is never learned |
| conditioning, POD ranks and modal shrinkage yield 1-3% | they optimize a component carrying 24% of the energy with R^2 ~ 0 |
| every method hugs the mean baseline | every method learned the conservation law |

It also gives a differentiated position relative to Williams et al. (arXiv:2411.18459,
JCP 2026): the trunk basis is not the binding constraint here; the spatial inductive bias
of the input-to-coefficient map is.

## Limitations

1. **Step budget is not controlled.** Every public run uses 1500 updates. `R^2 ~ 0` is not
   yet distinguished from undertraining. This is the single largest threat and is tested
   first (E0).
2. Equal update counts are not equal parameter counts or equal compute
   (MLP DeepONet 115,713 parameters). E3 addresses this.
3. One PDE, one coefficient (PDEBench 1D Burgers Nu0.001), one forecast time, one split.
4. `R^2` is computed on the test split already used for the original exploratory
   comparison; it is a redescription of existing runs, not fresh confirmation.
5. The Allen-Cahn and Advection tasks in later families have different fluctuation
   structure; this diagnostic has not been applied to them.

## E1 results (completed 2026-09-29)

`python scripts/diagnose_fluctuation.py --device cuda` recomputed 175 stored runs across
five families into `results/fluctuation_v1/`. **Every recomputed `mean_relative_l2`
matches its stored `result.json` value exactly (`max_abs_metric_difference = 0.00e+00`
in every family), so the reconstruction is faithful rather than approximate.**

Reference: oracle input-mean predictor has R^2 = 0 by construction, relL2 0.37451 (val)
and 0.37577 (test).

| Family | Split | Grouping | relL2 (mean +- sd) | **fluctuation R^2** |
|---|---|---|---|---|
| fixed_t1 | test | MLP DeepONet | 0.39357 +- 0.00907 | **-0.0056 +- 0.0084** |
| fixed_t1 | test | Fourier DeepONet | 0.41231 +- 0.01202 | **-0.0124 +- 0.0319** |
| fixed_t1 | test | **FNO** | **0.22835 +- 0.00608** | **+0.6282 +- 0.0167** |
| coefficients | test | 8 arms (arch x whiten x mean_lift) | 0.41594-0.43832 | **-0.0340 to -0.1089** |
| stability | val | 8 arms (arch x schedule x batch) | 0.44657-0.59345 | **-0.3767 to -1.2359** |
| scale1000 | val | 4 arms (arch x schedule) | 0.37481-0.43101 | **-0.0009 to -0.2386** |
| smallfit | val | 12 arms (arch x n_train x zero_init) | 0.38981-1.79116 | **-0.0400 to -15.3003** |

### Reading

1. **Every DeepONet arm in all 175 runs has R^2 <= 0.** Not one configuration, out of
   trunk architecture, whitening, mean lifting, zero initialization, constant versus
   cosine schedule, batch size 32/128 and training subsets of 32/128/1000 functions,
   produces positive fluctuation skill.
2. **The ceiling is zero, and more training pushes below it.** The end-to-end model at
   1500 updates sits at R^2 ~ 0 (-0.0056 / -0.0124); every family that trains the branch
   harder (3000 updates) is strictly negative, down to -1.24 for the stability family and
   -15.3 for n=32 without zero initialization. Interventions that improve fitting
   therefore *worsen* fluctuation skill, which is precisely the dissociation the
   manuscript reports.
3. **The best DeepONet anywhere in the project is MLP / n=1000 / cosine at
   R^2 = -0.0009 +- 0.0377 with relL2 0.37481**, against the oracle conservation-law
   predictor at 0.37451. The best configuration is statistically indistinguishable from
   predicting the conserved mean and nothing else.
4. **FNO is the only positive arm**, +0.6282 +- 0.0167, on identical data, split and
   1500-update budget. The two DeepONet variants and FNO are separated by more than
   0.62 in R^2 with no overlap in their seed distributions.
5. Whitening and mean lifting do not move R^2 (range -0.034 to -0.109 across all eight
   arms), so this is not a conditioning artifact of the frozen trunk basis.

## E0 results (2026-09-29): more updates do not buy fluctuation skill

`results/public_steps5k_v1` and `results/public_steps20k_v1` (15 runs each, same 3
architectures and 5 seeds) test whether the near-zero skill is undertraining. All
recomputations reproduce their stored metrics exactly.

| Budget | Model | relL2 | R^2 (energy) | **R^2 (per sample)** |
|---|---|---:|---:|---:|
| 1500 | MLP DeepONet | 0.39357 | -0.0056 +- 0.0084 | **-0.0042 +- 0.0064** |
| 1500 | Fourier DeepONet | 0.41231 | -0.0124 +- 0.0319 | **-0.0934 +- 0.0792** |
| 1500 | FNO | 0.22835 | +0.6282 +- 0.0167 | **+0.6503 +- 0.0199** |
| 5000 | MLP DeepONet | 0.39729 | +0.0165 +- 0.0142 | **-0.0640 +- 0.0450** |
| 5000 | Fourier DeepONet | 0.40250 | +0.0175 +- 0.0403 | **-0.1272 +- 0.0704** |
| 5000 | FNO | 0.21619 | +0.5864 +- 0.0148 | **+0.6830 +- 0.0291** |
| 20000 | MLP DeepONet | 0.39311 | +0.2126 +- 0.0498 | **-0.2470 +- 0.1166** |
| 20000 | Fourier DeepONet | 0.47689 | -0.1623 +- 0.0545 | **-0.8037 +- 0.2140** |
| 20000 | FNO | 0.22243 | +0.5538 +- 0.0125 | **+0.6660 +- 0.0152** |

### Reading

1. **Undertraining is not the explanation, and more training is actively worse on the
   per-sample measure.** MLP DeepONet goes -0.0042 -> -0.0640 -> -0.2470 and Fourier
   -0.0934 -> -0.1272 -> -0.8037 as the budget grows 13x. FNO is flat at +0.65 to +0.68.
2. **The energy-weighted measure can look like progress where the per-sample measure does
   not.** At 20000 updates MLP shows +0.2126 energy-weighted but -0.2470 per sample: the
   improvement is concentrated on high-energy samples while typical samples get worse.
   Reporting only the energy-weighted number would have produced a false positive. This is
   why both are reported and why the manuscript's per-sample metric decides the claim.
3. **Total relative L2 is flat across the whole ladder** (0.39357 / 0.39729 / 0.39311 for
   MLP; 0.22835 / 0.21619 / 0.22243 for FNO). The 13x budget therefore changes the ranking
   not at all: DeepONet stays at ~0.39, FNO at ~0.22.
4. The normalized mean error grows with budget for MLP DeepONet (0.081 -> 0.076 -> 0.111)
   while it stays at 0.024-0.034 for FNO, so extra updates trade conserved-mean accuracy
   for fluctuation fitting without net gain.

## E2 results (completed 2026-09-29): the failure is horizon-dependent, not absolute

A viscosity ladder needs additional multi-gigabyte downloads, so the difficulty ladder was
run over the **forecast horizon** instead, which the existing 201-layer file already
supports and which `the publication assessment` explicitly allows as a sensitivity check
(while warning that another time point is not a second PDE). `src.public_benchmark.prepare`
was called with `time_index` 25 and 50 and the same split seed 20260922 and sizes
(1000,200,200), so the trajectories are identical and only the target time changes.
Each horizon uses the same 1500-update budget, 5 seeds and 3 architectures as the original
family; `run_public_benchmark.py --data-dir` selects the cache.

Energy-weighted R^2:

| Horizon | fluctuation energy | MLP DeepONet | Fourier DeepONet | FNO |
|---|---:|---:|---:|---:|
| t = 0.25 | 0.4785 | **+0.2454 +- 0.0190** | **+0.4915 +- 0.0116** | +0.9327 +- 0.0075 |
| t = 0.50 | 0.3580 | **+0.1387 +- 0.0134** | **+0.3213 +- 0.0238** | +0.8149 +- 0.0114 |
| t = 1.00 | 0.2367 | **-0.0056 +- 0.0084** | **-0.0124 +- 0.0319** | +0.6282 +- 0.0167 |

Per-sample R^2 (the measure matching the manuscript metric):

| Horizon | MLP DeepONet | Fourier DeepONet | FNO |
|---|---:|---:|---:|
| t = 0.25 | +0.1695 +- 0.0138 | +0.4248 +- 0.0150 | +0.9474 +- 0.0052 |
| t = 0.50 | +0.1238 +- 0.0123 | +0.2510 +- 0.0348 | +0.8369 +- 0.0067 |
| t = 1.00 | -0.0042 +- 0.0064 | -0.0934 +- 0.0792 | +0.6503 +- 0.0199 |

Both measures agree in direction on this ladder, so the horizon conclusion does not depend
on the choice of weighting.

All recomputations again reproduce their stored metrics exactly (`max_abs_metric_difference
= 0.00e+00`).

### Reading

1. **DeepONet does have fluctuation skill — it decays to zero with horizon.**
   +0.245 / +0.492 at t=0.25, +0.139 / +0.321 at t=0.50, and about zero at t=1.00.
   "DeepONet cannot learn the fluctuation" is therefore wrong as an absolute statement;
   the correct statement is horizon-dependent.
2. **The decay is much steeper for DeepONet than for FNO.** Over the same horizons FNO
   goes 0.933 -> 0.815 -> 0.628 (retains 67% of its t=0.25 skill) while MLP DeepONet goes
   0.245 -> 0.139 -> -0.006 (loses essentially all of it).
3. **The fluctuation carries less energy at longer horizons** (0.4785 -> 0.3580 -> 0.2367),
   yet models do worse on it. The difficulty is therefore not the amount of fluctuation but
   its character: the task becomes transport- and shock-dominated, which is exactly where a
   spectral-convolutional architecture has an inductive advantage.
4. **The Fourier trunk helps at short horizons and stops helping at t=1.0**
   (+0.492 vs +0.245 at t=0.25; -0.012 vs -0.006 at t=1.00). Any claim about trunk features
   must therefore state the horizon.

### Caveats on this section

- A horizon ladder is a sensitivity check on one PDE, **not a second PDE**. The
  cross-PDE gap identified in the publication assessment is not closed by this experiment.
- The same 1000/200/200 trajectories are reused, so the three rows are paired but not
  independent samples of a PDE distribution.
- FNO and DeepONet differ in per-update cost even at matched update counts; parameter
  counts are close (below) but not identical.

## E3 partial result: parameter counts are close, so the gap is not a capacity artifact

| Model | parameters |
|---|---:|
| MLP DeepONet | 115,713 |
| Fourier DeepONet | 131,969 |
| FNO (width 32, 16 modes, 4 layers) | 136,481 |

The Fourier trunk DeepONet has 97% of FNO's parameter count (131,969 vs 136,481) yet scores
R^2 = -0.0124 against FNO's +0.6282 at t=1.0 on identical data, budget and evaluation. The
gap is therefore not explained by parameter count. It is **not** established that the gap
survives a compute-matched comparison, since FNO and DeepONet have different per-update
cost; that remains an explicit limitation rather than a claim.

### Caveats on this section

- R^2 is invariant to the mean by construction; the exact-mean predictor and the
  all-zero predictor both score 0. It measures fluctuation skill only.
- Branch families are evaluated on validation only (they never loaded the test split);
  the coefficients family has both and agrees.
- `smallfit` uses n=32/128 subsets drawn from the same 1000 functions whose trunks were
  already supervised, so those runs are a branch-fit diagnostic, not few-shot learning.
- E1 is a redescription of completed runs. It does not by itself settle E0.

Per-sample R^2 for the same families (all negative, confirming the energy-weighted reading):

| Family | MLP DeepONet | Fourier DeepONet |
|---|---:|---:|
| fixed_t1 | -0.0042 +- 0.0064 | -0.0934 +- 0.0792 |
| coefficients | -0.1286 +- 0.1054 | -0.3030 +- 0.1558 |
| stability | -0.6438 +- 0.2273 | -1.4062 +- 0.1736 |
| scale1000 | -0.0354 +- 0.0556 | -0.3181 +- 0.0895 |
| smallfit | -2.2911 +- 4.0722 | -5.5548 +- 8.7244 |

## Predeclared next experiments

| ID | Question | Design | Pass criterion |
|---|---|---|---|
| **E0** | is `R^2 ~ 0` undertraining? | `run_public_benchmark.py` at 5k / 20k / 50k updates, 3 architectures x 5 seeds, fresh `--out` dirs | report the relL2 and R^2 curves; no threshold claim, direction is the result |
| ~~**E1**~~ | ~~does any prior intervention move R^2?~~ | **completed**: 175 runs, every DeepONet arm R^2 <= 0, FNO +0.6282 | hypothesis supported: all DeepONet arms stay at or below zero |
| **E2** | is this specific to Nu0.001? | repeat on PDEBench 1D Burgers Nu0.1 and Nu0.01 with a separate prepared split | hypothesis supported if DeepONet R^2 rises as viscosity rises |
| **E3** | is the FNO gap a parameter/compute artifact? | parameter- or FLOP-matched FNO and DeepONet at the E0 budget | hypothesis supported if the gap persists at matched budget |

E0 and E1 are cheap: 1500 updates take about 6 s on this machine
(`train_seconds` recorded in each `result.json`), so a full ladder is minutes, not hours.

No manuscript claim about an architectural bottleneck is written until E0 completes.
