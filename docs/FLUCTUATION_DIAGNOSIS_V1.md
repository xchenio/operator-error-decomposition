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

## Predeclared next experiments

| ID | Question | Design | Pass criterion |
|---|---|---|---|
| **E0** | is `R^2 ~ 0` undertraining? | `run_public_benchmark.py` at 5k / 20k / 50k updates, 3 architectures x 5 seeds, fresh `--out` dirs | report the relL2 and R^2 curves; no threshold claim, direction is the result |
| **E1** | does any prior intervention move R^2? | recompute R^2 for all stored checkpoints in `public_fixed_t1_v1`, `public_coefficients_v1`, `public_smallfit_v1`, `public_stability_v1`, `public_scale1000_v1` | report R^2 per family; hypothesis supported if all DeepONet arms stay within +-0.05 of zero |
| **E2** | is this specific to Nu0.001? | repeat on PDEBench 1D Burgers Nu0.1 and Nu0.01 with a separate prepared split | hypothesis supported if DeepONet R^2 rises as viscosity rises |
| **E3** | is the FNO gap a parameter/compute artifact? | parameter- or FLOP-matched FNO and DeepONet at the E0 budget | hypothesis supported if the gap persists at matched budget |

E0 and E1 are cheap: 1500 updates take about 6 s on this machine
(`train_seconds` recorded in each `result.json`), so a full ladder is minutes, not hours.

No manuscript claim about an architectural bottleneck is written until E0 completes.
