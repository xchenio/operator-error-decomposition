# Frozen ridge with a regularized nonlinear residual: fresh confirmation

**2026-09-28 addendum.** The study now adds40 residual-training records and8 deterministic
ridge refits. Across the public-study families this yields245 neural records (including
20 historical repeated controls) and26 ridge-fit records. Counts are not independent
experimental units; six new ridge fits repeat old rank16 alpha settings.

## Protocol and separation

We freeze the previous rank16/alpha10 POD ridge model and add a zero-output one-hidden-layer
tanh correction in coefficient space. Width16/32 and correction-energy penalties0/0.1/1/10
give eight settings, each with five seeds. AdamW uses fixed weight decay1e-4, batch32 and
a maximum3000-step cosine schedule. Development mean relative L2 controls early stopping
(patience5 at100-step evaluations, min_delta1e-4, minimum500 steps); step0 is eligible.

Only the original1000 training and200 development trajectories enter fitting or selection.
The expanded linear comparison tunes eight alphas from1e-4 to1000 at the identical POD basis.
Equal numbers of hyperparameter settings do not equalize tuning cost: the nonlinear
procedure also selects checkpoints and trains five seeds.

Five hundred fresh IDs were reserved before training, excluding every historical split
and a conservative smoke-test prefix. Selection was committed as8881a9c before any
confirmation targets were extracted. The software checks this gate. No models or
selection rules changed after that commit or after viewing confirmation results.

## Results

The selected residual has width32 and correction penalty1. Its five checkpoint steps are
2200/1400/2300/2300/1100. Across the whole grid17/40 runs retained step0. The independent
audit verifies exact frozen parameters and replays every early-stopping decision.

| Method | Development relative L2 | Fresh confirmation relative L2 | Fresh confirmation RMSE |
|---|---:|---:|---:|
| Frozen ridge (alpha10) | 0.373656 | 0.405354 | 0.150931 |
| Expanded ridge (alpha100) | 0.373600 | 0.404518 | 0.150910 |
| Ridge + selected residual | 0.366400 | 0.405135 ± 0.001893 | 0.147701 ± 0.000916 |
| Initial spatial mean | 0.374509 | 0.404373 | 0.151370 |

Values average separate models' metrics, not ensemble predictions. Standard deviations
are across optimizer seeds. The primary endpoint averages five per-trajectory relative
errors, with paired bootstrapping over500 trajectories (5000 resamples). This is not
2500 independent observations.

The primary residual-minus-frozen-ridge difference is-0.000219, with95%
percentile interval [-0.007498, 0.008923]. Relative improvement is only
0.0540%. Thus the prespecified requirement of interval upper<0 and at least1% relative
improvement is not met. Expanded ridge and mean baseline are also slightly better in
confirmation mean relative L2. Failure to confirm superiority is not an equivalence test.

Secondary RMSE decreases in all five seeds against frozen and expanded ridge, with a
2.14% reduction in mean RMSE against frozen ridge. This is a useful metric-specific observation,
not grounds to replace the failed primary endpoint. Absolute squared error and average
per-trajectory relative error weight the data differently; identifying the cause of the
disagreement requires separate diagnostics rather than an asserted low-energy mechanism.

## Interpretation and novelty boundary

Regularization and development early stopping can preserve a baseline during model
selection and produce a promising development score, without establishing reliable
primary gains on fresh trajectories. Baseline-inclusive selection is not a guarantee
against unseen-data degradation. This extends the main diagnostic finding; it does
not establish a universally successful repair or a new state-of-the-art operator.

POD/DeepONet residual learning is already studied by
[Demo, Tezzele and Rozza](https://arxiv.org/abs/2302.12682), and zero-initialized residual
gating by [Bachlechner et al.](https://proceedings.mlr.press/v161/bachlechner21a.html).
Our coefficient correction is not identical to either architecture, but their existence
rules out a broad first-residual-method claim. These are targeted references, not an
exhaustive novelty review.

Future work should first diagnose training-loss/evaluation alignment on development
data, then test cross-fitted modal learnability-based shrinkage against ordinary
regularization under matched budgets and independent training splits/PDEs. This remains
a proposed research direction, not an implemented or established contribution. The
500 confirmation trajectories are now consumed; adaptive follow-ups require another
prospectively reserved set. Conditional trajectory intervals do not measure
training-split uncertainty or cross-PDE generalization.

See [the full protocol, grid and research analysis](../docs/RIDGE_RESIDUAL_V1.md).
