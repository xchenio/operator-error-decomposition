# Continuous uncertainty-penalized shrinkage — prospective protocol

2026-09-29. Preserve old continuous shrinkage AND all-mode nonlinear correction as
strong controls. Reuse frozen CV/final models for both Burgers splits, linear Advection
and custom synthetic Allen-Cahn. No new neural fitting and no new hard modal thresholds.

## Estimator and interpretation

Let r_im be the held-fold true coefficient minus its nested-tuned linear prediction,
delta_ism the frozen nonlinear correction, w_i the fold-training protected energy
weight, and p=256. Estimate N_m=mean(w_i*r_im*delta_ism)/p and
D_m=mean(w_i*delta_ism^2)/p across600 held trajectories and two probes per fold.
Old shrinkage is a0=clip(N/D,0,1), reproduced from the previous artifacts.

New shrinkage is a=clip(N/(D+1.96*s_N),0,1). This is the exact minimizer of
sum_m[(D_m+1.96*s_Nm)*a_m^2-2*N_m*a_m] under0<=a<=1. Uncertainty enters as a
quadratic penalty on correction amplitude. This is a soft uncertainty constraint in
the regularization sense, NOT a calibrated high-probability risk guarantee. The1.96
multiplier is a fixed penalty strength, not a claim of95% confidence coverage.
Positive old coefficients remain positive for finite moments; there is no new support
threshold, field-gain cutoff or mode rejection. Negative N still has the inherited
nonnegative box constraint; no sign-flipping correction is allowed.

Estimate s_N with2000 deterministic bootstrap replicates. Main hierarchy: resample
three fitted fold-model blocks, pick one whole seed probe per block, then resample200
held trajectories. A trajectory-only ablation holds folds fixed and averages the two
probes before resampling rows. These conditional sensitivity estimates do not capture
all retraining uncertainty: there are only3 folds/two probes and overlapping training
sets. They must not be advertised as calibrated epistemic uncertainty or safety bounds.
Each replicate has its own deterministic RNG seed; checkpoint every100 replicates.

## Controls and selection

Arms: none, all, old_shrink, uncertainty, trajectory_only, matched_global.
matched_global applies one scalar g to old_shrink so sum(D*(g*a0)^2)=sum(D*a^2)
on CV moments. This distinguishes mode-specific uncertainty from simply reducing
overall correction energy; matching need not hold on new targets. Same parameters,
same weights and same final seeds for every arm. Masking does not reduce network cost.

No strength sweep or development choice between uncertainty estimators. The main
candidate is hierarchical uncertainty, falling back to none only if development gain
over linear is less than1%. Keep all5 final seeds. The old/all baselines remain direct
comparators rather than being silently replaced by a weaker baseline.

Reserve NEW500 IDs per task; exclude all prior public train/dev/test/confirmation and
smoke IDs, as well as this round's other IDs from the same source. Allen-Cahn reserves
1700–2199, excluding train0–999,dev1000–1199,consumed1200–1699 and QA100000–100015.
No new targets loaded/generated until the candidate/model/hash seal is committed.

Four comparisons per task: selected versus linear, old_shrink, all, matched_global.
Sixteen-comparison family:>=1% relative mean error reduction AND paired trajectory
bootstrap interval below0; Bonferroni two-sided99.6875% percentile intervals,10000
resamples,seed2026093019. Average per-trajectory errors over5 seeds, not ensemble
predictions. RMSE and trajectory-only ablation are descriptive, not replacement endpoints.
Failure against either strong baseline prohibits claiming a superior shrinkage method.

Commands with D:/miniconda3/envs/pytorch/python.exe:

```text
scripts/run_uncertainty_modal.py plan
# commit protocol/source before development
scripts/run_uncertainty_modal.py develop
scripts/run_uncertainty_modal.py seal
# commit fixed selection before confirmation
scripts/run_uncertainty_modal.py confirm
```

Old confirmation data stay consumed. The new masks use training-internal predictions,
not past confirmation errors. Repeated adaptations of the overall research direction
remain disclosed; independent new confirmation does not establish global novelty.
