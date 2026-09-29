# Continuous uncertainty-penalized shrinkage — prospective protocol

Active machine protocol: plan_v2.json. Initial plan.json is preserved: its first run
stopped before bootstrap/evaluation because artifact enumeration required an omitted
cv_seeds key. Version2 adds the explicit[0,1] key and a new source hash; no scientific
settings, candidate criteria or results informed this repair. No new targets were opened.

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

## Development results before confirmation

Active source/protocol commit0e3217f. All four frozen-model analyses completed; zero
new neural fits. Each conditional bootstrap2000 replicates is independently replayed.

| Task | All modes | Old continuous | Hierarchical uncertainty | Trajectory only | Matched global |
|---|---:|---:|---:|---:|---:|
| Original Burgers | 0.363277 | 0.362297 | 0.362695 | 0.362492 | 0.362834 |
| Independent Burgers | 0.403154 | 0.402073 | 0.402668 | 0.402410 | 0.402678 |
| Advection | 0.019756122297 | 0.019756122211 | 0.019756123346 | 0.019756122988 | 0.019756123045 |
| Synthetic Allen-Cahn | 0.359048 | 0.359048 | 0.359048 | 0.359048 | 0.359048 |

Main candidates are uncertainty on both Burgers splits and Allen-Cahn; Advection
falls back to none. The new rule is worse than old continuous shrinkage on both Burgers
development sets. Its advantage over matching global attenuation is small. This is
recorded before confirmation; no penalty strength or candidate rule is adjusted.
Old/new nonzero mode counts remain11/11,9/9,13/13,16/16: no modes are newly deleted.
Matched global amplitude factors are0.891607,0.797415,0.404901,1.0. On Allen-Cahn the
box-constrained coefficients stay all1, so every nonlinear arm is an exact duplicate.
No uncertainty advantage can be inferred from those identical predictions.

## Confirmation results — primary endpoint FAILED (2026-09-29)

Selection lock `704d9a85431048700e1bbb3674a7ddccb0a0f2a75d8562b9a1061b68393d273e` was
committed at `bfd52cd`, THEN the four reserved 500-ID sets were opened. Zero new neural
fits; no candidate, penalty strength or rule was changed after confirmation.

| Task | selected | linear (none) | all | old_shrink | **uncertainty** | matched_global |
|---|---|---:|---:|---:|---:|---:|
| Original Burgers | uncertainty | 0.426201 | 0.420848 | **0.419806** | 0.419982 | 0.419670 |
| Independent Burgers | uncertainty | 0.398433 | 0.382781 | **0.382451** | 0.384878 | 0.384784 |
| Advection | none | 0.0178842 | 0.0178841 | 0.0178841 | 0.0178841 | 0.0178841 |
| Synthetic Allen-Cahn | uncertainty | 0.469687 | **0.369091** (all arms identical) | = | = | = |

Per-task comparisons versus the three strong controls (old_shrink, all, matched_global):

| Task | vs old_shrink | vs all | vs matched_global |
|---|---|---|---|
| Original Burgers | -0.042%, interval crosses 0, **fail** | +0.206%, interval crosses 0, **fail** | -0.075%, interval crosses 0, **fail** |
| Independent Burgers | -0.635%, interval [0.001133, 0.003818] above 0, **fail** | -0.548%, interval crosses 0, **fail** | -0.024%, interval crosses 0, **fail** |
| Advection | differences ~1e-7, **fail** | **fail** | **fail** |
| Synthetic Allen-Cahn | exactly 0 (degenerate), **fail** | 0, **fail** | 0, **fail** |

### Reading

1. **0 of the 12 comparisons against strong controls pass.** The only two passing
   comparisons in the 16-comparison family are "versus no correction" on the two Burgers
   splits, which restates an effect already established in earlier rounds; they do not
   support the new rule.
2. **The new uncertainty penalty is worse than the old continuous shrinkage on both
   Burgers splits** (0.419982 vs 0.419806; 0.384878 vs 0.382451). The development-ordering
   observed before confirmation is reproduced on fresh targets rather than reversed.
3. **Allen-Cahn is fully degenerate**: the box-constrained coefficients stay at 1, so all
   five nonlinear arms emit identical predictions. No modal conclusion can be drawn there.
4. **Advection differences are ~1e-7**, i.e. numerically meaningless; it correctly falls
   back to none.
5. Per the predeclared rule, failure against either strong baseline prohibits claiming a
   superior shrinkage method. **This line is closed; the 2000 new targets are consumed.**

### Closure

No sixth shrinkage variant, no new hard threshold and no retuning on these targets. The
masks, models and predictions are preserved for the record. Subsequent work moves to a
different question rather than another arm of the same family.
